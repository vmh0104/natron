"""
Natron Phase 3: Reinforcement Learning (PPO)
Author: Natron AI System
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
import gym
from gym import spaces
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.natron_transformer import NatronTransformer


class TradingEnvironment(gym.Env):
    """
    Custom trading environment for Natron RL.
    """
    
    def __init__(self, data, features, initial_balance=10000):
        """
        Args:
            data: OHLCV DataFrame
            features: Normalized features array
            initial_balance: Starting capital
        """
        super().__init__()
        
        self.data = data
        self.features = features
        self.initial_balance = initial_balance
        self.sequence_length = 96
        
        # Action space: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: sequence of features
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.sequence_length, features.shape[1]),
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        """Reset environment"""
        self.current_step = self.sequence_length
        self.balance = self.initial_balance
        self.position = 0  # 0=no position, 1=long, -1=short
        self.entry_price = 0
        self.total_profit = 0
        self.trades = 0
        self.winning_trades = 0
        self.max_balance = self.initial_balance
        self.max_drawdown = 0
        
        return self._get_observation()
    
    def _get_observation(self):
        """Get current market state"""
        start = self.current_step - self.sequence_length
        end = self.current_step
        obs = self.features[start:end]
        return obs
    
    def _calculate_reward(self, action, price_change):
        """
        Calculate reward based on action and market movement.
        
        Reward components:
        - Profit/loss from trade
        - Penalty for excessive trading (turnover)
        - Penalty for drawdown
        """
        reward = 0
        turnover_penalty = 0
        drawdown_penalty = 0
        
        current_price = self.data.iloc[self.current_step]['close']
        
        # Action: 0=hold, 1=buy, 2=sell
        if action == 1 and self.position == 0:  # Open long
            self.position = 1
            self.entry_price = current_price
            self.trades += 1
            turnover_penalty = -0.001  # Small penalty for trading
            
        elif action == 2 and self.position == 0:  # Open short
            self.position = -1
            self.entry_price = current_price
            self.trades += 1
            turnover_penalty = -0.001
            
        elif action == 2 and self.position == 1:  # Close long
            profit = (current_price - self.entry_price) / self.entry_price
            reward = profit * 100  # Scale reward
            self.total_profit += profit
            self.balance += self.balance * profit
            
            if profit > 0:
                self.winning_trades += 1
            
            self.position = 0
            self.entry_price = 0
            
        elif action == 1 and self.position == -1:  # Close short
            profit = (self.entry_price - current_price) / self.entry_price
            reward = profit * 100
            self.total_profit += profit
            self.balance += self.balance * profit
            
            if profit > 0:
                self.winning_trades += 1
            
            self.position = 0
            self.entry_price = 0
        
        # Calculate drawdown
        if self.balance > self.max_balance:
            self.max_balance = self.balance
        
        drawdown = (self.max_balance - self.balance) / self.max_balance
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        if drawdown > 0.1:  # Penalty for >10% drawdown
            drawdown_penalty = -drawdown * 10
        
        # Unrealized P&L for open positions
        if self.position == 1:  # Long
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            reward += unrealized_pnl * 0.1  # Small reward for floating profit
        elif self.position == -1:  # Short
            unrealized_pnl = (self.entry_price - current_price) / self.entry_price
            reward += unrealized_pnl * 0.1
        
        # Total reward
        total_reward = reward + turnover_penalty + drawdown_penalty
        
        return total_reward
    
    def step(self, action):
        """Execute one step"""
        # Get price change
        current_price = self.data.iloc[self.current_step]['close']
        next_price = self.data.iloc[self.current_step + 1]['close'] if self.current_step + 1 < len(self.data) else current_price
        price_change = (next_price - current_price) / current_price
        
        # Calculate reward
        reward = self._calculate_reward(action, price_change)
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= len(self.data) - 1
        
        # Get next observation
        obs = self._get_observation() if not done else np.zeros_like(self._get_observation())
        
        # Info
        info = {
            'balance': self.balance,
            'position': self.position,
            'total_profit': self.total_profit,
            'trades': self.trades,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.winning_trades / self.trades if self.trades > 0 else 0
        }
        
        return obs, reward, done, info


class PPOAgent:
    """
    Proximal Policy Optimization agent for trading.
    """
    
    def __init__(self, 
                 model: NatronTransformer,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 device: str = 'cuda'):
        """
        Args:
            model: Pretrained Natron model
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_epsilon: PPO clip parameter
        """
        self.model = model
        self.device = device
        
        # Add policy and value heads to model
        self.action_head = nn.Sequential(
            nn.Linear(model.d_model, 3),
            nn.Softmax(dim=-1)
        ).to(device)
        
        self.value_head = nn.Linear(model.d_model, 1).to(device)
        
        # PPO parameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        
        # Optimizer for policy and value heads only (keep transformer frozen or fine-tune)
        self.optimizer = optim.Adam(
            list(self.action_head.parameters()) + 
            list(self.value_head.parameters()),
            lr=lr
        )
        
        # Memory
        self.memory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
    
    def get_action(self, state):
        """
        Get action from policy.
        
        Args:
            state: Current state (96, features)
            
        Returns:
            action, log_prob, value
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get embeddings from transformer
            outputs = self.model(state_tensor, return_embeddings=True)
            embeddings = outputs['embeddings']
            
            # Get action probabilities
            action_probs = self.action_head(embeddings)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            
            # Get value
            value = self.value_head(embeddings)
            
        return action.item(), action_dist.log_prob(action).item(), value.item()
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        """Store transition in memory"""
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['rewards'].append(reward)
        self.memory['values'].append(value)
        self.memory['log_probs'].append(log_prob)
        self.memory['dones'].append(done)
    
    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        return advantages, returns
    
    def update(self):
        """Update policy using PPO"""
        # Convert memory to tensors
        states = torch.FloatTensor(np.array(self.memory['states'])).to(self.device)
        actions = torch.LongTensor(self.memory['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(self.memory['log_probs']).to(self.device)
        
        # Compute GAE
        advantages, returns = self.compute_gae(
            self.memory['rewards'],
            self.memory['values'],
            self.memory['dones']
        )
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(10):  # Multiple epochs
            # Get current policy
            outputs = self.model(states, return_embeddings=True)
            embeddings = outputs['embeddings']
            
            action_probs = self.action_head(embeddings)
            action_dist = torch.distributions.Categorical(action_probs)
            new_log_probs = action_dist.log_prob(actions)
            
            values = self.value_head(embeddings).squeeze()
            
            # Policy loss with clipping
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(values, returns)
            
            # Entropy bonus (for exploration)
            entropy = action_dist.entropy().mean()
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.action_head.parameters()) + list(self.value_head.parameters()),
                max_norm=0.5
            )
            self.optimizer.step()
        
        # Clear memory
        self.memory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }


def train_rl(config_path: str = "config/config.yaml"):
    """Main RL training function"""
    print("=" * 80)
    print("🧠 NATRON TRANSFORMER - PHASE 3: REINFORCEMENT LEARNING (PPO)")
    print("=" * 80)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    rl_config = config['reinforcement']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    from src.data.dataset_loader import NatronDataModule
    data_module = NatronDataModule(config_path=config_path)
    data_module.load_data()
    data_module.generate_features()
    
    # Create environment
    env = TradingEnvironment(
        data=data_module.raw_data,
        features=data_module.features.values
    )
    
    # Load pretrained model
    print("📥 Loading supervised model...")
    model = NatronTransformer(
        num_features=data_module.features.shape[1],
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_encoder_layers=config['model']['num_encoder_layers']
    ).to(device)
    
    checkpoint_path = config['supervised']['checkpoint_path'].replace('.pt', '_best.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded supervised model")
    
    # Freeze transformer (only train RL heads)
    model.freeze_encoder()
    
    # Create PPO agent
    agent = PPOAgent(
        model=model,
        lr=rl_config['learning_rate'],
        gamma=rl_config['gamma'],
        gae_lambda=rl_config['gae_lambda'],
        clip_epsilon=rl_config['clip_epsilon'],
        device=device
    )
    
    # Tensorboard
    writer = SummaryWriter(log_dir=config['logging']['tensorboard_dir'] + '/rl')
    
    # Training loop
    print(f"\n🚀 Starting RL Training...")
    print(f"   Episodes: {rl_config['episodes']}")
    
    best_reward = -float('inf')
    
    for episode in range(rl_config['episodes']):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Get action
            action, log_prob, value = agent.get_action(state)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, value, log_prob, done)
            
            episode_reward += reward
            state = next_state
        
        # Update policy
        update_info = agent.update()
        
        # Log metrics
        writer.add_scalar('rl/episode_reward', episode_reward, episode)
        writer.add_scalar('rl/balance', info['balance'], episode)
        writer.add_scalar('rl/total_profit', info['total_profit'], episode)
        writer.add_scalar('rl/trades', info['trades'], episode)
        writer.add_scalar('rl/win_rate', info['win_rate'], episode)
        writer.add_scalar('rl/max_drawdown', info['max_drawdown'], episode)
        
        for key, value in update_info.items():
            writer.add_scalar(f'rl/{key}', value, episode)
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save({
                'episode': episode,
                'model_state_dict': model.state_dict(),
                'action_head_state_dict': agent.action_head.state_dict(),
                'value_head_state_dict': agent.value_head.state_dict(),
                'best_reward': best_reward
            }, rl_config['checkpoint_path'].replace('.pt', '_best.pt'))
        
        if episode % 10 == 0:
            print(f"\n📊 Episode {episode}:")
            print(f"   Reward: {episode_reward:.2f}")
            print(f"   Balance: ${info['balance']:.2f}")
            print(f"   Profit: {info['total_profit']*100:.2f}%")
            print(f"   Trades: {info['trades']}")
            print(f"   Win Rate: {info['win_rate']*100:.1f}%")
            print(f"   Max DD: {info['max_drawdown']*100:.1f}%")
            print(f"   Best Reward: {best_reward:.2f}")
    
    writer.close()
    print("\n✅ RL training completed!")


if __name__ == "__main__":
    train_rl()
