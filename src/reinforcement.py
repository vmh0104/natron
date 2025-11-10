"""
Natron Transformer - Phase 3: Reinforcement Learning
PPO-based optimization for trading performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import os

from model_natron import NatronTransformer


class TradingEnvironment:
    """
    Trading environment for RL training
    Simulates market interactions and computes rewards
    """
    
    def __init__(
        self,
        sequences: np.ndarray,
        prices: np.ndarray,
        alpha_turnover: float = 0.01,
        beta_drawdown: float = 0.05
    ):
        """
        Args:
            sequences: Feature sequences (N, 96, 100)
            prices: Corresponding close prices (N,)
            alpha_turnover: Penalty weight for turnover
            beta_drawdown: Penalty weight for drawdown
        """
        self.sequences = sequences
        self.prices = prices
        self.alpha = alpha_turnover
        self.beta = beta_drawdown
        
        self.n_steps = len(sequences)
        self.current_step = 0
        self.position = 0.0  # -1: short, 0: neutral, 1: long
        self.portfolio_value = 10000.0  # Initial capital
        self.max_portfolio_value = 10000.0
        self.trades = 0
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.position = 0.0
        self.portfolio_value = 10000.0
        self.max_portfolio_value = 10000.0
        self.trades = 0
        return self.sequences[0]
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take action and return next state, reward, done, info
        
        Args:
            action: 0=sell, 1=hold, 2=buy
            
        Returns:
            next_state, reward, done, info
        """
        # Map action to position
        action_to_position = {0: -1.0, 1: 0.0, 2: 1.0}
        new_position = action_to_position[action]
        
        # Compute price change
        if self.current_step < self.n_steps - 1:
            price_change = (
                self.prices[self.current_step + 1] / self.prices[self.current_step] - 1
            )
        else:
            price_change = 0.0
        
        # Compute profit from position
        profit = self.position * price_change * self.portfolio_value
        
        # Update portfolio value
        self.portfolio_value += profit
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
        
        # Compute turnover penalty
        position_change = abs(new_position - self.position)
        turnover_penalty = self.alpha * position_change
        
        # Compute drawdown penalty
        drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        drawdown_penalty = self.beta * drawdown
        
        # Total reward
        reward = profit / 10000.0 - turnover_penalty - drawdown_penalty
        
        # Update state
        if position_change > 0:
            self.trades += 1
        self.position = new_position
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= self.n_steps - 1
        
        # Next state
        if not done:
            next_state = self.sequences[self.current_step]
        else:
            next_state = self.sequences[-1]
        
        # Info
        info = {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'trades': self.trades,
            'profit': profit,
            'drawdown': drawdown
        }
        
        return next_state, reward, done, info
    
    def get_state(self) -> np.ndarray:
        """Get current state"""
        return self.sequences[self.current_step]


class PPOAgent:
    """
    Proximal Policy Optimization agent for trading
    """
    
    def __init__(
        self,
        model: NatronTransformer,
        config: Dict,
        device: str = 'cuda'
    ):
        self.model = model
        self.config = config
        self.device = device
        
        # RL config
        self.rl_config = config['training']['reinforcement']
        self.gamma = self.rl_config['gamma']
        self.gae_lambda = self.rl_config['gae_lambda']
        self.clip_epsilon = self.rl_config['clip_epsilon']
        self.value_coef = self.rl_config['value_coef']
        self.entropy_coef = self.rl_config['entropy_coef']
        
        # Action head (3 actions: sell, hold, buy)
        self.action_head = nn.Sequential(
            nn.Linear(model.d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 3 actions
        ).to(device)
        
        # Value head (state value estimation)
        self.value_head = nn.Sequential(
            nn.Linear(model.d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(device)
        
        # Optimizer
        params = (
            list(self.model.parameters()) +
            list(self.action_head.parameters()) +
            list(self.value_head.parameters())
        )
        self.optimizer = optim.Adam(params, lr=1e-4)
        
        # Replay buffer
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
    
    def get_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float, float]:
        """
        Get action from policy
        
        Args:
            state: Current state (96, 100)
            training: Whether in training mode
            
        Returns:
            action, log_prob, value
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.set_grad_enabled(training):
            # Get model encoding
            encoded = self.model.get_encoder_output(state_tensor)
            pooled = encoded.mean(dim=1)  # (1, d_model)
            
            # Get action logits and value
            action_logits = self.action_head(pooled)
            value = self.value_head(pooled)
            
            # Sample action
            action_probs = torch.softmax(action_logits, dim=-1)
            
            if training:
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            else:
                action = action_probs.argmax(dim=-1)
                log_prob = torch.log(action_probs[0, action])
            
            return action.item(), log_prob.item(), value.item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Store transition in buffer"""
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['values'].append(value)
        self.buffer['log_probs'].append(log_prob)
        self.buffer['dones'].append(done)
    
    def compute_gae(self, next_value: float) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation
        
        Args:
            next_value: Value of next state
            
        Returns:
            advantages, returns
        """
        rewards = self.buffer['rewards']
        values = self.buffer['values'] + [next_value]
        dones = self.buffer['dones']
        
        advantages = []
        returns = []
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return advantages, returns
    
    def update(self):
        """Update policy using PPO"""
        if len(self.buffer['states']) == 0:
            return {}
        
        # Get next state value
        last_state = self.buffer['states'][-1]
        state_tensor = torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            encoded = self.model.get_encoder_output(state_tensor)
            pooled = encoded.mean(dim=1)
            next_value = self.value_head(pooled).item()
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.buffer['states'])).to(self.device)
        actions = torch.LongTensor(self.buffer['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer['log_probs']).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        encoded = self.model.get_encoder_output(states)
        pooled = encoded.mean(dim=1)
        
        action_logits = self.action_head(pooled)
        values = self.value_head(pooled).squeeze(-1)
        
        # Policy loss
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = nn.MSELoss()(values, returns)
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        # Clear buffer
        self.buffer = {k: [] for k in self.buffer}
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': loss.item()
        }


def run_reinforcement_learning(
    model: NatronTransformer,
    sequences: np.ndarray,
    prices: np.ndarray,
    config: Dict,
    device: str = 'cuda'
) -> NatronTransformer:
    """
    Run reinforcement learning training
    
    Args:
        model: Trained Natron model
        sequences: Feature sequences (N, 96, 100)
        prices: Close prices (N,)
        config: Configuration dict
        device: Device
        
    Returns:
        RL-optimized model
    """
    print(f"\n🚀 Starting Reinforcement Learning...")
    
    rl_config = config['training']['reinforcement']
    episodes = rl_config['episodes']
    steps_per_episode = rl_config['steps_per_episode']
    alpha = rl_config['reward']['alpha_turnover']
    beta = rl_config['reward']['beta_drawdown']
    
    # Create environment and agent
    env = TradingEnvironment(sequences, prices, alpha, beta)
    agent = PPOAgent(model, config, device)
    
    # Training loop
    episode_rewards = []
    
    for episode in range(1, episodes + 1):
        state = env.reset()
        episode_reward = 0
        
        for step in range(min(steps_per_episode, env.n_steps)):
            # Get action
            action, log_prob, value = agent.get_action(state, training=True)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, value, log_prob, done)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Update policy
        losses = agent.update()
        
        episode_rewards.append(episode_reward)
        
        # Log
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode}/{episodes}")
            print(f"  Avg Reward: {avg_reward:.4f}")
            print(f"  Portfolio Value: ${info['portfolio_value']:.2f}")
            print(f"  Trades: {info['trades']}")
            if losses:
                print(f"  Policy Loss: {losses['policy_loss']:.4f}")
    
    print(f"\n✅ Reinforcement learning complete!")
    
    return model


if __name__ == "__main__":
    # Test RL
    print("🧪 Testing Reinforcement Learning...")
    
    import yaml
    from model_natron import create_model
    
    # Load config
    with open('/workspace/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dummy data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_samples = 500
    
    sequences = np.random.randn(n_samples, 96, 100)
    prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)
    
    # Create model
    model = create_model(config, device)
    
    # Run 10 episodes for testing
    config['training']['reinforcement']['episodes'] = 10
    config['training']['reinforcement']['steps_per_episode'] = 100
    
    print("\n🔄 Running test RL training (10 episodes)...")
    rl_model = run_reinforcement_learning(model, sequences, prices, config, device)
    
    print("\n✅ RL test successful!")
