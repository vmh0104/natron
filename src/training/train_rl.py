"""
Natron V2 Phase 3: Reinforcement Learning Module
PPO-based training for trading optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import os
from typing import Dict, List, Tuple
from collections import deque

from src.models.natron_transformer import NatronTransformer


class TradingEnvironment:
    """
    Simplified trading environment for RL training.
    """
    
    def __init__(self, sequences: np.ndarray, labels: Dict, config: dict):
        """
        Args:
            sequences: Array of feature sequences (N, seq_len, features)
            labels: Dict with buy/sell/direction/regime labels
            config: Configuration dict
        """
        self.sequences = sequences
        self.labels = labels
        self.config = config
        
        self.current_idx = 0
        self.position = 0  # -1: short, 0: neutral, 1: long
        self.entry_price = 0
        self.balance = 10000
        self.equity = 10000
        self.trades = []
        
        self.max_idx = len(sequences) - 1
        
        # Reward parameters
        self.profit_weight = config['rl']['reward_profit_weight']
        self.turnover_penalty = config['rl']['reward_turnover_penalty']
        self.drawdown_penalty = config['rl']['reward_drawdown_penalty']
        
        self.max_equity = self.equity
        self.episode_trades = 0
    
    def reset(self, start_idx: int = 0):
        """Reset environment to initial state"""
        self.current_idx = start_idx
        self.position = 0
        self.entry_price = 0
        self.balance = 10000
        self.equity = 10000
        self.trades = []
        self.max_equity = self.equity
        self.episode_trades = 0
        
        return self.sequences[self.current_idx]
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: 0=hold, 1=buy, 2=sell, 3=close
            
        Returns:
            next_state, reward, done, info
        """
        # Get current and next prices (simplified - using close prices)
        current_price = 1.0  # Normalized price (we work with returns)
        
        # Execute action
        reward = 0.0
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = current_price
            self.episode_trades += 1
            reward -= self.turnover_penalty  # Transaction cost
        
        elif action == 2 and self.position == 0:  # Sell
            self.position = -1
            self.entry_price = current_price
            self.episode_trades += 1
            reward -= self.turnover_penalty
        
        elif action == 3 and self.position != 0:  # Close position
            # Calculate profit/loss
            if self.position == 1:  # Close long
                pnl = current_price - self.entry_price
            else:  # Close short
                pnl = self.entry_price - current_price
            
            self.equity += pnl * 1000  # Position size
            self.position = 0
            self.episode_trades += 1
            
            reward += pnl * self.profit_weight
            reward -= self.turnover_penalty
        
        # Update equity based on open position
        if self.position == 1:
            unrealized_pnl = (current_price - self.entry_price) * 1000
            self.equity = self.balance + unrealized_pnl
        elif self.position == -1:
            unrealized_pnl = (self.entry_price - current_price) * 1000
            self.equity = self.balance + unrealized_pnl
        
        # Drawdown penalty
        if self.equity > self.max_equity:
            self.max_equity = self.equity
        drawdown = (self.max_equity - self.equity) / self.max_equity
        reward -= drawdown * self.drawdown_penalty
        
        # Move to next state
        self.current_idx += 1
        done = self.current_idx >= self.max_idx
        
        next_state = self.sequences[self.current_idx] if not done else self.sequences[self.current_idx - 1]
        
        info = {
            'equity': self.equity,
            'position': self.position,
            'trades': self.episode_trades,
            'drawdown': drawdown
        }
        
        return next_state, reward, done, info


class PPOAgent:
    """
    Proximal Policy Optimization agent for trading.
    """
    
    def __init__(self, model: NatronTransformer, config: dict, device: str = 'cuda'):
        self.model = model
        self.config = config
        self.device = device
        
        # PPO parameters
        self.gamma = config['rl']['gamma']
        self.gae_lambda = config['rl']['gae_lambda']
        self.clip_epsilon = config['rl']['clip_epsilon']
        self.value_coef = config['rl']['value_coef']
        self.entropy_coef = config['rl']['entropy_coef']
        
        # Action head (4 actions: hold, buy, sell, close)
        hidden_dim = config['model']['d_model'] * 2  # Assuming hybrid pooling
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, config['model']['d_model'] // 2),
            nn.GELU(),
            nn.Linear(config['model']['d_model'] // 2, 4)
        ).to(device)
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, config['model']['d_model'] // 2),
            nn.GELU(),
            nn.Linear(config['model']['d_model'] // 2, 1)
        ).to(device)
        
        # Optimizer
        params = list(self.model.parameters()) + \
                list(self.action_head.parameters()) + \
                list(self.value_head.parameters())
        
        self.optimizer = torch.optim.AdamW(
            params,
            lr=config['training']['learning_rate'] * 0.1,  # Lower LR for RL
            weight_decay=config['training']['weight_decay']
        )
    
    def select_action(self, state: np.ndarray, deterministic: bool = False):
        """
        Select action based on current policy.
        
        Returns:
            action, log_prob, value
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(state_tensor, return_embeddings=True)
            pooled = outputs['pooled']
            
            action_logits = self.action_head(pooled)
            value = self.value_head(pooled)
            
            action_probs = F.softmax(action_logits, dim=-1)
            
            if deterministic:
                action = action_probs.argmax(dim=-1)
            else:
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
            
            log_prob = F.log_softmax(action_logits, dim=-1)[0, action]
        
        return action.item(), log_prob.item(), value.item()
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                   dones: List[bool]) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation.
        """
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
    
    def update(self, trajectories: List[Dict], epochs: int = 4):
        """
        Update policy using PPO.
        
        Args:
            trajectories: List of trajectory dicts with states, actions, etc.
            epochs: Number of update epochs
        """
        # Prepare batch data
        states = torch.FloatTensor(np.array([t['states'] for t in trajectories])).to(self.device)
        actions = torch.LongTensor([t['actions'] for t in trajectories]).to(self.device)
        old_log_probs = torch.FloatTensor([t['log_probs'] for t in trajectories]).to(self.device)
        advantages = torch.FloatTensor([t['advantages'] for t in trajectories]).to(self.device)
        returns = torch.FloatTensor([t['returns'] for t in trajectories]).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(epochs):
            # Forward pass
            outputs = self.model(states, return_embeddings=True)
            pooled = outputs['pooled']
            
            action_logits = self.action_head(pooled)
            values = self.value_head(pooled).squeeze(-1)
            
            # Calculate new log probs
            action_probs = F.softmax(action_logits, dim=-1)
            log_probs = F.log_softmax(action_logits, dim=-1)
            new_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
            
            # PPO clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, returns)
            
            # Entropy bonus
            entropy = -(action_probs * log_probs).sum(dim=-1).mean()
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + 
                list(self.action_head.parameters()) + 
                list(self.value_head.parameters()),
                self.config['training']['gradient_clip']
            )
            self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': loss.item()
        }


class RLTrainer:
    """
    Manages Phase 3 reinforcement learning training.
    """
    
    def __init__(self, config: dict, device: str = 'cuda', 
                 supervised_checkpoint: str = None):
        self.config = config
        self.device = device
        
        # Model
        self.model = NatronTransformer(config).to(device)
        
        # Load supervised weights
        if supervised_checkpoint and os.path.exists(supervised_checkpoint):
            checkpoint = torch.load(supervised_checkpoint, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded supervised weights from {supervised_checkpoint}")
        
        # PPO Agent
        self.agent = PPOAgent(self.model, config, device)
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir='logs/rl')
        
        self.best_reward = float('-inf')
    
    def train(self, train_data: Dict, num_episodes: int = 100, steps_per_episode: int = 1000):
        """
        Train using PPO in trading environment.
        
        Args:
            train_data: Training data dict with sequences and labels
            num_episodes: Number of training episodes
            steps_per_episode: Max steps per episode
        """
        print("=" * 80)
        print("PHASE 3: REINFORCEMENT LEARNING (PPO)")
        print("=" * 80)
        
        env = TradingEnvironment(train_data['sequences'], train_data['labels'], self.config)
        
        for episode in range(1, num_episodes + 1):
            # Collect trajectory
            trajectory = self._collect_trajectory(env, steps_per_episode)
            
            # Update policy
            update_info = self.agent.update([trajectory])
            
            # Logging
            episode_reward = sum(trajectory['rewards'])
            print(f"Episode {episode}/{num_episodes} - "
                  f"Reward: {episode_reward:.2f}, "
                  f"Trades: {trajectory['info'][-1]['trades']}, "
                  f"Final Equity: {trajectory['info'][-1]['equity']:.2f}")
            
            self.writer.add_scalar('RL/Episode_Reward', episode_reward, episode)
            self.writer.add_scalar('RL/Policy_Loss', update_info['policy_loss'], episode)
            self.writer.add_scalar('RL/Value_Loss', update_info['value_loss'], episode)
            self.writer.add_scalar('RL/Entropy', update_info['entropy'], episode)
            
            # Save best model
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.save_checkpoint('model/rl_best.pt')
                print(f"✓ Best RL model saved (reward: {self.best_reward:.2f})")
        
        print("\n" + "=" * 80)
        print("REINFORCEMENT LEARNING COMPLETE")
        print("=" * 80)
        
        self.writer.close()
    
    def _collect_trajectory(self, env: TradingEnvironment, max_steps: int) -> Dict:
        """Collect one trajectory (episode) of experience"""
        state = env.reset(start_idx=np.random.randint(0, len(env.sequences) - max_steps))
        
        states, actions, rewards, log_probs, values, dones, infos = [], [], [], [], [], [], []
        
        for step in range(max_steps):
            action, log_prob, value = self.agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            dones.append(done)
            infos.append(info)
            
            state = next_state
            
            if done:
                break
        
        # Compute advantages and returns
        advantages, returns = self.agent.compute_gae(rewards, values, dones)
        
        trajectory = {
            'states': np.array(states),
            'actions': actions,
            'rewards': rewards,
            'log_probs': log_probs,
            'values': values,
            'advantages': advantages,
            'returns': returns,
            'info': infos
        }
        
        return trajectory
    
    def save_checkpoint(self, path: str):
        """Save RL model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'action_head_state_dict': self.agent.action_head.state_dict(),
            'value_head_state_dict': self.agent.value_head.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'config': self.config,
            'best_reward': self.best_reward
        }, path)
