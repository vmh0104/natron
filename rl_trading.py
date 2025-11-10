"""
Phase 3: Reinforcement Learning Module - PPO for Trading

Optimize trading decisions using PPO algorithm with trading rewards.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from collections import deque

from model_natron import NatronModel
from losses import PPOActorCriticLoss, TradingReward


class TradingActorCritic(nn.Module):
    """Actor-Critic network for RL trading."""
    
    def __init__(self, base_model: NatronModel, action_dim: int = 3):
        """
        Args:
            base_model: Pretrained Natron model
            action_dim: Number of actions (0: hold, 1: buy, 2: sell)
        """
        super().__init__()
        self.base_model = base_model
        
        # Actor: policy head
        self.actor = nn.Sequential(
            nn.Linear(base_model.encoder.d_model, base_model.encoder.d_model // 2),
            nn.ReLU(),
            nn.Linear(base_model.encoder.d_model // 2, action_dim)
        )
        
        # Critic: value head
        self.critic = nn.Sequential(
            nn.Linear(base_model.encoder.d_model, base_model.encoder.d_model // 2),
            nn.ReLU(),
            nn.Linear(base_model.encoder.d_model // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input sequence (batch_size, seq_len, input_dim)
        
        Returns:
            action_logits: (batch_size, action_dim)
            value: (batch_size, 1)
        """
        # Get encoded representation
        encoded = self.base_model.encode(x)
        
        # Actor and critic outputs
        action_logits = self.actor(encoded)
        value = self.critic(encoded)
        
        return action_logits, value
    
    def get_action(self, x: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Returns:
            action: Sampled action
            log_prob: Log probability of action
            value: Value estimate
        """
        action_logits, value = self.forward(x)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Sample action
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, value.squeeze()


class TradingEnvironment:
    """Trading environment for RL."""
    
    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        initial_balance: float = 10000.0,
        commission: float = 0.001
    ):
        """
        Args:
            features: Feature sequences (N, seq_len, feature_dim)
            prices: Price array (N,)
            initial_balance: Starting balance
            commission: Trading commission rate
        """
        self.features = features
        self.prices = prices
        self.initial_balance = initial_balance
        self.commission = commission
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.balance = self.initial_balance
        self.position = 0  # -1: short, 0: flat, 1: long
        self.shares = 0
        self.current_step = 0
        self.trades_count = 0
        self.equity_history = [self.initial_balance]
        self.max_equity = self.initial_balance
        
        return self.features[self.current_step]
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action.
        
        Args:
            action: 0=hold, 1=buy, 2=sell
        
        Returns:
            next_state, reward, done, info
        """
        current_price = self.prices[self.current_step]
        prev_equity = self.get_equity(current_price)
        
        # Execute action
        if action == 1:  # Buy
            if self.position != 1:
                # Close existing position if any
                if self.position == -1:
                    self.balance += self.shares * current_price * (1 - self.commission)
                    self.trades_count += 1
                
                # Open long position
                self.shares = self.balance / current_price
                self.balance = 0
                self.position = 1
                self.trades_count += 1
        
        elif action == 2:  # Sell
            if self.position != -1:
                # Close existing position if any
                if self.position == 1:
                    self.balance = self.shares * current_price * (1 - self.commission)
                    self.shares = 0
                    self.trades_count += 1
                
                # Open short position (simplified)
                self.shares = -self.balance / current_price
                self.balance = 0
                self.position = -1
                self.trades_count += 1
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward
        if self.current_step < len(self.prices):
            next_price = self.prices[self.current_step]
            new_equity = self.get_equity(next_price)
            price_change = (next_price - current_price) / current_price
            
            # Calculate drawdown
            self.max_equity = max(self.max_equity, new_equity)
            drawdown = (self.max_equity - new_equity) / self.max_equity
            
            # Reward
            reward_fn = TradingReward()
            reward = reward_fn.calculate_reward(
                action, price_change, self.position, self.trades_count, drawdown
            )
            
            done = False
            next_state = self.features[self.current_step] if self.current_step < len(self.features) else None
        else:
            reward = 0
            done = True
            next_state = None
        
        info = {
            'equity': self.get_equity(current_price),
            'position': self.position,
            'trades': self.trades_count
        }
        
        return next_state, reward, done, info
    
    def get_equity(self, price: float) -> float:
        """Calculate current equity."""
        if self.position == 1:  # Long
            return self.shares * price
        elif self.position == -1:  # Short
            return -self.shares * price
        else:
            return self.balance


class PPOTrainer:
    """PPO trainer for trading."""
    
    def __init__(
        self,
        model: TradingActorCritic,
        env: TradingEnvironment,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: torch.device = None
    ):
        self.model = model
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.model.to(device)
        
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = PPOActorCriticLoss(clip_epsilon, value_coef, entropy_coef)
    
    def collect_rollout(self, n_steps: int) -> Dict[str, List]:
        """Collect rollout data."""
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        
        state = self.env.reset()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        for _ in range(n_steps):
            if state is None:
                break
            
            # Get action
            action, log_prob, value = self.model.get_action(state_tensor)
            
            # Step environment
            next_state, reward, done, info = self.env.step(action)
            
            # Store
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob.item())
            values.append(value.item())
            dones.append(done)
            
            if done:
                state = self.env.reset()
            else:
                state = next_state
            
            if state is not None:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'log_probs': log_probs,
            'values': values,
            'dones': dones
        }
    
    def compute_advantages(self, rewards: List[float], values: List[float], dones: List[bool]) -> Tuple[List[float], List[float]]:
        """Compute GAE advantages and returns."""
        advantages = []
        returns = []
        
        gae = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                gae = 0
                next_value = 0
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
            
            returns.insert(0, gae + values[t])
            next_value = values[t]
        
        return advantages, returns
    
    def update(self, rollout: Dict[str, List], n_epochs: int = 4, batch_size: int = 64):
        """Update policy using PPO."""
        states = torch.FloatTensor(np.array(rollout['states'])).to(self.device)
        actions = torch.LongTensor(rollout['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(rollout['log_probs']).to(self.device)
        old_values = torch.FloatTensor(rollout['values']).to(self.device)
        
        advantages, returns = self.compute_advantages(
            rollout['rewards'], rollout['values'], rollout['dones']
        )
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        n_samples = len(states)
        indices = np.arange(n_samples)
        
        for epoch in range(n_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_old_values = old_values[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Get new policy outputs
                action_logits, new_values = self.model(batch_states)
                action_probs = F.softmax(action_logits, dim=-1)
                new_log_probs = F.log_softmax(action_logits, dim=-1)
                
                # Get log prob for selected actions
                selected_log_probs = new_log_probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                
                # Calculate loss
                losses = self.criterion(
                    batch_old_log_probs,
                    selected_log_probs,
                    batch_advantages,
                    batch_old_values,
                    new_values.squeeze(),
                    batch_returns,
                    action_probs
                )
                
                # Update
                self.optimizer.zero_grad()
                losses['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
        
        return losses
