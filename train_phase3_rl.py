"""
Phase 3: Reinforcement Learning Training (PPO)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import Dict, List, Tuple
import random
from model import NatronModel


class TradingEnvironment:
    """Trading environment for RL"""
    
    def __init__(self, prices: np.ndarray, initial_balance: float = 10000.0):
        self.prices = prices
        self.initial_balance = initial_balance
        self.reset()
    
    def reset(self):
        """Reset environment"""
        self.balance = self.initial_balance
        self.position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = 0.0
        self.equity = [self.balance]
        self.trades = []
        self.current_step = 0
        return self._get_state()
    
    def _get_state(self):
        """Get current state (price change)"""
        if self.current_step == 0:
            return 0.0
        return (self.prices[self.current_step] - self.prices[self.current_step - 1]) / self.prices[self.current_step - 1]
    
    def step(self, action: int, price: float) -> Tuple[float, bool, Dict]:
        """
        Execute action
        
        Args:
            action: 0=hold, 1=buy, 2=sell
            price: Current price
            
        Returns:
            reward, done, info
        """
        reward = 0.0
        done = False
        
        # Execute action
        if action == 1 and self.position == 0:  # Open long
            self.position = 1
            self.entry_price = price
        elif action == 2 and self.position == 0:  # Open short
            self.position = -1
            self.entry_price = price
        elif action == 0 and self.position != 0:  # Close position
            # Calculate P&L
            if self.position == 1:
                pnl = (price - self.entry_price) / self.entry_price
            else:
                pnl = (self.entry_price - price) / self.entry_price
            
            reward = pnl * 100  # Scale reward
            self.balance *= (1 + pnl)
            self.position = 0
            self.trades.append(pnl)
        
        # Update equity
        if self.position != 0:
            if self.position == 1:
                current_value = self.balance * (1 + (price - self.entry_price) / self.entry_price)
            else:
                current_value = self.balance * (1 + (self.entry_price - price) / self.entry_price)
        else:
            current_value = self.balance
        
        self.equity.append(current_value)
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= len(self.prices) - 1
        
        # Calculate drawdown penalty
        if len(self.equity) > 1:
            peak = max(self.equity)
            drawdown = (peak - current_value) / peak if peak > 0 else 0
            reward -= drawdown * 10  # Penalize drawdown
        
        info = {
            'balance': self.balance,
            'equity': current_value,
            'position': self.position,
            'trades': len(self.trades)
        }
        
        return reward, done, info


class PPOTrainer:
    """Proximal Policy Optimization trainer"""
    
    def __init__(
        self,
        model: NatronModel,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        k_epochs: int = 4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Value head for PPO
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        ).to(device)
        
        self.value_optimizer = optim.Adam(self.value_head.parameters(), lr=lr)
    
    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, float]:
        """
        Select action using policy
        
        Returns:
            action, log_prob
        """
        with torch.no_grad():
            outputs = self.model(state.unsqueeze(0))
            buy_prob = outputs['buy'].item()
            sell_prob = outputs['sell'].item()
            
            # Convert to action: 0=hold, 1=buy, 2=sell
            if deterministic:
                if buy_prob > 0.6 and buy_prob > sell_prob:
                    action = 1
                elif sell_prob > 0.6:
                    action = 2
                else:
                    action = 0
            else:
                # Sample from policy
                probs = torch.tensor([1 - buy_prob - sell_prob, buy_prob, sell_prob])
                probs = torch.clamp(probs, min=0.01, max=0.99)
                probs = probs / probs.sum()
                action = torch.multinomial(probs, 1).item()
            
            log_prob = torch.log(probs[action] + 1e-8)
            
        return action, log_prob.item()
    
    def update(
        self,
        states: List[torch.Tensor],
        actions: List[int],
        old_log_probs: List[float],
        rewards: List[float],
        next_states: List[torch.Tensor],
        dones: List[bool]
    ):
        """Update policy using PPO"""
        
        # Convert to tensors
        states = torch.stack(states).to(self.device)
        old_log_probs = torch.tensor(old_log_probs).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        
        # Calculate discounted returns
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate values
        values = self.value_head(self.model(states, return_features=True)['features']).squeeze()
        
        # Calculate advantages
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.k_epochs):
            # Get current policy
            outputs = self.model(states)
            buy_probs = outputs['buy']
            sell_probs = outputs['sell']
            
            # Calculate action probabilities
            probs = torch.stack([
                1 - buy_probs - sell_probs,
                buy_probs,
                sell_probs
            ], dim=1)
            probs = torch.clamp(probs, min=0.01, max=0.99)
            probs = probs / probs.sum(dim=1, keepdim=True)
            
            # Get log probabilities for selected actions
            log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze() + 1e-8)
            
            # Calculate ratio
            ratio = torch.exp(log_probs - old_log_probs)
            
            # Calculate surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            new_values = self.value_head(self.model(states, return_features=True)['features']).squeeze()
            value_loss = nn.MSELoss()(new_values, returns)
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss
            
            # Update
            self.optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), max_norm=0.5)
            self.optimizer.step()
            self.value_optimizer.step()
    
    def train_episode(
        self,
        env: TradingEnvironment,
        sequence_data: np.ndarray,
        max_steps: int = 1000
    ) -> Dict:
        """Train on one episode"""
        state_seq = env.reset()
        states = []
        actions = []
        old_log_probs = []
        rewards = []
        next_states = []
        dones = []
        
        step = 0
        while step < max_steps and not env.current_step >= len(sequence_data) - 1:
            # Get current sequence (last 96 candles)
            start_idx = max(0, env.current_step - 95)
            end_idx = env.current_step + 1
            current_sequence = sequence_data[start_idx:end_idx]
            
            # Pad if necessary
            if len(current_sequence) < 96:
                padding = np.zeros((96 - len(current_sequence), current_sequence.shape[1]))
                current_sequence = np.vstack([padding, current_sequence])
            
            state_tensor = torch.FloatTensor(current_sequence).unsqueeze(0)
            
            # Select action
            action, log_prob = self.select_action(state_tensor)
            
            # Execute action
            current_price = env.prices[env.current_step]
            reward, done, info = env.step(action, current_price)
            
            # Store transition
            states.append(state_tensor.squeeze(0))
            actions.append(action)
            old_log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            
            if not done:
                # Get next state
                next_start_idx = max(0, env.current_step - 95)
                next_end_idx = env.current_step + 1
                next_sequence = sequence_data[next_start_idx:next_end_idx]
                if len(next_sequence) < 96:
                    padding = np.zeros((96 - len(next_sequence), next_sequence.shape[1]))
                    next_sequence = np.vstack([padding, next_sequence])
                next_states.append(torch.FloatTensor(next_sequence))
            else:
                next_states.append(states[-1])  # Use last state if done
            
            step += 1
            
            if done:
                break
        
        # Update policy
        if len(states) > 0:
            self.update(states, actions, old_log_probs, rewards, next_states, dones)
        
        return {
            'total_reward': sum(rewards),
            'final_equity': env.equity[-1] if env.equity else env.initial_balance,
            'num_trades': len(env.trades),
            'win_rate': sum(1 for t in env.trades if t > 0) / len(env.trades) if env.trades else 0
        }
