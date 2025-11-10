"""
Phase 3: Reinforcement Learning Training Script for Natron Transformer
PPO algorithm for optimizing trading decisions
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from collections import deque
from tqdm import tqdm
import argparse

from dataset_loader import SequenceCreator
from model_natron import NatronTransformer
from losses import RLLoss


class TradingEnvironment:
    """
    Trading environment for RL training
    """
    
    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0):
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = 0.0
        self.current_step = 0
        self.trades = []
        self.equity_curve = [initial_balance]
        
    def reset(self):
        """Reset environment"""
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.current_step = 0
        self.trades = []
        self.equity_curve = [self.initial_balance]
        return self.current_step
    
    def step(self, action: int, current_price: float):
        """
        Execute action
        
        Args:
            action: 0=hold, 1=buy, 2=sell
            current_price: Current market price
            
        Returns:
            reward, done, info
        """
        reward = 0.0
        
        # Action: 0=hold, 1=buy, 2=sell
        if action == 1 and self.position != 1:  # Buy
            if self.position == -1:  # Close short
                pnl = (self.entry_price - current_price) / self.entry_price
                reward = pnl * 100
                self.balance *= (1 + pnl)
            
            self.position = 1
            self.entry_price = current_price
        
        elif action == 2 and self.position != -1:  # Sell
            if self.position == 1:  # Close long
                pnl = (current_price - self.entry_price) / self.entry_price
                reward = pnl * 100
                self.balance *= (1 + pnl)
            
            self.position = -1
            self.entry_price = current_price
        
        # Unrealized PnL
        if self.position != 0:
            if self.position == 1:
                unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            else:
                unrealized_pnl = (self.entry_price - current_price) / self.entry_price
            reward += unrealized_pnl * 10  # Small reward for unrealized gains
        
        # Penalty for holding (encourage trading)
        reward -= 0.01
        
        # Update equity curve
        equity = self.balance
        if self.position != 0:
            if self.position == 1:
                equity *= (1 + (current_price - self.entry_price) / self.entry_price)
            else:
                equity *= (1 + (self.entry_price - current_price) / self.entry_price)
        self.equity_curve.append(equity)
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        info = {
            'balance': self.balance,
            'position': self.position,
            'equity': equity
        }
        
        return reward, done, info
    
    def get_current_price(self):
        """Get current close price"""
        if self.current_step < len(self.df):
            return self.df.iloc[self.current_step]['close']
        return self.df.iloc[-1]['close']


class PolicyNetwork(nn.Module):
    """
    Policy network for RL (wraps NatronTransformer)
    """
    
    def __init__(self, base_model: NatronTransformer):
        super().__init__()
        self.base_model = base_model
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x: torch.Tensor):
        """Forward pass"""
        outputs = self.base_model(x, return_embeddings=True)
        pooled = outputs['embeddings'].mean(dim=1)  # (batch, d_model)
        
        # Action probabilities (based on buy/sell predictions)
        buy_prob = outputs['buy']
        sell_prob = outputs['sell']
        hold_prob = 1 - (buy_prob + sell_prob) / 2
        
        # Normalize to probabilities
        action_probs = torch.cat([hold_prob, buy_prob, sell_prob], dim=1)
        action_probs = torch.softmax(action_probs, dim=1)
        
        # Value estimate
        value = self.value_head(pooled)
        
        return action_probs, value


def collect_rollouts(env, policy, sequence_creator, device, max_steps=1000):
    """Collect rollout data using current policy"""
    states = []
    actions = []
    rewards = []
    values = []
    log_probs = []
    dones = []
    
    env.reset()
    step = 0
    
    while step < max_steps and env.current_step < len(env.df) - 96:
        # Get current sequence
        start_idx = env.current_step
        end_idx = start_idx + 96
        
        if end_idx >= len(env.df):
            break
        
        # Extract sequence
        features_df = sequence_creator.feature_engine.fit_transform(env.df.iloc[start_idx:end_idx])
        feature_matrix = features_df[sequence_creator.feature_columns].values.astype(np.float32)
        state = torch.FloatTensor(feature_matrix).unsqueeze(0).to(device)
        
        # Get action from policy
        policy.eval()
        with torch.no_grad():
            action_probs, value = policy(state)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
        
        # Execute action
        current_price = env.get_current_price()
        reward, done, info = env.step(action.item(), current_price)
        
        # Store data
        states.append(state.cpu())
        actions.append(action.cpu())
        rewards.append(reward)
        values.append(value.cpu().item())
        log_probs.append(log_prob.cpu())
        dones.append(done)
        
        step += 1
        
        if done:
            break
    
    return {
        'states': torch.cat(states, dim=0),
        'actions': torch.stack(actions),
        'rewards': np.array(rewards),
        'values': np.array(values),
        'log_probs': torch.stack(log_probs),
        'dones': np.array(dones)
    }


def compute_returns(rewards, dones, gamma=0.99):
    """Compute discounted returns"""
    returns = np.zeros_like(rewards)
    running_return = 0
    
    for t in reversed(range(len(rewards))):
        if dones[t]:
            running_return = 0
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
    
    return returns


def compute_advantages(returns, values, gamma=0.99, lam=0.95):
    """Compute GAE advantages"""
    advantages = np.zeros_like(returns)
    last_gae = 0
    
    for t in reversed(range(len(returns))):
        if t == len(returns) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = returns[t] - values[t] + gamma * next_value
        advantages[t] = last_gae = delta + gamma * lam * last_gae
    
    return advantages


def train_rl_epoch(policy, rollouts, optimizer, device, epochs=10, clip_epsilon=0.2):
    """Train policy for one epoch"""
    states = rollouts['states'].to(device)
    actions = rollouts['actions'].to(device)
    old_log_probs = rollouts['log_probs'].to(device)
    returns = torch.FloatTensor(rollouts['returns']).to(device)
    advantages = torch.FloatTensor(rollouts['advantages']).to(device)
    old_values = torch.FloatTensor(rollouts['values']).to(device)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    total_loss = 0.0
    
    for _ in range(epochs):
        # Forward pass
        action_probs, values = policy(states)
        action_dist = torch.distributions.Categorical(action_probs)
        new_log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy().mean()
        
        # Compute PPO loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_clipped = old_values + torch.clamp(
            values.squeeze() - old_values,
            -clip_epsilon,
            clip_epsilon
        )
        value_loss1 = (values.squeeze() - returns).pow(2)
        value_loss2 = (value_clipped - returns).pow(2)
        value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / epochs


def main():
    parser = argparse.ArgumentParser(description='Natron Phase 3: Reinforcement Learning')
    parser.add_argument('--data', type=str, default='data_export.csv', help='Path to CSV data')
    parser.add_argument('--model', type=str, default='./models/natron_v2.pt', help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--lam', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='PPO clip epsilon')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='./models', help='Model save directory')
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    
    # Create sequence creator
    sequence_creator = SequenceCreator(sequence_length=96)
    _, _, _ = sequence_creator.create_dataset(df)
    
    # Load base model
    print(f"Loading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location=args.device)
    
    base_model = NatronTransformer(
        num_features=len(sequence_creator.feature_columns),
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        sequence_length=96
    )
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model = base_model.to(args.device)
    
    # Create policy network
    policy = PolicyNetwork(base_model).to(args.device)
    optimizer = optim.AdamW(policy.parameters(), lr=args.lr)
    
    # Create environment
    env = TradingEnvironment(df)
    
    print("\nStarting RL training...")
    for episode in range(1, args.episodes + 1):
        # Collect rollouts
        rollouts = collect_rollouts(env, policy, sequence_creator, args.device, max_steps=1000)
        
        if len(rollouts['rewards']) == 0:
            continue
        
        # Compute returns and advantages
        returns = compute_returns(rollouts['rewards'], rollouts['dones'], args.gamma)
        advantages = compute_advantages(returns, rollouts['values'], args.gamma, args.lam)
        
        rollouts['returns'] = returns
        rollouts['advantages'] = advantages
        
        # Train policy
        avg_loss = train_rl_epoch(policy, rollouts, optimizer, args.device, epochs=10, clip_epsilon=args.clip_epsilon)
        
        # Evaluate
        total_reward = rollouts['rewards'].sum()
        avg_reward = rollouts['rewards'].mean()
        final_equity = env.equity_curve[-1] if len(env.equity_curve) > 0 else env.initial_balance
        
        print(f"Episode {episode}/{args.episodes}")
        print(f"  Total Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}")
        print(f"  Final Equity: {final_equity:.2f}, Return: {(final_equity/env.initial_balance - 1)*100:.2f}%")
        print(f"  Loss: {avg_loss:.4f}")
        
        # Save model
        if episode % 10 == 0:
            save_path = os.path.join(args.save_dir, f'natron_rl_episode{episode}.pt')
            torch.save({
                'episode': episode,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)
    
    # Save final model
    final_path = os.path.join(args.save_dir, 'natron_rl_final.pt')
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    print(f"\nFinal model saved to {final_path}")


if __name__ == '__main__':
    main()
