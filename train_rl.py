"""
Phase 3: Reinforcement Learning Training
PPO algorithm for optimizing trading decisions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import os
from tqdm import tqdm
import numpy as np
from collections import deque

from model_natron import NatronTransformer
from dataset_loader import SequenceCreator
from feature_engine import FeatureEngine
from label_generator import LabelGenerator
import pandas as pd


class TradingEnvironment:
    """
    Trading environment for RL training.
    Simulates trading based on model predictions.
    """
    
    def __init__(self, df: pd.DataFrame, features_df: pd.DataFrame, initial_balance: float = 10000.0):
        self.df = df.reset_index(drop=True)
        self.features_df = features_df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.reset()
        
    def reset(self):
        """Reset environment"""
        self.balance = self.initial_balance
        self.position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = 0.0
        self.equity_history = [self.initial_balance]
        self.trades = []
        self.current_step = 0
        self.max_drawdown = 0.0
        self.peak_equity = self.initial_balance
        
        return self._get_state()
    
    def _get_state(self):
        """Get current state index"""
        return self.current_step
    
    def step(self, action: int, price: float):
        """
        Execute action in environment.
        
        Args:
            action: 0 = hold, 1 = buy, 2 = sell
            price: Current close price
            
        Returns:
            reward, done, info
        """
        reward = 0.0
        done = False
        
        # Calculate current equity
        if self.position != 0:
            if self.position == 1:  # Long
                unrealized_pnl = (price - self.entry_price) / self.entry_price * self.balance
            else:  # Short
                unrealized_pnl = (self.entry_price - price) / self.entry_price * self.balance
            equity = self.balance + unrealized_pnl
        else:
            equity = self.balance
        
        # Update peak and drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity
        drawdown = (self.peak_equity - equity) / self.peak_equity
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        
        # Execute action
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = price
            self.trades.append({'type': 'buy', 'price': price, 'step': self.current_step})
        elif action == 2 and self.position == 0:  # Sell (short)
            self.position = -1
            self.entry_price = price
            self.trades.append({'type': 'sell', 'price': price, 'step': self.current_step})
        elif action == 0 and self.position != 0:  # Close position
            if self.position == 1:  # Close long
                pnl = (price - self.entry_price) / self.entry_price
            else:  # Close short
                pnl = (self.entry_price - price) / self.entry_price
            
            profit = pnl * self.balance
            self.balance += profit
            reward = profit / self.initial_balance  # Normalize reward
            
            self.trades.append({
                'type': 'close',
                'price': price,
                'pnl': profit,
                'step': self.current_step
            })
            self.position = 0
        
        # Calculate reward components
        profit_reward = reward
        turnover_penalty = len(self.trades) * 0.001  # Penalty for frequent trading
        drawdown_penalty = drawdown * 0.1
        
        total_reward = profit_reward - turnover_penalty - drawdown_penalty
        
        # Update equity history
        if self.position != 0:
            if self.position == 1:
                unrealized_pnl = (price - self.entry_price) / self.entry_price * self.balance
            else:
                unrealized_pnl = (self.entry_price - price) / self.entry_price * self.balance
            equity = self.balance + unrealized_pnl
        else:
            equity = self.balance
        self.equity_history.append(equity)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        info = {
            'equity': equity,
            'drawdown': drawdown,
            'num_trades': len(self.trades)
        }
        
        return total_reward, done, info


class PPOAgent:
    """
    PPO (Proximal Policy Optimization) agent for trading.
    """
    
    def __init__(self, model: NatronTransformer, config: dict, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.gamma = config['training']['rl']['gamma']
        self.clip_epsilon = config['training']['rl']['clip_epsilon']
        
        # Value head (critic)
        d_model = config['model']['d_model']
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        ).to(device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            list(model.parameters()) + list(self.value_head.parameters()),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
    def select_action(self, state: torch.Tensor, deterministic: bool = False):
        """
        Select action using policy.
        
        Args:
            state: (batch_size, seq_len, feature_dim) or (seq_len, feature_dim)
            deterministic: If True, use deterministic policy
            
        Returns:
            action, log_prob, value
        """
        if len(state.shape) == 2:
            state = state.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            predictions = self.model(state)
            
            # Combine buy/sell probabilities into action probabilities
            buy_prob = predictions['buy'].squeeze()
            sell_prob = predictions['sell'].squeeze()
            hold_prob = 1.0 - buy_prob - sell_prob
            hold_prob = torch.clamp(hold_prob, min=0.0)
            
            # Normalize
            action_probs = torch.stack([hold_prob, buy_prob, sell_prob], dim=-1)
            action_probs = action_probs / (action_probs.sum(dim=-1, keepdim=True) + 1e-8)
            
            if deterministic:
                action = action_probs.argmax(dim=-1)
            else:
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            # Value estimate
            encoded = self.model.encode(state)
            if self.model.use_pooling:
                pooled = encoded.mean(dim=1)
            else:
                pooled = encoded[:, -1, :]
            value = self.value_head(pooled).squeeze()
        
        if deterministic:
            return action.cpu().numpy(), None, value.cpu().numpy()
        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()
    
    def update(self, states, actions, rewards, log_probs_old, values_old, next_values):
        """
        Update policy using PPO.
        """
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        log_probs_old = torch.FloatTensor(log_probs_old).to(self.device)
        values_old = torch.FloatTensor(values_old).to(self.device)
        next_values = torch.FloatTensor(next_values).to(self.device)
        
        # Compute advantages
        returns = []
        advantage = 0
        for t in reversed(range(len(rewards))):
            advantage = rewards[t] + self.gamma * next_values[t] - values_old[t]
            returns.insert(0, advantage + values_old[t])
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = returns - values_old
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get current policy predictions
        predictions = self.model(states)
        buy_prob = predictions['buy'].squeeze()
        sell_prob = predictions['sell'].squeeze()
        hold_prob = 1.0 - buy_prob - sell_prob
        hold_prob = torch.clamp(hold_prob, min=0.0)
        action_probs = torch.stack([hold_prob, buy_prob, sell_prob], dim=-1)
        action_probs = action_probs / (action_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        
        # Get current values
        encoded = self.model.encode(states)
        if self.model.use_pooling:
            pooled = encoded.mean(dim=1)
        else:
            pooled = encoded[:, -1, :]
        values = self.value_head(pooled).squeeze()
        
        # PPO loss
        ratio = torch.exp(log_probs - log_probs_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = nn.MSELoss()(values, returns)
        
        # Entropy bonus
        entropy = dist.entropy().mean()
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.value_head.parameters()), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss.item()
        }


def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config['paths']['model_dir'], exist_ok=True)
    os.makedirs(config['paths']['logs_dir'], exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(config['data']['csv_path'])
    
    # Generate features
    print("Generating features...")
    feature_engine = FeatureEngine()
    features_df = feature_engine.generate_all_features(df)
    
    # Create sequences
    print("Creating sequences...")
    sequence_creator = SequenceCreator(config)
    
    # Create dummy labels for dataset
    dummy_labels = pd.DataFrame({
        'buy': [0] * len(features_df),
        'sell': [0] * len(features_df),
        'direction': [0] * len(features_df),
        'regime': [0] * len(features_df)
    }, index=features_df.index)
    
    train_dataset, val_dataset, test_dataset = sequence_creator.create_datasets(
        features_df, dummy_labels
    )
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    
    # Load supervised model
    print("Loading supervised model...")
    model = NatronTransformer(
        feature_dim=config['model']['feature_dim'],
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_layers=config['model']['num_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
        activation=config['model']['activation']
    ).to(device)
    
    model_path = os.path.join(config['paths']['model_dir'], 'natron_v2.pt')
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Supervised model loaded!")
    else:
        print("Warning: No supervised model found, training from scratch.")
    
    # Create RL agent
    agent = PPOAgent(model, config, device)
    
    # Create environment
    env = TradingEnvironment(df, features_df)
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(config['paths']['logs_dir'], 'rl'))
    
    # Training loop
    num_epochs = config['training']['num_epochs_rl']
    print(f"Starting RL training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Collect trajectories
        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        
        env.reset()
        state_idx = 0
        
        for batch in train_loader:
            if state_idx >= len(df) - 1:
                break
            
            sequence = batch['sequence'].squeeze(0).to(device)  # (seq_len, feature_dim)
            price = df.iloc[state_idx + sequence_creator.sequence_length - 1]['close']
            
            # Select action
            action, log_prob, value = agent.select_action(sequence, deterministic=False)
            
            # Execute in environment
            reward, done, info = env.step(action[0], price)
            
            states.append(sequence.cpu().numpy())
            actions.append(action[0])
            rewards.append(reward)
            log_probs.append(log_prob[0] if log_prob is not None else 0.0)
            values.append(value[0])
            
            state_idx += 1
            
            if done:
                break
        
        # Compute next values for last state
        if len(states) > 0:
            last_state = torch.FloatTensor([states[-1]]).to(device)
            _, _, next_value = agent.select_action(last_state, deterministic=True)
            next_values = values[1:] + [next_value[0]]
        else:
            next_values = values
        
        # Update agent
        if len(states) > 0:
            update_info = agent.update(states, actions, rewards, log_probs, values, next_values)
            
            # Logging
            total_reward = sum(rewards)
            final_equity = env.equity_history[-1]
            returns_pct = (final_equity - env.initial_balance) / env.initial_balance * 100
            
            writer.add_scalar('Reward/Total', total_reward, epoch)
            writer.add_scalar('Reward/Average', np.mean(rewards), epoch)
            writer.add_scalar('Equity/Final', final_equity, epoch)
            writer.add_scalar('Returns/Percentage', returns_pct, epoch)
            writer.add_scalar('Metrics/Drawdown', env.max_drawdown, epoch)
            writer.add_scalar('Metrics/NumTrades', len(env.trades), epoch)
            writer.add_scalar('Loss/Policy', update_info['policy_loss'], epoch)
            writer.add_scalar('Loss/Value', update_info['value_loss'], epoch)
            writer.add_scalar('Loss/Total', update_info['total_loss'], epoch)
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Total Reward: {total_reward:.4f}, Final Equity: {final_equity:.2f}")
            print(f"Returns: {returns_pct:.2f}%, Drawdown: {env.max_drawdown:.4f}")
            print(f"Trades: {len(env.trades)}")
        
        # Save model periodically
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'value_head_state_dict': agent.value_head.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(config['paths']['model_dir'], f'natron_rl_epoch_{epoch+1}.pt'))
    
    # Save final model
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'value_head_state_dict': agent.value_head.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
    }
    torch.save(checkpoint, os.path.join(config['paths']['model_dir'], 'natron_rl_final.pt'))
    
    writer.close()
    print("RL training complete!")


if __name__ == '__main__':
    main()
