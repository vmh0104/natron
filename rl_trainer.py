"""
Reinforcement learning module for Natron leveraging PPO to fine-tune the policy.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F

from dataset_loader import NatronDataModule
from model_natron import NatronModel


@dataclass
class PPOConfig:
    episodes: int = 50
    horizon: int = 256
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    target_kl: float = 0.02
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 3e-4
    max_grad_norm: float = 0.5


class TradingEnvironment:
    ACTIONS = [-1, 0, 1]  # short, flat, long

    def __init__(self, data_module: NatronDataModule, gamma: float, horizon: int, alpha: float = 0.001, beta: float = 0.001):
        self.data_module = data_module
        self.gamma = gamma
        self.horizon = horizon
        self.alpha = alpha
        self.beta = beta

        train_ds, _ = data_module.supervised_datasets()
        self.dataset = train_ds
        if data_module.raw_df is None:
            raise ValueError("Data module must retain raw dataframe for RL.")
        self.close_prices = data_module.raw_df["close"].values.astype(np.float32)

        self.reset()

    def reset(self) -> torch.Tensor:
        self.t = 0
        self.position = 0
        self.equity = 1.0
        self.peak_equity = 1.0
        self.index = random.randint(0, len(self.dataset) - self.horizon - 2)
        return torch.from_numpy(self.dataset.sequences[self.index])

    def step(self, action_id: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        action = self.ACTIONS[action_id]
        prev_position = self.position
        self.position = action

        dataset_index = self.dataset.indices[self.index + self.t]
        next_index = min(dataset_index + 1, len(self.close_prices) - 1)
        price_change = (self.close_prices[next_index] - self.close_prices[dataset_index]) / (
            self.close_prices[dataset_index] + 1e-9
        )

        pnl = self.position * price_change
        self.equity *= (1.0 + pnl)
        self.peak_equity = max(self.peak_equity, self.equity)
        drawdown = (self.peak_equity - self.equity) / self.peak_equity

        turnover_cost = self.alpha * abs(self.position - prev_position)
        drawdown_cost = self.beta * drawdown
        reward = pnl - turnover_cost - drawdown_cost

        self.t += 1
        done = self.t >= self.horizon

        if not done:
            obs = torch.from_numpy(self.dataset.sequences[self.index + self.t])
        else:
            obs = torch.zeros_like(torch.from_numpy(self.dataset.sequences[0]))

        info = {
            "pnl": pnl,
            "turnover_cost": turnover_cost,
            "drawdown_cost": drawdown_cost,
            "equity": self.equity,
        }
        return obs, reward, done, info


class ActorCritic(nn.Module):
    def __init__(self, base_model: NatronModel):
        super().__init__()
        self.encoder = base_model.encoder
        d_model = base_model.config.d_model
        self.policy_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        squeeze = False
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
            squeeze = True
        _, pooled = self.encoder(obs)
        logits = self.policy_head(pooled)
        value = self.value_head(pooled)
        if squeeze:
            logits = logits.squeeze(0)
            value = value.squeeze(0)
        return logits, value


class PPOTrainer:
    def __init__(self, model: NatronModel, data_module: NatronDataModule, cfg: Dict, device: torch.device):
        self.device = device
        self.config = PPOConfig(**cfg)
        self.env = TradingEnvironment(
            data_module,
            gamma=self.config.gamma,
            horizon=self.config.horizon,
            alpha=cfg.get("alpha", 0.001),
            beta=cfg.get("beta", 0.001),
        )
        self.actor_critic = ActorCritic(model).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.config.lr)

    def train(self) -> None:
        for episode in range(1, self.config.episodes + 1):
            batch = self.collect_episode()
            metrics = self.update_policy(batch)
            print(
                f"[RL] Episode {episode}/{self.config.episodes} "
                f"reward={metrics['episode_return']:.4f} "
                f"policy_loss={metrics['policy_loss']:.4f} value_loss={metrics['value_loss']:.4f}"
            )

    def collect_episode(self) -> Dict[str, torch.Tensor]:
        obs = self.env.reset()
        obs = obs.to(self.device)
        observations, actions, logprobs, rewards, values = [], [], [], [], []

        for _ in range(self.config.horizon):
            logits, value = self.actor_critic(obs)
            dist = Categorical(logits=logits)
            action = dist.sample()
            next_obs, reward, done, _ = self.env.step(action.item())

            observations.append(obs)
            actions.append(action)
            logprobs.append(dist.log_prob(action))
            rewards.append(torch.tensor(reward, device=self.device, dtype=torch.float32))
            values.append(value)

            obs = next_obs.to(self.device)
            if done:
                break

        with torch.no_grad():
            _, next_value = self.actor_critic(obs)

        returns, advantages = self.compute_gae(rewards, values, next_value)

        return {
            "observations": torch.stack(observations),
            "actions": torch.stack(actions),
            "logprobs": torch.stack(logprobs),
            "returns": returns,
            "advantages": advantages,
            "episode_return": returns[0].item(),
        }

    def compute_gae(
        self,
        rewards: List[torch.Tensor],
        values: List[torch.Tensor],
        next_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        advantages = []
        gae = 0.0
        values = values + [next_value]

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.config.gamma * values[step + 1] - values[step]
            gae = delta + self.config.gamma * self.config.lam * gae
            advantages.insert(0, gae)

        advantages = torch.stack(advantages)
        returns = advantages + torch.stack(values[:-1])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns.detach(), advantages.detach()

    def update_policy(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        observations = batch["observations"]
        actions = batch["actions"]
        old_logprobs = batch["logprobs"]
        returns = batch["returns"]
        advantages = batch["advantages"]

        policy_losses = []
        value_losses = []
        entropy_losses = []

        for _ in range(4):
            logits, values = self.actor_critic(observations)
            dist = Categorical(logits=logits)
            logprobs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(logprobs - old_logprobs)
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * advantages
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            value_loss = F.mse_loss(values, returns)

            loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy.item())

            kl = (old_logprobs - logprobs).mean().item()
            if kl > 1.5 * self.config.target_kl:
                break

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropy_losses)),
            "episode_return": batch["episode_return"],
        }


__all__ = ["PPOTrainer", "TradingEnvironment", "PPOConfig"]
