"""
Reinforcement learning module (PPO) for Natron V2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from model_natron import NatronTransformer


@dataclass
class PPOConfig:
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    steps_per_epoch: int = 2048
    mini_batch_size: int = 256
    update_epochs: int = 10
    turnover_penalty: float = 1e-3
    drawdown_penalty: float = 0.2
    device: str = "cuda"


class TradingEnvironment:
    """
    Minimal trading environment using precomputed sequences and forward returns.
    """

    ACTION_MAP = {
        0: -1.0,  # Short
        1: 0.0,   # Flat
        2: 1.0,   # Long
    }

    def __init__(
        self,
        sequences: np.ndarray,
        direction_scores: np.ndarray,
        turnover_penalty: float,
        drawdown_penalty: float,
    ) -> None:
        self.sequences = sequences
        self.direction_scores = direction_scores
        self.turnover_penalty = turnover_penalty
        self.drawdown_penalty = drawdown_penalty

        self.ptr = 0
        self.prev_position = 0.0
        self.equity = 1.0
        self.max_equity = 1.0

    def reset(self) -> Tuple[np.ndarray, Dict]:
        self.ptr = 0
        self.prev_position = 0.0
        self.equity = 1.0
        self.max_equity = 1.0
        return self.sequences[self.ptr], {}

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict]:
        position = self.ACTION_MAP[action_idx]
        reward, info = self._compute_reward(position)

        self.prev_position = position
        self.ptr += 1
        done = self.ptr >= len(self.sequences) - 1
        next_obs = self.sequences[self.ptr] if not done else self.sequences[-1]
        return next_obs, reward, done, info

    def _compute_reward(self, position: float) -> Tuple[float, Dict]:
        price_move = self.direction_scores[self.ptr]
        pnl = position * price_move
        turnover = abs(position - self.prev_position)
        self.equity += pnl
        self.max_equity = max(self.max_equity, self.equity)
        drawdown = (self.max_equity - self.equity) / (self.max_equity + 1e-6)
        reward = pnl - self.turnover_penalty * turnover - self.drawdown_penalty * drawdown
        info = {
            "pnl": pnl,
            "turnover": turnover,
            "drawdown": drawdown,
            "equity": self.equity,
        }
        return reward, info


class PolicyValueNet(nn.Module):
    def __init__(self, backbone: NatronTransformer, hidden_dim: int = 128, train_backbone: bool = False) -> None:
        super().__init__()
        self.backbone = backbone
        if not train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.policy_head = nn.Sequential(
            nn.LayerNorm(backbone.config.d_model),
            nn.Linear(backbone.config.d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(backbone.config.d_model),
            nn.Linear(backbone.config.d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_output, _ = self.backbone.encode(obs)
        logits = self.policy_head(cls_output)
        value = self.value_head(cls_output).squeeze(-1)
        return logits, value


class NatronPPOAgent:
    def __init__(self, policy_net: PolicyValueNet, config: PPOConfig) -> None:
        self.policy_net = policy_net.to(config.device)
        self.config = config
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.policy_net.parameters()),
            lr=config.lr,
        )

    def collect_trajectory(self, env: TradingEnvironment) -> Dict[str, torch.Tensor]:
        obs_buf, act_buf, logp_buf, rew_buf, val_buf = [], [], [], [], []
        obs, _ = env.reset()
        device = self.config.device

        for _ in range(self.config.steps_per_epoch):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, value = self.policy_net(obs_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)

            next_obs, reward, done, info = env.step(action.item())

            obs_buf.append(obs_tensor.squeeze(0))
            act_buf.append(action)
            logp_buf.append(logp)
            rew_buf.append(torch.tensor(reward, dtype=torch.float32, device=device))
            val_buf.append(value.squeeze(0))

            obs = next_obs
            if done:
                obs, _ = env.reset()

        buffer = {
            "obs": torch.stack(obs_buf),
            "actions": torch.stack(act_buf),
            "logp": torch.stack(logp_buf),
            "rewards": torch.stack(rew_buf),
            "values": torch.stack(val_buf),
        }
        return buffer

    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = rewards.device
        advantages = torch.zeros_like(rewards, device=device)
        last_advantage = 0.0
        values = torch.cat([values, torch.tensor([0.0], device=device)])

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.config.gamma * values[t + 1] - values[t]
            advantages[t] = last_advantage = delta + self.config.gamma * self.config.gae_lambda * last_advantage

        returns = advantages + values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        return advantages, returns

    def update(self, buffer: Dict[str, torch.Tensor]) -> Dict[str, float]:
        advantages, target_values = self.compute_gae(buffer["rewards"], buffer["values"])

        obs = buffer["obs"]
        actions = buffer["actions"]
        old_logp = buffer["logp"]

        metrics = {}
        for _ in range(self.config.update_epochs):
            idx = torch.randperm(len(obs))
            for start in range(0, len(obs), self.config.mini_batch_size):
                batch_idx = idx[start : start + self.config.mini_batch_size]
                batch_obs = obs[batch_idx].to(self.config.device)
                batch_actions = actions[batch_idx].to(self.config.device)
                batch_advantages = advantages[batch_idx].to(self.config.device)
                batch_returns = target_values[batch_idx].to(self.config.device)
                batch_old_logp = old_logp[batch_idx].to(self.config.device)

                logits, values = self.policy_net(batch_obs)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - batch_old_logp)
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                value_loss = (batch_returns - values) ** 2
                value_loss = 0.5 * value_loss.mean()

                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
                self.optimizer.step()

                metrics = {
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "entropy": entropy.item(),
                }
        return metrics


def run_ppo_training(
    backbone: NatronTransformer,
    sequences: np.ndarray,
    direction_scores: np.ndarray,
    config: PPOConfig,
) -> Dict[str, float]:
    """
    Execute PPO training for specified epochs.
    """
    env = TradingEnvironment(
        sequences=sequences,
        direction_scores=direction_scores,
        turnover_penalty=config.turnover_penalty,
        drawdown_penalty=config.drawdown_penalty,
    )
    policy_net = PolicyValueNet(backbone, train_backbone=False)
    agent = NatronPPOAgent(policy_net, config)

    history: List[Dict[str, float]] = []
    for epoch in range(config.update_epochs):
        buffer = agent.collect_trajectory(env)
        metrics = agent.update(buffer)
        history.append(metrics)
    return history[-1] if history else {}
