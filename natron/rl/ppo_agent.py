"""
Lightweight PPO implementation tailored for the Natron trading environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


@dataclass(slots=True)
class PPOConfig:
    """Hyperparameters for PPO training."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    learning_rate: float = 3e-4
    batch_size: int = 256
    mini_batch_size: int = 64
    update_epochs: int = 10
    max_grad_norm: float = 0.5


class ActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int) -> None:
        super().__init__()
        hidden = 512
        self.body = nn.Sequential(
            nn.LayerNorm(observation_dim),
            nn.Linear(observation_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.policy_head = nn.Linear(hidden, action_dim)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.policy_head(h), self.value_head(h)


class PPOAgent:
    """Proximal Policy Optimization agent."""

    def __init__(self, observation_space, action_space, config: PPOConfig, device: torch.device) -> None:
        self.obs_dim = int(np.prod(observation_space.shape))
        self.action_dim = action_space.n
        self.config = config
        self.device = device
        self.policy = ActorCritic(self.obs_dim, self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(array.astype(np.float32)).view(-1, self.obs_dim)
        return tensor.to(self.device)

    def act(self, observation: np.ndarray) -> Tuple[int, float, float]:
        obs_tensor = self._to_tensor(observation[np.newaxis, ...])
        logits, value = self.policy(obs_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.squeeze(-1).item())

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.policy(observations)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values.squeeze(-1)

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards, device=self.device)
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.config.gamma * next_value * mask - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * mask * gae
            advantages[t] = gae
            next_value = values[t]
        returns = advantages + values
        return advantages, returns

    def update(self, buffer: Dict[str, torch.Tensor]) -> Dict[str, float]:
        observations = buffer["observations"]
        actions = buffer["actions"]
        old_log_probs = buffer["log_probs"]
        returns = buffer["returns"]
        advantages = buffer["advantages"]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_samples = observations.size(0)
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, self.config.mini_batch_size):
                end = start + self.config.mini_batch_size
                idx = permutation[start:end]

                obs_batch = observations[idx]
                action_batch = actions[idx]
                old_log_prob_batch = old_log_probs[idx]
                return_batch = returns[idx]
                advantage_batch = advantages[idx]

                log_probs, entropy, values = self.evaluate_actions(obs_batch, action_batch)
                ratio = torch.exp(log_probs - old_log_prob_batch)
                surr1 = ratio * advantage_batch
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range) * advantage_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values, return_batch)
                entropy_bonus = entropy.mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy_bonus
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy_bonus.item()

        for key in metrics:
            metrics[key] /= max(self.config.update_epochs * (num_samples / self.config.mini_batch_size), 1)
        return metrics
