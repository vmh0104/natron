"""
Reinforcement learning integration for Natron Transformer.

Provides a lightweight PPO training loop using Stable-Baselines3 on a custom
trading environment derived from Natron feature sequences.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from pathlib import Path

from dataset_loader import NatronDataModule, NatronSequenceDataset
from model_natron import NatronTransformer

LOGGER = logging.getLogger("natron.rl")

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None
    spaces = None

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:  # pragma: no cover
    PPO = None
    DummyVecEnv = None


@dataclass
class RewardConfig:
    profit_weight: float = 1.0
    turnover_weight: float = 0.1
    drawdown_weight: float = 0.2


class NatronTradingEnv(gym.Env):  # type: ignore[misc]
    """Custom trading environment leveraging Natron Transformer signals."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        model: NatronTransformer,
        dataset: NatronSequenceDataset,
        feature_columns,
        reward_cfg: RewardConfig,
        device: torch.device,
    ):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.features = dataset.features
        self.labels = dataset.labels
        self.seq_len = dataset.sequence_length
        self.max_index = dataset.size - 1
        self.feature_columns = feature_columns
        self.device = device
        self.reward_cfg = reward_cfg
        self.position = 0.0
        self.prev_action = 0.0
        self.equity = 0.0
        self.peak_equity = 0.0
        self.ptr = 0

        self.log_return_idx = (
            feature_columns.index("log_return_1") if "log_return_1" in feature_columns else None
        )

        feature_dim = self.features.shape[1]
        extra_dim = 1 + 2 + 6 + 1 + 1  # buy, sell, direction(2), regime(6), position, equity
        obs_dim = feature_dim + extra_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if self.max_index <= 0:
            raise RuntimeError("RL environment requires dataset with sufficient samples.")

        self.ptr = np.random.randint(0, self.max_index)
        self.position = 0.0
        self.prev_action = 0.0
        self.equity = 0.0
        self.peak_equity = 0.0
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        action_value = float(np.clip(action[0], -1.0, 1.0))
        reward, info = self._compute_reward(action_value)
        self.prev_action = self.position
        self.position = action_value
        self.ptr += 1
        terminated = self.ptr >= self.max_index
        truncated = False
        observation = self._get_observation()
        return observation, reward, terminated, truncated, info

    def _compute_reward(self, action_value: float):
        idx = self.ptr + self.seq_len - 1
        if idx >= len(self.features):
            return 0.0, {}

        log_return = self.features[idx, self.log_return_idx] if self.log_return_idx is not None else 0.0
        realized_return = float(np.expm1(log_return))
        profit = self.reward_cfg.profit_weight * action_value * realized_return
        turnover = self.reward_cfg.turnover_weight * abs(action_value - self.prev_action)

        self.equity += profit
        self.peak_equity = max(self.peak_equity, self.equity)
        drawdown = self.peak_equity - self.equity
        drawdown_penalty = self.reward_cfg.drawdown_weight * drawdown

        reward = profit - turnover - drawdown_penalty
        info = {
            "realized_return": realized_return,
            "profit_component": profit,
            "turnover_penalty": turnover,
            "drawdown_penalty": drawdown_penalty,
            "equity": self.equity,
        }
        return reward, info

    def _get_observation(self) -> np.ndarray:
        start = min(self.ptr, self.max_index)
        end = start + self.seq_len
        if end > len(self.features):
            end = len(self.features)
            start = end - self.seq_len

        window = torch.from_numpy(self.features[start:end]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(window)
            buy_prob = torch.sigmoid(outputs["buy_logits"]).item()
            sell_prob = torch.sigmoid(outputs["sell_logits"]).item()
            direction_probs = torch.softmax(outputs["direction_logits"], dim=-1).squeeze().cpu().numpy()
            regime_probs = torch.softmax(outputs["regime_logits"], dim=-1).squeeze().cpu().numpy()

        last_features = self.features[end - 1].astype(np.float32)
        obs = np.concatenate(
            [
                last_features,
                np.array([buy_prob, sell_prob], dtype=np.float32),
                direction_probs.astype(np.float32),
                regime_probs.astype(np.float32),
                np.array([self.position, self.equity], dtype=np.float32),
            ]
        )
        return obs

    def render(self):  # pragma: no cover - visualization placeholder
        LOGGER.info("Step %d | Position %.3f | Equity %.5f", self.ptr, self.position, self.equity)


def maybe_train_rl(
    model: NatronTransformer,
    config: Dict,
    device: torch.device,
    data_module: NatronDataModule,
) -> None:
    if gym is None or PPO is None:
        LOGGER.warning("gymnasium or stable-baselines3 not available; skipping RL phase.")
        return

    rl_cfg = config.get("reinforcement", {})
    reward_cfg = RewardConfig(**rl_cfg.get("reward", {}))
    rollout_length = rl_cfg.get("rollout_length", 2048)
    mini_batch_size = rl_cfg.get("mini_batch_size", 256)
    total_timesteps = rl_cfg.get("total_timesteps", 200_000)
    learning_rate = rl_cfg.get("learning_rate", 3e-4)
    gamma = rl_cfg.get("gamma", 0.99)
    lam = rl_cfg.get("lam", 0.95)
    clip_range = rl_cfg.get("clip_range", 0.2)
    entropy_coef = rl_cfg.get("entropy_coef", 0.01)
    value_coef = rl_cfg.get("value_coef", 0.5)

    dataset = data_module.datasets.get("train")
    if dataset is None:
        LOGGER.warning("Training dataset unavailable for RL; skipping phase.")
        return

    model.eval()
    env = DummyVecEnv(
        [
            lambda: NatronTradingEnv(
                model=model,
                dataset=dataset,
                feature_columns=data_module.feature_columns,
                reward_cfg=reward_cfg,
                device=device,
            )
        ]
    )

    LOGGER.info("Starting PPO training for %d timesteps", total_timesteps)
    agent = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=rollout_length,
        batch_size=mini_batch_size,
        gamma=gamma,
        gae_lambda=lam,
        clip_range=clip_range,
        ent_coef=entropy_coef,
        vf_coef=value_coef,
        verbose=1,
        device=device,
    )
    agent.learn(total_timesteps=total_timesteps)

    checkpoint_dir = config.get("project", {}).get("checkpoint_dir", "models")
    policy_path = Path(checkpoint_dir) / "natron_policy.zip"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(policy_path))
    LOGGER.info("Saved PPO policy to %s", policy_path)
