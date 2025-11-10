"""
Market trading environment for Natron reinforcement learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(slots=True)
class MarketEnvConfig:
    """Configuration parameters for the trading environment."""

    sequence_length: int = 96
    feature_dim: int = 100
    episode_length: int = 512
    transaction_cost: float = 0.0005
    turnover_penalty: float = 0.1
    drawdown_penalty: float = 0.2
    reward_scale: float = 100.0
    seed: int = 42


class MarketEnvironment(gym.Env):
    """
    Custom environment exposing Natron feature sequences to an RL agent.

    Observation: recent sequence of engineered features (sequence_length, feature_dim)
    Action space: {0: flat, 1: long, 2: short}
    Reward: profit - alpha * turnover - beta * drawdown
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        config: MarketEnvConfig,
    ) -> None:
        super().__init__()
        if features.shape[0] != prices.shape[0]:
            raise ValueError("Features and prices must be aligned along time dimension.")
        self.features = features.astype(np.float32)
        self.prices = prices.astype(np.float32)
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(config.sequence_length, config.feature_dim),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)  # flat, long, short

        self.current_step: int = 0
        self.start_index: int = 0
        self.position: int = 0
        self.equity_curve: float = 1.0
        self.peak_equity: float = 1.0
        self.episode_steps: int = 0

    def _get_observation(self) -> np.ndarray:
        end = self.start_index + self.current_step + self.config.sequence_length
        start = end - self.config.sequence_length
        return self.features[start:end]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        max_start = self.features.shape[0] - (self.config.sequence_length + self.config.episode_length + 1)
        if max_start <= 0:
            raise RuntimeError("Insufficient data to create an episode. Consider shortening episode_length.")
        self.start_index = self.rng.integers(0, max_start)
        self.current_step = 0
        self.position = 0
        self.equity_curve = 1.0
        self.peak_equity = 1.0
        self.episode_steps = 0

        observation = self._get_observation()
        return observation, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError("Invalid action.")

        prev_position = self.position
        if action == 0:
            self.position = 0
        elif action == 1:
            self.position = 1
        elif action == 2:
            self.position = -1

        price_idx = self.start_index + self.current_step + self.config.sequence_length
        next_price_idx = price_idx + 1
        price_return = (self.prices[next_price_idx] - self.prices[price_idx]) / (self.prices[price_idx] + 1e-9)

        profit = self.position * price_return
        turnover = abs(self.position - prev_position)
        transaction_cost = turnover * self.config.transaction_cost

        self.equity_curve += profit - transaction_cost
        self.peak_equity = max(self.peak_equity, self.equity_curve)
        drawdown = (self.peak_equity - self.equity_curve)

        reward = (
            profit
            - self.config.turnover_penalty * turnover
            - self.config.drawdown_penalty * drawdown
        ) * self.config.reward_scale

        self.current_step += 1
        self.episode_steps += 1
        terminated = self.episode_steps >= self.config.episode_length
        truncated = next_price_idx >= self.prices.shape[0] - 1

        observation = self._get_observation()
        info = {
            "profit": profit,
            "turnover": turnover,
            "drawdown": drawdown,
            "equity": self.equity_curve,
        }

        return observation, float(reward), terminated, truncated, info

    def render(self) -> None:
        print(
            f"Step {self.episode_steps} | Position {self.position} | "
            f"Equity {self.equity_curve:.4f} | Peak {self.peak_equity:.4f}"
        )

    def close(self) -> None:
        pass
