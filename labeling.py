"""
Label generation module for Natron V2.

Implements institutional heuristics for buy/sell classification,
directional movement, and 6-state market regime labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class LabelGeneratorConfig:
    """Configuration for label generation."""

    buy_vote_threshold: int = 2
    sell_vote_threshold: int = 2
    direction_horizon: int = 1
    regime_trend_window: int = 96
    bull_trend_threshold: float = 0.02
    bear_trend_threshold: float = -0.02
    adx_threshold: float = 25.0
    volatility_percentile: float = 0.9
    volume_spike_threshold: float = 1.5


class LabelGenerator:
    """Derives supervised labels from OHLCV and engineered features."""

    def __init__(self, config: Optional[LabelGeneratorConfig] = None) -> None:
        self.config = config or LabelGeneratorConfig()

    def generate(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate labels aligned with the final candle in each sequence.

        Args:
            df: Original OHLCV dataframe.
            features: Engineered feature dataframe.

        Returns:
            labels dataframe with columns: buy, sell, direction, regime.
        """
        if len(df) != len(features):
            raise ValueError("Input dataframe and features must be aligned in length.")

        close = df["close"]
        volume = df["volume"]

        # Pre-compute helper series
        ma20 = features["sma_20"]
        ma50 = features["sma_50"]
        rsi = features["rsi_14"]
        macd_hist = features["macd_hist"]
        macd_hist_diff = macd_hist.diff()
        bb_mid = features["bb_mid"]
        ma20_slope = features["sma_slope_20"]
        ma20_slope = ma20_slope.fillna(0.0)
        volume_ma20 = features["volume_sma_20"]
        close_location = features["close_location"]

        volume_spike = volume > (self.config.volume_spike_threshold * (volume_ma20 + 1e-6))
        recent_rsi_oversold = rsi.rolling(5, min_periods=1).min() < 30
        recent_rsi_overbought = rsi.rolling(5, min_periods=1).max() > 70

        buy_conditions = [
            (close > ma20) & (ma20 > ma50),
            (rsi > 50) | ((rsi > 30) & recent_rsi_oversold.shift(1, fill_value=False)),
            (close > bb_mid) & (ma20_slope > 0),
            volume_spike,
            close_location >= 0.7,
            (macd_hist > 0) & (macd_hist_diff > 0),
        ]

        sell_conditions = [
            (close < ma20) & (ma20 < ma50),
            (rsi < 50) | ((rsi < 70) & recent_rsi_overbought.shift(1, fill_value=False)),
            (close < bb_mid) & (ma20_slope < 0),
            volume_spike & (close_location <= 0.3),
            (macd_hist < 0) & (macd_hist_diff < 0),
        ]

        buy_votes = sum(cond.astype(int) for cond in buy_conditions)
        sell_votes = sum(cond.astype(int) for cond in sell_conditions)

        buy_label = (buy_votes >= self.config.buy_vote_threshold).astype(int)
        sell_label = (sell_votes >= self.config.sell_vote_threshold).astype(int)

        # Direction label (binary up / down)
        future_close = close.shift(-self.config.direction_horizon)
        direction = (future_close > close).astype(int)
        direction = direction.fillna(method="ffill").fillna(0).astype(int)
        direction_score = (future_close - close) / (close + 1e-6)

        # Regime classification (6 classes)
        regime = self._regime_classification(df, features)

        labels = pd.DataFrame(
            {
                "buy": buy_label,
                "sell": sell_label,
                "direction": direction,
                "regime": regime,
                "buy_votes": buy_votes,
                "sell_votes": sell_votes,
                "direction_score": direction_score,
            }
        )
        return labels

    # ------------------------------------------------------------------ #
    # Regime logic
    # ------------------------------------------------------------------ #

    def _regime_classification(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        close = df["close"]
        adx = features.get("adx_14", pd.Series(index=df.index, data=0.0))
        atr = features.get("atr_14", pd.Series(index=df.index, data=0.0))
        volume_ratio = features.get("volume_ratio_20", pd.Series(index=df.index, data=1.0))

        trend_window = self.config.regime_trend_window
        trend = close.pct_change(trend_window)

        bull_strong = (trend > self.config.bull_trend_threshold) & (adx > self.config.adx_threshold)
        bull_weak = (trend > 0) & (trend <= self.config.bull_trend_threshold) & (adx <= self.config.adx_threshold)
        bear_strong = (trend < self.config.bear_trend_threshold) & (adx > self.config.adx_threshold)
        bear_weak = (trend < 0) & (trend >= self.config.bear_trend_threshold) & (adx <= self.config.adx_threshold)

        atr_threshold = atr.quantile(self.config.volatility_percentile)
        volatile = (atr > atr_threshold) | (volume_ratio > self.config.volume_spike_threshold * 1.5)

        regime = np.full(len(df), 2)  # default RANGE
        regime[bull_strong] = 0
        regime[bull_weak] = 1
        regime[bear_weak] = 3
        regime[bear_strong] = 4
        regime[volatile] = 5

        return pd.Series(regime, index=df.index, dtype=int)
