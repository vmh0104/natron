"""
Label generation for Natron Transformer tasks.

Implements institutional heuristics for buy/sell signals, directional targets,
and six-state market regimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


@dataclass
class LabelConfig:
    buy_threshold: int = 2
    sell_threshold: int = 2
    direction_horizon: int = 4
    trend_window: int = 20
    adx_threshold: float = 25.0
    trend_pct_threshold: float = 0.02
    range_threshold: float = 0.005
    volume_spike_multiplier: float = 1.5
    atr_percentile: float = 0.9


class LabelGenerator:
    """Generate buy, sell, direction, and regime labels from features."""

    def __init__(self, config: Optional[LabelConfig] = None):
        self.config = config or LabelConfig()

    def generate(self, raw_df: DataFrame, feature_df: DataFrame) -> DataFrame:
        """
        Produce target labels aligned with feature_df index.

        Args:
            raw_df: Original OHLCV DataFrame (time-indexed).
            feature_df: Engineered features aligned to raw_df.

        Returns:
            DataFrame with columns [buy, sell, direction, regime].
        """
        df = raw_df.copy()
        df = df.loc[feature_df.index]
        labels = pd.DataFrame(index=feature_df.index)

        labels["buy"] = self._buy_signal(df, feature_df).astype(int)
        labels["sell"] = self._sell_signal(df, feature_df).astype(int)
        labels["direction"] = self._direction_signal(df).astype(int)
        labels["regime"] = self._regime_class(df, feature_df).astype(int)
        return labels

    # --- Buy / Sell ---
    def _buy_signal(self, df: DataFrame, feat: DataFrame) -> Series:
        cfg = self.config
        ma20 = feat.get("sma_20", df["close"].rolling(20).mean())
        ma50 = feat.get("sma_50", df["close"].rolling(50).mean())
        rsi = feat.get("rsi_14", pd.Series(index=df.index))
        bb_mid = feat.get("bb_mid", df["close"].rolling(20).mean())
        slope20 = feat.get("slope_close_20", pd.Series(index=df.index))
        macd_hist = feat.get("macd_hist", pd.Series(index=df.index))
        close_position = feat.get("close_position_pct", (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-9))

        cond1 = (df["close"] > ma20) & (ma20 > ma50)
        cond2 = (rsi > 50) | ((rsi.shift(1) < 30) & (rsi > 30))
        cond3 = (df["close"] > bb_mid) & (slope20 > 0)
        cond4 = (df["volume"] > cfg.volume_spike_multiplier * df["volume"].rolling(20).mean())
        cond5 = close_position >= 0.7
        cond6 = (macd_hist > 0) & (macd_hist.diff() > 0)

        score = cond1.astype(int) + cond2.astype(int) + cond3.astype(int) + cond4.astype(int) + cond5.astype(int) + cond6.astype(int)
        return (score >= cfg.buy_threshold)

    def _sell_signal(self, df: DataFrame, feat: DataFrame) -> Series:
        cfg = self.config
        ma20 = feat.get("sma_20", df["close"].rolling(20).mean())
        ma50 = feat.get("sma_50", df["close"].rolling(50).mean())
        rsi = feat.get("rsi_14", pd.Series(index=df.index))
        bb_mid = feat.get("bb_mid", df["close"].rolling(20).mean())
        slope20 = feat.get("slope_close_20", pd.Series(index=df.index))
        macd_hist = feat.get("macd_hist", pd.Series(index=df.index))
        close_position = feat.get("close_position_pct", (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-9))

        cond1 = (df["close"] < ma20) & (ma20 < ma50)
        cond2 = (rsi < 50) | ((rsi.shift(1) > 70) & (rsi < 70))
        cond3 = (df["close"] < bb_mid) & (slope20 < 0)
        cond4 = (df["volume"] > cfg.volume_spike_multiplier * df["volume"].rolling(20).mean()) & (close_position <= 0.3)
        cond5 = (macd_hist < 0) & (macd_hist.diff() < 0)

        score = (
            cond1.astype(int)
            + cond2.astype(int)
            + cond3.astype(int)
            + cond4.astype(int)
            + cond5.astype(int)
        )
        return (score >= cfg.sell_threshold)

    # --- Direction ---
    def _direction_signal(self, df: DataFrame) -> Series:
        future_return = df["close"].shift(-self.config.direction_horizon) / df["close"] - 1.0
        direction = (future_return > 0).astype(int)
        direction = direction.fillna(method="ffill").fillna(0)
        return direction

    # --- Regime ---
    def _regime_class(self, df: DataFrame, feat: DataFrame) -> Series:
        cfg = self.config
        trend_pct = df["close"].pct_change(cfg.trend_window)
        adx = feat.get("adx", pd.Series(index=df.index))
        atr = feat.get("atr_14", pd.Series(index=df.index))

        volume_threshold = df["volume"].rolling(100).quantile(cfg.atr_percentile).fillna(df["volume"].median())
        atr_threshold = atr.rolling(200).quantile(cfg.atr_percentile).fillna(atr.median())
        vol_spike = (atr > atr_threshold) | (df["volume"] > volume_threshold)

        regime = pd.Series(2, index=df.index)  # default RANGE

        bull_strong = (trend_pct > cfg.trend_pct_threshold) & (adx > cfg.adx_threshold)
        bull_weak = (trend_pct > 0) & (trend_pct <= cfg.trend_pct_threshold) & (adx <= cfg.adx_threshold)
        bear_strong = (trend_pct < -cfg.trend_pct_threshold) & (adx > cfg.adx_threshold)
        bear_weak = (trend_pct < 0) & (trend_pct >= -cfg.trend_pct_threshold) & (adx <= cfg.adx_threshold)
        range_market = trend_pct.abs() <= cfg.range_threshold

        regime[bull_strong] = 0
        regime[bull_weak] = 1
        regime[range_market & (adx < cfg.adx_threshold)] = 2
        regime[bear_weak] = 3
        regime[bear_strong] = 4
        regime[vol_spike.fillna(False)] = 5

        regime = regime.fillna(2).astype(int)
        return regime


__all__ = ["LabelGenerator", "LabelConfig"]
