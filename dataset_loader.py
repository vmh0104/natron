"""
Natron dataset loader and feature engineering pipeline.

This module ingests raw OHLCV candles, engineers a rich set of technical features,
generates institutional-style labels, and produces sequence datasets for both
unsupervised pretraining and supervised multi-task learning.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame, Series
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset


LOGGER = logging.getLogger("natron.dataset")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _ensure_datetime(df: DataFrame) -> DataFrame:
    if not np.issubdtype(df["time"].dtype, np.datetime64):
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.sort_values("time").reset_index(drop=True)
    return df


def _sma(series: Series, window: int) -> Series:
    return series.rolling(window, min_periods=1).mean()


def _ema(series: Series, span: int) -> Series:
    return series.ewm(span=span, adjust=False, min_periods=1).mean()


def _rsi(close: Series, window: int = 14) -> Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _true_range(high: Series, low: Series, close: Series) -> Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def _atr(high: Series, low: Series, close: Series, window: int = 14) -> Series:
    return _true_range(high, low, close).rolling(window, min_periods=1).mean()


def _rolling_std(series: Series, window: int) -> Series:
    return series.rolling(window, min_periods=1).std()


def _macd(close: Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[Series, Series, Series]:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _stochastic(high: Series, low: Series, close: Series, window: int = 14) -> Tuple[Series, Series]:
    lowest_low = low.rolling(window, min_periods=1).min()
    highest_high = high.rolling(window, min_periods=1).max()
    k = ((close - lowest_low) / (highest_high - lowest_low + 1e-9)) * 100
    d = k.rolling(3, min_periods=1).mean()
    return k, d


def _willr(high: Series, low: Series, close: Series, window: int = 14) -> Series:
    highest_high = high.rolling(window, min_periods=1).max()
    lowest_low = low.rolling(window, min_periods=1).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low + 1e-9)


def _cci(high: Series, low: Series, close: Series, window: int = 20) -> Series:
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window, min_periods=1).mean()
    mad = (typical_price - sma_tp).abs().rolling(window, min_periods=1).mean()
    return (typical_price - sma_tp) / (0.015 * (mad + 1e-9))


def _obv(close: Series, volume: Series) -> Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def _mfi(high: Series, low: Series, close: Series, volume: Series, window: int = 14) -> Series:
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    direction = np.sign(typical_price.diff()).fillna(0.0)
    pos_flow = raw_money_flow.where(direction > 0, 0.0)
    neg_flow = raw_money_flow.where(direction < 0, 0.0).abs()
    pos_mf = pos_flow.rolling(window, min_periods=1).sum()
    neg_mf = neg_flow.rolling(window, min_periods=1).sum()
    ratio = pos_mf / (neg_mf + 1e-9)
    return 100 - (100 / (1 + ratio))


def _adx(high: Series, low: Series, close: Series, window: int = 14) -> Tuple[Series, Series, Series]:
    plus_dm = (high.diff()).clip(lower=0.0)
    minus_dm = (-low.diff()).clip(lower=0.0)
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0.0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0.0)
    tr = _true_range(high, low, close)
    atr = tr.rolling(window, min_periods=1).mean()
    plus_di = 100 * (plus_dm.rolling(window, min_periods=1).sum() / (atr + 1e-9))
    minus_di = 100 * (minus_dm.rolling(window, min_periods=1).sum() / (atr + 1e-9))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    adx = dx.rolling(window, min_periods=1).mean()
    return adx, plus_di, minus_di


def _hurst(series: Series, window: int = 100) -> Series:
    # Simple R/S Hurst exponent estimator
    def _hurst_window(values: np.ndarray) -> float:
        if len(values) < 20:
            return np.nan
        ts = values - values.mean()
        cumulative = np.cumsum(ts)
        r = cumulative.max() - cumulative.min()
        s = values.std()
        if s == 0:
            return 0.5
        return math.log(r / s + 1e-9) / (math.log(len(values) + 1e-9))

    return series.rolling(window, min_periods=window).apply(_hurst_window, raw=True)


def _entropy(series: Series, window: int = 48, bins: int = 20) -> Series:
    def _window_entropy(values: np.ndarray) -> float:
        hist, _ = np.histogram(values, bins=bins, density=True)
        hist = hist[hist > 0]
        return -(hist * np.log(hist + 1e-9)).sum()

    return series.rolling(window, min_periods=window // 2).apply(_window_entropy, raw=True)


def _robust_zscore(series: Series, window: int = 20) -> Series:
    median = series.rolling(window, min_periods=1).median()
    mad = (series - median).abs().rolling(window, min_periods=1).median()
    return (series - median) / (mad + 1e-9)


def _percentage_position(close: Series, high: Series, low: Series) -> Series:
    return (close - low) / (high - low + 1e-9)


def _swing_points(series: Series, left: int = 3, right: int = 3) -> Series:
    """
    Marks swing highs/lows: returns 1 for swing high, -1 for swing low, 0 otherwise.
    """
    values = series.values
    res = np.zeros_like(values, dtype=float)
    length = len(values)
    for idx in range(left, length - right):
        window = values[idx - left : idx + right + 1]
        center = values[idx]
        if center == window.max():
            res[idx] = 1.0
        elif center == window.min():
            res[idx] = -1.0
    return pd.Series(res, index=series.index)


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------


@dataclass
class FeatureSpec:
    input_columns: Tuple[str, ...] = ("open", "high", "low", "close", "volume")
    sequence_length: int = 96
    cache_path: Optional[Path] = None
    scaler_type: str = "robust"


class FeatureEngineer:
    """Generates a comprehensive feature matrix for Natron."""

    def __init__(self, spec: FeatureSpec):
        self.spec = spec

    def transform(self, df: DataFrame, *, allow_cache: bool = True) -> Tuple[DataFrame, object]:
        df = _ensure_datetime(df)
        if allow_cache and self.spec.cache_path and self.spec.cache_path.exists():
            LOGGER.info("Loading cached features from %s", self.spec.cache_path)
            features = pd.read_parquet(self.spec.cache_path)
            scaler_path = self.spec.cache_path.with_suffix(".scaler.pkl")
            scaler = joblib.load(scaler_path) if scaler_path.exists() else None
            return features, scaler

        features = self._build_features(df)
        scaler = self._fit_scaler(features)

        scaled = pd.DataFrame(
            scaler.transform(features),
            index=features.index,
            columns=features.columns,
        )

        if allow_cache and self.spec.cache_path:
            self.spec.cache_path.parent.mkdir(parents=True, exist_ok=True)
            scaled.to_parquet(self.spec.cache_path)
            scaler_path = self.spec.cache_path.with_suffix(".scaler.pkl")
            joblib.dump(scaler, scaler_path)
            LOGGER.info("Cached features to %s", self.spec.cache_path)

        return scaled, scaler

    def _fit_scaler(self, features: DataFrame):
        if self.spec.scaler_type.lower() == "standard":
            scaler = StandardScaler()
        else:
            scaler = RobustScaler()
        scaler.fit(features)
        return scaler

    def _build_features(self, df: DataFrame) -> DataFrame:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        open_ = df["open"]
        volume = df["volume"]
        typical_price = (high + low + close) / 3

        feats: Dict[str, Series] = {}

        # Moving averages and slopes
        ma_windows = [5, 8, 10, 12, 16, 20, 24, 30, 34, 40, 50, 60, 72, 96, 120]
        for window in ma_windows:
            sma = _sma(close, window)
            ema = _ema(close, window)
            feats[f"sma_{window}"] = sma
            feats[f"ema_{window}"] = ema
            feats[f"price_to_sma_{window}"] = close / (sma + 1e-9)
            feats[f"sma_slope_{window}"] = sma.diff()
            feats[f"ema_slope_{window}"] = ema.diff()

        # Momentum features
        for window in (7, 14, 21, 28):
            feats[f"rsi_{window}"] = _rsi(close, window)
            feats[f"roc_{window}"] = close.pct_change(periods=window)
            feats[f"momentum_{window}"] = close.diff(periods=window)

        feats["roc_1"] = close.pct_change()
        feats["roc_5"] = close.pct_change(5)
        feats["roc_10"] = close.pct_change(10)
        feats["price_acceleration"] = feats["momentum_7"].diff()

        cci = _cci(high, low, close, 20)
        feats["cci_20"] = cci
        stoch_k, stoch_d = _stochastic(high, low, close, 14)
        feats["stoch_k_14"] = stoch_k
        feats["stoch_d_14"] = stoch_d
        feats["willr_14"] = _willr(high, low, close, 14)
        macd_line, macd_signal, macd_hist = _macd(close)
        feats["macd_line"] = macd_line
        feats["macd_signal"] = macd_signal
        feats["macd_hist"] = macd_hist
        feats["macd_hist_diff"] = macd_hist.diff()

        # Volatility features
        for window in (10, 14, 21, 30):
            feats[f"atr_{window}"] = _atr(high, low, close, window)
            feats[f"true_range_{window}"] = _true_range(high, low, close).rolling(window, min_periods=1).mean()
            feats[f"volatility_{window}"] = _rolling_std(close, window)

        bb_mid = _sma(close, 20)
        bb_std = _rolling_std(close, 20)
        feats["bb_mid"] = bb_mid
        feats["bb_upper"] = bb_mid + 2 * bb_std
        feats["bb_lower"] = bb_mid - 2 * bb_std
        feats["bb_width"] = (feats["bb_upper"] - feats["bb_lower"]) / (bb_mid + 1e-9)
        feats["bb_pct"] = (close - feats["bb_lower"]) / (feats["bb_upper"] - feats["bb_lower"] + 1e-9)

        ema20 = _ema(close, 20)
        atr14 = _atr(high, low, close, 14)
        feats["keltner_upper"] = ema20 + 1.5 * atr14
        feats["keltner_lower"] = ema20 - 1.5 * atr14
        feats["keltner_width"] = (feats["keltner_upper"] - feats["keltner_lower"]) / (ema20 + 1e-9)

        # Volume features
        feats["obv"] = _obv(close, volume)
        feats["obv_slope_10"] = feats["obv"].diff(10)
        feats["mfi_14"] = _mfi(high, low, close, volume, 14)
        feats["volume_ma_20"] = _sma(volume, 20)
        feats["volume_ma_50"] = _sma(volume, 50)
        feats["volume_ratio_20"] = volume / (feats["volume_ma_20"] + 1e-9)
        feats["volume_ratio_50"] = volume / (feats["volume_ma_50"] + 1e-9)
        feats["volume_roc_5"] = volume.pct_change(5)
        feats["volume_roc_10"] = volume.pct_change(10)
        feats["volume_zscore_20"] = _robust_zscore(volume, 20)
        feats["vwap"] = (volume * typical_price).cumsum() / (volume.cumsum() + 1e-9)
        feats["price_to_vwap"] = close / (feats["vwap"] + 1e-9)

        # Price pattern features
        candle_range = high - low
        body = (close - open_).abs()
        feats["candle_body_pct"] = body / (candle_range + 1e-9)
        feats["upper_shadow_pct"] = (high - np.maximum(open_, close)) / (candle_range + 1e-9)
        feats["lower_shadow_pct"] = (np.minimum(open_, close) - low) / (candle_range + 1e-9)
        feats["candle_gap"] = open_ - close.shift(1)
        feats["doji"] = (feats["candle_body_pct"] < 0.1).astype(float)
        feats["body_direction"] = np.sign(close - open_)
        feats["position_in_range"] = _percentage_position(close, high, low)

        # Return-based features
        log_close = np.log(close + 1e-9)
        feats["log_return_1"] = log_close.diff()
        feats["log_return_5"] = log_close.diff(5)
        feats["log_return_20"] = log_close.diff(20)
        feats["intraday_return"] = (close - open_) / (open_ + 1e-9)
        feats["high_low_return"] = (high - low) / (close.shift(1) + 1e-9)
        feats["cumulative_return_96"] = (close / close.shift(self.spec.sequence_length)) - 1
        feats["rolling_sharpe_20"] = feats["log_return_1"].rolling(20, min_periods=5).mean() / (
            feats["log_return_1"].rolling(20, min_periods=5).std() + 1e-9
        )

        # Trend strength features
        adx, plus_di, minus_di = _adx(high, low, close, 14)
        feats["adx_14"] = adx
        feats["+di_14"] = plus_di
        feats["-di_14"] = minus_di
        feats["trend_strength"] = feats["+di_14"] - feats["-di_14"]
        feats["trend_direction_20"] = close / (close.shift(20) + 1e-9) - 1
        feats["trend_direction_50"] = close / (close.shift(50) + 1e-9) - 1

        # Statistical features
        for window in (20, 40, 60):
            feats[f"rolling_skew_{window}"] = close.rolling(window, min_periods=20).skew()
            feats[f"rolling_kurt_{window}"] = close.rolling(window, min_periods=20).kurt()
            feats[f"rolling_zscore_{window}"] = (close - _sma(close, window)) / (_rolling_std(close, window) + 1e-9)

        feats["hurst_100"] = _hurst(close, 100)
        feats["price_entropy_48"] = _entropy(close, 48, bins=24)
        feats["volume_entropy_48"] = _entropy(volume, 48, bins=24)

        # Support / Resistance distances
        for window in (20, 50):
            rolling_high = high.rolling(window, min_periods=1).max()
            rolling_low = low.rolling(window, min_periods=1).min()
            feats[f"dist_to_high_{window}"] = (rolling_high - close) / (rolling_high + 1e-9)
            feats[f"dist_to_low_{window}"] = (close - rolling_low) / (rolling_low + 1e-9)
            feats[f"range_width_{window}"] = (rolling_high - rolling_low) / (rolling_low + 1e-9)

        # Smart Money Concepts (simplified)
        swing = _swing_points(close, 3, 3)
        feats["swing_high_flag"] = (swing == 1).astype(float)
        feats["swing_low_flag"] = (swing == -1).astype(float)
        feats["swing_distance_high"] = close - close.shift(1).where(swing.shift(1) == 1)
        feats["swing_distance_low"] = close - close.shift(1).where(swing.shift(1) == -1)
        feats["bos_flag"] = (close > close.shift(5).rolling(5, min_periods=1).max()).astype(float)
        feats["choch_flag"] = (close < close.shift(5).rolling(5, min_periods=1).min()).astype(float)
        feats["liquidity_sweep"] = (high > high.shift(10)) & (close < close.shift(10))
        feats["liquidity_sweep"] = feats["liquidity_sweep"].astype(float)

        # Market profile approximations
        for window in (48, 96):
            rolling_close = close.rolling(window, min_periods=window // 2)
            mp_mean = rolling_close.mean()
            mp_std = rolling_close.std()
            feats[f"mp_value_area_high_{window}"] = mp_mean + mp_std
            feats[f"mp_value_area_low_{window}"] = mp_mean - mp_std
            feats[f"mp_point_of_control_{window}"] = mp_mean
            feats[f"mp_price_position_{window}"] = (close - mp_mean) / (mp_std + 1e-9)

        feats["time_index"] = np.arange(len(df))
        feats["hour_sin"] = np.sin(df["time"].dt.hour / 24 * 2 * np.pi)
        feats["hour_cos"] = np.cos(df["time"].dt.hour / 24 * 2 * np.pi)
        feats["dayofweek_sin"] = np.sin(df["time"].dt.dayofweek / 7 * 2 * np.pi)
        feats["dayofweek_cos"] = np.cos(df["time"].dt.dayofweek / 7 * 2 * np.pi)

        features = pd.DataFrame(feats).replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method="bfill").fillna(method="ffill").fillna(0.0)

        LOGGER.info("Feature matrix generated with %d columns", features.shape[1])
        if features.shape[1] < 100:
            raise ValueError(f"Expected at least 100 features, got {features.shape[1]}")

        return features


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------


@dataclass
class LabelSpec:
    buy_conditions_required: int = 2
    sell_conditions_required: int = 2


class LabelGenerator:
    """Generates Buy/Sell/Direction/Regime labels based on institutional heuristics."""

    def __init__(self, spec: LabelSpec):
        self.spec = spec

    def generate(self, df: DataFrame, features: DataFrame) -> DataFrame:
        close = df["close"]
        volume = df["volume"]

        ma20 = features.get("sma_20", _sma(close, 20))
        ma50 = features.get("sma_50", _sma(close, 50))
        ema20_slope = features.get("ema_slope_20", _ema(close, 20).diff())
        bb_mid = features.get("bb_mid", _sma(close, 20))
        rsi_14 = features.get("rsi_14", _rsi(close, 14))
        macd_hist = features.get("macd_hist", _macd(close)[2])
        volume_ma20 = features.get("volume_ma_20", _sma(volume, 20))
        atr_14 = features.get("atr_14", _atr(df["high"], df["low"], close, 14))
        adx_14 = features.get("adx_14", _adx(df["high"], df["low"], close, 14)[0])
        plus_di = features.get("+di_14", _adx(df["high"], df["low"], close, 14)[1])
        minus_di = features.get("-di_14", _adx(df["high"], df["low"], close, 14)[2])

        # BUY conditions
        cond_buy = [
            (close > ma20) & (ma20 > ma50),
            (rsi_14 > 50) | ((rsi_14.shift(1) < 30) & (rsi_14 > 30)),
            (close > bb_mid) & (ema20_slope > 0),
            (volume > 1.5 * volume_ma20),
            (features["position_in_range"] >= 0.7),
            (macd_hist > 0) & (macd_hist.diff() > 0),
        ]
        buy_signal = sum(cond_buy) >= self.spec.buy_conditions_required

        # SELL conditions
        cond_sell = [
            (close < ma20) & (ma20 < ma50),
            (rsi_14 < 50) | ((rsi_14.shift(1) > 70) & (rsi_14 < rsi_14.shift(1))),
            (close < bb_mid) & (ema20_slope < 0),
            (volume > 1.5 * volume_ma20) & (features["position_in_range"] <= 0.3),
            (macd_hist < 0) & (macd_hist.diff() < 0),
        ]
        # Additional condition using DI
        cond_sell.append(minus_di > plus_di)
        sell_signal = sum(cond_sell) >= self.spec.sell_conditions_required

        direction = (close.diff().fillna(0.0) >= 0).astype(int)

        trend_lookback = 32
        trend = close.pct_change(trend_lookback)
        volatile = (atr_14 > atr_14.quantile(0.9)) | (volume > volume.quantile(0.9))

        regime = pd.Series(2, index=df.index, dtype=int)  # default RANGE
        regime = regime.mask((trend > 0.02) & (adx_14 > 25), 0)  # BULL_STRONG
        regime = regime.mask((trend > 0.0) & (trend <= 0.02) & (adx_14 <= 25), 1)  # BULL_WEAK
        regime = regime.mask((trend >= -0.02) & (trend < 0.0) & (adx_14 <= 25), 3)  # BEAR_WEAK
        regime = regime.mask((trend < -0.02) & (adx_14 > 25), 4)  # BEAR_STRONG
        regime = regime.mask(volatile, 5)  # VOLATILE

        labels = pd.DataFrame(
            {
                "buy": buy_signal.astype(float),
                "sell": sell_signal.astype(float),
                "direction": direction.astype(int),
                "regime": regime.fillna(2).astype(int),
            },
            index=df.index,
        )

        return labels


# ---------------------------------------------------------------------------
# Sequence datasets
# ---------------------------------------------------------------------------


class SequenceDataset(Dataset):
    """Supervised dataset yielding sequences and multi-task targets."""

    def __init__(
        self,
        sequences: np.ndarray,
        buy: np.ndarray,
        sell: np.ndarray,
        direction: np.ndarray,
        regime: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.sequences = sequences.astype(np.float32)
        self.buy = buy.astype(np.float32)
        self.sell = sell.astype(np.float32)
        self.direction = direction.astype(np.int64)
        self.regime = regime.astype(np.int64)
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "sequence": torch.from_numpy(self.sequences[idx]),
            "buy": torch.tensor(self.buy[idx]),
            "sell": torch.tensor(self.sell[idx]),
            "direction": torch.tensor(self.direction[idx]),
            "regime": torch.tensor(self.regime[idx]),
            "index": torch.tensor(self.indices[idx]),
        }


class UnsupervisedSequenceDataset(Dataset):
    """Dataset for masked modeling / contrastive learning."""

    def __init__(self, sequences: np.ndarray) -> None:
        self.sequences = sequences.astype(np.float32)

    def __len__(self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.sequences[idx])


# ---------------------------------------------------------------------------
# Data module orchestration
# ---------------------------------------------------------------------------


@dataclass
class DataModuleConfig:
    csv_path: Path
    sequence_length: int = 96
    train_val_split: float = 0.8
    num_workers: int = 4
    pin_memory: bool = True
    cache_features: Optional[Path] = None
    cache_labels: Optional[Path] = None
    cache_sequences: Optional[Path] = None
    scaler_type: str = "robust"


class NatronDataModule:
    """High-level access point to create all Natron datasets and loaders."""

    def __init__(self, config: DataModuleConfig):
        self.config = config
        self.feature_spec = FeatureSpec(
            cache_path=config.cache_features,
            sequence_length=config.sequence_length,
            scaler_type=config.scaler_type,
        )
        self.label_spec = LabelSpec()
        self.feature_engineer = FeatureEngineer(self.feature_spec)
        self.label_generator = LabelGenerator(self.label_spec)
        self.scaler = None
        self._features: Optional[DataFrame] = None
        self._labels: Optional[DataFrame] = None
        self.raw_df: Optional[DataFrame] = None

    def prepare(self) -> None:
        df = self._load_csv(self.config.csv_path)
        self.raw_df = df.copy()
        features, scaler = self.feature_engineer.transform(df, allow_cache=True)
        self.scaler = scaler
        labels = self._load_or_generate_labels(df, features)
        self._features, self._labels = features, labels
        LOGGER.info("Data module prepared with %d samples", len(df))

    def supervised_datasets(self) -> Tuple[SequenceDataset, SequenceDataset]:
        if self._features is None or self._labels is None:
            raise RuntimeError("Data module not prepared. Call prepare() first.")

        sequences, buy, sell, direction, regime, indices = self._build_sequences(self._features, self._labels)
        split_idx = int(len(sequences) * self.config.train_val_split)
        train_ds = SequenceDataset(
            sequences[:split_idx],
            buy[:split_idx],
            sell[:split_idx],
            direction[:split_idx],
            regime[:split_idx],
            indices[:split_idx],
        )
        val_ds = SequenceDataset(
            sequences[split_idx:],
            buy[split_idx:],
            sell[split_idx:],
            direction[split_idx:],
            regime[split_idx:],
            indices[split_idx:],
        )
        return train_ds, val_ds

    def unsupervised_dataset(self) -> UnsupervisedSequenceDataset:
        if self._features is None:
            raise RuntimeError("Data module not prepared. Call prepare() first.")
        sequences, *_ = self._build_sequences(self._features, None)
        return UnsupervisedSequenceDataset(sequences)

    def dataloaders(
        self,
        batch_size_sup: int,
        batch_size_unsup: Optional[int] = None,
    ) -> Dict[str, DataLoader]:
        train_ds, val_ds = self.supervised_datasets()
        loaders = {
            "train": DataLoader(
                train_ds,
                batch_size=batch_size_sup,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
            ),
            "val": DataLoader(
                val_ds,
                batch_size=batch_size_sup,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
            ),
        }
        if batch_size_unsup:
            unsup_ds = self.unsupervised_dataset()
            loaders["unsupervised"] = DataLoader(
                unsup_ds,
                batch_size=batch_size_unsup,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
            )
        return loaders

    def _load_csv(self, path: Path) -> DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found at {path}")
        df = pd.read_csv(path)
        missing = set(self.feature_spec.input_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        return df

    def _load_or_generate_labels(self, df: DataFrame, features: DataFrame) -> DataFrame:
        if self.config.cache_labels and self.config.cache_labels.exists():
            LOGGER.info("Loading cached labels from %s", self.config.cache_labels)
            return pd.read_parquet(self.config.cache_labels)
        labels = self.label_generator.generate(df, features)
        if self.config.cache_labels:
            self.config.cache_labels.parent.mkdir(parents=True, exist_ok=True)
            labels.to_parquet(self.config.cache_labels)
            LOGGER.info("Cached labels to %s", self.config.cache_labels)
        return labels

    def _build_sequences(
        self,
        features: DataFrame,
        labels: Optional[DataFrame],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        seq_len = self.config.sequence_length
        feature_array = features.to_numpy(dtype=np.float32)
        total = len(features) - seq_len + 1
        if total <= 0:
            raise ValueError("Not enough samples to build sequences.")

        sequences = np.lib.stride_tricks.sliding_window_view(feature_array, (seq_len, feature_array.shape[1]))[:, 0, :, :]
        buy = np.zeros(total, dtype=np.float32)
        sell = np.zeros(total, dtype=np.float32)
        direction = np.zeros(total, dtype=np.int64)
        regime = np.zeros(total, dtype=np.int64)
        target_indices = np.arange(seq_len - 1, len(features), dtype=np.int64)

        if labels is not None:
            buy = labels["buy"].values[seq_len - 1 :].astype(np.float32)
            sell = labels["sell"].values[seq_len - 1 :].astype(np.float32)
            direction = labels["direction"].values[seq_len - 1 :].astype(np.int64)
            regime = labels["regime"].values[seq_len - 1 :].astype(np.int64)

        if self.config.cache_sequences:
            payload = {
                "sequences": sequences,
                "buy": buy,
                "sell": sell,
                "direction": direction,
                "regime": regime,
                "indices": target_indices,
            }
            self.config.cache_sequences.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, self.config.cache_sequences)
            LOGGER.info("Cached sequence tensors to %s", self.config.cache_sequences)

        return sequences, buy, sell, direction, regime, target_indices


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def load_config(path: Path) -> Dict:
    if path.suffix in {".yml", ".yaml"}:
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    raise ValueError(f"Unsupported config format: {path.suffix}")


def create_data_module_from_config(config: Dict) -> NatronDataModule:
    paths = config.get("paths", {})
    data_cfg = config.get("data", {})
    module_config = DataModuleConfig(
        csv_path=Path(paths.get("data_csv", "data/data_export.csv")),
        sequence_length=data_cfg.get("sequence_length", 96),
        train_val_split=data_cfg.get("train_val_split", 0.8),
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
        cache_features=Path(paths["feature_cache"]) if paths.get("feature_cache") else None,
        cache_labels=Path(paths["label_cache"]) if paths.get("label_cache") else None,
        cache_sequences=Path(paths["sequence_cache"]) if paths.get("sequence_cache") else None,
        scaler_type=data_cfg.get("scaler", "robust"),
    )
    return NatronDataModule(module_config)


def build_dataloaders_from_config(config: Dict) -> Dict[str, DataLoader]:
    data_module = create_data_module_from_config(config)
    data_module.prepare()

    pre_cfg = config.get("pretraining", {})
    sup_cfg = config.get("supervised", {})
    batch_unsup = pre_cfg.get("batch_size") if pre_cfg.get("enabled", True) else None

    loaders = data_module.dataloaders(
        batch_size_sup=sup_cfg.get("batch_size", 48),
        batch_size_unsup=batch_unsup,
    )
    LOGGER.info("Constructed dataloaders: %s", list(loaders.keys()))
    return loaders


__all__ = [
    "FeatureEngineer",
    "LabelGenerator",
    "SequenceDataset",
    "UnsupervisedSequenceDataset",
    "NatronDataModule",
    "DataModuleConfig",
    "load_config",
    "create_data_module_from_config",
    "build_dataloaders_from_config",
]
