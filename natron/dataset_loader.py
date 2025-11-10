"""
Natron data ingestion, feature engineering, labeling, and dataset builders.

This module converts raw OHLCV price series into a dense 100-dimensional
feature representation, generates multi-task trading labels, and exposes
PyTorch-ready datasets for the Natron Transformer training pipeline.

Author: GPT-5 Codex High (Lead Engineer)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class NatronDatasetConfig:
    """Configuration container for dataset preparation."""

    sequence_length: int = 96
    feature_count: int = 100
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    masking_probability: float = 0.15
    normalization: bool = True
    cache_features: bool = True
    feature_cache_path: Optional[Path] = None
    random_state: int = 42
    min_samples: int = 1500

    def __post_init__(self) -> None:
        total = self.train_split + self.val_split + self.test_split
        if not np.isclose(total, 1.0):
            raise ValueError("train/val/test splits must sum to 1.0")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.feature_count != 100:
            raise ValueError(
                "Natron feature space must remain 100-dimensional. "
                f"Received feature_count={self.feature_count}."
            )


# ---------------------------------------------------------------------------
# Technical indicator helpers
# ---------------------------------------------------------------------------


def _ema(values: pd.Series, span: int) -> pd.Series:
    return values.ewm(span=span, adjust=False, min_periods=span).mean()


def _sma(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window=window, min_periods=window).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _roc(series: pd.Series, period: int) -> pd.Series:
    return series.pct_change(periods=period) * 100


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    shifted_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - shifted_close).abs(),
            (low - shifted_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.rolling(window=period, min_periods=period).mean()


def _bollinger_bands(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(window=period, min_periods=period).min()
    highest_high = high.rolling(window=period, min_periods=period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d = k.rolling(window=3, min_periods=3).mean()
    return k, d


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    highest_high = high.rolling(window=period, min_periods=period).max()
    lowest_low = low.rolling(window=period, min_periods=period).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)


def _typical_price(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    return (high + low + close) / 3.0


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    typical_price = _typical_price(high, low, close)
    raw_money_flow = typical_price * volume
    positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0.0)
    negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0.0)
    pos_sum = positive_flow.rolling(window=period, min_periods=period).sum()
    neg_sum = negative_flow.rolling(window=period, min_periods=period).sum()
    money_ratio = pos_sum / neg_sum.replace(0, np.nan)
    mfi = 100 - 100 / (1 + money_ratio)
    return mfi.fillna(50.0)


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (volume * direction).fillna(0).cumsum()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    up_move = high.diff()
    down_move = low.diff(-1) * -1
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = _true_range(high, low, close)
    atr = tr.rolling(window=period, min_periods=period).mean()

    plus_di = 100 * (plus_dm.rolling(window=period, min_periods=period).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period, min_periods=period).sum() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.rolling(window=period, min_periods=period).mean()
    return adx, plus_di, minus_di


def _aroon(high: pd.Series, low: pd.Series, period: int = 25) -> Tuple[pd.Series, pd.Series]:
    def aroon_up(series: pd.Series) -> pd.Series:
        return 100 * (series.rolling(window=period, min_periods=period).apply(np.argmax, raw=True) / period)

    def aroon_down(series: pd.Series) -> pd.Series:
        return 100 * (series.rolling(window=period, min_periods=period).apply(np.argmin, raw=True) / period)

    up = aroon_up(high)
    down = aroon_down(low)
    return up, down


def _hurst_exponent(series: pd.Series, window: int = 100) -> pd.Series:
    """Compute rolling Hurst exponent using rescaled range."""
    hurst_values = []
    for i in range(len(series)):
        if i < window:
            hurst_values.append(np.nan)
            continue
        subset = series.iloc[i - window + 1 : i + 1]
        cumulative = subset.cumsum()
        r = cumulative.max() - cumulative.min()
        s = subset.std()
        if s == 0:
            hurst_values.append(0.5)
        else:
            hurst_values.append(np.log(r / s + 1e-9) / np.log(window))
    return pd.Series(hurst_values, index=series.index)


def _rolling_entropy(series: pd.Series, window: int = 20, bins: int = 10) -> pd.Series:
    """Rolling Shannon entropy of the sign of returns."""
    returns = series.pct_change()
    entropy_values = returns.rolling(window=window).apply(
        lambda s: stats.entropy(np.histogram(s.dropna(), bins=bins)[0] + 1e-9), raw=False
    )
    return entropy_values


def _swing_points(series: pd.Series, left: int = 3, right: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Identify swing highs/lows using a moving window."""
    highs = series.rolling(window=left + right + 1, center=True).apply(lambda s: int(s[left] == s.max()), raw=True)
    lows = series.rolling(window=left + right + 1, center=True).apply(lambda s: int(s[left] == s.min()), raw=True)
    return highs.fillna(0), lows.fillna(0)


def _volume_profile(close: pd.Series, volume: pd.Series, window: int = 20, bins: int = 10) -> pd.DataFrame:
    """Approximate rolling market profile metrics."""
    typical_price = close
    vp_columns = {
        "volume_poc_ratio": [],
        "vah_distance": [],
        "val_distance": [],
        "profile_width": [],
        "volume_entropy": [],
        "volume_climax": [],
        "time_above_vwap": [],
        "time_below_vwap": [],
        "session_range_ratio": [],
        "closing_imbalance": [],
        "poc_price": [],
        "value_area_mid": [],
    }

    vwap = (volume * typical_price).rolling(window=window, min_periods=window).sum() / volume.rolling(
        window=window, min_periods=window
    ).sum()

    for idx in range(len(close)):
        if idx < window:
            for key in vp_columns:
                vp_columns[key].append(np.nan)
            continue

        prices_window = typical_price.iloc[idx - window + 1 : idx + 1]
        volume_window = volume.iloc[idx - window + 1 : idx + 1]
        hist, bin_edges = np.histogram(prices_window, bins=bins, weights=volume_window)
        poc_index = hist.argmax()
        poc_price = (bin_edges[poc_index] + bin_edges[poc_index + 1]) / 2
        cumulative_volume = np.cumsum(hist) / hist.sum() if hist.sum() > 0 else np.zeros_like(hist)
        try:
            vah_index = np.where(cumulative_volume >= 0.85)[0][0]
            val_index = np.where(cumulative_volume >= 0.15)[0][0]
        except IndexError:
            vah_index = poc_index
            val_index = poc_index

        vah_price = (bin_edges[vah_index] + bin_edges[vah_index + 1]) / 2
        val_price = (bin_edges[val_index] + bin_edges[val_index + 1]) / 2
        profile_width = vah_price - val_price

        entropy = stats.entropy(hist + 1e-9) if hist.sum() > 0 else 0.0
        volume_climax = hist.max() / (hist.mean() + 1e-9) if hist.mean() > 0 else 0.0
        session_range = prices_window.max() - prices_window.min()
        session_range_ratio = session_range / (prices_window.mean() + 1e-9)
        close_session = prices_window.iloc[-1]

        vp_columns["volume_poc_ratio"].append((close_session - poc_price) / (profile_width + 1e-9))
        vp_columns["vah_distance"].append((close_session - vah_price) / (profile_width + 1e-9))
        vp_columns["val_distance"].append((close_session - val_price) / (profile_width + 1e-9))
        vp_columns["profile_width"].append(profile_width)
        vp_columns["volume_entropy"].append(entropy)
        vp_columns["volume_climax"].append(volume_climax)
        vp_columns["time_above_vwap"].append((prices_window > vwap.iloc[idx]).sum() / window)
        vp_columns["time_below_vwap"].append((prices_window < vwap.iloc[idx]).sum() / window)
        vp_columns["session_range_ratio"].append(session_range_ratio)
        vp_columns["closing_imbalance"].append((close_session - prices_window.mean()) / (session_range + 1e-9))
        vp_columns["poc_price"].append(poc_price)
        vp_columns["value_area_mid"].append((vah_price + val_price) / 2)

    return pd.DataFrame(vp_columns, index=close.index)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


class NatronFeatureEngineer:
    """
    Generate ~100 engineered features from OHLCV data.

    Groups:
    - Moving Average (13)
    - Momentum (13)
    - Volatility (15)
    - Volume (9)
    - Price Pattern (8)
    - Returns (8)
    - Trend Strength (6)
    - Statistical (6)
    - Support/Resistance (4)
    - Smart Money Concepts (6)
    - Market Profile (12)
    """

    def __init__(
        self,
        config: NatronDatasetConfig,
        cache_path: Optional[Path] = None,
        use_cache: Optional[bool] = None,
    ) -> None:
        self.config = config
        self.cache_path = cache_path or config.feature_cache_path
        self.use_cache = config.cache_features if use_cache is None else use_cache

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute feature matrix from raw OHLCV candles.

        Parameters
        ----------
        df:
            DataFrame with columns: time, open, high, low, close, volume
        """
        if self.use_cache and self.cache_path and Path(self.cache_path).exists():
            return pd.read_parquet(self.cache_path)

        features = self._compute_features(df)
        if self.use_cache and self.cache_path:
            Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
            features.to_parquet(self.cache_path)
        return features

    # pylint: disable=too-many-statements, too-many-locals
    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        open_ = df["open"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        volume = df["volume"].astype(float)

        tp = _typical_price(high, low, close)
        returns = close.pct_change().fillna(0.0)
        log_returns = np.log(close / close.shift(1)).fillna(0.0)

        features: Dict[str, pd.Series] = {}

        # Moving Average Features (13)
        ma_windows = [5, 10, 20, 50, 100]
        for window in ma_windows:
            features[f"ma_{window}"] = _sma(close, window)
            features[f"ema_{window}"] = _ema(close, window)

        features["ma20_slope"] = features["ma_20"].diff()
        features["ma50_slope"] = features["ma_50"].diff()
        features["price_ma20_ratio"] = close / (features["ma_20"] + 1e-9)
        features["price_ma50_ratio"] = close / (features["ma_50"] + 1e-9)
        features["ma_cross_signal"] = np.sign(features["ma_20"] - features["ma_50"])

        # Momentum Features (13)
        features["rsi_14"] = _rsi(close, 14)
        features["rsi_7"] = _rsi(close, 7)
        features["rsi_21"] = _rsi(close, 21)
        features["roc_5"] = _roc(close, 5)
        features["roc_10"] = _roc(close, 10)
        features["cci_20"] = (
            (tp - tp.rolling(window=20, min_periods=20).mean())
            / (0.015 * tp.rolling(window=20, min_periods=20).apply(lambda s: np.abs(s - s.mean()).mean()))
        )
        stoch_k, stoch_d = _stochastic(high, low, close, 14)
        features["stoch_k"] = stoch_k
        features["stoch_d"] = stoch_d
        ema12 = _ema(close, 12)
        ema26 = _ema(close, 26)
        macd = ema12 - ema26
        signal = _ema(macd, 9)
        features["macd_line"] = macd
        features["macd_signal"] = signal
        features["macd_hist"] = macd - signal
        features["williams_r"] = _williams_r(high, low, close, 14)
        features["momentum_10"] = close - close.shift(10)

        # Volatility Features (15)
        atr14 = _atr(high, low, close, 14)
        atr30 = _atr(high, low, close, 30)
        boll_up, boll_mid, boll_low = _bollinger_bands(close, 20, 2.0)
        features["atr_14"] = atr14
        features["atr_30"] = atr30
        features["boll_upper"] = boll_up
        features["boll_mid"] = boll_mid
        features["boll_lower"] = boll_low
        features["boll_bandwidth"] = (boll_up - boll_low) / (boll_mid + 1e-9)
        features["boll_pctb"] = (close - boll_low) / (boll_up - boll_low + 1e-9)
        tr = _true_range(high, low, close)
        features["true_range"] = tr
        features["keltner_upper"] = _ema(tp, 20) + 2 * atr14
        features["keltner_lower"] = _ema(tp, 20) - 2 * atr14
        features["std_10"] = close.rolling(window=10, min_periods=10).std()
        features["std_20"] = close.rolling(window=20, min_periods=20).std()
        features["std_50"] = close.rolling(window=50, min_periods=50).std()
        features["hv_30"] = log_returns.rolling(window=30, min_periods=30).std() * np.sqrt(252)
        features["donchian_high_20"] = high.rolling(window=20, min_periods=20).max()
        features["donchian_low_20"] = low.rolling(window=20, min_periods=20).min()
        features["chaikin_volatility"] = (high - low).ewm(span=10, adjust=False).mean()

        # Volume Features (9)
        features["volume"] = volume
        features["volume_ema_20"] = _ema(volume, 20)
        features["volume_ratio_20"] = volume / (volume.rolling(window=20, min_periods=20).mean() + 1e-9)
        obv = _obv(close, volume)
        features["obv"] = obv
        features["obv_slope_5"] = obv.diff(5)
        features["mfi_14"] = _mfi(high, low, close, volume, 14)
        features["vwap_20"] = (volume * tp).rolling(window=20, min_periods=20).sum() / (
            volume.rolling(window=20, min_periods=20).sum() + 1e-9
        )
        features["volume_zscore_20"] = (volume - volume.rolling(window=20, min_periods=20).mean()) / (
            volume.rolling(window=20, min_periods=20).std() + 1e-9
        )
        features["accumulation_distribution"] = ((close - low) - (high - close)) / (high - low + 1e-9) * volume

        # Price Pattern Features (8)
        candle_range = high - low
        body = (close - open_).abs()
        features["candle_body_pct"] = body / (candle_range + 1e-9)
        features["upper_shadow_pct"] = (high - close.clip(lower=open_)) / (candle_range + 1e-9)
        features["lower_shadow_pct"] = (open_.clip(upper=close) - low) / (candle_range + 1e-9)
        features["is_doji"] = (body / (candle_range + 1e-9) < 0.1).astype(float)
        features["gap_up_pct"] = (open_ - close.shift(1)) / (close.shift(1) + 1e-9)
        features["gap_down_pct"] = (open_.shift(-1) - close) / (close + 1e-9)
        features["candle_range"] = candle_range
        features["close_position"] = (close - low) / (candle_range + 1e-9)

        # Return Features (8)
        features["log_return_1"] = log_returns
        features["log_return_5"] = np.log(close / close.shift(5)).replace([np.inf, -np.inf], 0.0)
        features["log_return_10"] = np.log(close / close.shift(10)).replace([np.inf, -np.inf], 0.0)
        features["intraday_return"] = (close - open_) / (open_ + 1e-9)
        features["high_low_pct"] = candle_range / (close + 1e-9)
        features["cumulative_return"] = np.log(close / close.iloc[0]).replace([np.inf, -np.inf], 0.0)
        features["rolling_return_20"] = close.pct_change(20)
        features["rolling_return_50"] = close.pct_change(50)

        # Trend Strength Features (6)
        adx, plus_di, minus_di = _adx(high, low, close, 14)
        features["adx_14"] = adx
        features["plus_di_14"] = plus_di
        features["minus_di_14"] = minus_di
        aroon_up, aroon_down = _aroon(high, low, 25)
        features["aroon_up_25"] = aroon_up
        features["aroon_down_25"] = aroon_down
        features["aroon_oscillator"] = aroon_up - aroon_down

        # Statistical Features (6)
        features["skew_20"] = close.rolling(window=20, min_periods=20).skew()
        features["kurtosis_20"] = close.rolling(window=20, min_periods=20).kurt()
        features["zscore_20"] = (close - close.rolling(window=20, min_periods=20).mean()) / (
            close.rolling(window=20, min_periods=20).std() + 1e-9
        )
        features["hurst_100"] = _hurst_exponent(log_returns.fillna(0.0), 100)
        features["rolling_median_20"] = close.rolling(window=20, min_periods=20).median()
        features["rolling_iqr_20"] = close.rolling(window=20, min_periods=20).quantile(0.75) - close.rolling(
            window=20, min_periods=20
        ).quantile(0.25)

        # Support/Resistance Features (4)
        features["dist_high_20"] = (close - high.rolling(window=20, min_periods=20).max()) / (close + 1e-9)
        features["dist_low_20"] = (close - low.rolling(window=20, min_periods=20).min()) / (close + 1e-9)
        features["dist_high_50"] = (close - high.rolling(window=50, min_periods=50).max()) / (close + 1e-9)
        features["dist_low_50"] = (close - low.rolling(window=50, min_periods=50).min()) / (close + 1e-9)

        # Smart Money Concepts (6)
        swing_highs, swing_lows = _swing_points(high, 3, 3)
        features["swing_high"] = swing_highs
        features["swing_low"] = swing_lows
        features["bos_signal"] = np.where(close > high.shift(1), 1.0, 0.0)
        features["choch_signal"] = np.where(close < low.shift(1), 1.0, 0.0)
        features["premium_zone"] = np.where(close > (high.rolling(window=20).max() + low.rolling(window=20).min()) / 2, 1.0, 0.0)
        features["liquidity_grab"] = np.where((high > high.shift(1)) & (close < close.shift(1)), 1.0, 0.0)

        # Market Profile Features (12)
        vp_features = _volume_profile(close, volume, window=20, bins=12)
        for column in vp_features.columns:
            features[f"mp_{column}"] = vp_features[column]
        features["mp_vwap_dev"] = close - features["vwap_20"]
        features["mp_value_area_ratio"] = (
            (close - vp_features["value_area_mid"]) / (vp_features["profile_width"] + 1e-9)
        )

        feature_df = pd.DataFrame(features).replace([np.inf, -np.inf], np.nan)
        feature_df = feature_df.dropna().astype(np.float32)

        if feature_df.shape[1] != self.config.feature_count:
            raise RuntimeError(
                f"Expected {self.config.feature_count} features, got {feature_df.shape[1]}."
            )
        if len(feature_df) < self.config.min_samples:
            raise RuntimeError(
                f"Insufficient samples after feature engineering ({len(feature_df)}). "
                f"Need at least {self.config.min_samples}."
            )

        return feature_df


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------


class NatronLabelGenerator:
    """Generate institutional-style buy/sell/direction/regime labels."""

    def __init__(self, feature_df: pd.DataFrame, ohlcv_df: pd.DataFrame):
        if len(feature_df) != len(ohlcv_df.loc[feature_df.index]):
            raise ValueError("Feature frame and OHLCV frame must be aligned.")
        self.features = feature_df
        self.ohlcv = ohlcv_df.loc[feature_df.index]

    def generate(self) -> pd.DataFrame:
        close = self.ohlcv["close"]
        volume = self.ohlcv["volume"]
        ma20 = self.features["ma_20"]
        ma50 = self.features["ma_50"]
        rsi = self.features["rsi_14"]
        macd_hist = self.features["macd_hist"]
        boll_mid = self.features["boll_mid"]
        ma20_slope = self.features["ma20_slope"]
        position = self.features["close_position"]
        volume_ratio = self.features["volume_ratio_20"]
        adx = self.features["adx_14"]

        # Buy signal conditions
        cond_buy = [
            (close > ma20) & (ma20 > ma50),
            (rsi > 50) | ((rsi.shift(1) < 30) & (rsi > 30)),
            (close > boll_mid) & (ma20_slope > 0),
            volume_ratio > 1.5,
            position >= 0.7,
            (macd_hist > 0) & (macd_hist.diff() > 0),
        ]
        buy_votes = sum(cond.astype(float) for cond in cond_buy)
        buy_label = (buy_votes >= 2).astype(int)

        # Sell signal conditions
        cond_sell = [
            (close < ma20) & (ma20 < ma50),
            (rsi < 50) | ((rsi.shift(1) > 70) & (rsi < 70)),
            (close < boll_mid) & (ma20_slope < 0),
            (volume_ratio > 1.5) & (position <= 0.3),
            (macd_hist < 0) & (macd_hist.diff() < 0),
        ]
        sell_votes = sum(cond.astype(float) for cond in cond_sell)
        sell_label = (sell_votes >= 2).astype(int)

        # Direction label (future close vs current close)
        direction = (close.shift(-1) > close).astype(int)

        # Regime classification
        trend_pct = (ma20 - ma50) / (ma50 + 1e-9) * 100
        atr = self.features["atr_14"]
        atr_threshold = atr.rolling(window=200, min_periods=200).quantile(0.9).fillna(method="bfill")
        volatile = (atr > atr_threshold) | (volume_ratio > 2.0)

        regime = pd.Series(2, index=self.features.index)  # default RANGE
        regime = regime.mask((trend_pct > 2) & (adx > 25), 0)  # BULL_STRONG
        regime = regime.mask((trend_pct > 0) & (trend_pct <= 2) & (adx <= 25), 1)  # BULL_WEAK
        regime = regime.mask((trend_pct < 0) & (trend_pct >= -2) & (adx <= 25), 3)  # BEAR_WEAK
        regime = regime.mask((trend_pct < -2) & (adx > 25), 4)  # BEAR_STRONG
        regime = regime.mask(volatile, 5)  # VOLATILE

        labels = pd.DataFrame(
            {
                "buy": buy_label.astype(int),
                "sell": sell_label.astype(int),
                "direction": direction.fillna(method="ffill").fillna(0).astype(int),
                "regime": regime.fillna(2).astype(int),
            },
            index=self.features.index,
        )
        labels = labels.dropna()
        return labels


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------


def _normalize_features(
    features: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-6
    normalized = (features - mean) / std
    return normalized.astype(np.float32), mean.squeeze(0), std.squeeze(0)


def _create_sequences(
    features: np.ndarray,
    labels: pd.DataFrame,
    sequence_length: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    num_samples = len(features) - sequence_length + 1
    sequences = np.stack(
        [features[i : i + sequence_length] for i in range(num_samples)],
        axis=0,
    )
    label_indices = labels.iloc[sequence_length - 1 : sequence_length - 1 + num_samples]
    label_tensors = {
        "buy": label_indices["buy"].to_numpy(dtype=np.float32),
        "sell": label_indices["sell"].to_numpy(dtype=np.float32),
        "direction": label_indices["direction"].to_numpy(dtype=np.int64),
        "regime": label_indices["regime"].to_numpy(dtype=np.int64),
    }
    return sequences.astype(np.float32), label_tensors


class NatronSequenceDataset(Dataset):
    """Supervised dataset returning Natron multi-task labels."""

    def __init__(
        self,
        sequences: np.ndarray,
        labels: Dict[str, np.ndarray],
    ) -> None:
        self.sequences = torch.from_numpy(sequences)
        self.labels = {
            key: torch.from_numpy(value) for key, value in labels.items()
        }

    def __len__(self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "sequence": self.sequences[idx],
            "buy": self.labels["buy"][idx],
            "sell": self.labels["sell"][idx],
            "direction": self.labels["direction"][idx],
            "regime": self.labels["regime"][idx],
        }


class NatronMaskedModelingDataset(Dataset):
    """Dataset for masked time-series modeling."""

    def __init__(
        self,
        sequences: np.ndarray,
        mask_probability: float,
    ) -> None:
        self.sequences = torch.from_numpy(sequences)
        self.mask_probability = mask_probability

    def __len__(self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        mask = torch.bernoulli(torch.full(sequence.shape[:-1], self.mask_probability)).bool()
        masked_sequence = sequence.clone()
        masked_sequence[mask] = 0.0  # zero masking
        return {
            "input": masked_sequence,
            "target": sequence,
            "mask": mask,
        }


class NatronContrastiveDataset(Dataset):
    """Dataset returning two augmented sequence views for contrastive pretraining."""

    def __init__(
        self,
        sequences: np.ndarray,
        jitter_std: float = 0.01,
        time_dropout_prob: float = 0.05,
    ) -> None:
        self.sequences = torch.from_numpy(sequences)
        self.jitter_std = jitter_std
        self.time_dropout_prob = time_dropout_prob

    def __len__(self) -> int:
        return self.sequences.shape[0]

    def _augment(self, sequence: torch.Tensor) -> torch.Tensor:
        augmented = sequence.clone()
        noise = torch.randn_like(augmented) * self.jitter_std
        augmented = augmented + noise
        dropout_mask = torch.rand(sequence.shape[0]) < self.time_dropout_prob
        augmented[dropout_mask] = 0.0
        return augmented

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        return {
            "view_1": self._augment(sequence),
            "view_2": self._augment(sequence),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_ohlcv(csv_path: Path) -> pd.DataFrame:
    """Load OHLCV data from CSV, ensuring datetime index."""
    df = pd.read_csv(csv_path)
    if "time" not in df.columns:
        raise ValueError("CSV must contain a 'time' column.")
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df.set_index("time")


def prepare_natron_datasets(
    csv_path: Path,
    config: NatronDatasetConfig,
) -> Dict[str, Dataset]:
    """
    Build Natron datasets for all training phases.

    Returns
    -------
    dict with keys: 'supervised_train', 'supervised_val', 'supervised_test',
    'masked_pretrain', 'contrastive_pretrain', plus normalization stats.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    raw_df = load_ohlcv(csv_path)
    feature_engineer = NatronFeatureEngineer(config=config)
    feature_df = feature_engineer.fit_transform(raw_df)
    aligned_ohlcv = raw_df.loc[feature_df.index]

    label_generator = NatronLabelGenerator(feature_df, aligned_ohlcv)
    label_df = label_generator.generate()
    feature_df = feature_df.loc[label_df.index]
    aligned_labels = label_df

    features_np = feature_df.to_numpy(dtype=np.float32)
    if config.normalization:
        features_np, feature_mean, feature_std = _normalize_features(features_np)
    else:
        feature_mean = np.zeros(config.feature_count, dtype=np.float32)
        feature_std = np.ones(config.feature_count, dtype=np.float32)

    sequences, label_tensors = _create_sequences(features_np, aligned_labels, config.sequence_length)

    num_samples = len(sequences)
    train_end = int(num_samples * config.train_split)
    val_end = train_end + int(num_samples * config.val_split)

    supervised_train = NatronSequenceDataset(sequences[:train_end], {k: v[:train_end] for k, v in label_tensors.items()})
    supervised_val = NatronSequenceDataset(
        sequences[train_end:val_end], {k: v[train_end:val_end] for k, v in label_tensors.items()}
    )
    supervised_test = NatronSequenceDataset(
        sequences[val_end:], {k: v[val_end:] for k, v in label_tensors.items()}
    )

    masked_dataset = NatronMaskedModelingDataset(sequences, config.masking_probability)
    contrastive_dataset = NatronContrastiveDataset(sequences)

    return {
        "supervised_train": supervised_train,
        "supervised_val": supervised_val,
        "supervised_test": supervised_test,
        "masked_pretrain": masked_dataset,
        "contrastive_pretrain": contrastive_dataset,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
    }
