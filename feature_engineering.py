"""
Feature engineering utilities for Natron Transformer.

Generates ~100 engineered features per OHLCV candle to enrich the raw market state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

try:
    import ta
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "The 'ta' package is required for feature generation. Install via `pip install ta`."
    ) from exc

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    rolling_windows: Iterable[int] = (5, 10, 14, 20, 30, 50, 96)
    volatility_windows: Iterable[int] = (10, 20, 50)
    regression_windows: Iterable[int] = (5, 10, 20)
    market_profile_bins: int = 12
    hurst_window: int = 96
    fillna: bool = True
    clip_sigma: Optional[float] = 5.0


def _log_return(series: Series, periods: int = 1) -> Series:
    return np.log(series / series.shift(periods))


def _zscore(series: Series, window: int) -> Series:
    return (series - series.rolling(window).mean()) / (series.rolling(window).std() + 1e-9)


def _slope(series: Series, window: int) -> Series:
    idx = np.arange(window)
    weights = idx - idx.mean()
    weights /= np.sum(weights**2)
    return series.rolling(window).apply(lambda x: np.dot(weights, x), raw=True)


def _percentile_rank(series: Series, window: int) -> Series:
    return series.rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])


def _candle_features(df: DataFrame) -> DataFrame:
    high, low, close, open_ = df["high"], df["low"], df["close"], df["open"]
    body = close - open_
    range_ = (high - low).replace(0, np.nan)
    features = pd.DataFrame(index=df.index)
    features["candle_body"] = body
    features["candle_range"] = range_
    features["body_pct"] = body / (range_ + 1e-9)
    features["upper_shadow"] = (high - close).where(close >= open_, high - open_)
    features["lower_shadow"] = (open_ - low).where(close >= open_, close - low)
    features["upper_shadow_pct"] = features["upper_shadow"] / (range_ + 1e-9)
    features["lower_shadow_pct"] = features["lower_shadow"] / (range_ + 1e-9)
    features["is_doji"] = (features["body_pct"].abs() < 0.1).astype(float)
    features["gap_up"] = (open_ > close.shift(1)).astype(float)
    features["gap_down"] = (open_ < close.shift(1)).astype(float)
    features["close_position_pct"] = (close - low) / (range_ + 1e-9)
    features["hl_ratio"] = high / (low + 1e-9)
    features["close_to_high"] = (high - close) / (range_ + 1e-9)
    features["close_to_low"] = (close - low) / (range_ + 1e-9)
    return features


def _market_profile(df: DataFrame, bins: int) -> DataFrame:
    close = df["close"]
    vol = df["volume"]
    features = pd.DataFrame(index=df.index)
    window = 96

    def _profile(window_close: Series, window_volume: Series):
        hist, edges = np.histogram(
            window_close, bins=bins, weights=window_volume, density=False
        )
        total = np.sum(hist) + 1e-9
        poc_idx = np.argmax(hist)
        poc = (edges[poc_idx] + edges[poc_idx + 1]) / 2
        value_area = np.cumsum(np.sort(hist)[::-1]) / total
        vah = np.quantile(window_close, 0.7)
        val = np.quantile(window_close, 0.3)
        entropy = -np.sum((hist / total) * np.log(hist / total + 1e-9))
        return poc, vah, val, entropy

    poc, vah, val, entropy = [], [], [], []
    for i in range(len(close)):
        if i < window:
            poc.append(np.nan)
            vah.append(np.nan)
            val.append(np.nan)
            entropy.append(np.nan)
            continue
        _poc, _vah, _val, _entropy = _profile(
            close.iloc[i - window : i], vol.iloc[i - window : i]
        )
        poc.append(_poc)
        vah.append(_vah)
        val.append(_val)
        entropy.append(_entropy)

    features["profile_poc"] = poc
    features["profile_vah"] = vah
    features["profile_val"] = val
    features["profile_entropy"] = entropy
    features["profile_poc_distance"] = close - features["profile_poc"]
    return features


def _hurst_exponent(series: Series, window: int) -> Series:
    def hurst_exponent(ts: np.ndarray) -> float:
        if np.all(ts == ts[0]):
            return 0.0
        lags = range(2, min(100, len(ts) // 2))
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0

    return series.rolling(window).apply(lambda x: hurst_exponent(np.array(x)), raw=False)


@dataclass
class FeatureEngineer:
    config: FeatureConfig = field(default_factory=FeatureConfig)

    def transform(self, df: DataFrame) -> DataFrame:
        """
        Generate engineered features for the provided OHLCV DataFrame.

        Args:
            df: DataFrame with columns [time, open, high, low, close, volume]

        Returns:
            Feature-enriched DataFrame aligned with original index.
        """
        df = df.copy()
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.sort_values("time")

        features = []

        # Moving Averages & Slopes
        for window in self.config.rolling_windows:
            sma = df["close"].rolling(window).mean().rename(f"sma_{window}")
            ema = df["close"].ewm(span=window, adjust=False).mean().rename(f"ema_{window}")
            slope = _slope(df["close"], window).rename(f"slope_close_{window}")
            close_to_sma = (df["close"] / (sma + 1e-9)).rename(f"close_to_sma_{window}")
            close_to_ema = (df["close"] / (ema + 1e-9)).rename(f"close_to_ema_{window}")
            features.extend([sma, ema, slope, close_to_sma, close_to_ema])

        # Crossovers
        for short, long in zip(self.config.rolling_windows, self.config.rolling_windows[1:]):
            sma_short = df["close"].rolling(short).mean()
            sma_long = df["close"].rolling(long).mean()
            features.append((sma_short - sma_long).rename(f"sma_diff_{short}_{long}"))
            ema_short = df["close"].ewm(span=short).mean()
            ema_long = df["close"].ewm(span=long).mean()
            features.append((ema_short - ema_long).rename(f"ema_diff_{short}_{long}"))

        # Momentum Features
        features.append(ta.momentum.RSIIndicator(df["close"], window=14).rsi().rename("rsi_14"))
        features.append(ta.momentum.RSIIndicator(df["close"], window=7).rsi().rename("rsi_7"))
        features.append(ta.momentum.ROCIndicator(df["close"], window=5).roc().rename("roc_5"))
        features.append(ta.momentum.ROCIndicator(df["close"], window=12).roc().rename("roc_12"))
        cci = ta.trend.CCIIndicator(df["high"], df["low"], df["close"], window=20).cci()
        features.append(cci.rename("cci_20"))
        features.append(ta.momentum.StochRSIIndicator(df["close"]).stochrsi().rename("stoch_rsi"))
        stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
        features.append(stoch.stoch().rename("stoch_k"))
        features.append(stoch.stoch_signal().rename("stoch_d"))
        macd = ta.trend.MACD(df["close"])
        features.append(macd.macd().rename("macd_line"))
        features.append(macd.macd_signal().rename("macd_signal"))
        features.append(macd.macd_diff().rename("macd_hist"))
        features.append(ta.momentum.WilliamsRIndicator(df["high"], df["low"], df["close"]).williams_r().rename("williams_r"))
        features.append(ta.momentum.KAMAIndicator(df["close"], window=10).kama().rename("kama_10"))
        features.append(ta.momentum.AwesomeOscillatorIndicator(df["high"], df["low"]).awesome_oscillator().rename("awesome_osc"))
        features.append(ta.trend.DPOIndicator(df["close"]).dpo().rename("dpo"))
        features.append(ta.momentum.TSIIndicator(df["close"]).tsi().rename("tsi"))
        features.append(ta.momentum.UltimateOscillator(df["high"], df["low"], df["close"]).ultimate_oscillator().rename("ultimate_osc"))
        features.append(ta.momentum.PercentagePriceOscillator(df["close"]).ppo().rename("ppo"))
        features.append(ta.momentum.PercentageVolumeOscillator(df["volume"]).pvo().rename("pvo"))

        # Volatility Features
        atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"])
        features.append(atr.average_true_range().rename("atr_14"))
        bb = ta.volatility.BollingerBands(df["close"])
        features.append(bb.bollinger_hband().rename("bb_high"))
        features.append(bb.bollinger_lband().rename("bb_low"))
        features.append(bb.bollinger_mavg().rename("bb_mid"))
        features.append(bb.bollinger_hband_indicator().rename("bb_high_ind"))
        features.append(bb.bollinger_lband_indicator().rename("bb_low_ind"))
        features.append(bb.bollinger_pband().rename("bb_pband"))
        for window in self.config.volatility_windows:
            features.append(df["close"].rolling(window).std().rename(f"std_{window}"))
            features.append((_log_return(df["close"], 1).rolling(window).std()).rename(f"realized_vol_{window}"))
            features.append((atr.average_true_range() / df["close"]).rolling(window).mean().rename(f"atr_ratio_{window}"))
        keltner = ta.volatility.KeltnerChannel(df["high"], df["low"], df["close"])
        features.append(keltner.keltner_channel_hband().rename("kc_high"))
        features.append(keltner.keltner_channel_lband().rename("kc_low"))
        donchian = ta.volatility.DonchianChannel(df["high"], df["low"], df["close"])
        features.append(donchian.donchian_channel_hband().rename("donchian_high"))
        features.append(donchian.donchian_channel_lband().rename("donchian_low"))

        # Volume Features
        features.append(ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume().rename("obv"))
        features.append(ta.volume.VolumeWeightedAveragePrice(df["high"], df["low"], df["close"], df["volume"]).volume_weighted_average_price().rename("vwap"))
        features.append(ta.volume.MFIIndicator(df["high"], df["low"], df["close"], df["volume"]).money_flow_index().rename("mfi"))
        ad = ta.volume.AccDistIndexIndicator(df["high"], df["low"], df["close"], df["volume"]).acc_dist_index()
        features.append(ad.rename("acc_dist_index"))
        features.append(ta.volume.ChaikinMoneyFlowIndicator(df["high"], df["low"], df["close"], df["volume"]).chaikin_money_flow().rename("cmf"))
        features.append(ta.volume.EaseOfMovementIndicator(df["high"], df["low"], df["volume"]).ease_of_movement().rename("eom"))
        features.append(ta.volume.VolumePriceTrendIndicator(df["close"], df["volume"]).volume_price_trend().rename("vpt"))
        features.append(ta.volume.ForceIndexIndicator(df["close"], df["volume"]).force_index().rename("force_index"))
        features.append((_log_return(df["volume"].replace(0, np.nan), 1)).rename("volume_log_return"))
        features.append((df["volume"] / df["volume"].rolling(20).mean()).rename("volume_vs_20"))

        # Price Pattern & Returns
        candle_features = _candle_features(df)
        features.append(candle_features["body_pct"])
        features.append(candle_features["upper_shadow_pct"])
        features.append(candle_features["lower_shadow_pct"])
        features.append(candle_features["close_position_pct"])
        features.append(candle_features["close_to_high"])
        features.append(candle_features["close_to_low"])
        features.append(candle_features["is_doji"])
        features.append(candle_features["gap_up"])
        features.append(candle_features["gap_down"])
        returns = _log_return(df["close"])
        features.append(returns.rename("log_return_1"))
        features.append(_log_return(df["close"], 4).rename("log_return_4"))
        features.append((_log_return(df["close"], 12)).rename("log_return_12"))
        features.append((returns.rolling(20).sum()).rename("cumulative_return_20"))
        features.append(((df["high"] - df["low"]) / df["close"]).rename("intraday_range_pct"))
        features.append((_log_return(df["high"] / df["low"], 1)).rename("range_return"))

        # Trend Strength
        adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"])
        features.append(adx.adx().rename("adx"))
        features.append(adx.adx_pos().rename("adx_pos"))
        features.append(adx.adx_neg().rename("adx_neg"))
        aroon = ta.trend.AroonIndicator(df["close"])
        features.append(aroon.aroon_up().rename("aroon_up"))
        features.append(aroon.aroon_down().rename("aroon_down"))
        features.append((aroon.aroon_up() - aroon.aroon_down()).rename("aroon_diff"))

        # Statistical Features
        for window in self.config.regression_windows:
            features.append(_zscore(df["close"], window).rename(f"zscore_close_{window}"))
            features.append(df["close"].rolling(window).skew().rename(f"skew_{window}"))
            features.append(df["close"].rolling(window).kurt().rename(f"kurt_{window}"))
            features.append((_slope(returns, window)).rename(f"return_slope_{window}"))
            features.append((_percentile_rank(df["close"], window)).rename(f"close_pct_rank_{window}"))
        features.append(_hurst_exponent(df["close"], self.config.hurst_window).rename("hurst"))

        # Support/Resistance
        for window in (20, 30, 50):
            features.append((df["close"] - df["high"].rolling(window).max()).rename(f"dist_high_{window}"))
            features.append((df["close"] - df["low"].rolling(window).min()).rename(f"dist_low_{window}"))
            features.append((df["high"].rolling(window).max() - df["low"].rolling(window).min()).rename(f"range_{window}"))

        # Smart Money Concepts (approximate)
        swing_high = df["high"].rolling(5, center=True).max()
        swing_low = df["low"].rolling(5, center=True).min()
        features.append((df["high"] == swing_high).astype(float).rename("swing_high"))
        features.append((df["low"] == swing_low).astype(float).rename("swing_low"))
        bos = (df["high"] > swing_high.shift(1)).astype(float)
        choch = (df["low"] < swing_low.shift(1)).astype(float)
        features.append(bos.rename("bos"))
        features.append(choch.rename("choch"))
        features.append(((bos - choch).rolling(10).sum()).rename("structure_bias"))
        features.append(((df["close"] > df["open"]).astype(int).rolling(4).sum()).rename("bullish_count_4"))
        features.append(((df["close"] < df["open"]).astype(int).rolling(4).sum()).rename("bearish_count_4"))

        # Market Profile
        market_profile = _market_profile(df, self.config.market_profile_bins)
        features.extend(
            [
                market_profile["profile_poc"],
                market_profile["profile_vah"],
                market_profile["profile_val"],
                market_profile["profile_entropy"],
                market_profile["profile_poc_distance"],
            ]
        )

        # Aggregation
        feature_df = pd.concat(features, axis=1)
        feature_df.columns = [col.lower() for col in feature_df.columns]
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

        if self.config.clip_sigma:
            numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
            mu = feature_df[numeric_cols].mean()
            sigma = feature_df[numeric_cols].std() + 1e-9
            feature_df[numeric_cols] = feature_df[numeric_cols].clip(
                lower=mu - self.config.clip_sigma * sigma,
                upper=mu + self.config.clip_sigma * sigma,
            )

        if self.config.fillna:
            feature_df = feature_df.fillna(method="ffill").fillna(method="bfill")

        feature_df = feature_df.fillna(0.0)
        logger.info("Generated %d features", feature_df.shape[1])
        return feature_df


__all__ = ["FeatureEngineer", "FeatureConfig"]
