"""
Feature engineering module for Natron V2.

Generates ~100 engineered technical features from OHLCV data.
Built for vectorised pandas operations with optional caching support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureEngineerConfig:
    """Configuration container for the feature engineer."""

    rolling_windows: Tuple[int, ...] = (5, 10, 14, 20, 30, 50, 96)
    atr_windows: Tuple[int, ...] = (14, 20)
    bollinger_window: int = 20
    bollinger_num_std: float = 2.0
    regime_trend_window: int = 96
    volume_spike_threshold: float = 1.5
    enable_market_profile: bool = True
    enable_smc: bool = True
    cache_intermediate: bool = False
    cache: Dict[str, pd.Series] = field(default_factory=dict)


class FeatureEngineer:
    """
    Generates technical indicators, statistical descriptors, SMC features,
    and market profile aggregates.
    """

    def __init__(self, config: Optional[FeatureEngineerConfig] = None) -> None:
        self.config = config or FeatureEngineerConfig()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute engineered features.

        Args:
            df: DataFrame with columns time, open, high, low, close, volume.

        Returns:
            features_df: DataFrame aligned with df index containing engineered features.
        """
        self._validate(df)
        df = df.copy()
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)

        features = {}

        # Basic derived series
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        hl_range = df["high"] - df["low"]
        body = (df["close"] - df["open"]).abs()
        features["true_range"] = self._true_range(df)

        # Moving averages & slopes
        for window in self.config.rolling_windows:
            features[f"sma_{window}"] = df["close"].rolling(window, min_periods=1).mean()
            features[f"ema_{window}"] = df["close"].ewm(span=window, adjust=False).mean()
            features[f"wma_{window}"] = (
                df["close"]
                .rolling(window, min_periods=1)
                .apply(lambda x: np.dot(x, np.arange(1, len(x) + 1)) / ((len(x) * (len(x) + 1)) / 2), raw=True)
            )
            features[f"price_to_sma_{window}"] = df["close"] / (features[f"sma_{window}"] + 1e-6)
            features[f"price_to_ema_{window}"] = df["close"] / (features[f"ema_{window}"] + 1e-6)

            sma = features[f"sma_{window}"]
            ema = features[f"ema_{window}"]
            features[f"sma_slope_{window}"] = sma.diff(window) / (window + 1e-6)
            features[f"ema_slope_{window}"] = ema.diff(window) / (window + 1e-6)

        # Momentum indicators
        for window in (7, 14, 21):
            features[f"rsi_{window}"] = self._rsi(df["close"], window)
        for window in (5, 10, 20):
            features[f"roc_{window}"] = df["close"].pct_change(window)
        features["cci_20"] = self._cci(df, 20)
        stoch_k, stoch_d = self._stochastic(df, k_period=14, d_period=3)
        features["stoch_k_14"] = stoch_k
        features["stoch_d_14"] = stoch_d
        macd, macd_signal, macd_hist = self._macd(df["close"])
        features["macd_line"] = macd
        features["macd_signal"] = macd_signal
        features["macd_hist"] = macd_hist
        features["tsi"] = self._tsi(df["close"])
        features["willr_14"] = self._williams_r(df, window=14)

        # Volatility indicators
        for window in self.config.atr_windows:
            features[f"atr_{window}"] = self._atr(df, window)
        bb_mid, bb_upper, bb_lower, bb_width, bb_pct = self._bollinger(df["close"], self.config.bollinger_window, self.config.bollinger_num_std)
        features["bb_mid"] = bb_mid
        features["bb_upper"] = bb_upper
        features["bb_lower"] = bb_lower
        features["bb_width"] = bb_width
        features["bb_pctb"] = bb_pct
        kel_mid, kel_upper, kel_lower = self._keltner(df, ema_window=20, atr_window=20, multiplier=1.5)
        features["kel_mid"] = kel_mid
        features["kel_upper"] = kel_upper
        features["kel_lower"] = kel_lower
        features["kel_width"] = kel_upper - kel_lower
        for window in (10, 20, 50):
            std = df["close"].rolling(window, min_periods=1).std()
            features[f"rolling_std_{window}"] = std
            features[f"realized_vol_{window}"] = df["close"].pct_change().rolling(window, min_periods=1).std() * np.sqrt(window)
        donchian_high = df["high"].rolling(20, min_periods=1).max()
        donchian_low = df["low"].rolling(20, min_periods=1).min()
        features["donchian_high_20"] = donchian_high
        features["donchian_low_20"] = donchian_low
        features["donchian_width_20"] = donchian_high - donchian_low

        # Volume indicators
        for window in (20, 50):
            features[f"volume_sma_{window}"] = df["volume"].rolling(window, min_periods=1).mean()
        features["volume_ratio_20"] = df["volume"] / (features["volume_sma_20"] + 1e-6)
        features["volume_zscore_20"] = (df["volume"] - features["volume_sma_20"]) / (df["volume"].rolling(20, min_periods=1).std() + 1e-6)
        features["obv"] = self._obv(df)
        features["vwap_96"] = (df["close"] * df["volume"]).rolling(96, min_periods=1).sum() / (df["volume"].rolling(96, min_periods=1).sum() + 1e-6)
        features["mfi_14"] = self._mfi(df, window=14)
        features["cmf_20"] = self._chaikin_money_flow(df, window=20)
        features["acc_dist"] = self._accumulation_distribution(df)

        # Price pattern indicators
        features["candle_body"] = body
        features["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
        features["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]
        features["body_pct"] = body / (hl_range.replace(0, np.nan))
        features["upper_shadow_pct"] = features["upper_shadow"] / (hl_range.replace(0, np.nan))
        features["lower_shadow_pct"] = features["lower_shadow"] / (hl_range.replace(0, np.nan))
        features["is_doji"] = (features["body_pct"] < 0.1).astype(float)
        features["gap_up"] = ((df["open"] - df["close"].shift(1)) / df["close"].shift(1)).clip(lower=0)
        features["gap_down"] = ((df["open"] - df["close"].shift(1)) / df["close"].shift(1)).clip(upper=0).abs()
        features["close_location"] = (df["close"] - df["low"]) / (hl_range + 1e-6)

        # Returns and statistics
        log_ret = np.log(df["close"] / df["close"].shift(1))
        features["log_return"] = log_ret
        features["abs_log_return"] = log_ret.abs()
        features["rolling_return_5"] = df["close"].pct_change(5)
        features["rolling_return_20"] = df["close"].pct_change(20)
        features["cumulative_return"] = df["close"] / df["close"].iloc[0] - 1.0
        features["rolling_sharpe_20"] = (
            df["close"].pct_change().rolling(20, min_periods=3).mean()
            / (df["close"].pct_change().rolling(20, min_periods=3).std() + 1e-6)
        )
        features["zscore_20"] = (df["close"] - df["close"].rolling(20, min_periods=1).mean()) / (
            df["close"].rolling(20, min_periods=1).std() + 1e-6
        )
        features["rolling_skew_20"] = df["close"].pct_change().rolling(20, min_periods=5).skew()
        features["rolling_kurt_20"] = df["close"].pct_change().rolling(20, min_periods=5).kurt()
        features["hurst_50"] = self._hurst_exponent(df["close"], window=50)

        # Trend strength
        adx, plus_di, minus_di = self._adx(df, window=14)
        features["adx_14"] = adx
        features["plus_di_14"] = plus_di
        features["minus_di_14"] = minus_di
        aroon_up, aroon_down = self._aroon(df, window=25)
        features["aroon_up_25"] = aroon_up
        features["aroon_down_25"] = aroon_down
        features["aroon_oscillator_25"] = aroon_up - aroon_down

        # Support / Resistance distances
        for window in (20, 50):
            rolling_high = df["high"].rolling(window, min_periods=1).max()
            rolling_low = df["low"].rolling(window, min_periods=1).min()
            features[f"dist_to_high_{window}"] = (rolling_high - df["close"]) / (rolling_high + 1e-6)
            features[f"dist_to_low_{window}"] = (df["close"] - rolling_low) / (rolling_low + 1e-6)

        # Smart Money Concepts (approximate)
        if self.config.enable_smc:
            smc_feats = self._smc_features(df, lookback=20)
            features.update(smc_feats)

        # Market Profile summaries
        if self.config.enable_market_profile:
            profile_feats = self._market_profile(tp, df["volume"], window=96)
            features.update(profile_feats)

        features_df = pd.DataFrame(features)
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(method="ffill").fillna(method="bfill")
        features_df = features_df.fillna(0.0)

        return features_df

    # -----------------------------
    # Technical indicator helpers
    # -----------------------------

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        required = {"time", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Dataframe missing required columns: {missing}")

    @staticmethod
    def _true_range(df: pd.DataFrame) -> pd.Series:
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.fillna(0.0)

    @staticmethod
    def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = pd.Series(gain).rolling(window, min_periods=1).mean()
        avg_loss = pd.Series(loss).rolling(window, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-6)
        rsi = 100 - (100 / (1 + rs))
        return pd.Series(rsi, index=series.index)

    @staticmethod
    def _cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        sma = tp.rolling(window, min_periods=1).mean()
        mad = (tp - sma).abs().rolling(window, min_periods=1).mean()
        return (tp - sma) / (0.015 * mad + 1e-6)

    @staticmethod
    def _stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        low_min = df["low"].rolling(window=k_period, min_periods=1).min()
        high_max = df["high"].rolling(window=k_period, min_periods=1).max()
        percent_k = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-6)
        percent_d = percent_k.rolling(window=d_period, min_periods=1).mean()
        return percent_k, percent_d

    @staticmethod
    def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist

    @staticmethod
    def _tsi(series: pd.Series, long: int = 25, short: int = 13) -> pd.Series:
        diff = series.diff()
        delta2 = diff.diff()
        ema1 = diff.ewm(span=long, adjust=False).mean()
        ema2 = ema1.ewm(span=short, adjust=False).mean()
        ema1_abs = diff.abs().ewm(span=long, adjust=False).mean()
        ema2_abs = ema1_abs.ewm(span=short, adjust=False).mean()
        tsi = 100 * (ema2 / (ema2_abs + 1e-6))
        return tsi.fillna(0.0)

    @staticmethod
    def _williams_r(df: pd.DataFrame, window: int = 14) -> pd.Series:
        highest_high = df["high"].rolling(window, min_periods=1).max()
        lowest_low = df["low"].rolling(window, min_periods=1).min()
        willr = -100 * (highest_high - df["close"]) / (highest_high - lowest_low + 1e-6)
        return willr

    @staticmethod
    def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        tr = FeatureEngineer._true_range(df)
        atr = tr.rolling(window=window, min_periods=1).mean()
        return atr

    @staticmethod
    def _bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        mid = series.rolling(window, min_periods=1).mean()
        std = series.rolling(window, min_periods=1).std()
        upper = mid + num_std * std
        lower = mid - num_std * std
        width = (upper - lower) / (mid + 1e-6)
        pctb = (series - lower) / (upper - lower + 1e-6)
        return mid, upper, lower, width, pctb

    @staticmethod
    def _keltner(df: pd.DataFrame, ema_window: int = 20, atr_window: int = 20, multiplier: float = 1.5) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema = df["close"].ewm(span=ema_window, adjust=False).mean()
        atr = FeatureEngineer._atr(df, atr_window)
        upper = ema + multiplier * atr
        lower = ema - multiplier * atr
        return ema, upper, lower

    @staticmethod
    def _obv(df: pd.DataFrame) -> pd.Series:
        direction = np.where(df["close"] > df["close"].shift(1), 1, np.where(df["close"] < df["close"].shift(1), -1, 0))
        obv = (direction * df["volume"]).cumsum()
        return pd.Series(obv, index=df.index)

    @staticmethod
    def _mfi(df: pd.DataFrame, window: int = 14) -> pd.Series:
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        raw_money_flow = tp * df["volume"]
        positive_flow = np.where(tp > tp.shift(1), raw_money_flow, 0.0)
        negative_flow = np.where(tp < tp.shift(1), raw_money_flow, 0.0)
        pos_mf = pd.Series(positive_flow).rolling(window, min_periods=1).sum()
        neg_mf = pd.Series(negative_flow).rolling(window, min_periods=1).sum()
        mfi = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-6)))
        return pd.Series(mfi, index=df.index)

    @staticmethod
    def _chaikin_money_flow(df: pd.DataFrame, window: int = 20) -> pd.Series:
        money_flow_multiplier = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"] + 1e-6)
        money_flow_volume = money_flow_multiplier * df["volume"]
        cmf = money_flow_volume.rolling(window, min_periods=1).sum() / (df["volume"].rolling(window, min_periods=1).sum() + 1e-6)
        return cmf

    @staticmethod
    def _accumulation_distribution(df: pd.DataFrame) -> pd.Series:
        clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"] + 1e-6)
        ad = (clv * df["volume"]).cumsum()
        return ad

    @staticmethod
    def _adx(df: pd.DataFrame, window: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        up_move = df["high"].diff()
        down_move = df["low"].diff(-1)
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), -down_move, 0.0)

        tr = FeatureEngineer._true_range(df)
        atr = tr.rolling(window=window, min_periods=1).mean()
        plus_di = 100 * (pd.Series(plus_dm).rolling(window, min_periods=1).sum() / (atr + 1e-6))
        minus_di = 100 * (pd.Series(minus_dm).rolling(window, min_periods=1).sum() / (atr + 1e-6))
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-6) * 100
        adx = dx.rolling(window, min_periods=1).mean()
        return adx, plus_di, minus_di

    @staticmethod
    def _aroon(df: pd.DataFrame, window: int = 25) -> Tuple[pd.Series, pd.Series]:
        def aroon_up(series: pd.Series) -> pd.Series:
            return 100 * series.rolling(window, min_periods=1).apply(lambda x: (np.argmax(x[::-1]) + 1) / len(x), raw=True)

        def aroon_down(series: pd.Series) -> pd.Series:
            return 100 * series.rolling(window, min_periods=1).apply(lambda x: (np.argmin(x[::-1]) + 1) / len(x), raw=True)

        return aroon_up(df["high"]), aroon_down(df["low"])

    @staticmethod
    def _hurst_exponent(series: pd.Series, window: int = 50) -> pd.Series:
        def hurst(arr: np.ndarray) -> float:
            if len(arr) < 10:
                return np.nan
            arr = arr - arr.mean()
            cumulative = np.cumsum(arr)
            range_ = cumulative.max() - cumulative.min()
            std = arr.std()
            if std == 0:
                return np.nan
            return np.log(range_ / std + 1e-6) / np.log(len(arr) + 1e-6)

        return series.rolling(window, min_periods=window).apply(hurst, raw=True)

    # -----------------------------
    # Smart Money Concepts features
    # -----------------------------

    def _smc_features(self, df: pd.DataFrame, lookback: int = 20) -> Dict[str, pd.Series]:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        swing_high = (high == high.rolling(window=lookback, center=True, min_periods=1).max()).astype(float)
        swing_low = (low == low.rolling(window=lookback, center=True, min_periods=1).min()).astype(float)

        prev_swing_high = high.where(swing_high == 1).ffill()
        prev_swing_low = low.where(swing_low == 1).ffill()

        bos = ((close > prev_swing_high.shift(1)) | (close < prev_swing_low.shift(1))).astype(float)
        choch = (
            ((close > prev_swing_high.shift(1)) & (close < prev_swing_low.shift(2)))
            | ((close < prev_swing_low.shift(1)) & (close > prev_swing_high.shift(2)))
        ).astype(float)
        liquidity_grab = (
            ((high > prev_swing_high.shift(1)) & (close < prev_swing_high.shift(1)))
            | ((low < prev_swing_low.shift(1)) & (close > prev_swing_low.shift(1)))
        ).astype(float)
        structure_bias = np.where(close > prev_swing_high, 1, np.where(close < prev_swing_low, -1, 0))

        return {
            "smc_swing_high": swing_high,
            "smc_swing_low": swing_low,
            "smc_bos": bos,
            "smc_choch": choch,
            "smc_liquidity_grab": liquidity_grab,
            "smc_structure_bias": pd.Series(structure_bias, index=df.index),
        }

    # -----------------------------
    # Market Profile features
    # -----------------------------

    @staticmethod
    def _market_profile(tp: pd.Series, volume: pd.Series, window: int = 96) -> Dict[str, pd.Series]:
        def profile_metrics(tp_arr: np.ndarray, vol_arr: np.ndarray) -> Tuple[float, float, float, float, float, float, float, float]:
            if len(tp_arr) == 0 or np.all(np.isnan(tp_arr)):
                return (np.nan,) * 8
            valid = ~np.isnan(tp_arr)
            tp_arr = tp_arr[valid]
            vol_arr = vol_arr[valid]
            if len(tp_arr) == 0:
                return (np.nan,) * 8
            weights = vol_arr / (vol_arr.sum() + 1e-6)
            vwap = (tp_arr * weights).sum()
            vah = np.quantile(tp_arr, 0.7)
            val = np.quantile(tp_arr, 0.3)
            width = vah - val
            hist, bin_edges = np.histogram(tp_arr, bins=min(20, len(tp_arr)), weights=weights, density=False)
            prob = hist / (hist.sum() + 1e-6)
            entropy = -np.sum(prob * np.log(prob + 1e-6))
            skew = ((tp_arr - tp_arr.mean()) ** 3).mean() / (tp_arr.std() ** 3 + 1e-6)
            kurt = ((tp_arr - tp_arr.mean()) ** 4).mean() / (tp_arr.std() ** 4 + 1e-6)
            balance = np.abs((tp_arr[tp_arr >= vwap].sum() - tp_arr[tp_arr < vwap].sum())) / (tp_arr.sum() + 1e-6)
            volume_balance = np.abs(vol_arr[tp_arr >= vwap].sum() - vol_arr[tp_arr < vwap].sum()) / (vol_arr.sum() + 1e-6)
            return vwap, vah, val, width, entropy, balance, volume_balance, skew + kurt

        metrics = tp.rolling(window, min_periods=10).apply(
            lambda x: profile_metrics(x, volume.loc[x.index].to_numpy()),
            raw=False,
        )
        # rolling apply returns Series of tuples; convert
        metric_df = metrics.apply(pd.Series)
        metric_df.columns = [
            "profile_vwap",
            "profile_vah",
            "profile_val",
            "profile_width",
            "profile_entropy",
            "profile_balance",
            "profile_volume_balance",
            "profile_shape",
        ]
        return metric_df.to_dict(orient="series")
