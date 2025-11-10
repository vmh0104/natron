"""
Regime Labeling Module for Natron AI Trading System
Labels market regimes based on trend, volatility, and breakout patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from enum import IntEnum


class RegimeType(IntEnum):
    """Market regime types."""
    BULL_STRONG = 0
    BULL_WEAK = 1
    BEAR_STRONG = 2
    BEAR_WEAK = 3
    RANGE = 4
    VOLATILE = 5


class RegimeLabeler:
    """
    Labels market data into 6 regime classes based on trend strength,
    volatility, and breakout patterns.
    """
    
    def __init__(self, 
                 trend_period: int = 50,
                 atr_percentile_window: int = 100,
                 breakout_threshold: float = 0.02):
        """
        Initialize Regime Labeler.
        
        Args:
            trend_period: Period for trend slope calculation
            atr_percentile_window: Window for ATR percentile calculation
            breakout_threshold: Threshold for breakout detection (2% default)
        """
        self.trend_period = trend_period
        self.atr_percentile_window = atr_percentile_window
        self.breakout_threshold = breakout_threshold
    
    def compute_trend_slope(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Compute trend slope using linear regression.
        
        Args:
            prices: Price series
            period: Rolling window period
            
        Returns:
            Slope series
        """
        def calc_slope(x):
            if len(x) < period:
                return np.nan
            return np.polyfit(range(len(x)), x.values, 1)[0]
        
        slope = prices.rolling(window=period).apply(calc_slope, raw=False)
        return slope
    
    def compute_atr_percentile(self, atr: pd.Series, window: int) -> pd.Series:
        """
        Compute ATR percentile within rolling window.
        
        Args:
            atr: ATR series
            window: Rolling window size
            
        Returns:
            ATR percentile series (0-1)
        """
        percentile = atr.rolling(window=window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == window else np.nan,
            raw=False
        )
        return percentile
    
    def detect_breakout(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Detect breakout patterns.
        
        Args:
            df: DataFrame with OHLC columns
            period: Lookback period for range calculation
            
        Returns:
            Breakout signal series (1 for breakout up, -1 for breakout down, 0 for no breakout)
        """
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        range_size = high_max - low_min
        
        # Breakout up: close above previous high + threshold
        breakout_up = (df['close'] > high_max.shift(1) * (1 + self.breakout_threshold)).astype(int)
        
        # Breakout down: close below previous low - threshold
        breakout_down = (df['close'] < low_min.shift(1) * (1 - self.breakout_threshold)).astype(int)
        
        breakout_signal = breakout_up - breakout_down
        return breakout_signal
    
    def label_regime(self, 
                    df: pd.DataFrame,
                    trend_slope: pd.Series,
                    atr_percentile: pd.Series,
                    breakout_signal: pd.Series) -> pd.Series:
        """
        Label regimes based on trend, volatility, and breakout logic.
        
        Args:
            df: DataFrame with OHLC columns
            trend_slope: Trend slope series
            atr_percentile: ATR percentile series
            breakout_signal: Breakout signal series
            
        Returns:
            Regime labels series (0-5)
        """
        labels = pd.Series(index=df.index, dtype=int)
        
        # Normalize trend slope by price to get percentage slope
        price_normalized_slope = trend_slope / df['close'] * 100
        
        # Define thresholds
        strong_trend_threshold = 0.1  # 0.1% per period
        weak_trend_threshold = 0.02   # 0.02% per period
        high_volatility_threshold = 0.75  # 75th percentile
        low_volatility_threshold = 0.25    # 25th percentile
        
        # Initialize labels
        for idx in df.index:
            slope_val = price_normalized_slope.loc[idx]
            atr_pct = atr_percentile.loc[idx]
            breakout = breakout_signal.loc[idx]
            
            if pd.isna(slope_val) or pd.isna(atr_pct):
                labels.loc[idx] = RegimeType.RANGE
                continue
            
            # High volatility regime
            if atr_pct > high_volatility_threshold:
                labels.loc[idx] = RegimeType.VOLATILE
                continue
            
            # Bull market
            if slope_val > strong_trend_threshold:
                if atr_pct < low_volatility_threshold:
                    labels.loc[idx] = RegimeType.BULL_STRONG
                else:
                    labels.loc[idx] = RegimeType.BULL_WEAK
            elif slope_val > weak_trend_threshold:
                labels.loc[idx] = RegimeType.BULL_WEAK
            
            # Bear market
            elif slope_val < -strong_trend_threshold:
                if atr_pct < low_volatility_threshold:
                    labels.loc[idx] = RegimeType.BEAR_STRONG
                else:
                    labels.loc[idx] = RegimeType.BEAR_WEAK
            elif slope_val < -weak_trend_threshold:
                labels.loc[idx] = RegimeType.BEAR_WEAK
            
            # Range-bound or low volatility
            else:
                # Check for breakout
                if abs(breakout) > 0:
                    if breakout > 0:
                        labels.loc[idx] = RegimeType.BULL_WEAK
                    else:
                        labels.loc[idx] = RegimeType.BEAR_WEAK
                else:
                    labels.loc[idx] = RegimeType.RANGE
        
        return labels
    
    def label_data(self, df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.Series:
        """
        Label entire dataset with regime classes.
        
        Args:
            df: Original OHLCV DataFrame
            feature_df: Feature DataFrame (must contain 'atr' column)
            
        Returns:
            Regime labels series
        """
        if 'atr' not in feature_df.columns:
            raise ValueError("feature_df must contain 'atr' column")
        
        # Compute trend slope
        trend_slope = self.compute_trend_slope(df['close'], self.trend_period)
        
        # Compute ATR percentile
        atr_percentile = self.compute_atr_percentile(
            feature_df['atr'], 
            self.atr_percentile_window
        )
        
        # Detect breakouts
        breakout_signal = self.detect_breakout(df, period=20)
        
        # Label regimes
        labels = self.label_regime(df, trend_slope, atr_percentile, breakout_signal)
        
        return labels
    
    def get_regime_names(self) -> Dict[int, str]:
        """
        Get mapping of regime IDs to names.
        
        Returns:
            Dictionary mapping regime ID to name
        """
        return {
            RegimeType.BULL_STRONG: "BULL_STRONG",
            RegimeType.BULL_WEAK: "BULL_WEAK",
            RegimeType.BEAR_STRONG: "BEAR_STRONG",
            RegimeType.BEAR_WEAK: "BEAR_WEAK",
            RegimeType.RANGE: "RANGE",
            RegimeType.VOLATILE: "VOLATILE"
        }
    
    def print_regime_distribution(self, labels: pd.Series):
        """
        Print distribution of regime labels.
        
        Args:
            labels: Regime labels series
        """
        regime_names = self.get_regime_names()
        counts = labels.value_counts().sort_index()
        
        print("\n=== Regime Distribution ===")
        total = len(labels)
        for regime_id, count in counts.items():
            pct = (count / total) * 100
            regime_name = regime_names.get(regime_id, f"UNKNOWN_{regime_id}")
            print(f"{regime_name:15s}: {count:6d} ({pct:5.2f}%)")
        print(f"{'Total':15s}: {total:6d} (100.00%)")
