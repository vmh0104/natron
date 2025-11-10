"""
Regime Labeling Module for Natron
Labels market regimes: BULL_STRONG, BULL_WEAK, BEAR_STRONG, BEAR_WEAK, RANGE, VOLATILE
"""

import pandas as pd
import numpy as np
from typing import Optional


class RegimeLabeler:
    """
    Labels market data into 6 regime classes based on trend, volatility, and breakout patterns.
    """
    
    REGIME_CLASSES = {
        0: 'BULL_STRONG',
        1: 'BULL_WEAK',
        2: 'BEAR_STRONG',
        3: 'BEAR_WEAK',
        4: 'RANGE',
        5: 'VOLATILE'
    }
    
    def __init__(self, trend_window: int = 20, atr_percentile_window: int = 50,
                 breakout_threshold: float = 0.02, volatility_threshold: float = 0.75):
        """
        Initialize regime labeler.
        
        Args:
            trend_window: Window for trend slope calculation
            atr_percentile_window: Window for ATR percentile calculation
            breakout_threshold: Threshold for breakout detection (e.g., 0.02 = 2%)
            volatility_threshold: Percentile threshold for VOLATILE regime (e.g., 0.75 = 75th)
        """
        self.trend_window = trend_window
        self.atr_percentile_window = atr_percentile_window
        self.breakout_threshold = breakout_threshold
        self.volatility_threshold = volatility_threshold
    
    def compute_trend_slope(self, prices: pd.Series, window: int) -> pd.Series:
        """
        Compute trend slope using linear regression.
        
        Args:
            prices: Price series
            window: Rolling window size
            
        Returns:
            Trend slope series (positive = uptrend, negative = downtrend)
        """
        slopes = []
        indices = np.arange(window)
        
        for i in range(len(prices)):
            if i < window:
                slopes.append(0.0)
            else:
                window_prices = prices.iloc[i-window+1:i+1].values
                if len(window_prices) == window and not np.isnan(window_prices).any():
                    slope = np.polyfit(indices, window_prices, 1)[0]
                    slopes.append(slope)
                else:
                    slopes.append(0.0)
        
        return pd.Series(slopes, index=prices.index)
    
    def detect_breakout(self, prices: pd.Series, window: int, threshold: float) -> pd.Series:
        """
        Detect breakout patterns.
        
        Args:
            prices: Price series
            window: Window for range calculation
            threshold: Breakout threshold (e.g., 0.02 = 2%)
            
        Returns:
            Breakout signal series (1 = breakout up, -1 = breakout down, 0 = no breakout)
        """
        rolling_max = prices.rolling(window=window).max()
        rolling_min = prices.rolling(window=window).min()
        range_size = rolling_max - rolling_min
        
        breakout = pd.Series(0, index=prices.index)
        
        # Breakout up
        breakout_up = (prices > rolling_max.shift(1)) & \
                     ((prices - rolling_max.shift(1)) / (range_size.shift(1) + 1e-8) > threshold)
        
        # Breakout down
        breakout_down = (prices < rolling_min.shift(1)) & \
                       ((rolling_min.shift(1) - prices) / (range_size.shift(1) + 1e-8) > threshold)
        
        breakout[breakout_up] = 1
        breakout[breakout_down] = -1
        
        return breakout
    
    def label_regime(self, df: pd.DataFrame, atr: pd.Series) -> pd.Series:
        """
        Label market regime for each candle.
        
        Args:
            df: DataFrame with OHLCV and feature columns
            atr: ATR series
            
        Returns:
            Regime labels series (0-5 corresponding to REGIME_CLASSES)
        """
        prices = df['close']
        
        # Compute trend slope
        trend_slope = self.compute_trend_slope(prices, self.trend_window)
        trend_slope_normalized = trend_slope / (prices.rolling(self.trend_window).std() + 1e-8)
        
        # Compute ATR percentile
        atr_percentile = atr.rolling(window=self.atr_percentile_window).rank(pct=True)
        
        # Detect breakouts
        breakout = self.detect_breakout(prices, window=20, threshold=self.breakout_threshold)
        
        # Initialize regime labels
        regime = pd.Series(4, index=df.index, dtype=int)  # Default: RANGE
        
        # VOLATILE regime (high ATR percentile)
        volatile_mask = atr_percentile > self.volatility_threshold
        regime[volatile_mask] = 5
        
        # BULL regimes
        bull_mask = trend_slope_normalized > 0.5
        strong_bull_mask = bull_mask & (trend_slope_normalized > 1.0) & (breakout == 1)
        weak_bull_mask = bull_mask & ~strong_bull_mask & ~volatile_mask
        
        regime[strong_bull_mask] = 0  # BULL_STRONG
        regime[weak_bull_mask] = 1  # BULL_WEAK
        
        # BEAR regimes
        bear_mask = trend_slope_normalized < -0.5
        strong_bear_mask = bear_mask & (trend_slope_normalized < -1.0) & (breakout == -1)
        weak_bear_mask = bear_mask & ~strong_bear_mask & ~volatile_mask
        
        regime[strong_bear_mask] = 2  # BEAR_STRONG
        regime[weak_bear_mask] = 3  # BEAR_WEAK
        
        # RANGE regime (low trend, low volatility, no breakout)
        range_mask = (np.abs(trend_slope_normalized) < 0.3) & \
                    (atr_percentile < 0.5) & \
                    (breakout == 0) & \
                    ~volatile_mask
        regime[range_mask] = 4
        
        return regime.fillna(4)  # Default to RANGE if NaN
    
    def get_regime_name(self, regime_code: int) -> str:
        """
        Get regime name from code.
        
        Args:
            regime_code: Regime code (0-5)
            
        Returns:
            Regime name string
        """
        return self.REGIME_CLASSES.get(regime_code, 'UNKNOWN')
    
    def get_regime_distribution(self, regimes: pd.Series) -> dict:
        """
        Get distribution of regimes.
        
        Args:
            regimes: Series of regime codes
            
        Returns:
            Dictionary with regime counts and percentages
        """
        counts = regimes.value_counts().sort_index()
        total = len(regimes)
        
        distribution = {}
        for code, count in counts.items():
            regime_name = self.get_regime_name(code)
            distribution[regime_name] = {
                'count': int(count),
                'percentage': float(count / total * 100)
            }
        
        return distribution
