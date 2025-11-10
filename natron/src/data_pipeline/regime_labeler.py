"""
Regime Labeling Module for Natron Trading System

This module labels market regimes based on trend strength, volatility,
and breakout patterns. Regimes: BULL_STRONG, BULL_WEAK, BEAR_STRONG,
BEAR_WEAK, RANGE, VOLATILE
"""

import pandas as pd
import numpy as np
from typing import Optional


class RegimeLabeler:
    """
    Classifies market data into 6 regime categories based on
    trend slope, ATR percentile, and breakout logic.
    """
    
    REGIME_CLASSES = [
        'BULL_STRONG',
        'BULL_WEAK',
        'BEAR_STRONG',
        'BEAR_WEAK',
        'RANGE',
        'VOLATILE'
    ]
    
    REGIME_MAP = {i: regime for i, regime in enumerate(REGIME_CLASSES)}
    
    def __init__(self, 
                 trend_window: int = 20,
                 atr_percentile_window: int = 100,
                 volatility_threshold: float = 0.75,
                 trend_strength_threshold: float = 0.02):
        """
        Initialize regime labeler.
        
        Args:
            trend_window: Window for computing trend slope
            atr_percentile_window: Window for ATR percentile calculation
            volatility_threshold: ATR percentile threshold for volatile regime
            trend_strength_threshold: Minimum slope for strong trend
        """
        self.trend_window = trend_window
        self.atr_percentile_window = atr_percentile_window
        self.volatility_threshold = volatility_threshold
        self.trend_strength_threshold = trend_strength_threshold
    
    def compute_trend_slope(self, prices: pd.Series, window: int) -> pd.Series:
        """
        Compute trend slope using linear regression.
        
        Args:
            prices: Price series
            window: Rolling window size
            
        Returns:
            Series with trend slopes
        """
        slopes = []
        for i in range(len(prices)):
            if i < window:
                slopes.append(0.0)
            else:
                y = prices.iloc[i-window:i].values
                x = np.arange(len(y))
                # Normalize slope by price level
                slope = np.polyfit(x, y, 1)[0] / (prices.iloc[i] + 1e-8)
                slopes.append(slope)
        
        return pd.Series(slopes, index=prices.index)
    
    def compute_atr_percentile(self, atr: pd.Series, window: int) -> pd.Series:
        """
        Compute ATR percentile within rolling window.
        
        Args:
            atr: ATR series
            window: Rolling window size
            
        Returns:
            Series with ATR percentiles (0-1)
        """
        percentiles = []
        for i in range(len(atr)):
            if i < window:
                percentiles.append(0.5)  # Default to median
            else:
                window_data = atr.iloc[i-window:i]
                current_atr = atr.iloc[i]
                percentile = (window_data < current_atr).sum() / len(window_data)
                percentiles.append(percentile)
        
        return pd.Series(percentiles, index=atr.index)
    
    def detect_breakout(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Detect breakout patterns (price breaking above/below recent range).
        
        Args:
            df: DataFrame with OHLC columns
            window: Window for range calculation
            
        Returns:
            Series with breakout signals (1=up, -1=down, 0=none)
        """
        high_max = df['high'].rolling(window=window).max()
        low_min = df['low'].rolling(window=window).min()
        
        breakout_up = (df['close'] > high_max.shift(1)).astype(int)
        breakout_down = (df['close'] < low_min.shift(1)).astype(int)
        
        breakout = breakout_up - breakout_down
        return breakout
    
    def label_regime(self, 
                     df: pd.DataFrame,
                     trend_slope: Optional[pd.Series] = None,
                     atr: Optional[pd.Series] = None,
                     atr_percentile: Optional[pd.Series] = None) -> pd.Series:
        """
        Label market regime for each timestamp.
        
        Args:
            df: DataFrame with OHLC columns and features
            trend_slope: Pre-computed trend slope (optional)
            atr: Pre-computed ATR (optional)
            atr_percentile: Pre-computed ATR percentile (optional)
            
        Returns:
            Series with regime labels (0-5)
        """
        # Compute trend slope if not provided
        if trend_slope is None:
            if 'trend_slope_20' in df.columns:
                trend_slope = df['trend_slope_20']
            else:
                trend_slope = self.compute_trend_slope(df['close'], self.trend_window)
        
        # Compute ATR if not provided
        if atr is None:
            if 'atr' in df.columns:
                atr = df['atr']
            else:
                # Simple ATR approximation
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = true_range.rolling(window=14).mean()
        
        # Compute ATR percentile if not provided
        if atr_percentile is None:
            if 'atr_percentile' in df.columns:
                atr_percentile = df['atr_percentile']
            else:
                atr_percentile = self.compute_atr_percentile(atr, self.atr_percentile_window)
        
        # Detect breakouts
        breakout = self.detect_breakout(df)
        
        # Initialize regime labels
        regimes = pd.Series(index=df.index, dtype=int)
        
        # Rule 1: VOLATILE regime (high ATR percentile)
        volatile_mask = atr_percentile >= self.volatility_threshold
        regimes[volatile_mask] = 5  # VOLATILE
        
        # Rule 2: Strong trends (high absolute slope)
        strong_bull_mask = (
            (trend_slope > self.trend_strength_threshold) &
            (atr_percentile < self.volatility_threshold) &
            (~volatile_mask)
        )
        strong_bear_mask = (
            (trend_slope < -self.trend_strength_threshold) &
            (atr_percentile < self.volatility_threshold) &
            (~volatile_mask)
        )
        
        regimes[strong_bull_mask] = 0  # BULL_STRONG
        regimes[strong_bear_mask] = 2  # BEAR_STRONG
        
        # Rule 3: Weak trends (low absolute slope, but directional)
        weak_bull_mask = (
            (trend_slope > 0) &
            (trend_slope <= self.trend_strength_threshold) &
            (atr_percentile < self.volatility_threshold) &
            (~volatile_mask) &
            (regimes.isna())
        )
        weak_bear_mask = (
            (trend_slope < 0) &
            (trend_slope >= -self.trend_strength_threshold) &
            (atr_percentile < self.volatility_threshold) &
            (~volatile_mask) &
            (regimes.isna())
        )
        
        regimes[weak_bull_mask] = 1  # BULL_WEAK
        regimes[weak_bear_mask] = 3  # BEAR_WEAK
        
        # Rule 4: RANGE regime (low volatility, no clear trend)
        range_mask = (
            (np.abs(trend_slope) < 0.005) &
            (atr_percentile < 0.5) &
            (~volatile_mask) &
            (regimes.isna())
        )
        regimes[range_mask] = 4  # RANGE
        
        # Fill any remaining NaN with RANGE (default)
        regimes = regimes.fillna(4)
        
        return regimes.astype(int)
    
    def get_regime_name(self, regime_id: int) -> str:
        """
        Get regime name from ID.
        
        Args:
            regime_id: Regime ID (0-5)
            
        Returns:
            Regime name string
        """
        return self.REGIME_MAP.get(regime_id, 'UNKNOWN')
    
    def add_regime_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add regime labels to DataFrame.
        
        Args:
            df: DataFrame with OHLC and feature columns
            
        Returns:
            DataFrame with added 'regime' and 'regime_name' columns
        """
        df = df.copy()
        
        # Label regimes
        regime_labels = self.label_regime(df)
        df['regime'] = regime_labels
        
        # Add regime names
        df['regime_name'] = regime_labels.map(self.REGIME_MAP)
        
        return df
