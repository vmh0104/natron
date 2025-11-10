"""
Natron Regime Labeler
Labels market regimes based on trend, volatility, and breakout patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegimeLabeler:
    """
    Labels market data into 6 regime classes:
    - BULL_STRONG: Strong uptrend
    - BULL_WEAK: Weak uptrend
    - BEAR_STRONG: Strong downtrend
    - BEAR_WEAK: Weak downtrend
    - RANGE: Sideways/consolidation
    - VOLATILE: High volatility regime
    """
    
    REGIME_CLASSES = [
        'BULL_STRONG',
        'BULL_WEAK',
        'BEAR_STRONG',
        'BEAR_WEAK',
        'RANGE',
        'VOLATILE'
    ]
    
    def __init__(self):
        """Initialize regime labeler."""
        logger.info("RegimeLabeler initialized")
    
    def label_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label regimes for each row in the dataframe.
        
        Args:
            df: DataFrame with OHLCV and feature columns
            
        Returns:
            DataFrame with 'regime' column added
        """
        df = df.copy()
        
        # Ensure required features exist
        required_features = ['ema_20', 'ema_50', 'atr_percentile', 'volatility_20']
        missing = [f for f in required_features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing required features for regime labeling: {missing}")
        
        # Calculate trend slope (using EMA)
        df['trend_slope'] = self._calculate_trend_slope(df)
        
        # Calculate volatility regime
        df['is_volatile'] = df['atr_percentile'] > 0.75
        
        # Detect breakouts
        df['is_breakout'] = self._detect_breakouts(df)
        
        # Label regimes
        df['regime'] = df.apply(self._label_single_regime, axis=1)
        
        # Map to numeric labels for model training
        regime_map = {regime: idx for idx, regime in enumerate(self.REGIME_CLASSES)}
        df['regime_label'] = df['regime'].map(regime_map)
        
        logger.info(f"Regime distribution:\n{df['regime'].value_counts()}")
        
        return df
    
    def _calculate_trend_slope(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate trend slope using EMA and price action.
        
        Returns:
            Series with trend slope values (positive = uptrend, negative = downtrend)
        """
        # EMA slope over 10 periods
        ema_slope = (df['ema_20'] - df['ema_20'].shift(10)) / df['ema_20'].shift(10)
        
        # Price vs EMA position
        price_above_ema = (df['close'] > df['ema_20']).astype(int) - 0.5
        
        # Combined trend strength
        trend_slope = ema_slope * 100 + price_above_ema * 0.02
        
        return trend_slope.fillna(0)
    
    def _detect_breakouts(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect breakout patterns using volatility and price action.
        
        Returns:
            Series with breakout flags
        """
        # High volatility spike
        atr_spike = df['atr_percentile'] > 0.8
        
        # Price movement beyond recent range
        price_range_20 = df['high'].rolling(20).max() - df['low'].rolling(20).min()
        price_deviation = np.abs(df['close'] - df['close'].rolling(20).mean()) / (price_range_20 + 1e-8)
        range_breakout = price_deviation > 0.7
        
        # Volume confirmation (if available)
        volume_spike = False
        if 'volume_ratio' in df.columns:
            volume_spike = df['volume_ratio'] > 1.5
        
        breakout = (atr_spike | range_breakout) & (volume_spike if 'volume_ratio' in df.columns else True)
        
        return breakout.fillna(False).astype(int)
    
    def _label_single_regime(self, row: pd.Series) -> str:
        """
        Label a single row's regime based on features.
        
        Args:
            row: Single row from DataFrame
            
        Returns:
            Regime class string
        """
        trend_slope = row.get('trend_slope', 0)
        is_volatile = row.get('is_volatile', False)
        is_breakout = row.get('is_breakout', 0)
        ema_20 = row.get('ema_20', row.get('close', 0))
        ema_50 = row.get('ema_50', row.get('close', 0))
        
        # High volatility regime takes precedence
        if is_volatile or is_breakout:
            return 'VOLATILE'
        
        # Determine trend direction
        ema_alignment = ema_20 > ema_50
        price_above_ema20 = row.get('close', 0) > ema_20
        
        # Strong trends
        if trend_slope > 0.01 and ema_alignment and price_above_ema20:
            return 'BULL_STRONG'
        elif trend_slope < -0.01 and not ema_alignment and not price_above_ema20:
            return 'BEAR_STRONG'
        
        # Weak trends
        elif trend_slope > 0.005:
            return 'BULL_WEAK'
        elif trend_slope < -0.005:
            return 'BEAR_WEAK'
        
        # Range/consolidation
        else:
            return 'RANGE'
