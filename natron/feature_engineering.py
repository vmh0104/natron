"""
Natron Feature Engineering
Computes 50-70 technical indicators and derived features from OHLCV data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering module for Natron.
    Computes technical indicators, candle patterns, and volatility metrics.
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        self.feature_count = 0
        logger.info("FeatureEngineer initialized")
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps.
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with all engineered features
        """
        df = df.copy()
        
        # Basic price features
        df = self._add_price_features(df)
        
        # Moving averages
        df = self._add_moving_averages(df)
        
        # Momentum indicators
        df = self._add_momentum_indicators(df)
        
        # Volatility indicators
        df = self._add_volatility_indicators(df)
        
        # Volume indicators
        df = self._add_volume_indicators(df)
        
        # Candle patterns
        df = self._add_candle_features(df)
        
        # Statistical features
        df = self._add_statistical_features(df)
        
        # Cross-features and interactions
        df = self._add_interaction_features(df)
        
        feature_cols = [col for col in df.columns 
                       if col not in ['time', 'open', 'high', 'low', 'close', 'volume']]
        logger.info(f"Engineered {len(feature_cols)} features")
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-derived features."""
        # Returns
        df['returns'] = df['close'].pct_change()
        df['returns_5'] = df['close'].pct_change(5)
        df['returns_10'] = df['close'].pct_change(10)
        df['returns_20'] = df['close'].pct_change(20)
        
        # Price position in range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        
        # High-Low spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        
        return df
    
    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add EMA and SMA indicators."""
        # EMAs
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # SMAs
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # Price vs MA ratios
        df['price_ema20_ratio'] = df['close'] / (df['ema_20'] + 1e-8)
        df['price_ema50_ratio'] = df['close'] / (df['ema_50'] + 1e-8)
        df['ema20_ema50_ratio'] = df['ema_20'] / (df['ema_50'] + 1e-8)
        df['ema50_ema200_ratio'] = df['ema_50'] / (df['ema_200'] + 1e-8)
        
        # MA slopes
        df['ema20_slope'] = df['ema_20'].diff(5) / df['ema_20']
        df['ema50_slope'] = df['ema_50'].diff(10) / df['ema_50']
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI, MACD, and momentum indicators."""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        df['macd_cross'] = ((df['macd'] > df['macd_signal']) & 
                           (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        
        # Momentum
        df['momentum_10'] = df['close'].pct_change(10)
        df['momentum_20'] = df['close'].pct_change(20)
        
        # Rate of Change
        df['roc_10'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
        df['roc_20'] = ((df['close'] - df['close'].shift(20)) / df['close'].shift(20)) * 100
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ATR, Bollinger Bands, and volatility metrics."""
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        df['atr_pct'] = df['atr'] / df['close']
        df['atr_percentile'] = df['atr'].rolling(window=100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
        bb_std_val = df['close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std_val * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std_val * bb_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).quantile(0.2)).astype(int)
        
        # Volatility (rolling std of returns)
        df['volatility_10'] = df['returns'].rolling(window=10).std()
        df['volatility_20'] = df['returns'].rolling(window=20).std()
        df['volatility_ratio'] = df['volatility_10'] / (df['volatility_20'] + 1e-8)
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        # Volume MA
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma_20'] + 1e-8)
        
        # Volume-price trend
        df['vpt'] = (df['volume'] * df['returns']).cumsum()
        
        # On-Balance Volume
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv_ema'] = df['obv'].ewm(span=20, adjust=False).mean()
        
        # Volume-weighted average price (simplified)
        df['vwap'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
        df['price_vwap_ratio'] = df['close'] / (df['vwap'] + 1e-8)
        
        return df
    
    def _add_candle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candle pattern features."""
        # Body and wick calculations
        df['body'] = np.abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['total_range'] = df['high'] - df['low']
        
        # Body percentage
        df['body_pct'] = df['body'] / (df['total_range'] + 1e-8)
        
        # Wick ratios
        df['upper_wick_ratio'] = df['upper_wick'] / (df['total_range'] + 1e-8)
        df['lower_wick_ratio'] = df['lower_wick'] / (df['total_range'] + 1e-8)
        df['wick_ratio'] = (df['upper_wick'] + df['lower_wick']) / (df['body'] + 1e-8)
        
        # Candle direction
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        df['is_bearish'] = (df['close'] < df['open']).astype(int)
        df['is_doji'] = (df['body_pct'] < 0.1).astype(int)
        
        # Hammer pattern (simplified)
        df['is_hammer'] = ((df['lower_wick_ratio'] > 2) & 
                          (df['upper_wick_ratio'] < 0.3) & 
                          (df['body_pct'] < 0.3)).astype(int)
        
        # Engulfing pattern (simplified)
        prev_bullish = df['is_bullish'].shift(1)
        prev_bearish = df['is_bearish'].shift(1)
        df['bullish_engulfing'] = ((df['is_bullish'] == 1) & 
                                   (prev_bearish == 1) &
                                   (df['close'] > df['open'].shift(1)) &
                                   (df['open'] < df['close'].shift(1))).astype(int)
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features over rolling windows."""
        # Z-scores
        df['zscore_20'] = (df['close'] - df['close'].rolling(20).mean()) / (df['close'].rolling(20).std() + 1e-8)
        df['zscore_50'] = (df['close'] - df['close'].rolling(50).mean()) / (df['close'].rolling(50).std() + 1e-8)
        
        # Percentiles
        df['price_percentile_20'] = df['close'].rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        
        # Skewness and Kurtosis
        df['returns_skew_20'] = df['returns'].rolling(20).skew()
        df['returns_kurt_20'] = df['returns'].rolling(20).kurt()
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-features and interactions."""
        # Trend strength (EMA alignment)
        df['trend_strength'] = (
            (df['ema_20'] > df['ema_50']).astype(int) +
            (df['ema_50'] > df['ema_200']).astype(int) +
            (df['close'] > df['ema_20']).astype(int)
        ) / 3.0
        
        # Volatility regime (high/low volatility)
        df['volatility_regime'] = (df['atr_percentile'] > 0.7).astype(int)
        
        # Momentum divergence
        df['momentum_divergence'] = np.abs(df['rsi'] - df['macd_histogram'] / (df['macd'] + 1e-8) * 100)
        
        return df
