"""
Feature Engineering Module for Natron
Computes 50-70 technical indicators and derived features from OHLCV data.
"""

import pandas as pd
import numpy as np
from typing import Optional


class FeatureEngineer:
    """
    Engine for computing technical indicators and derived features.
    """
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.feature_names = []
    
    def compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Compute Average True Range (ATR).
        
        Args:
            df: DataFrame with OHLC columns
            period: ATR period (default: 14)
            
        Returns:
            ATR series
        """
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def compute_ema(self, series: pd.Series, period: int) -> pd.Series:
        """
        Compute Exponential Moving Average (EMA).
        
        Args:
            series: Price series (typically close)
            period: EMA period
            
        Returns:
            EMA series
        """
        return series.ewm(span=period, adjust=False).mean()
    
    def compute_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """
        Compute Relative Strength Index (RSI).
        
        Args:
            series: Price series
            period: RSI period (default: 14)
            
        Returns:
            RSI series (0-100)
        """
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def compute_macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """
        Compute MACD (Moving Average Convergence Divergence).
        
        Args:
            series: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            
        Returns:
            Tuple of (MACD, Signal, Histogram)
        """
        ema_fast = self.compute_ema(series, fast)
        ema_slow = self.compute_ema(series, slow)
        
        macd = ema_fast - ema_slow
        signal_line = self.compute_ema(macd, signal)
        histogram = macd - signal_line
        
        return macd, signal_line, histogram
    
    def compute_bollinger_bands(self, series: pd.Series, period: int = 20, std_dev: int = 2) -> tuple:
        """
        Compute Bollinger Bands.
        
        Args:
            series: Price series
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            Tuple of (upper, middle, lower, width)
        """
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        width = (upper - lower) / middle
        
        return upper, middle, lower, width
    
    def compute_candle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute candle body and wick features.
        
        Args:
            df: DataFrame with OHLC columns
            
        Returns:
            DataFrame with body_pct and wick_ratio columns
        """
        body = np.abs(df['close'] - df['open'])
        total_range = df['high'] - df['low']
        
        body_pct = body / total_range
        body_pct = body_pct.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        total_wick = upper_wick + lower_wick
        
        wick_ratio = total_wick / (total_range + 1e-8)  # Avoid division by zero
        wick_ratio = wick_ratio.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        result = pd.DataFrame({
            'body_pct': body_pct,
            'wick_ratio': wick_ratio
        })
        
        return result
    
    def compute_volatility_regime(self, df: pd.DataFrame, atr: pd.Series, window: int = 20) -> pd.Series:
        """
        Compute volatility regime based on ATR percentiles.
        
        Args:
            df: DataFrame with price data
            atr: ATR series
            window: Rolling window for percentile calculation
            
        Returns:
            Volatility regime series (0=low, 1=normal, 2=high)
        """
        atr_pct = atr.rolling(window=window).quantile([0.33, 0.67])
        
        regime = pd.Series(index=df.index, dtype=int)
        regime[:] = 1  # Normal by default
        
        low_threshold = atr.rolling(window=window).quantile(0.33)
        high_threshold = atr.rolling(window=window).quantile(0.67)
        
        regime[atr < low_threshold] = 0  # Low volatility
        regime[atr > high_threshold] = 2  # High volatility
        
        return regime.fillna(1)
    
    def compute_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute additional price-based features.
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with additional price features
        """
        features = pd.DataFrame(index=df.index)
        
        # Returns
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price position within range
        features['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        
        # Volume features
        if 'volume' in df.columns:
            features['volume_ma'] = df['volume'].rolling(window=20).mean()
            features['volume_ratio'] = df['volume'] / (features['volume_ma'] + 1e-8)
            features['volume_price_trend'] = (df['volume'] * features['returns']).rolling(window=10).sum()
        
        # Momentum
        features['momentum_5'] = df['close'].pct_change(5)
        features['momentum_10'] = df['close'].pct_change(10)
        features['momentum_20'] = df['close'].pct_change(20)
        
        # Volatility
        features['volatility_5'] = features['returns'].rolling(window=5).std()
        features['volatility_20'] = features['returns'].rolling(window=20).std()
        
        return features.fillna(0)
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to compute all features.
        
        Args:
            df: DataFrame with OHLCV columns (time, open, high, low, close, volume)
            
        Returns:
            DataFrame with all engineered features
        """
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            raise ValueError("DataFrame must contain 'open', 'high', 'low', 'close' columns")
        
        features_df = df.copy()
        
        # ATR
        features_df['ATR'] = self.compute_atr(df, period=14)
        
        # EMAs
        features_df['EMA_20'] = self.compute_ema(df['close'], 20)
        features_df['EMA_50'] = self.compute_ema(df['close'], 50)
        features_df['EMA_200'] = self.compute_ema(df['close'], 200)
        
        # EMA relationships
        features_df['EMA_20_50_diff'] = features_df['EMA_20'] - features_df['EMA_50']
        features_df['EMA_50_200_diff'] = features_df['EMA_50'] - features_df['EMA_200']
        features_df['price_EMA20_diff'] = df['close'] - features_df['EMA_20']
        features_df['price_EMA50_diff'] = df['close'] - features_df['EMA_50']
        
        # RSI
        features_df['RSI'] = self.compute_rsi(df['close'], period=14)
        
        # MACD
        macd, signal, hist = self.compute_macd(df['close'])
        features_df['MACD'] = macd
        features_df['MACD_signal'] = signal
        features_df['MACD_hist'] = hist
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower, bb_width = self.compute_bollinger_bands(df['close'])
        features_df['BB_upper'] = bb_upper
        features_df['BB_middle'] = bb_middle
        features_df['BB_lower'] = bb_lower
        features_df['BB_width'] = bb_width
        features_df['BB_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
        
        # Candle features
        candle_features = self.compute_candle_features(df)
        features_df['body_pct'] = candle_features['body_pct']
        features_df['wick_ratio'] = candle_features['wick_ratio']
        
        # Volatility regime
        features_df['volatility_regime'] = self.compute_volatility_regime(df, features_df['ATR'])
        
        # Additional price features
        price_features = self.compute_price_features(df)
        features_df = pd.concat([features_df, price_features], axis=1)
        
        # Fill NaN values
        features_df = features_df.fillna(method='bfill').fillna(0)
        
        # Store feature names
        self.feature_names = [col for col in features_df.columns 
                             if col not in ['time', 'open', 'high', 'low', 'close', 'volume']]
        
        return features_df
    
    def get_feature_names(self) -> list:
        """Return list of engineered feature names."""
        return self.feature_names
