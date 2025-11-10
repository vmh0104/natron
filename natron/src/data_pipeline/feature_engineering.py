"""
Feature Engineering Module for Natron Trading System

This module computes 50-70 technical indicators and derived features
from OHLCV market data for use in machine learning models.
"""

import pandas as pd
import numpy as np
from typing import Optional


class FeatureEngineer:
    """
    Feature engineering class that computes technical indicators
    and derived features from OHLCV data.
    """
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.feature_names = []
    
    def compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Compute Average True Range (ATR).
        
        Args:
            df: DataFrame with OHLC columns
            period: ATR period (default 14)
            
        Returns:
            Series with ATR values
        """
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def compute_ema(self, series: pd.Series, period: int) -> pd.Series:
        """
        Compute Exponential Moving Average (EMA).
        
        Args:
            series: Price series
            period: EMA period
            
        Returns:
            Series with EMA values
        """
        return series.ewm(span=period, adjust=False).mean()
    
    def compute_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """
        Compute Relative Strength Index (RSI).
        
        Args:
            series: Price series
            period: RSI period (default 14)
            
        Returns:
            Series with RSI values (0-100)
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
            Tuple of (MACD line, Signal line, Histogram)
        """
        ema_fast = self.compute_ema(series, fast)
        ema_slow = self.compute_ema(series, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.compute_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def compute_bollinger_bands(self, series: pd.Series, period: int = 20, std_dev: float = 2.0) -> tuple:
        """
        Compute Bollinger Bands.
        
        Args:
            series: Price series
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        middle_band = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return upper_band, middle_band, lower_band
    
    def compute_candle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute candle body and wick features.
        
        Args:
            df: DataFrame with OHLC columns
            
        Returns:
            DataFrame with candle features
        """
        features = pd.DataFrame(index=df.index)
        
        # Body percentage (body size relative to range)
        body_size = np.abs(df['close'] - df['open'])
        candle_range = df['high'] - df['low']
        features['body_pct'] = body_size / (candle_range + 1e-8)
        
        # Upper wick ratio
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        features['upper_wick_ratio'] = upper_wick / (candle_range + 1e-8)
        
        # Lower wick ratio
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        features['lower_wick_ratio'] = lower_wick / (candle_range + 1e-8)
        
        # Candle direction
        features['is_bullish'] = (df['close'] > df['open']).astype(int)
        features['is_bearish'] = (df['close'] < df['open']).astype(int)
        features['is_doji'] = (body_size / (df['close'] + 1e-8) < 0.001).astype(int)
        
        return features
    
    def compute_volatility_features(self, df: pd.DataFrame, atr: pd.Series) -> pd.DataFrame:
        """
        Compute volatility regime features.
        
        Args:
            df: DataFrame with OHLC columns
            atr: ATR series
            
        Returns:
            DataFrame with volatility features
        """
        features = pd.DataFrame(index=df.index)
        
        # ATR percentile
        features['atr_percentile'] = atr.rolling(window=100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        
        # Price volatility (rolling std)
        features['price_volatility'] = df['close'].rolling(window=20).std()
        features['price_volatility_pct'] = features['price_volatility'] / (df['close'] + 1e-8)
        
        # Volume volatility
        if 'volume' in df.columns:
            features['volume_volatility'] = df['volume'].rolling(window=20).std()
            features['volume_ma_ratio'] = df['volume'] / (df['volume'].rolling(window=20).mean() + 1e-8)
        
        return features
    
    def compute_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute momentum and rate of change features.
        
        Args:
            df: DataFrame with OHLC columns
            
        Returns:
            DataFrame with momentum features
        """
        features = pd.DataFrame(index=df.index)
        
        # Rate of change
        for period in [1, 3, 5, 10, 20]:
            features[f'roc_{period}'] = df['close'].pct_change(periods=period)
        
        # Momentum
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        # Price position in range
        for period in [10, 20, 50]:
            rolling_high = df['high'].rolling(window=period).max()
            rolling_low = df['low'].rolling(window=period).min()
            features[f'price_position_{period}'] = (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-8)
        
        return features
    
    def compute_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute trend strength and direction features.
        
        Args:
            df: DataFrame with OHLC columns
            
        Returns:
            DataFrame with trend features
        """
        features = pd.DataFrame(index=df.index)
        
        # EMA slopes
        for period in [20, 50, 200]:
            ema = self.compute_ema(df['close'], period)
            features[f'ema_{period}'] = ema
            features[f'ema_{period}_slope'] = ema.diff()
            features[f'price_vs_ema_{period}'] = (df['close'] - ema) / (ema + 1e-8)
        
        # EMA crossovers
        ema_20 = self.compute_ema(df['close'], 20)
        ema_50 = self.compute_ema(df['close'], 50)
        ema_200 = self.compute_ema(df['close'], 200)
        
        features['ema_20_50_cross'] = (ema_20 > ema_50).astype(int)
        features['ema_50_200_cross'] = (ema_50 > ema_200).astype(int)
        features['golden_cross'] = ((ema_20 > ema_50) & (ema_50 > ema_200)).astype(int)
        features['death_cross'] = ((ema_20 < ema_50) & (ema_50 < ema_200)).astype(int)
        
        # Trend slope (linear regression slope)
        for period in [10, 20, 50]:
            slopes = []
            for i in range(len(df)):
                if i < period:
                    slopes.append(np.nan)
                else:
                    y = df['close'].iloc[i-period:i].values
                    x = np.arange(len(y))
                    slope = np.polyfit(x, y, 1)[0]
                    slopes.append(slope)
            features[f'trend_slope_{period}'] = slopes
        
        return features
    
    def compute_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute volume-based features.
        
        Args:
            df: DataFrame with volume column
            
        Returns:
            DataFrame with volume features
        """
        features = pd.DataFrame(index=df.index)
        
        if 'volume' not in df.columns:
            return features
        
        # Volume moving averages
        for period in [10, 20, 50]:
            features[f'volume_ma_{period}'] = df['volume'].rolling(window=period).mean()
            features[f'volume_ratio_{period}'] = df['volume'] / (features[f'volume_ma_{period}'] + 1e-8)
        
        # On Balance Volume (OBV)
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        features['obv'] = obv
        features['obv_ema'] = self.compute_ema(obv, 20)
        
        # Volume Price Trend (VPT)
        vpt = (df['close'].pct_change() * df['volume']).fillna(0).cumsum()
        features['vpt'] = vpt
        
        return features
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to compute all features.
        
        Args:
            df: DataFrame with OHLCV columns (time, open, high, low, close, volume)
            
        Returns:
            DataFrame with all engineered features
        """
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")
        
        # Start with base features
        feature_df = pd.DataFrame(index=df.index)
        
        # Compute ATR
        atr = self.compute_atr(df)
        feature_df['atr'] = atr
        feature_df['atr_pct'] = atr / (df['close'] + 1e-8)
        
        # Compute EMAs
        for period in [20, 50, 200]:
            ema = self.compute_ema(df['close'], period)
            feature_df[f'ema_{period}'] = ema
        
        # Compute RSI
        rsi = self.compute_rsi(df['close'])
        feature_df['rsi'] = rsi
        feature_df['rsi_overbought'] = (rsi > 70).astype(int)
        feature_df['rsi_oversold'] = (rsi < 30).astype(int)
        
        # Compute MACD
        macd_line, signal_line, histogram = self.compute_macd(df['close'])
        feature_df['macd'] = macd_line
        feature_df['macd_signal'] = signal_line
        feature_df['macd_histogram'] = histogram
        feature_df['macd_cross'] = (macd_line > signal_line).astype(int)
        
        # Compute Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.compute_bollinger_bands(df['close'])
        feature_df['bb_upper'] = bb_upper
        feature_df['bb_middle'] = bb_middle
        feature_df['bb_lower'] = bb_lower
        feature_df['bb_width'] = (bb_upper - bb_lower) / (bb_middle + 1e-8)
        feature_df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
        feature_df['bb_squeeze'] = (feature_df['bb_width'] < feature_df['bb_width'].rolling(20).quantile(0.2)).astype(int)
        
        # Compute candle features
        candle_features = self.compute_candle_features(df)
        feature_df = pd.concat([feature_df, candle_features], axis=1)
        
        # Compute volatility features
        volatility_features = self.compute_volatility_features(df, atr)
        feature_df = pd.concat([feature_df, volatility_features], axis=1)
        
        # Compute momentum features
        momentum_features = self.compute_momentum_features(df)
        feature_df = pd.concat([feature_df, momentum_features], axis=1)
        
        # Compute trend features
        trend_features = self.compute_trend_features(df)
        feature_df = pd.concat([feature_df, trend_features], axis=1)
        
        # Compute volume features
        if 'volume' in df.columns:
            volume_features = self.compute_volume_features(df)
            feature_df = pd.concat([feature_df, volume_features], axis=1)
        
        # Store feature names
        self.feature_names = feature_df.columns.tolist()
        
        return feature_df
