"""
Feature Engineering Module for Natron AI Trading System
Computes 50-70 technical indicators and derived features from OHLCV data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Feature engineering class that computes technical indicators and derived features.
    """
    
    def __init__(self):
        self.feature_names = []
    
    def compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Compute Average True Range (ATR).
        
        Args:
            df: DataFrame with OHLC columns
            period: ATR period (default 14)
            
        Returns:
            ATR series
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
            EMA series
        """
        return series.ewm(span=period, adjust=False).mean()
    
    def compute_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """
        Compute Relative Strength Index (RSI).
        
        Args:
            series: Price series
            period: RSI period (default 14)
            
        Returns:
            RSI series (0-100)
        """
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def compute_macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        Compute MACD (Moving Average Convergence Divergence).
        
        Args:
            series: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            Dictionary with 'macd', 'signal', 'histogram'
        """
        ema_fast = self.compute_ema(series, fast)
        ema_slow = self.compute_ema(series, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.compute_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def compute_bollinger_bands(self, series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        Compute Bollinger Bands.
        
        Args:
            series: Price series
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            Dictionary with 'upper', 'middle', 'lower' bands
        """
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        
        return {
            'upper': sma + (std * std_dev),
            'middle': sma,
            'lower': sma - (std * std_dev)
        }
    
    def compute_candle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute candle body and wick features.
        
        Args:
            df: DataFrame with OHLC columns
            
        Returns:
            DataFrame with candle features
        """
        features = pd.DataFrame(index=df.index)
        
        # Body percentage (close - open) / close
        features['body_pct'] = ((df['close'] - df['open']) / df['close']) * 100
        
        # Upper wick ratio
        features['upper_wick_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        
        # Lower wick ratio
        features['lower_wick_ratio'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
        
        # Total wick ratio
        features['wick_ratio'] = features['upper_wick_ratio'] + features['lower_wick_ratio']
        
        # Candle range percentage
        features['range_pct'] = ((df['high'] - df['low']) / df['close']) * 100
        
        return features
    
    def compute_volatility_features(self, df: pd.DataFrame, atr: pd.Series) -> pd.DataFrame:
        """
        Compute volatility-related features.
        
        Args:
            df: DataFrame with OHLC columns
            atr: ATR series
            
        Returns:
            DataFrame with volatility features
        """
        features = pd.DataFrame(index=df.index)
        
        # ATR percentage of price
        features['atr_pct'] = (atr / df['close']) * 100
        
        # Rolling volatility (std of returns)
        returns = df['close'].pct_change()
        features['volatility_20'] = returns.rolling(window=20).std() * np.sqrt(252) * 100
        features['volatility_50'] = returns.rolling(window=50).std() * np.sqrt(252) * 100
        
        # ATR percentile (rolling 100 period)
        features['atr_percentile'] = atr.rolling(window=100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        
        return features
    
    def compute_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute trend-related features.
        
        Args:
            df: DataFrame with OHLC columns
            
        Returns:
            DataFrame with trend features
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']
        
        # Price slope (linear regression over 20, 50, 200 periods)
        for period in [20, 50, 200]:
            slope = close.rolling(window=period).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == period else np.nan,
                raw=False
            )
            features[f'trend_slope_{period}'] = slope
        
        # Price position relative to EMA bands
        ema_20 = self.compute_ema(close, 20)
        ema_50 = self.compute_ema(close, 50)
        ema_200 = self.compute_ema(close, 200)
        
        features['price_vs_ema20'] = (close - ema_20) / close * 100
        features['price_vs_ema50'] = (close - ema_50) / close * 100
        features['price_vs_ema200'] = (close - ema_200) / close * 100
        
        # EMA crossovers
        features['ema20_above_ema50'] = (ema_20 > ema_50).astype(int)
        features['ema50_above_ema200'] = (ema_50 > ema_200).astype(int)
        
        return features
    
    def compute_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute momentum indicators.
        
        Args:
            df: DataFrame with OHLC columns
            
        Returns:
            DataFrame with momentum features
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']
        
        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            features[f'roc_{period}'] = close.pct_change(period) * 100
        
        # Stochastic Oscillator
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        features['stoch_k'] = 100 * ((close - low_14) / (high_14 - low_14))
        features['stoch_d'] = features['stoch_k'].rolling(window=3).mean()
        
        # Williams %R
        features['williams_r'] = -100 * ((high_14 - close) / (high_14 - low_14))
        
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
        volume = df['volume']
        
        # Volume moving averages
        features['volume_ma_20'] = volume.rolling(window=20).mean()
        features['volume_ma_50'] = volume.rolling(window=50).mean()
        
        # Volume ratio
        features['volume_ratio'] = volume / features['volume_ma_20']
        
        # On-Balance Volume (OBV)
        price_change = df['close'].diff()
        obv = (volume * np.sign(price_change)).fillna(0).cumsum()
        features['obv'] = obv
        features['obv_ema'] = self.compute_ema(obv, 20)
        
        # Volume Price Trend (VPT)
        vpt = (volume * price_change / df['close'].shift()).fillna(0).cumsum()
        features['vpt'] = vpt
        
        return features
    
    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features (50-70 features total).
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with all computed features
        """
        feature_df = pd.DataFrame(index=df.index)
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # ATR
        atr = self.compute_atr(df, period=14)
        feature_df['atr'] = atr
        
        # EMAs
        close = df['close']
        feature_df['ema_20'] = self.compute_ema(close, 20)
        feature_df['ema_50'] = self.compute_ema(close, 50)
        feature_df['ema_200'] = self.compute_ema(close, 200)
        
        # RSI
        feature_df['rsi'] = self.compute_rsi(close, period=14)
        feature_df['rsi_9'] = self.compute_rsi(close, period=9)
        feature_df['rsi_21'] = self.compute_rsi(close, period=21)
        
        # MACD
        macd_dict = self.compute_macd(close)
        feature_df['macd'] = macd_dict['macd']
        feature_df['macd_signal'] = macd_dict['signal']
        feature_df['macd_histogram'] = macd_dict['histogram']
        
        # Bollinger Bands
        bb_dict = self.compute_bollinger_bands(close, period=20, std_dev=2.0)
        feature_df['bb_upper'] = bb_dict['upper']
        feature_df['bb_middle'] = bb_dict['middle']
        feature_df['bb_lower'] = bb_dict['lower']
        feature_df['bb_width'] = (bb_dict['upper'] - bb_dict['lower']) / bb_dict['middle'] * 100
        feature_df['bb_position'] = (close - bb_dict['lower']) / (bb_dict['upper'] - bb_dict['lower'])
        
        # Candle features
        candle_features = self.compute_candle_features(df)
        feature_df = pd.concat([feature_df, candle_features], axis=1)
        
        # Volatility features
        vol_features = self.compute_volatility_features(df, atr)
        feature_df = pd.concat([feature_df, vol_features], axis=1)
        
        # Trend features
        trend_features = self.compute_trend_features(df)
        feature_df = pd.concat([feature_df, trend_features], axis=1)
        
        # Momentum features
        momentum_features = self.compute_momentum_features(df)
        feature_df = pd.concat([feature_df, momentum_features], axis=1)
        
        # Volume features
        volume_features = self.compute_volume_features(df)
        feature_df = pd.concat([feature_df, volume_features], axis=1)
        
        # Additional derived features
        # Price change percentages
        for period in [1, 5, 10, 20]:
            feature_df[f'price_change_{period}'] = close.pct_change(period) * 100
        
        # High-Low spread
        feature_df['hl_spread'] = (df['high'] - df['low']) / close * 100
        
        # Close position in daily range
        feature_df['close_position'] = (close - df['low']) / (df['high'] - df['low'])
        
        # Fill NaN values with forward fill then backward fill
        feature_df = feature_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        self.feature_names = list(feature_df.columns)
        
        return feature_df
