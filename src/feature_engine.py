"""
Feature Engineering Module - Generates ~100 Technical Indicators
Optimized for GPU-accelerated computation

Feature Groups:
- Moving Averages (13)
- Momentum (13)
- Volatility (15)
- Volume (9)
- Price Patterns (8)
- Returns (8)
- Trend Strength (6)
- Statistical (6)
- Support/Resistance (4)
- Smart Money Concepts (6)
- Market Profile (10)
"""

import numpy as np
import pandas as pd
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


class FeatureEngine:
    """Generate 100+ technical features from OHLCV data"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.feature_names = []
        
    def generate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all 100+ features from OHLCV data
        
        Args:
            df: DataFrame with columns [time, open, high, low, close, volume]
            
        Returns:
            DataFrame with ~100 feature columns
        """
        if self.verbose:
            print("🔧 Generating features...")
            
        features_df = df[['time', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        # Group 1: Moving Averages (13 features)
        features_df = self._add_moving_average_features(features_df)
        
        # Group 2: Momentum Indicators (13 features)
        features_df = self._add_momentum_features(features_df)
        
        # Group 3: Volatility Indicators (15 features)
        features_df = self._add_volatility_features(features_df)
        
        # Group 4: Volume Indicators (9 features)
        features_df = self._add_volume_features(features_df)
        
        # Group 5: Price Patterns (8 features)
        features_df = self._add_price_pattern_features(features_df)
        
        # Group 6: Returns (8 features)
        features_df = self._add_return_features(features_df)
        
        # Group 7: Trend Strength (6 features)
        features_df = self._add_trend_strength_features(features_df)
        
        # Group 8: Statistical Features (6 features)
        features_df = self._add_statistical_features(features_df)
        
        # Group 9: Support/Resistance (4 features)
        features_df = self._add_support_resistance_features(features_df)
        
        # Group 10: Smart Money Concepts (6 features)
        features_df = self._add_smc_features(features_df)
        
        # Group 11: Market Profile (10 features)
        features_df = self._add_market_profile_features(features_df)
        
        # Fill NaN values
        features_df = features_df.fillna(method='bfill').fillna(0)
        
        if self.verbose:
            feature_cols = [c for c in features_df.columns if c not in ['time', 'open', 'high', 'low', 'close', 'volume']]
            print(f"✅ Generated {len(feature_cols)} features")
            
        return features_df
    
    def _add_moving_average_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Moving Average features (13)"""
        close = df['close']
        
        # Simple Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'MA_{period}'] = close.rolling(period).mean()
            
        # Exponential Moving Averages
        for period in [12, 26]:
            df[f'EMA_{period}'] = close.ewm(span=period, adjust=False).mean()
            
        # MA slopes
        df['MA20_slope'] = df['MA_20'].diff(5) / df['MA_20']
        df['MA50_slope'] = df['MA_50'].diff(10) / df['MA_50']
        
        # Price to MA ratios
        df['price_to_MA20'] = close / df['MA_20']
        df['price_to_MA50'] = close / df['MA_50']
        
        # MA crossovers
        df['MA_cross_20_50'] = (df['MA_20'] > df['MA_50']).astype(int)
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum indicators (13)"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # RSI (14-period)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # RSI variants
        df['RSI_7'] = self._calculate_rsi(close, 7)
        df['RSI_21'] = self._calculate_rsi(close, 21)
        
        # Rate of Change
        df['ROC_12'] = ((close - close.shift(12)) / close.shift(12)) * 100
        df['ROC_25'] = ((close - close.shift(25)) / close.shift(25)) * 100
        
        # Stochastic Oscillator
        low_14 = low.rolling(window=14).min()
        high_14 = high.rolling(window=14).max()
        df['Stoch_K'] = 100 * ((close - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # CCI (Commodity Channel Index)
        tp = (high + low + close) / 3
        df['CCI_20'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
        
        # Momentum
        df['MOM_10'] = close - close.shift(10)
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility indicators (15)"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # ATR (Average True Range)
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR_14'] = tr.rolling(window=14).mean()
        df['ATR_20'] = tr.rolling(window=20).mean()
        
        # Bollinger Bands
        for period in [20, 50]:
            ma = close.rolling(period).mean()
            std = close.rolling(period).std()
            df[f'BB_upper_{period}'] = ma + (2 * std)
            df[f'BB_lower_{period}'] = ma - (2 * std)
            df[f'BB_width_{period}'] = (df[f'BB_upper_{period}'] - df[f'BB_lower_{period}']) / ma
            df[f'BB_position_{period}'] = (close - df[f'BB_lower_{period}']) / (df[f'BB_upper_{period}'] - df[f'BB_lower_{period}'])
        
        # Keltner Channels
        ma_20 = close.rolling(20).mean()
        df['Keltner_upper'] = ma_20 + (2 * df['ATR_20'])
        df['Keltner_lower'] = ma_20 - (2 * df['ATR_20'])
        
        # Standard Deviation
        df['STD_20'] = close.rolling(20).std()
        df['STD_50'] = close.rolling(50).std()
        
        # Historical Volatility
        returns = np.log(close / close.shift())
        df['HV_20'] = returns.rolling(20).std() * np.sqrt(252)
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume indicators (9)"""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Volume Moving Averages
        df['Volume_MA_20'] = volume.rolling(20).mean()
        df['Volume_MA_50'] = volume.rolling(50).mean()
        
        # Volume Ratio
        df['Volume_ratio'] = volume / df['Volume_MA_20']
        
        # On-Balance Volume (OBV)
        obv = np.where(close > close.shift(), volume, 
                      np.where(close < close.shift(), -volume, 0))
        df['OBV'] = pd.Series(obv).cumsum()
        
        # VWAP (Volume Weighted Average Price)
        tp = (high + low + close) / 3
        df['VWAP'] = (tp * volume).rolling(20).sum() / volume.rolling(20).sum()
        
        # Money Flow Index
        tp_diff = tp.diff()
        positive_flow = np.where(tp_diff > 0, tp * volume, 0)
        negative_flow = np.where(tp_diff < 0, tp * volume, 0)
        positive_mf = pd.Series(positive_flow).rolling(14).sum()
        negative_mf = pd.Series(negative_flow).rolling(14).sum()
        mfi_ratio = positive_mf / negative_mf
        df['MFI_14'] = 100 - (100 / (1 + mfi_ratio))
        
        # Volume Oscillator
        df['Volume_osc'] = ((df['Volume_MA_20'] - df['Volume_MA_50']) / df['Volume_MA_50']) * 100
        
        # Accumulation/Distribution
        clv = ((close - low) - (high - close)) / (high - low)
        df['AD_line'] = (clv * volume).cumsum()
        
        return df
    
    def _add_price_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price pattern features (8)"""
        open_p = df['open']
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Candle body and shadows
        df['body'] = np.abs(close - open_p)
        df['upper_shadow'] = high - np.maximum(open_p, close)
        df['lower_shadow'] = np.minimum(open_p, close) - low
        df['body_pct'] = df['body'] / (high - low)
        
        # Doji detection
        df['is_doji'] = (df['body'] / (high - low) < 0.1).astype(int)
        
        # Price position in candle
        df['close_position'] = (close - low) / (high - low)
        
        # Gap detection
        df['gap_up'] = (low > high.shift()).astype(int)
        df['gap_down'] = (high < low.shift()).astype(int)
        
        return df
    
    def _add_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return-based features (8)"""
        close = df['close']
        open_p = df['open']
        
        # Simple returns
        df['return_1'] = close.pct_change(1)
        df['return_5'] = close.pct_change(5)
        df['return_10'] = close.pct_change(10)
        df['return_20'] = close.pct_change(20)
        
        # Log returns
        df['log_return'] = np.log(close / close.shift())
        
        # Intraday return
        df['intraday_return'] = (close - open_p) / open_p
        
        # Cumulative returns
        df['cum_return_20'] = (close / close.shift(20)) - 1
        df['cum_return_50'] = (close / close.shift(50)) - 1
        
        return df
    
    def _add_trend_strength_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trend strength indicators (6)"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # ADX (Average Directional Index)
        high_diff = high.diff()
        low_diff = -low.diff()
        
        pos_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        neg_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        tr = pd.concat([high - low, 
                       np.abs(high - close.shift()), 
                       np.abs(low - close.shift())], axis=1).max(axis=1)
        
        atr_14 = tr.rolling(14).mean()
        pos_di = 100 * (pd.Series(pos_dm).rolling(14).mean() / atr_14)
        neg_di = 100 * (pd.Series(neg_dm).rolling(14).mean() / atr_14)
        
        df['ADX_14'] = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di)
        df['Plus_DI'] = pos_di
        df['Minus_DI'] = neg_di
        
        # Aroon Indicator
        aroon_period = 25
        df['Aroon_up'] = 100 * high.rolling(aroon_period).apply(
            lambda x: aroon_period - (aroon_period - 1 - x.argmax())
        ) / aroon_period
        df['Aroon_down'] = 100 * low.rolling(aroon_period).apply(
            lambda x: aroon_period - (aroon_period - 1 - x.argmin())
        ) / aroon_period
        df['Aroon_osc'] = df['Aroon_up'] - df['Aroon_down']
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Statistical features (6)"""
        close = df['close']
        returns = close.pct_change()
        
        # Skewness
        df['skew_20'] = returns.rolling(20).skew()
        df['skew_50'] = returns.rolling(50).skew()
        
        # Kurtosis
        df['kurt_20'] = returns.rolling(20).kurt()
        
        # Z-score
        df['zscore_20'] = (close - close.rolling(20).mean()) / close.rolling(20).std()
        
        # Hurst Exponent (simplified version)
        df['hurst_50'] = self._calculate_hurst(close, 50)
        
        # Autocorrelation
        df['autocorr_5'] = returns.rolling(20).apply(
            lambda x: x.autocorr(lag=5) if len(x) > 5 else 0
        )
        
        return df
    
    def _add_support_resistance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Support/Resistance features (4)"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Distance to recent highs/lows
        df['dist_to_high_20'] = (high.rolling(20).max() - close) / close
        df['dist_to_low_20'] = (close - low.rolling(20).min()) / close
        df['dist_to_high_50'] = (high.rolling(50).max() - close) / close
        df['dist_to_low_50'] = (close - low.rolling(50).min()) / close
        
        return df
    
    def _add_smc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Smart Money Concepts features (6)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Swing High/Low (simplified)
        df['swing_high'] = high.rolling(5, center=True).max()
        df['swing_low'] = low.rolling(5, center=True).min()
        df['is_swing_high'] = (high == df['swing_high']).astype(int)
        df['is_swing_low'] = (low == df['swing_low']).astype(int)
        
        # Break of Structure (BOS) - simplified
        df['BOS_bull'] = (close > df['swing_high'].shift()).astype(int)
        df['BOS_bear'] = (close < df['swing_low'].shift()).astype(int)
        
        return df
    
    def _add_market_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market Profile features (10)"""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Value Area High/Low (approximation using quantiles)
        df['VAH_20'] = close.rolling(20).quantile(0.7)
        df['VAL_20'] = close.rolling(20).quantile(0.3)
        df['POC_20'] = close.rolling(20).median()  # Point of Control approximation
        
        # Price distribution features
        df['price_range_20'] = high.rolling(20).max() - low.rolling(20).min()
        df['volume_profile_position'] = (close - df['VAL_20']) / (df['VAH_20'] - df['VAL_20'])
        
        # Volume distribution
        df['volume_std_20'] = volume.rolling(20).std()
        df['volume_skew_20'] = volume.rolling(20).skew()
        
        # Price entropy (measure of randomness)
        df['price_entropy'] = self._calculate_entropy(close, 20)
        
        # Volume at price levels
        df['volume_weighted_close'] = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
        df['vwap_deviation'] = (close - df['volume_weighted_close']) / df['volume_weighted_close']
        
        return df
    
    # Helper functions
    def _calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_hurst(self, series: pd.Series, window: int) -> pd.Series:
        """Simplified Hurst exponent calculation"""
        def hurst_calc(x):
            if len(x) < 20:
                return 0.5
            lags = range(2, min(20, len(x) // 2))
            tau = [np.sqrt(np.std(np.subtract(x[lag:], x[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        
        return series.rolling(window).apply(hurst_calc, raw=False)
    
    def _calculate_entropy(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate price entropy"""
        def entropy_calc(x):
            if len(x) < 5:
                return 0
            hist, _ = np.histogram(x, bins=10)
            hist = hist / hist.sum()
            hist = hist[hist > 0]
            return -np.sum(hist * np.log(hist))
        
        return series.rolling(window).apply(entropy_calc, raw=False)


if __name__ == "__main__":
    # Test feature generation
    print("Testing Feature Engine...")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=1000, freq='15min')
    np.random.seed(42)
    
    sample_df = pd.DataFrame({
        'time': dates,
        'open': 100 + np.random.randn(1000).cumsum() * 0.5,
        'high': 101 + np.random.randn(1000).cumsum() * 0.5,
        'low': 99 + np.random.randn(1000).cumsum() * 0.5,
        'close': 100 + np.random.randn(1000).cumsum() * 0.5,
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    engine = FeatureEngine(verbose=True)
    features = engine.generate_all_features(sample_df)
    
    print(f"\n✅ Feature shape: {features.shape}")
    print(f"✅ Feature columns: {len([c for c in features.columns if c not in ['time', 'open', 'high', 'low', 'close', 'volume']])}")
