"""
Natron Feature Engine - Generates ~100 technical features from OHLCV data
"""
import numpy as np
import pandas as pd
from typing import Optional
from scipy import stats
from scipy.signal import find_peaks


class FeatureEngine:
    """Extracts ~100 technical features from OHLCV data"""
    
    def __init__(self):
        self.feature_names = []
    
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all feature groups from OHLCV DataFrame
        Returns DataFrame with ~100 features
        """
        features_df = pd.DataFrame(index=df.index)
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Extract feature groups
        features_df = pd.concat([
            features_df,
            self._moving_average_features(df),
            self._momentum_features(df),
            self._volatility_features(df),
            self._volume_features(df),
            self._price_pattern_features(df),
            self._returns_features(df),
            self._trend_strength_features(df),
            self._statistical_features(df),
            self._support_resistance_features(df),
            self._smc_features(df),
            self._market_profile_features(df)
        ], axis=1)
        
        # Fill NaN values
        features_df = features_df.ffill().fillna(0)
        
        return features_df
    
    def _moving_average_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """13 Moving Average features"""
        features = pd.DataFrame(index=df.index)
        
        # MAs
        for period in [5, 10, 20, 50, 100, 200]:
            features[f'MA{period}'] = df['close'].rolling(period).mean()
            features[f'EMA{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # MA20 slope
        features['MA20_slope'] = features['MA20'].diff()
        
        # Crossovers
        features['MA5_MA20_cross'] = (features['MA5'] > features['MA20']).astype(int)
        features['MA20_MA50_cross'] = (features['MA20'] > features['MA50']).astype(int)
        
        # Price to MA ratios
        features['price_to_MA20'] = df['close'] / features['MA20']
        features['price_to_MA50'] = df['close'] / features['MA50']
        
        return features
    
    def _momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """13 Momentum features"""
        features = pd.DataFrame(index=df.index)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['RSI'] = 100 - (100 / (1 + rs))
        
        # ROC
        features['ROC_10'] = df['close'].pct_change(10) * 100
        features['ROC_20'] = df['close'].pct_change(20) * 100
        
        # CCI
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        features['CCI'] = (tp - sma_tp) / (0.015 * mad)
        
        # Stochastic
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        features['Stoch_K'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        features['Stoch_D'] = features['Stoch_K'].rolling(3).mean()
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        features['MACD'] = ema12 - ema26
        features['MACD_signal'] = features['MACD'].ewm(span=9, adjust=False).mean()
        features['MACD_hist'] = features['MACD'] - features['MACD_signal']
        
        # Williams %R
        features['Williams_R'] = -100 * ((high_14 - df['close']) / (high_14 - low_14))
        
        # Momentum
        features['Momentum_10'] = df['close'] - df['close'].shift(10)
        
        return features
    
    def _volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """15 Volatility features"""
        features = pd.DataFrame(index=df.index)
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features['ATR_14'] = tr.rolling(14).mean()
        features['ATR_20'] = tr.rolling(20).mean()
        
        # Bollinger Bands
        ma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        features['BB_upper'] = ma20 + (std20 * 2)
        features['BB_mid'] = ma20
        features['BB_lower'] = ma20 - (std20 * 2)
        features['BB_width'] = features['BB_upper'] - features['BB_lower']
        features['BB_position'] = (df['close'] - features['BB_lower']) / features['BB_width']
        
        # Keltner Channels
        kc_middle = df['close'].ewm(span=20, adjust=False).mean()
        kc_range = tr.rolling(20).mean()
        features['KC_upper'] = kc_middle + (kc_range * 1.5)
        features['KC_lower'] = kc_middle - (kc_range * 1.5)
        
        # Standard Deviation
        features['StdDev_20'] = df['close'].rolling(20).std()
        features['StdDev_50'] = df['close'].rolling(50).std()
        
        # Volatility ratio
        features['volatility_ratio'] = features['StdDev_20'] / features['StdDev_50']
        
        return features
    
    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """9 Volume features"""
        features = pd.DataFrame(index=df.index)
        
        # OBV
        features['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # VWAP
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        features['VWAP'] = (typical_price * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        
        # MFI
        money_flow = typical_price * df['volume']
        positive_flow = money_flow.where(df['close'] > df['close'].shift(), 0).rolling(14).sum()
        negative_flow = money_flow.where(df['close'] < df['close'].shift(), 0).rolling(14).sum()
        mfi_ratio = positive_flow / negative_flow
        features['MFI'] = 100 - (100 / (1 + mfi_ratio))
        
        # Volume ratios
        features['volume_MA20'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_MA20']
        features['volume_MA50'] = df['volume'].rolling(50).mean()
        features['volume_ratio_50'] = df['volume'] / features['volume_MA50']
        
        # Volume price trend
        features['VPT'] = (df['close'].pct_change() * df['volume']).fillna(0).cumsum()
        
        return features
    
    def _price_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """8 Price pattern features"""
        features = pd.DataFrame(index=df.index)
        
        # Body and shadows
        body = np.abs(df['close'] - df['open'])
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        total_range = df['high'] - df['low']
        
        features['body_pct'] = body / (total_range + 1e-8)
        features['upper_shadow_pct'] = upper_shadow / (total_range + 1e-8)
        features['lower_shadow_pct'] = lower_shadow / (total_range + 1e-8)
        
        # Doji pattern
        features['is_doji'] = (body / (total_range + 1e-8) < 0.1).astype(int)
        
        # Gaps
        features['gap_up'] = (df['low'] > df['high'].shift(1)).astype(int)
        features['gap_down'] = (df['high'] < df['low'].shift(1)).astype(int)
        
        # Position within range
        features['position_in_range'] = (df['close'] - df['low']) / (total_range + 1e-8)
        
        return features
    
    def _returns_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """8 Returns features"""
        features = pd.DataFrame(index=df.index)
        
        # Log returns
        features['log_return_1'] = np.log(df['close'] / df['close'].shift(1))
        features['log_return_5'] = np.log(df['close'] / df['close'].shift(5))
        features['log_return_20'] = np.log(df['close'] / df['close'].shift(20))
        
        # Intraday return
        features['intraday_return'] = (df['close'] - df['open']) / df['open']
        
        # Cumulative returns
        features['cumulative_return_5'] = df['close'].pct_change(5).fillna(0)
        features['cumulative_return_20'] = df['close'].pct_change(20).fillna(0)
        
        # Return volatility
        features['return_volatility'] = features['log_return_1'].rolling(20).std()
        
        return features
    
    def _trend_strength_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """6 Trend strength features"""
        features = pd.DataFrame(index=df.index)
        
        # ADX calculation
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        tr = self._calculate_tr(df)
        atr = tr.rolling(14).mean()
        
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        features['ADX'] = dx.rolling(14).mean()
        features['plus_DI'] = plus_di
        features['minus_DI'] = minus_di
        
        # Aroon
        aroon_period = 14
        aroon_up = df['high'].rolling(aroon_period).apply(
            lambda x: (aroon_period - x.argmax()) / aroon_period * 100, raw=True
        )
        aroon_down = df['low'].rolling(aroon_period).apply(
            lambda x: (aroon_period - x.argmin()) / aroon_period * 100, raw=True
        )
        features['Aroon_Up'] = aroon_up
        features['Aroon_Down'] = aroon_down
        
        return features
    
    def _calculate_tr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    def _statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """6 Statistical features"""
        features = pd.DataFrame(index=df.index)
        
        window = 20
        returns = df['close'].pct_change()
        
        # Skewness
        features['skewness'] = returns.rolling(window).skew()
        
        # Kurtosis
        features['kurtosis'] = returns.rolling(window).kurtosis()
        
        # Z-score
        mean = df['close'].rolling(window).mean()
        std = df['close'].rolling(window).std()
        features['z_score'] = (df['close'] - mean) / (std + 1e-8)
        
        # Hurst exponent (simplified)
        def hurst(ts):
            lags = range(2, 20)
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        
        features['hurst'] = df['close'].rolling(50).apply(hurst, raw=True)
        
        # Price position
        features['price_position'] = (df['close'] - df['low'].rolling(20).min()) / (
            df['high'].rolling(20).max() - df['low'].rolling(20).min() + 1e-8
        )
        
        return features
    
    def _support_resistance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """4 Support/Resistance features"""
        features = pd.DataFrame(index=df.index)
        
        # Distance to recent highs/lows
        high_20 = df['high'].rolling(20).max()
        low_20 = df['low'].rolling(20).min()
        high_50 = df['high'].rolling(50).max()
        low_50 = df['low'].rolling(50).min()
        
        features['dist_to_high_20'] = (high_20 - df['close']) / df['close']
        features['dist_to_low_20'] = (df['close'] - low_20) / df['close']
        features['dist_to_high_50'] = (high_50 - df['close']) / df['close']
        features['dist_to_low_50'] = (df['close'] - low_50) / df['close']
        
        return features
    
    def _smc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """6 Smart Money Concepts features"""
        features = pd.DataFrame(index=df.index)
        
        # Swing High/Low
        window = 5
        features['swing_high'] = (
            (df['high'] == df['high'].rolling(window * 2 + 1, center=True).max())
        ).astype(int)
        features['swing_low'] = (
            (df['low'] == df['low'].rolling(window * 2 + 1, center=True).min())
        ).astype(int)
        
        # Break of Structure (BOS) - price breaks previous swing high
        swing_highs = df['high'].where(features['swing_high'] == 1)
        last_swing_high = swing_highs.ffill()
        features['BOS'] = (df['close'] > last_swing_high.shift(1)).astype(int)
        
        # Change of Character (CHOCH) - price breaks previous swing low
        swing_lows = df['low'].where(features['swing_low'] == 1)
        last_swing_low = swing_lows.ffill()
        features['CHOCH'] = (df['close'] < last_swing_low.shift(1)).astype(int)
        
        # Order blocks (simplified - large body candles)
        body = np.abs(df['close'] - df['open'])
        large_body = body > body.rolling(20).quantile(0.8)
        features['order_block_bullish'] = (large_body & (df['close'] > df['open'])).astype(int)
        features['order_block_bearish'] = (large_body & (df['close'] < df['open'])).astype(int)
        
        return features
    
    def _market_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """10 Market Profile features"""
        features = pd.DataFrame(index=df.index)
        
        window = 20
        
        # POC (Point of Control) - price level with highest volume
        # Simplified: use typical price as proxy
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        features['POC'] = typical_price.rolling(window).apply(
            lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else x.mean()
        )
        
        # VAH/VAL (Value Area High/Low) - simplified
        features['VAH'] = df['high'].rolling(window).quantile(0.7)
        features['VAL'] = df['low'].rolling(window).quantile(0.3)
        
        # Entropy (price distribution measure)
        price_bins = pd.cut(df['close'], bins=10, labels=False)
        features['entropy'] = price_bins.rolling(window).apply(
            lambda x: stats.entropy(x.value_counts(normalize=True), base=2) if len(x.dropna()) > 0 else 0
        )
        
        # Volume profile features
        features['volume_weighted_price'] = (typical_price * df['volume']).rolling(window).sum() / df['volume'].rolling(window).sum()
        
        # Price distribution
        features['price_distribution_skew'] = df['close'].rolling(window).skew()
        features['price_distribution_kurt'] = df['close'].rolling(window).kurtosis()
        
        # Market structure
        features['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        features['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        
        # Trend consistency
        features['trend_consistency'] = (
            (df['close'] > df['close'].shift(1)).rolling(window).mean()
        )
        
        return features
