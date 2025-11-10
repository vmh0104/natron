"""
FeatureEngine: Generate ~100 technical features from OHLCV data
"""
import numpy as np
import pandas as pd
from typing import Optional


class FeatureEngine:
    """Generate comprehensive technical features for financial time series"""
    
    def __init__(self):
        self.feature_names = []
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ~100 technical features from OHLCV data
        
        Args:
            df: DataFrame with columns [time, open, high, low, close, volume]
            
        Returns:
            DataFrame with original columns + 100 feature columns
        """
        features_df = df.copy()
        
        # Ensure numeric columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in features_df.columns:
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
        
        # Moving Averages (13 features)
        features_df = self._add_moving_averages(features_df)
        
        # Momentum (13 features)
        features_df = self._add_momentum(features_df)
        
        # Volatility (15 features)
        features_df = self._add_volatility(features_df)
        
        # Volume (9 features)
        features_df = self._add_volume_features(features_df)
        
        # Price Patterns (8 features)
        features_df = self._add_price_patterns(features_df)
        
        # Returns (8 features)
        features_df = self._add_returns(features_df)
        
        # Trend Strength (6 features)
        features_df = self._add_trend_strength(features_df)
        
        # Statistical (6 features)
        features_df = self._add_statistical(features_df)
        
        # Support/Resistance (4 features)
        features_df = self._add_support_resistance(features_df)
        
        # SMC - Smart Money Concepts (6 features)
        features_df = self._add_smc_features(features_df)
        
        # Market Profile (10 features)
        features_df = self._add_market_profile(features_df)
        
        # Fill NaN values
        features_df = features_df.bfill().fillna(0)
        
        return features_df
    
    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Moving Average features (13)"""
        close = df['close']
        
        # Simple MAs
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'MA{period}'] = close.rolling(period).mean()
            df[f'MA{period}_slope'] = df[f'MA{period}'].diff()
        
        # EMA
        df['EMA12'] = close.ewm(span=12, adjust=False).mean()
        df['EMA26'] = close.ewm(span=26, adjust=False).mean()
        
        # Price to MA ratios
        df['price_to_MA20'] = close / df['MA20']
        df['price_to_MA50'] = close / df['MA50']
        
        # MA crossovers
        df['MA5_MA20_cross'] = (df['MA5'] > df['MA20']).astype(int)
        df['MA20_MA50_cross'] = (df['MA20'] > df['MA50']).astype(int)
        
        return df
    
    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum indicators (13)"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI_signal'] = (df['RSI'] > 50).astype(int)
        
        # ROC
        df['ROC'] = close.pct_change(periods=10) * 100
        
        # CCI
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        df['CCI'] = (tp - sma_tp) / (0.015 * mad)
        
        # Stochastic
        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        df['Stoch_K'] = 100 * ((close - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
        
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        df['MACD_hist_diff'] = df['MACD_hist'].diff()
        
        # Momentum
        df['Momentum'] = close.pct_change(periods=10)
        
        # Williams %R
        df['Williams_R'] = -100 * ((high_14 - close) / (high_14 - low_14))
        
        return df
    
    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility features (15)"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        df['ATR_pct'] = (df['ATR'] / close) * 100
        
        # Bollinger Bands
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df['BB_upper'] = ma20 + (std20 * 2)
        df['BB_lower'] = ma20 - (std20 * 2)
        df['BB_mid'] = ma20
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        df['BB_position'] = (close - df['BB_lower']) / (df['BB_width'] + 1e-8)
        
        # Keltner Channels
        kc_middle = close.rolling(20).mean()
        kc_range = (high - low).rolling(20).mean()
        df['KC_upper'] = kc_middle + (kc_range * 1.5)
        df['KC_lower'] = kc_middle - (kc_range * 1.5)
        df['KC_position'] = (close - df['KC_lower']) / (df['KC_upper'] - df['KC_lower'] + 1e-8)
        
        # Standard Deviation
        df['StdDev_20'] = close.rolling(20).std()
        df['StdDev_50'] = close.rolling(50).std()
        
        # Volatility ratio
        df['Volatility_ratio'] = df['StdDev_20'] / (df['StdDev_50'] + 1e-8)
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume indicators (9)"""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # OBV
        df['OBV'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        
        # VWAP (simplified daily)
        typical_price = (high + low + close) / 3
        df['VWAP'] = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
        
        # Volume MA
        df['Volume_MA20'] = volume.rolling(20).mean()
        df['Volume_MA50'] = volume.rolling(50).mean()
        df['Volume_ratio'] = volume / (df['Volume_MA20'] + 1e-8)
        
        # MFI (Money Flow Index)
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(14).sum()
        mfi_ratio = positive_flow / (negative_flow + 1e-8)
        df['MFI'] = 100 - (100 / (1 + mfi_ratio))
        
        # Volume price trend
        df['VPT'] = (close.pct_change() * volume).fillna(0).cumsum()
        
        # Volume spike
        df['Volume_spike'] = (volume > df['Volume_MA20'] * 1.5).astype(int)
        
        return df
    
    def _add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price pattern recognition (8)"""
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Body and shadow calculations
        body = abs(close - open_price)
        upper_shadow = high - np.maximum(close, open_price)
        lower_shadow = np.minimum(close, open_price) - low
        total_range = high - low
        
        # Doji pattern
        df['Doji'] = (body < (total_range * 0.1)).astype(int)
        
        # Body percentage
        df['Body_pct'] = body / (total_range + 1e-8)
        
        # Upper shadow percentage
        df['Upper_shadow_pct'] = upper_shadow / (total_range + 1e-8)
        
        # Lower shadow percentage
        df['Lower_shadow_pct'] = lower_shadow / (total_range + 1e-8)
        
        # Price position in range
        df['Price_position'] = (close - low) / (total_range + 1e-8)
        
        # Gap detection
        df['Gap_up'] = (low > high.shift(1)).astype(int)
        df['Gap_down'] = (high < low.shift(1)).astype(int)
        
        # Hammer pattern (simplified)
        df['Hammer'] = ((lower_shadow > body * 2) & (upper_shadow < body * 0.1)).astype(int)
        
        return df
    
    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return features (8)"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Log returns
        df['Log_return_1'] = np.log(close / close.shift(1))
        df['Log_return_5'] = np.log(close / close.shift(5))
        df['Log_return_20'] = np.log(close / close.shift(20))
        
        # Simple returns
        df['Return_1'] = close.pct_change(1)
        df['Return_5'] = close.pct_change(5)
        
        # Intraday return
        df['Intraday_return'] = (close - df['open']) / df['open']
        
        # Cumulative return (rolling)
        df['Cumulative_return_20'] = (close / close.shift(20)) - 1
        
        # High-low range
        df['HL_range_pct'] = ((high - low) / close) * 100
        
        return df
    
    def _add_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trend strength indicators (6)"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # ADX calculation
        tr = self._calculate_tr(df)
        atr = tr.rolling(14).mean()
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        df['ADX'] = dx.rolling(14).mean()
        df['Plus_DI'] = plus_di
        df['Minus_DI'] = minus_di
        
        # Aroon
        period = 14
        aroon_up = high.rolling(period + 1).apply(lambda x: (period - x.argmax()) / period * 100)
        aroon_down = low.rolling(period + 1).apply(lambda x: (period - x.argmin()) / period * 100)
        df['Aroon_Up'] = aroon_up
        df['Aroon_Down'] = aroon_down
        df['Aroon_Oscillator'] = aroon_up - aroon_down
        
        return df
    
    def _calculate_tr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr
    
    def _add_statistical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Statistical features (6)"""
        close = df['close']
        
        # Rolling statistics
        window = 20
        rolling_mean = close.rolling(window).mean()
        rolling_std = close.rolling(window).std()
        
        # Z-score
        df['Z_score'] = (close - rolling_mean) / (rolling_std + 1e-8)
        
        # Skewness
        df['Skewness'] = close.rolling(window).skew()
        
        # Kurtosis
        df['Kurtosis'] = close.rolling(window).kurt()
        
        # Hurst exponent (simplified)
        def hurst(ts):
            lags = range(2, 20)
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        
        df['Hurst'] = close.rolling(100).apply(hurst, raw=True)
        
        # Price deviation
        df['Price_deviation'] = (close - rolling_mean) / rolling_mean
        
        # Coefficient of variation
        df['CV'] = rolling_std / (rolling_mean + 1e-8)
        
        return df
    
    def _add_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Support/Resistance levels (4)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Distance to recent highs/lows
        high_20 = high.rolling(20).max()
        high_50 = high.rolling(50).max()
        low_20 = low.rolling(20).min()
        low_50 = low.rolling(50).min()
        
        df['Dist_to_high_20'] = (high_20 - close) / close
        df['Dist_to_high_50'] = (high_50 - close) / close
        df['Dist_to_low_20'] = (close - low_20) / close
        df['Dist_to_low_50'] = (close - low_50) / close
        
        return df
    
    def _add_smc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Smart Money Concepts features (6)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Swing High/Low
        window = 5
        df['Swing_High'] = (high == high.rolling(window * 2 + 1, center=True).max()).astype(int)
        df['Swing_Low'] = (low == low.rolling(window * 2 + 1, center=True).min()).astype(int)
        
        # Break of Structure (BOS) - price breaks previous swing high
        swing_highs = high.where(df['Swing_High'] == 1)
        last_swing_high = swing_highs.rolling(20, min_periods=1).max()
        df['BOS'] = (close > last_swing_high.shift(1)).astype(int)
        
        # Change of Character (CHOCH) - price breaks previous swing low
        swing_lows = low.where(df['Swing_Low'] == 1)
        last_swing_low = swing_lows.rolling(20, min_periods=1).min()
        df['CHOCH'] = (close < last_swing_low.shift(1)).astype(int)
        
        # Order blocks (simplified - last candle before strong move)
        strong_move = abs(close.pct_change()) > 0.01
        df['Order_Block_Bullish'] = (strong_move & (close > df['open'])).astype(int)
        df['Order_Block_Bearish'] = (strong_move & (close < df['open'])).astype(int)
        
        return df
    
    def _add_market_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market Profile features (10)"""
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        # Price levels (simplified market profile)
        price_range = high - low
        num_levels = 10
        
        # VAH/VAL/POC approximation
        typical_price = (high + low + close) / 3
        volume_weighted_price = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
        
        df['POC'] = volume_weighted_price  # Point of Control
        
        # Value Area High/Low (simplified)
        rolling_high = high.rolling(20).max()
        rolling_low = low.rolling(20).min()
        df['VAH'] = rolling_high * 0.7 + rolling_low * 0.3
        df['VAL'] = rolling_high * 0.3 + rolling_low * 0.7
        
        # Distance to POC
        df['Dist_to_POC'] = (close - df['POC']) / df['POC']
        
        # Volume profile features
        df['Volume_profile_high'] = volume.where(close > df['POC'], 0).rolling(20).sum()
        df['Volume_profile_low'] = volume.where(close < df['POC'], 0).rolling(20).sum()
        
        # Entropy (price distribution measure)
        price_bins = pd.cut(close, bins=num_levels, labels=False)
        df['Price_entropy'] = price_bins.rolling(20).apply(
            lambda x: -sum((pd.Series(x).value_counts() / len(x)) * 
                          np.log2(pd.Series(x).value_counts() / len(x) + 1e-8))
        )
        
        # Market profile position
        df['MP_position'] = (close - df['VAL']) / (df['VAH'] - df['VAL'] + 1e-8)
        
        # Single prints (price levels with low volume)
        volume_ma = volume.rolling(20).mean()
        df['Single_print'] = (volume < volume_ma * 0.5).astype(int)
        
        # Balance/imbalance
        df['Balance'] = (abs(close - df['POC']) < price_range * 0.1).astype(int)
        df['Imbalance'] = (abs(close - df['POC']) > price_range * 0.3).astype(int)
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """Get list of feature column names (excluding original OHLCV)"""
        original_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df.columns if col not in original_cols]
        return feature_cols
