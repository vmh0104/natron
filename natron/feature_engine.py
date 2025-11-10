"""
Natron Feature Engine - Generates ~100 technical features from OHLCV data
"""
import numpy as np
import pandas as pd
from typing import Optional


class FeatureEngine:
    """Generates comprehensive technical features for financial time series."""
    
    def __init__(self):
        self.feature_names = []
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ~100 technical features from OHLCV data.
        
        Args:
            df: DataFrame with columns [time, open, high, low, close, volume]
            
        Returns:
            DataFrame with original columns + 100 feature columns
        """
        features_df = df.copy()
        
        # Ensure numeric columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
        
        # Fill NaN values
        features_df = features_df.fillna(method='ffill').fillna(method='bfill')
        
        # Group 1: Moving Averages (13 features)
        features_df = self._add_moving_averages(features_df)
        
        # Group 2: Momentum (13 features)
        features_df = self._add_momentum(features_df)
        
        # Group 3: Volatility (15 features)
        features_df = self._add_volatility(features_df)
        
        # Group 4: Volume (9 features)
        features_df = self._add_volume_features(features_df)
        
        # Group 5: Price Patterns (8 features)
        features_df = self._add_price_patterns(features_df)
        
        # Group 6: Returns (8 features)
        features_df = self._add_returns(features_df)
        
        # Group 7: Trend Strength (6 features)
        features_df = self._add_trend_strength(features_df)
        
        # Group 8: Statistical (6 features)
        features_df = self._add_statistical(features_df)
        
        # Group 9: Support/Resistance (4 features)
        features_df = self._add_support_resistance(features_df)
        
        # Group 10: SMC - Smart Money Concepts (6 features)
        features_df = self._add_smc_features(features_df)
        
        # Group 11: Market Profile (10 features)
        features_df = self._add_market_profile(features_df)
        
        # Fill remaining NaN
        features_df = features_df.fillna(0)
        
        return features_df
    
    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 13 moving average features."""
        # Simple MAs
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'MA{period}'] = df['close'].rolling(period).mean()
        
        # EMAs
        df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # MA slopes
        df['MA20_slope'] = df['MA20'].diff(5) / df['MA20'].shift(5)
        df['MA50_slope'] = df['MA50'].diff(5) / df['MA50'].shift(5)
        
        # Crossovers
        df['MA5_MA20_cross'] = (df['MA5'] > df['MA20']).astype(int)
        df['MA20_MA50_cross'] = (df['MA20'] > df['MA50']).astype(int)
        
        # Price to MA ratio
        df['price_to_MA20'] = df['close'] / df['MA20']
        
        return df
    
    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 13 momentum indicators."""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI_oversold'] = (df['RSI'] < 30).astype(int)
        df['RSI_overbought'] = (df['RSI'] > 70).astype(int)
        
        # ROC (Rate of Change)
        df['ROC_10'] = df['close'].pct_change(10) * 100
        df['ROC_20'] = df['close'].pct_change(20) * 100
        
        # CCI (Commodity Channel Index)
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        df['CCI'] = (tp - sma_tp) / (0.015 * mad)
        
        # Stochastic
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['Stoch_K'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
        
        # MACD
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        df['MACD_hist_change'] = df['MACD_hist'].diff()
        
        # Momentum
        df['Momentum_10'] = df['close'] / df['close'].shift(10) - 1
        
        return df
    
    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 15 volatility features."""
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR_14'] = tr.rolling(14).mean()
        df['ATR_20'] = tr.rolling(20).mean()
        
        # Bollinger Bands
        df['BB_mid'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['BB_upper'] = df['BB_mid'] + (bb_std * 2)
        df['BB_lower'] = df['BB_mid'] - (bb_std * 2)
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        df['BB_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Keltner Channels
        kc_mid = df['close'].ewm(span=20).mean()
        kc_range = df['ATR_20'] * 1.5
        df['KC_upper'] = kc_mid + kc_range
        df['KC_lower'] = kc_mid - kc_range
        
        # Standard Deviation
        df['StdDev_20'] = df['close'].rolling(20).std()
        df['StdDev_50'] = df['close'].rolling(50).std()
        
        # Volatility ratios
        df['volatility_ratio'] = df['StdDev_20'] / df['StdDev_50']
        
        # True Range percentiles
        df['ATR_pct_rank'] = df['ATR_14'].rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0
        )
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 9 volume features."""
        # OBV (On-Balance Volume)
        df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # VWAP (Volume Weighted Average Price)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['VWAP'] = (typical_price * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        
        # MFI (Money Flow Index)
        money_flow = typical_price * df['volume']
        positive_flow = money_flow.where(df['close'] > df['close'].shift(), 0).rolling(14).sum()
        negative_flow = money_flow.where(df['close'] < df['close'].shift(), 0).rolling(14).sum()
        mfi_ratio = positive_flow / negative_flow
        df['MFI'] = 100 - (100 / (1 + mfi_ratio))
        
        # Volume ratios
        df['volume_MA20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_MA20']
        df['volume_spike'] = (df['volume'] > 1.5 * df['volume_MA20']).astype(int)
        
        # Price-Volume correlation
        df['price_volume_corr'] = df['close'].rolling(20).corr(df['volume'])
        
        # Volume trend
        df['volume_trend'] = df['volume'].rolling(10).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
        
        return df
    
    def _add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 8 price pattern features."""
        # Candle body and shadows
        body = np.abs(df['close'] - df['open'])
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        total_range = df['high'] - df['low']
        
        df['body_pct'] = body / (total_range + 1e-8)
        df['upper_shadow_pct'] = upper_shadow / (total_range + 1e-8)
        df['lower_shadow_pct'] = lower_shadow / (total_range + 1e-8)
        
        # Doji pattern (small body)
        df['is_doji'] = (df['body_pct'] < 0.1).astype(int)
        
        # Gaps
        df['gap_up'] = (df['low'] > df['high'].shift(1)).astype(int)
        df['gap_down'] = (df['high'] < df['low'].shift(1)).astype(int)
        
        # Position within range
        df['position_in_range'] = (df['close'] - df['low']) / (total_range + 1e-8)
        
        # Hammer pattern (long lower shadow, small body)
        df['is_hammer'] = ((df['lower_shadow_pct'] > 0.6) & (df['body_pct'] < 0.3)).astype(int)
        
        return df
    
    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 8 return features."""
        # Simple returns
        df['return_1'] = df['close'].pct_change(1)
        df['return_5'] = df['close'].pct_change(5)
        df['return_10'] = df['close'].pct_change(10)
        df['return_20'] = df['close'].pct_change(20)
        
        # Log returns
        df['log_return_1'] = np.log(df['close'] / df['close'].shift(1))
        df['log_return_5'] = np.log(df['close'] / df['close'].shift(5))
        
        # Intraday return
        df['intraday_return'] = (df['close'] - df['open']) / df['open']
        
        # Cumulative return
        df['cumulative_return'] = (1 + df['return_1']).cumprod() - 1
        
        return df
    
    def _add_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 6 trend strength features (ADX, DI+, DI-, Aroon)."""
        # ADX calculation
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        tr = self._calculate_tr(df)
        atr_14 = tr.rolling(14).mean()
        
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        df['ADX'] = dx.rolling(14).mean()
        df['DI_plus'] = plus_di
        df['DI_minus'] = minus_di
        
        # Aroon
        period = 14
        aroon_up = df['high'].rolling(period + 1).apply(
            lambda x: (period - x.argmax()) / period * 100
        )
        aroon_down = df['low'].rolling(period + 1).apply(
            lambda x: (period - x.argmin()) / period * 100
        )
        df['Aroon_Up'] = aroon_up
        df['Aroon_Down'] = aroon_down
        
        return df
    
    def _calculate_tr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    def _add_statistical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 6 statistical features."""
        window = 20
        
        # Skewness
        df['skewness_20'] = df['close'].rolling(window).skew()
        
        # Kurtosis
        df['kurtosis_20'] = df['close'].rolling(window).kurtosis()
        
        # Z-score
        mean = df['close'].rolling(window).mean()
        std = df['close'].rolling(window).std()
        df['z_score'] = (df['close'] - mean) / (std + 1e-8)
        
        # Hurst exponent (simplified)
        returns = df['close'].pct_change().dropna()
        lags = range(2, 20)
        hurst_vals = []
        for i in range(len(df)):
            if i < window:
                hurst_vals.append(0)
            else:
                window_returns = returns.iloc[max(0, i-window):i]
                if len(window_returns) < 10:
                    hurst_vals.append(0)
                else:
                    # Simplified Hurst calculation
                    lag_vars = []
                    for lag in lags[:5]:
                        if len(window_returns) > lag:
                            lag_vars.append(np.var(window_returns.diff(lag).dropna()))
                    if len(lag_vars) > 1:
                        hurst = 0.5 * (1 + np.log(lag_vars[-1] / lag_vars[0]) / np.log(len(lag_vars)))
                        hurst_vals.append(hurst)
                    else:
                        hurst_vals.append(0.5)
        df['hurst'] = pd.Series(hurst_vals, index=df.index)
        
        # Price position percentile
        df['price_percentile_50'] = df['close'].rolling(50).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )
        
        return df
    
    def _add_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 4 support/resistance features."""
        # Distance to recent high/low
        df['dist_to_high_20'] = (df['high'].rolling(20).max() - df['close']) / df['close']
        df['dist_to_low_20'] = (df['close'] - df['low'].rolling(20).min()) / df['close']
        df['dist_to_high_50'] = (df['high'].rolling(50).max() - df['close']) / df['close']
        df['dist_to_low_50'] = (df['close'] - df['low'].rolling(50).min()) / df['close']
        
        return df
    
    def _add_smc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 6 Smart Money Concepts features."""
        # Swing High/Low
        window = 5
        df['swing_high'] = (df['high'] == df['high'].rolling(window*2+1, center=True).max()).astype(int)
        df['swing_low'] = (df['low'] == df['low'].rolling(window*2+1, center=True).min()).astype(int)
        
        # Break of Structure (BOS) - price breaks previous swing high
        swing_highs = df[df['swing_high'] == 1]['high']
        if len(swing_highs) > 0:
            last_swing_high = swing_highs.iloc[-1] if len(swing_highs) > 0 else df['high'].iloc[0]
            df['BOS'] = (df['close'] > last_swing_high).astype(int)
        else:
            df['BOS'] = 0
        
        # Change of Character (CHOCH) - trend change signal
        df['CHOCH'] = ((df['swing_low'] == 1) & (df['close'] > df['close'].shift(5))).astype(int)
        
        # Order blocks (simplified - last significant candle before move)
        df['order_block_bullish'] = ((df['close'] > df['open']) & 
                                     (df['close'].shift(-5) > df['close'] * 1.01)).astype(int)
        df['order_block_bearish'] = ((df['close'] < df['open']) & 
                                     (df['close'].shift(-5) < df['close'] * 0.99)).astype(int)
        
        return df
    
    def _add_market_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 10 market profile features."""
        window = 20
        
        # POC (Point of Control) - price level with most volume
        # Simplified: use VWAP as proxy
        df['POC'] = df['VWAP']
        
        # VAH/VAL (Value Area High/Low) - simplified
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['VAH'] = typical_price.rolling(window).quantile(0.8)
        df['VAL'] = typical_price.rolling(window).quantile(0.2)
        
        # Entropy (price distribution measure)
        price_bins = pd.cut(df['close'], bins=10, labels=False)
        df['entropy'] = price_bins.rolling(window).apply(
            lambda x: -sum((pd.Series(x).value_counts(normalize=True) * 
                           np.log2(pd.Series(x).value_counts(normalize=True) + 1e-8)).values)
        )
        
        # Market profile position
        df['above_POC'] = (df['close'] > df['POC']).astype(int)
        df['in_value_area'] = ((df['close'] >= df['VAL']) & (df['close'] <= df['VAH'])).astype(int)
        
        # Volume profile features
        df['volume_at_price'] = df.groupby(pd.cut(df['close'], bins=20))['volume'].transform('mean')
        df['volume_profile_skew'] = df['volume_at_price'].rolling(window).skew()
        
        # Time-based features
        df['time_of_day'] = pd.to_datetime(df['time']).dt.hour if 'time' in df.columns else 0
        df['day_of_week'] = pd.to_datetime(df['time']).dt.dayofweek if 'time' in df.columns else 0
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """Get list of feature column names (excluding OHLCV)."""
        base_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        return [col for col in df.columns if col not in base_cols]
