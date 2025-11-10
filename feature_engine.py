"""
FeatureEngine - Automated Technical Feature Generation
Generates ~100 engineered features from OHLCV data for Natron Transformer.
"""

import numpy as np
import pandas as pd
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


class FeatureEngine:
    """
    Generates comprehensive technical features for financial time series.
    Output: ~100 features across multiple categories.
    """
    
    def __init__(self, lookback_periods: list = [5, 10, 20, 50, 100, 200]):
        """
        Args:
            lookback_periods: List of periods for moving averages and indicators
        """
        self.lookback_periods = lookback_periods
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features from OHLCV DataFrame.
        
        Args:
            df: DataFrame with columns ['time', 'open', 'high', 'low', 'close', 'volume']
            
        Returns:
            DataFrame with original columns + ~100 feature columns
        """
        features_df = df.copy()
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in features_df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Generate feature groups
        features_df = self._add_moving_averages(features_df)
        features_df = self._add_momentum_indicators(features_df)
        features_df = self._add_volatility_indicators(features_df)
        features_df = self._add_volume_indicators(features_df)
        features_df = self._add_price_patterns(features_df)
        features_df = self._add_returns(features_df)
        features_df = self._add_trend_strength(features_df)
        features_df = self._add_statistical_features(features_df)
        features_df = self._add_support_resistance(features_df)
        features_df = self._add_smc_features(features_df)
        features_df = self._add_market_profile(features_df)
        
        # Fill NaN values
        features_df = features_df.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        return features_df
    
    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Moving Average Features (13 features)"""
        close = df['close']
        
        # Simple Moving Averages
        for period in [5, 10, 20, 50]:
            df[f'MA{period}'] = close.rolling(period).mean()
            df[f'price_to_MA{period}'] = close / df[f'MA{period}'] - 1
        
        # Exponential Moving Averages
        for period in [10, 20, 50]:
            df[f'EMA{period}'] = close.ewm(span=period, adjust=False).mean()
        
        # MA Slopes
        df['MA20_slope'] = df['MA20'].diff(5) / df['MA20']
        df['MA50_slope'] = df['MA50'].diff(10) / df['MA50']
        
        # MA Crossovers
        df['MA5_MA20_cross'] = (df['MA5'] > df['MA20']).astype(int)
        df['MA20_MA50_cross'] = (df['MA20'] > df['MA50']).astype(int)
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum Indicators (13 features)"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # RSI
        for period in [14, 21]:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'RSI{period}'] = 100 - (100 / (1 + rs))
        
        # ROC (Rate of Change)
        for period in [10, 20]:
            df[f'ROC{period}'] = close.pct_change(period) * 100
        
        # CCI (Commodity Channel Index)
        period = 20
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(period).mean()
        mad = typical_price.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        df['CCI'] = (typical_price - sma_tp) / (0.015 * mad)
        
        # Stochastic Oscillator
        period = 14
        lowest_low = low.rolling(period).min()
        highest_high = high.rolling(period).max()
        df['Stoch_K'] = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
        
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        df['MACD_hist_change'] = df['MACD_hist'].diff()
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility Indicators (15 features)"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # ATR (Average True Range)
        for period in [14, 21]:
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df[f'ATR{period}'] = tr.rolling(period).mean()
            df[f'ATR{period}_pct'] = df[f'ATR{period}'] / close
        
        # Bollinger Bands
        for period in [20, 50]:
            ma = close.rolling(period).mean()
            std = close.rolling(period).std()
            df[f'BB_upper{period}'] = ma + (2 * std)
            df[f'BB_lower{period}'] = ma - (2 * std)
            df[f'BB_midband{period}'] = ma
            df[f'BB_width{period}'] = (df[f'BB_upper{period}'] - df[f'BB_lower{period}']) / ma
            df[f'BB_position{period}'] = (close - df[f'BB_lower{period}']) / (df[f'BB_upper{period}'] - df[f'BB_lower{period}'])
        
        # Keltner Channels
        period = 20
        ema = close.ewm(span=period, adjust=False).mean()
        atr = df['ATR14']
        df['KC_upper'] = ema + (1.5 * atr)
        df['KC_lower'] = ema - (1.5 * atr)
        df['KC_position'] = (close - df['KC_lower']) / (df['KC_upper'] - df['KC_lower'])
        
        # Standard Deviation
        for period in [20, 50]:
            df[f'StdDev{period}'] = close.rolling(period).std()
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume Indicators (9 features)"""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # OBV (On-Balance Volume)
        df['OBV'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        df['OBV_MA'] = df['OBV'].rolling(20).mean()
        
        # VWAP (Volume Weighted Average Price)
        typical_price = (high + low + close) / 3
        df['VWAP'] = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
        df['price_to_VWAP'] = close / df['VWAP'] - 1
        
        # MFI (Money Flow Index)
        period = 14
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(period).sum()
        mfi_ratio = positive_flow / negative_flow
        df['MFI'] = 100 - (100 / (1 + mfi_ratio))
        
        # Volume Ratios
        df['volume_MA20'] = volume.rolling(20).mean()
        df['volume_ratio'] = volume / df['volume_MA20']
        df['volume_trend'] = volume.rolling(5).mean() / volume.rolling(20).mean()
        
        return df
    
    def _add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price Pattern Features (8 features)"""
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Body and Shadow Calculations
        body = abs(close - open_price)
        upper_shadow = high - np.maximum(open_price, close)
        lower_shadow = np.minimum(open_price, close) - low
        total_range = high - low
        
        df['body_pct'] = body / (total_range + 1e-10)
        df['upper_shadow_pct'] = upper_shadow / (total_range + 1e-10)
        df['lower_shadow_pct'] = lower_shadow / (total_range + 1e-10)
        
        # Doji Pattern (body < 20% of range)
        df['is_doji'] = (df['body_pct'] < 0.2).astype(int)
        
        # Gaps
        df['gap_up'] = ((low > close.shift(1))).astype(int)
        df['gap_down'] = ((high < close.shift(1))).astype(int)
        
        # Position within range
        df['position_in_range'] = (close - low) / (total_range + 1e-10)
        
        return df
    
    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return Features (8 features)"""
        close = df['close']
        open_price = df['open']
        high = df['high']
        low = df['low']
        
        # Log Returns
        for period in [1, 5, 10, 20]:
            df[f'log_return{period}'] = np.log(close / close.shift(period))
        
        # Intraday Return
        df['intraday_return'] = (close - open_price) / open_price
        
        # Cumulative Returns
        df['cumulative_return_5'] = close.pct_change(5).fillna(0).cumsum()
        df['cumulative_return_20'] = close.pct_change(20).fillna(0).cumsum()
        
        return df
    
    def _add_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trend Strength Indicators (6 features)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # ADX (Average Directional Index)
        period = 14
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = self._true_range(df)
        atr = tr.rolling(period).mean()
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['ADX'] = dx.rolling(period).mean()
        df['+DI'] = plus_di
        df['-DI'] = minus_di
        
        # Aroon Indicator
        period = 25
        aroon_up = high.rolling(period + 1).apply(lambda x: (period - x.argmax()) / period * 100, raw=True)
        aroon_down = low.rolling(period + 1).apply(lambda x: (period - x.argmin()) / period * 100, raw=True)
        df['Aroon_Up'] = aroon_up
        df['Aroon_Down'] = aroon_down
        df['Aroon_Oscillator'] = aroon_up - aroon_down
        
        return df
    
    def _true_range(self, df: pd.DataFrame) -> pd.Series:
        """Calculate True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Statistical Features (6 features)"""
        close = df['close']
        
        # Rolling Statistics
        for period in [20, 50]:
            rolling = close.rolling(period)
            df[f'skewness{period}'] = rolling.skew()
            df[f'kurtosis{period}'] = rolling.kurt()
            mean = rolling.mean()
            std = rolling.std()
            df[f'zscore{period}'] = (close - mean) / (std + 1e-10)
        
        # Hurst Exponent (simplified)
        period = 50
        returns = close.pct_change().dropna()
        if len(returns) >= period:
            lags = range(2, min(20, period // 2))
            tau = [np.std(np.subtract(returns[lag:], returns[:-lag])) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            df['Hurst'] = poly[0] * 2
        else:
            df['Hurst'] = 0.5
        
        return df
    
    def _add_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Support/Resistance Features (4 features)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Distance to High/Low
        for period in [20, 50]:
            highest = high.rolling(period).max()
            lowest = low.rolling(period).min()
            df[f'dist_to_high{period}'] = (highest - close) / close
            df[f'dist_to_low{period}'] = (close - lowest) / close
        
        return df
    
    def _add_smc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Smart Money Concepts Features (6 features)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Swing High/Low
        window = 5
        df['swing_high'] = (high == high.rolling(window * 2 + 1, center=True).max()).astype(int)
        df['swing_low'] = (low == low.rolling(window * 2 + 1, center=True).min()).astype(int)
        
        # Break of Structure (BOS) - price breaks previous swing high
        swing_highs = high.where(df['swing_high'] == 1)
        last_swing_high = swing_highs.ffill()
        df['BOS'] = (close > last_swing_high.shift(1)).astype(int)
        
        # Change of Character (CHOCH) - price breaks previous swing low
        swing_lows = low.where(df['swing_low'] == 1)
        last_swing_low = swing_lows.ffill()
        df['CHOCH'] = (close < last_swing_low.shift(1)).astype(int)
        
        # Order Blocks (simplified - large bullish/bearish candles)
        body = abs(close - df['open'])
        body_ma = body.rolling(20).mean()
        df['bullish_order_block'] = ((close > df['open']) & (body > 1.5 * body_ma)).astype(int)
        df['bearish_order_block'] = ((close < df['open']) & (body > 1.5 * body_ma)).astype(int)
        
        return df
    
    def _add_market_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market Profile Features (10 features)"""
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        # Price-Volume Distribution
        period = 20
        typical_price = (high + low + close) / 3
        
        # POC (Point of Control) - price level with highest volume
        bins = 20
        price_range = high.rolling(period).max() - low.rolling(period).min()
        price_min = low.rolling(period).min()
        
        def calc_poc(row_idx):
            if row_idx < period:
                return typical_price.iloc[row_idx]
            window_high = high.iloc[row_idx-period+1:row_idx+1]
            window_low = low.iloc[row_idx-period+1:row_idx+1]
            window_vol = volume.iloc[row_idx-period+1:row_idx+1]
            window_tp = typical_price.iloc[row_idx-period+1:row_idx+1]
            
            if len(window_high) == 0:
                return typical_price.iloc[row_idx]
            
            price_bins = np.linspace(window_low.min(), window_high.max(), bins)
            bin_indices = np.digitize(window_tp, price_bins)
            volume_per_bin = pd.Series(window_vol.values, index=bin_indices).groupby(bin_indices).sum()
            poc_bin = volume_per_bin.idxmax()
            return price_bins[poc_bin] if poc_bin < len(price_bins) else typical_price.iloc[row_idx]
        
        df['POC'] = pd.Series([calc_poc(i) for i in range(len(df))], index=df.index)
        df['price_to_POC'] = close / (df['POC'] + 1e-10) - 1
        
        # VAH/VAL (Value Area High/Low) - simplified
        df['VAH'] = high.rolling(period).quantile(0.75)
        df['VAL'] = low.rolling(period).quantile(0.25)
        df['price_to_VAH'] = close / (df['VAH'] + 1e-10) - 1
        df['price_to_VAL'] = close / (df['VAL'] + 1e-10) - 1
        
        # Volume Profile Entropy
        def calc_entropy(row_idx):
            if row_idx < period:
                return 0.5
            window_vol = volume.iloc[row_idx-period+1:row_idx+1]
            if window_vol.sum() == 0:
                return 0.5
            probs = window_vol / window_vol.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            return entropy / np.log(len(probs))  # Normalized
        
        df['volume_entropy'] = pd.Series([calc_entropy(i) for i in range(len(df))], index=df.index)
        
        # Additional Market Profile metrics
        df['TPO_count'] = (typical_price > df['POC']).rolling(period).sum()
        df['volume_weighted_price'] = (typical_price * volume).rolling(period).sum() / volume.rolling(period).sum()
        df['price_to_VWP'] = close / (df['volume_weighted_price'] + 1e-10) - 1
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """Get list of feature column names (excluding original OHLCV)"""
        original_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df.columns if col not in original_cols]
        return feature_cols
