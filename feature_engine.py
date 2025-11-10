"""
Natron Feature Engine
Generates ~100 technical features from OHLCV data for Transformer input.
"""

import numpy as np
import pandas as pd
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


class FeatureEngine:
    """
    Generates comprehensive technical features for financial time series.
    Output: ~100 features across 10 categories.
    """
    
    def __init__(self):
        self.feature_names = []
    
    def generate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all feature groups from OHLCV DataFrame.
        
        Args:
            df: DataFrame with columns ['time', 'open', 'high', 'low', 'close', 'volume']
        
        Returns:
            DataFrame with ~100 feature columns
        """
        features_df = df.copy()
        
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
        
        # Group 10: SMC (6 features)
        features_df = self._add_smc_features(features_df)
        
        # Group 11: Market Profile (10 features)
        features_df = self._add_market_profile(features_df)
        
        # Select only feature columns (exclude original OHLCV)
        feature_cols = [c for c in features_df.columns 
                       if c not in ['time', 'open', 'high', 'low', 'close', 'volume']]
        
        return features_df[feature_cols]
    
    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group 1: Moving Averages (13 features)"""
        close = df['close']
        
        # Simple MAs
        df['ma5'] = close.rolling(5).mean()
        df['ma10'] = close.rolling(10).mean()
        df['ma20'] = close.rolling(20).mean()
        df['ma50'] = close.rolling(50).mean()
        
        # EMAs
        df['ema12'] = close.ewm(span=12, adjust=False).mean()
        df['ema26'] = close.ewm(span=26, adjust=False).mean()
        df['ema50'] = close.ewm(span=50, adjust=False).mean()
        
        # MA slopes
        df['ma20_slope'] = df['ma20'].diff(5) / df['ma20']
        df['ma50_slope'] = df['ma50'].diff(10) / df['ma50']
        
        # Crossovers
        df['ma5_ma20_cross'] = (df['ma5'] > df['ma20']).astype(int)
        df['ma20_ma50_cross'] = (df['ma20'] > df['ma50']).astype(int)
        
        # Price to MA ratios
        df['price_ma20_ratio'] = close / df['ma20']
        df['price_ma50_ratio'] = close / df['ma50']
        
        return df
    
    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group 2: Momentum (13 features)"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ROC
        df['roc5'] = close.pct_change(5) * 100
        df['roc10'] = close.pct_change(10) * 100
        
        # CCI
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        df['cci'] = (tp - sma_tp) / (0.015 * mad + 1e-10)
        
        # Stochastic
        lowest_low = low.rolling(14).min()
        highest_high = high.rolling(14).max()
        df['stoch_k'] = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_hist_change'] = df['macd_hist'].diff()
        
        # Williams %R
        df['williams_r'] = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
        
        # Momentum
        df['momentum10'] = close.diff(10)
        
        return df
    
    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group 3: Volatility (15 features)"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr14'] = tr.rolling(14).mean()
        df['atr20'] = tr.rolling(20).mean()
        
        # Bollinger Bands
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df['bb_upper'] = ma20 + 2 * std20
        df['bb_mid'] = ma20
        df['bb_lower'] = ma20 - 2 * std20
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # Keltner Channels
        ema20 = close.ewm(span=20, adjust=False).mean()
        df['kc_upper'] = ema20 + 1.5 * df['atr20']
        df['kc_lower'] = ema20 - 1.5 * df['atr20']
        df['kc_width'] = df['kc_upper'] - df['kc_lower']
        
        # Standard Deviation
        df['std10'] = close.rolling(10).std()
        df['std20'] = close.rolling(20).std()
        
        # True Range percentiles
        df['atr_pct'] = df['atr14'] / close
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group 4: Volume (9 features)"""
        close = df['close']
        volume = df['volume']
        
        # OBV
        df['obv'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        
        # VWAP (simplified - using rolling window)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap20'] = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
        
        # MFI
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical_price * volume
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(), 0).rolling(14).sum()
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(), 0).rolling(14).sum()
        mfi_ratio = positive_flow / (negative_flow + 1e-10)
        df['mfi'] = 100 - (100 / (1 + mfi_ratio))
        
        # Volume ratios
        df['volume_ma20'] = volume.rolling(20).mean()
        df['volume_ratio'] = volume / (df['volume_ma20'] + 1e-10)
        df['volume_ma50'] = volume.rolling(50).mean()
        df['volume_ratio50'] = volume / (df['volume_ma50'] + 1e-10)
        
        # Volume change
        df['volume_change'] = volume.pct_change()
        
        return df
    
    def _add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group 5: Price Patterns (8 features)"""
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Body and shadows
        body = abs(close - open_price)
        upper_shadow = high - np.maximum(open_price, close)
        lower_shadow = np.minimum(open_price, close) - low
        total_range = high - low
        
        df['body_pct'] = body / (total_range + 1e-10)
        df['upper_shadow_pct'] = upper_shadow / (total_range + 1e-10)
        df['lower_shadow_pct'] = lower_shadow / (total_range + 1e-10)
        
        # Doji pattern (body < 20% of range)
        df['is_doji'] = (df['body_pct'] < 0.2).astype(int)
        
        # Gap detection
        df['gap_up'] = ((low > close.shift())).astype(int)
        df['gap_down'] = ((high < close.shift())).astype(int)
        
        # Position in range
        df['position_in_range'] = (close - low) / (total_range + 1e-10)
        
        return df
    
    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group 6: Returns (8 features)"""
        close = df['close']
        
        # Log returns
        df['log_return1'] = np.log(close / close.shift(1))
        df['log_return5'] = np.log(close / close.shift(5))
        df['log_return10'] = np.log(close / close.shift(10))
        
        # Simple returns
        df['return1'] = close.pct_change(1)
        df['return5'] = close.pct_change(5)
        
        # Intraday return
        df['intraday_return'] = (df['close'] - df['open']) / df['open']
        
        # Cumulative return (rolling)
        df['cumulative_return20'] = close.pct_change(20)
        
        return df
    
    def _add_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group 7: Trend Strength (6 features)"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # ADX calculation
        tr = self._calculate_tr(df)
        atr14 = tr.rolling(14).mean()
        
        plus_dm = high.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm = -low.diff()
        minus_dm[minus_dm < 0] = 0
        
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.rolling(14).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        # Aroon
        period = 14
        aroon_up = high.rolling(period + 1).apply(lambda x: (period - x.argmax()) / period * 100, raw=True)
        aroon_down = low.rolling(period + 1).apply(lambda x: (period - x.argmin()) / period * 100, raw=True)
        df['aroon_up'] = aroon_up
        df['aroon_down'] = aroon_down
        
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
        """Group 8: Statistical (6 features)"""
        close = df['close']
        
        # Rolling statistics
        rolling20 = close.rolling(20)
        df['skewness20'] = rolling20.apply(lambda x: x.skew() if len(x) == 20 else np.nan)
        df['kurtosis20'] = rolling20.apply(lambda x: x.kurtosis() if len(x) == 20 else np.nan)
        
        # Z-score
        mean20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df['zscore20'] = (close - mean20) / (std20 + 1e-10)
        
        # Hurst exponent (simplified)
        returns = close.pct_change().dropna()
        if len(returns) > 20:
            lags = [2, 5, 10]
            hurst_vals = []
            for lag in lags:
                if len(returns) >= lag * 2:
                    var_lag = returns.rolling(lag).var().mean()
                    var_1 = returns.rolling(1).var().mean()
                    if var_1 > 0:
                        hurst_vals.append(0.5 * np.log(var_lag / var_1) / np.log(lag))
            df['hurst'] = np.nan
            if hurst_vals:
                df.loc[df.index[-1], 'hurst'] = np.mean(hurst_vals) if hurst_vals else np.nan
        
        # Price position in distribution
        df['price_percentile20'] = close.rolling(20).apply(lambda x: (x.iloc[-1] > x).sum() / len(x))
        
        return df
    
    def _add_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group 9: Support/Resistance (4 features)"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Distance to recent highs/lows
        high20 = high.rolling(20).max()
        low20 = low.rolling(20).min()
        high50 = high.rolling(50).max()
        low50 = low.rolling(50).min()
        
        df['dist_to_high20'] = (high20 - close) / close
        df['dist_to_low20'] = (close - low20) / close
        df['dist_to_high50'] = (high50 - close) / close
        df['dist_to_low50'] = (close - low50) / close
        
        return df
    
    def _add_smc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group 10: Smart Money Concepts (6 features)"""
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
        df['bos'] = (close > last_swing_high.shift(1)).astype(int)
        
        # Change of Character (CHOCH) - price breaks previous swing low
        swing_lows = low.where(df['swing_low'] == 1)
        last_swing_low = swing_lows.ffill()
        df['choch'] = (close < last_swing_low.shift(1)).astype(int)
        
        # Order blocks (simplified - strong moves)
        strong_up = (close > close.shift(3)) & (df['volume_ratio'] > 1.5)
        strong_down = (close < close.shift(3)) & (df['volume_ratio'] > 1.5)
        df['order_block_up'] = strong_up.astype(int)
        df['order_block_down'] = strong_down.astype(int)
        
        return df
    
    def _add_market_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group 11: Market Profile (10 features)"""
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        # Simplified Market Profile
        typical_price = (high + low + close) / 3
        
        # POC (Point of Control) - price level with highest volume
        bins = 20
        price_range = high.max() - low.min()
        bin_size = price_range / bins
        
        # Volume profile (simplified)
        df['poc_distance'] = np.nan  # Distance to POC (simplified)
        df['vah'] = typical_price.rolling(20).quantile(0.8)  # Value Area High
        df['val'] = typical_price.rolling(20).quantile(0.2)  # Value Area Low
        df['vah_distance'] = (close - df['vah']) / close
        df['val_distance'] = (close - df['val']) / close
        
        # Entropy (price distribution)
        returns = close.pct_change().dropna()
        if len(returns) > 20:
            hist, _ = np.histogram(returns[-20:], bins=10)
            hist = hist[hist > 0]
            if len(hist) > 0:
                probs = hist / hist.sum()
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                df.loc[df.index[-1], 'entropy'] = entropy
        
        # Market profile position
        df['profile_position'] = (close - df['val']) / (df['vah'] - df['val'] + 1e-10)
        
        # Additional profile features
        df['profile_width'] = df['vah'] - df['val']
        df['profile_center'] = (df['vah'] + df['val']) / 2
        df['profile_deviation'] = (close - df['profile_center']) / close
        
        return df
