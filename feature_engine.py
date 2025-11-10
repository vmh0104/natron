"""
FeatureEngine - Generates ~100 Technical Features for Financial Trading
Part of Natron Transformer Multi-Task Trading System
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from typing import Optional


class FeatureEngine:
    """
    Generates comprehensive technical features from OHLCV data.
    Output: ~100 engineered features grouped by category.
    """
    
    def __init__(self):
        self.feature_names = []
        
    def generate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point: generates all ~100 features.
        
        Args:
            df: DataFrame with columns ['time', 'open', 'high', 'low', 'close', 'volume']
            
        Returns:
            DataFrame with 100 feature columns
        """
        features_dict = {}
        
        # Group 1: Moving Averages (13 features)
        ma_features = self._moving_average_features(df)
        features_dict.update(ma_features)
        
        # Group 2: Momentum (13 features)
        momentum_features = self._momentum_features(df)
        features_dict.update(momentum_features)
        
        # Group 3: Volatility (15 features)
        volatility_features = self._volatility_features(df)
        features_dict.update(volatility_features)
        
        # Group 4: Volume (9 features)
        volume_features = self._volume_features(df)
        features_dict.update(volume_features)
        
        # Group 5: Price Pattern (8 features)
        pattern_features = self._price_pattern_features(df)
        features_dict.update(pattern_features)
        
        # Group 6: Returns (8 features)
        returns_features = self._returns_features(df)
        features_dict.update(returns_features)
        
        # Group 7: Trend Strength (6 features)
        trend_features = self._trend_strength_features(df)
        features_dict.update(trend_features)
        
        # Group 8: Statistical (6 features)
        statistical_features = self._statistical_features(df)
        features_dict.update(statistical_features)
        
        # Group 9: Support/Resistance (4 features)
        sr_features = self._support_resistance_features(df)
        features_dict.update(sr_features)
        
        # Group 10: SMC - Smart Money Concepts (6 features)
        smc_features = self._smc_features(df)
        features_dict.update(smc_features)
        
        # Group 11: Market Profile (10 features)
        profile_features = self._market_profile_features(df)
        features_dict.update(profile_features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_dict, index=df.index)
        
        # Fill NaN values (forward fill then backward fill)
        features_df = features_df.ffill().bfill().fillna(0)
        
        self.feature_names = list(features_df.columns)
        return features_df
    
    def _moving_average_features(self, df: pd.DataFrame) -> dict:
        """Group 1: Moving Average Features (13 features)"""
        features = {}
        close = df['close']
        
        # Simple Moving Averages
        for period in [5, 10, 20, 50, 100]:
            ma = close.rolling(period).mean()
            features[f'MA{period}'] = ma
            features[f'price_to_MA{period}'] = close / ma - 1
        
        # Exponential Moving Averages
        for period in [12, 26]:
            ema = close.ewm(span=period, adjust=False).mean()
            features[f'EMA{period}'] = ema
        
        # MA Crossovers
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        features['MA20_above_MA50'] = (ma20 > ma50).astype(float)
        features['MA20_slope'] = ma20.diff(5) / ma20.shift(5)
        features['MA50_slope'] = ma50.diff(10) / ma50.shift(10)
        
        return features
    
    def _momentum_features(self, df: pd.DataFrame) -> dict:
        """Group 2: Momentum Indicators (13 features)"""
        features = {}
        close = df['close']
        high = df['high']
        low = df['low']
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        features['RSI'] = rsi / 100  # Normalize to [0, 1]
        features['RSI_oversold'] = (rsi < 30).astype(float)
        features['RSI_overbought'] = (rsi > 70).astype(float)
        
        # ROC (Rate of Change)
        for period in [5, 10, 20]:
            roc = close.pct_change(period)
            features[f'ROC{period}'] = roc
        
        # CCI (Commodity Channel Index)
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (tp - sma_tp) / (0.015 * mad)
        features['CCI'] = cci / 100  # Normalize
        
        # Stochastic
        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        stoch_k = 100 * ((close - low_14) / (high_14 - low_14))
        features['Stoch_K'] = stoch_k / 100
        features['Stoch_D'] = stoch_k.rolling(3).mean() / 100
        
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal_line
        features['MACD'] = macd_line / close  # Normalize
        features['MACD_signal'] = signal_line / close
        features['MACD_hist'] = macd_hist / close
        
        return features
    
    def _volatility_features(self, df: pd.DataFrame) -> dict:
        """Group 3: Volatility Indicators (15 features)"""
        features = {}
        close = df['close']
        high = df['high']
        low = df['low']
        
        # ATR (Average True Range)
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        for period in [14, 20]:
            atr = tr.rolling(period).mean()
            features[f'ATR{period}'] = atr / close  # Normalize
        
        # Bollinger Bands
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_upper = ma20 + 2 * std20
        bb_lower = ma20 - 2 * std20
        features['BB_upper'] = bb_upper / close
        features['BB_mid'] = ma20 / close
        features['BB_lower'] = bb_lower / close
        features['BB_width'] = (bb_upper - bb_lower) / close
        features['BB_position'] = (close - bb_lower) / (bb_upper - bb_lower)
        
        # Keltner Channels
        kc_middle = close.ewm(span=20, adjust=False).mean()
        kc_range = tr.rolling(20).mean()
        kc_upper = kc_middle + 1.5 * kc_range
        kc_lower = kc_middle - 1.5 * kc_range
        features['KC_upper'] = kc_upper / close
        features['KC_lower'] = kc_lower / close
        features['KC_width'] = (kc_upper - kc_lower) / close
        
        # Standard Deviation
        for period in [10, 20]:
            std = close.rolling(period).std()
            features[f'StdDev{period}'] = std / close
        
        return features
    
    def _volume_features(self, df: pd.DataFrame) -> dict:
        """Group 4: Volume Indicators (9 features)"""
        features = {}
        close = df['close']
        volume = df['volume']
        
        # OBV (On-Balance Volume)
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        features['OBV'] = obv / (obv.abs().rolling(20).mean() + 1e-8)
        
        # VWAP (Volume Weighted Average Price)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        features['VWAP'] = vwap / close
        features['price_to_VWAP'] = close / vwap - 1
        
        # MFI (Money Flow Index)
        money_flow = typical_price * volume
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(14).sum()
        mfi = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-8)))
        features['MFI'] = mfi / 100
        
        # Volume ratios
        for period in [10, 20]:
            vol_ma = volume.rolling(period).mean()
            features[f'volume_ratio{period}'] = volume / (vol_ma + 1e-8)
        
        # Volume trend
        features['volume_trend'] = volume.rolling(5).mean().diff(3) / (volume.rolling(5).mean().shift(3) + 1e-8)
        
        return features
    
    def _price_pattern_features(self, df: pd.DataFrame) -> dict:
        """Group 5: Price Pattern Recognition (8 features)"""
        features = {}
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Body and shadows
        body = abs(close - open_price)
        upper_shadow = high - df[['open', 'close']].max(axis=1)
        lower_shadow = df[['open', 'close']].min(axis=1) - low
        total_range = high - low
        
        features['body_pct'] = body / (total_range + 1e-8)
        features['upper_shadow_pct'] = upper_shadow / (total_range + 1e-8)
        features['lower_shadow_pct'] = lower_shadow / (total_range + 1e-8)
        
        # Doji pattern (body < 20% of range)
        features['is_doji'] = (features['body_pct'] < 0.2).astype(float)
        
        # Gap detection
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        gap_up = (low > prev_high).astype(float)
        gap_down = (high < prev_low).astype(float)
        features['gap_up'] = gap_up
        features['gap_down'] = gap_down
        
        # Price position in range
        features['price_position'] = (close - low) / (total_range + 1e-8)
        
        return features
    
    def _returns_features(self, df: pd.DataFrame) -> dict:
        """Group 6: Return Features (8 features)"""
        features = {}
        close = df['close']
        open_price = df['open']
        high = df['high']
        low = df['low']
        
        # Log returns
        for period in [1, 5, 10, 20]:
            log_ret = np.log(close / close.shift(period))
            features[f'log_return{period}'] = log_ret
        
        # Intraday return
        features['intraday_return'] = (close - open_price) / open_price
        
        # High-low return
        features['hl_return'] = (high - low) / low
        
        # Cumulative return (rolling 20)
        features['cumulative_return20'] = close.pct_change(20)
        
        return features
    
    def _trend_strength_features(self, df: pd.DataFrame) -> dict:
        """Group 7: Trend Strength Indicators (6 features)"""
        features = {}
        close = df['close']
        high = df['high']
        low = df['low']
        
        # ADX (Average Directional Index)
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = self._true_range(df)
        atr14 = tr.rolling(14).mean()
        
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        adx = dx.rolling(14).mean()
        
        features['ADX'] = adx / 100
        features['plus_DI'] = plus_di / 100
        features['minus_DI'] = minus_di / 100
        
        # Aroon
        period = 14
        aroon_up = high.rolling(period + 1).apply(lambda x: (period - x.argmax()) / period * 100)
        aroon_down = low.rolling(period + 1).apply(lambda x: (period - x.argmin()) / period * 100)
        features['Aroon_Up'] = aroon_up / 100
        features['Aroon_Down'] = aroon_down / 100
        
        return features
    
    def _true_range(self, df: pd.DataFrame) -> pd.Series:
        """Helper: Calculate True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    def _statistical_features(self, df: pd.DataFrame) -> dict:
        """Group 8: Statistical Features (6 features)"""
        features = {}
        close = df['close']
        
        # Rolling statistics
        window = 20
        rolling_mean = close.rolling(window).mean()
        rolling_std = close.rolling(window).std()
        
        # Z-score
        features['z_score'] = (close - rolling_mean) / (rolling_std + 1e-8)
        
        # Skewness
        features['skewness'] = close.rolling(window).skew()
        
        # Kurtosis
        features['kurtosis'] = close.rolling(window).kurt()
        
        # Hurst exponent (simplified)
        def hurst_approx(series):
            if len(series) < 10:
                return 0.5
            lags = range(2, min(10, len(series)))
            tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        
        features['hurst'] = close.rolling(30).apply(hurst_approx, raw=False)
        
        # Price momentum (rate of change)
        features['momentum'] = close.pct_change(10)
        
        return features
    
    def _support_resistance_features(self, df: pd.DataFrame) -> dict:
        """Group 9: Support/Resistance Levels (4 features)"""
        features = {}
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Distance to recent highs/lows
        for period in [20, 50]:
            period_high = high.rolling(period).max()
            period_low = low.rolling(period).min()
            features[f'dist_to_high{period}'] = (period_high - close) / close
            features[f'dist_to_low{period}'] = (close - period_low) / close
        
        return features
    
    def _smc_features(self, df: pd.DataFrame) -> dict:
        """Group 10: Smart Money Concepts (6 features)"""
        features = {}
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Swing High/Low detection
        window = 5
        swing_high = high.rolling(window * 2 + 1, center=True).max() == high
        swing_low = low.rolling(window * 2 + 1, center=True).min() == low
        
        features['is_swing_high'] = swing_high.astype(float)
        features['is_swing_low'] = swing_low.astype(float)
        
        # Break of Structure (BOS) - price breaks previous swing high/low
        swing_high_values = high.where(swing_high)
        swing_low_values = low.where(swing_low)
        
        last_swing_high = swing_high_values.ffill()
        last_swing_low = swing_low_values.ffill()
        
        bos_up = (close > last_swing_high.shift(1)).astype(float)
        bos_down = (close < last_swing_low.shift(1)).astype(float)
        features['BOS_up'] = bos_up
        features['BOS_down'] = bos_down
        
        # Change of Character (CHOCH) - trend change signal
        # Simplified: when price crosses above/below recent swing points
        features['CHOCH'] = (bos_up - bos_down)
        
        return features
    
    def _market_profile_features(self, df: pd.DataFrame) -> dict:
        """Group 11: Market Profile Features (10 features)"""
        features = {}
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        # Price-Volume Distribution (simplified)
        window = 20
        typical_price = (high + low + close) / 3
        
        # POC (Point of Control) - price level with highest volume
        # Simplified: use VWAP as proxy
        vwap = (typical_price * volume).rolling(window).sum() / volume.rolling(window).sum()
        features['POC'] = vwap / close
        
        # Value Area High/Low (VAH/VAL) - 70% of volume range
        # Simplified approximation
        price_range = high.rolling(window).max() - low.rolling(window).min()
        features['VAH'] = (high.rolling(window).max() - price_range * 0.15) / close
        features['VAL'] = (low.rolling(window).min() + price_range * 0.15) / close
        
        # Market profile width
        features['profile_width'] = price_range / close
        
        # Entropy (price distribution measure)
        def price_entropy(series):
            if len(series) < 5:
                return 0.5
            hist, _ = np.histogram(series, bins=10)
            hist = hist + 1e-8  # Avoid log(0)
            prob = hist / hist.sum()
            entropy = -np.sum(prob * np.log2(prob))
            return entropy / np.log2(len(hist))  # Normalize
        
        features['entropy'] = close.rolling(20).apply(price_entropy, raw=False)
        
        # Volume profile features
        vol_high = volume.rolling(window).max()
        vol_low = volume.rolling(window).min()
        features['volume_profile_range'] = (vol_high - vol_low) / (vol_high + 1e-8)
        
        # Price-Volume correlation
        features['price_volume_corr'] = close.rolling(window).corr(volume)
        
        # Additional profile metrics
        features['profile_balance'] = (close - typical_price.rolling(window).mean()) / close
        features['profile_trend'] = typical_price.rolling(5).mean().diff(3) / typical_price.shift(3)
        
        return features
