"""
Natron FeatureEngine - Generate ~100 Technical Features from OHLCV Data

Feature Groups:
- Moving Averages (13): MA, EMA, slope, crossovers, price-to-MA ratio
- Momentum (13): RSI, ROC, CCI, Stochastic, MACD
- Volatility (15): ATR, Bollinger Bands, Keltner, StdDev
- Volume (9): OBV, VWAP, MFI, Volume ratio
- Price Pattern (8): Doji, Gaps, Shadows, Body%, Position
- Returns (8): Log return, intraday, cumulative
- Trend Strength (6): ADX, +DI, -DI, Aroon
- Statistical (6): Skewness, Kurtosis, Z-score, Hurst exponent
- Support/Resistance (4): Distance to High/Low 20-50
- SMC (6): Swing High/Low, BOS/CHOCH
- Market Profile (10): POC, VAH, VAL, Entropy
"""

import numpy as np
import pandas as pd
from typing import Optional
from scipy import stats
from scipy.stats import linregress


class FeatureEngine:
    """Generate comprehensive technical features from OHLCV data."""
    
    def __init__(self):
        self.feature_names = []
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features from OHLCV DataFrame.
        
        Args:
            df: DataFrame with columns ['time', 'open', 'high', 'low', 'close', 'volume']
        
        Returns:
            DataFrame with ~100 engineered features
        """
        df = df.copy()
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        assert all(col in df.columns for col in required_cols), \
            f"Missing required columns. Found: {df.columns.tolist()}"
        
        features_list = []
        
        # 1. Moving Averages (13 features)
        ma_features = self._moving_averages(df)
        features_list.append(ma_features)
        
        # 2. Momentum (13 features)
        momentum_features = self._momentum(df)
        features_list.append(momentum_features)
        
        # 3. Volatility (15 features)
        volatility_features = self._volatility(df)
        features_list.append(volatility_features)
        
        # 4. Volume (9 features)
        volume_features = self._volume(df)
        features_list.append(volume_features)
        
        # 5. Price Pattern (8 features)
        pattern_features = self._price_patterns(df)
        features_list.append(pattern_features)
        
        # 6. Returns (8 features)
        returns_features = self._returns(df)
        features_list.append(returns_features)
        
        # 7. Trend Strength (6 features)
        trend_features = self._trend_strength(df)
        features_list.append(trend_features)
        
        # 8. Statistical (6 features)
        stat_features = self._statistical(df)
        features_list.append(stat_features)
        
        # 9. Support/Resistance (4 features)
        sr_features = self._support_resistance(df)
        features_list.append(sr_features)
        
        # 10. SMC - Smart Money Concepts (6 features)
        smc_features = self._smc_features(df)
        features_list.append(smc_features)
        
        # 11. Market Profile (10 features)
        profile_features = self._market_profile(df)
        features_list.append(profile_features)
        
        # Combine all features
        features_df = pd.concat(features_list, axis=1)
        
        # Fill NaN values (forward fill, then backward fill)
        features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        self.feature_names = features_df.columns.tolist()
        
        return features_df
    
    def _moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate 13 moving average features."""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        
        # Simple Moving Averages
        features['ma5'] = close.rolling(5).mean()
        features['ma10'] = close.rolling(10).mean()
        features['ma20'] = close.rolling(20).mean()
        features['ma50'] = close.rolling(50).mean()
        
        # Exponential Moving Averages
        features['ema12'] = close.ewm(span=12, adjust=False).mean()
        features['ema26'] = close.ewm(span=26, adjust=False).mean()
        features['ema50'] = close.ewm(span=50, adjust=False).mean()
        
        # MA Slopes
        features['ma20_slope'] = features['ma20'].diff(5) / features['ma20']
        features['ma50_slope'] = features['ma50'].diff(5) / features['ma50']
        
        # Price to MA ratios
        features['close_to_ma20'] = close / features['ma20'] - 1
        features['close_to_ma50'] = close / features['ma50'] - 1
        
        # MA Crossovers
        features['ma5_ma20_cross'] = (features['ma5'] > features['ma20']).astype(int)
        features['ma20_ma50_cross'] = (features['ma20'] > features['ma50']).astype(int)
        
        return features
    
    def _momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate 13 momentum features."""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # ROC (Rate of Change)
        features['roc5'] = close.pct_change(5) * 100
        features['roc10'] = close.pct_change(10) * 100
        
        # CCI (Commodity Channel Index)
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        features['cci'] = (tp - sma_tp) / (0.015 * mad)
        
        # Stochastic Oscillator
        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        features['stoch_k'] = 100 * (close - low_14) / (high_14 - low_14)
        features['stoch_d'] = features['stoch_k'].rolling(3).mean()
        
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # Williams %R
        features['williams_r'] = -100 * (high_14 - close) / (high_14 - low_14)
        
        # Momentum
        features['momentum'] = close.diff(10)
        
        return features
    
    def _volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate 15 volatility features."""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']
        
        # ATR (Average True Range)
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        features['atr'] = tr.rolling(14).mean()
        features['atr_pct'] = features['atr'] / close
        
        # Bollinger Bands
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        features['bb_upper'] = ma20 + (std20 * 2)
        features['bb_lower'] = ma20 - (std20 * 2)
        features['bb_mid'] = ma20
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_mid']
        features['bb_position'] = (close - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # Keltner Channels
        ema20 = close.ewm(span=20, adjust=False).mean()
        features['kc_upper'] = ema20 + (features['atr'] * 1.5)
        features['kc_lower'] = ema20 - (features['atr'] * 1.5)
        features['kc_width'] = (features['kc_upper'] - features['kc_lower']) / ema20
        
        # Standard Deviation
        features['std10'] = close.rolling(10).std()
        features['std20'] = close.rolling(20).std()
        features['std50'] = close.rolling(50).std()
        
        return features
    
    def _volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate 9 volume features."""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # OBV (On-Balance Volume)
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        features['obv'] = obv
        features['obv_ema'] = obv.ewm(span=20, adjust=False).mean()
        
        # VWAP (Volume Weighted Average Price)
        typical_price = (high + low + close) / 3
        features['vwap'] = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
        features['close_to_vwap'] = close / features['vwap'] - 1
        
        # MFI (Money Flow Index)
        money_flow = typical_price * volume
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(14).sum()
        mfi_ratio = positive_flow / negative_flow
        features['mfi'] = 100 - (100 / (1 + mfi_ratio))
        
        # Volume ratios
        features['volume_ma20'] = volume.rolling(20).mean()
        features['volume_ratio'] = volume / features['volume_ma20']
        features['volume_trend'] = volume.rolling(5).mean() / volume.rolling(20).mean()
        
        return features
    
    def _price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate 8 price pattern features."""
        features = pd.DataFrame(index=df.index)
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Body and Shadow calculations
        body = abs(close - open_price)
        upper_shadow = high - np.maximum(close, open_price)
        lower_shadow = np.minimum(close, open_price) - low
        total_range = high - low
        
        # Doji pattern (body < 10% of range)
        features['doji'] = (body / (total_range + 1e-8) < 0.1).astype(int)
        
        # Body percentage
        features['body_pct'] = body / (total_range + 1e-8)
        
        # Shadow ratios
        features['upper_shadow_ratio'] = upper_shadow / (total_range + 1e-8)
        features['lower_shadow_ratio'] = lower_shadow / (total_range + 1e-8)
        
        # Price position in range
        features['price_position'] = (close - low) / (total_range + 1e-8)
        
        # Gaps
        features['gap_up'] = (open_price > close.shift()).astype(int)
        features['gap_down'] = (open_price < close.shift()).astype(int)
        features['gap_size'] = (open_price - close.shift()) / close.shift()
        
        return features
    
    def _returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate 8 return features."""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        open_price = df['open']
        high = df['high']
        low = df['low']
        
        # Log returns
        features['log_return_1'] = np.log(close / close.shift(1))
        features['log_return_5'] = np.log(close / close.shift(5))
        features['log_return_20'] = np.log(close / close.shift(20))
        
        # Simple returns
        features['return_1'] = close.pct_change(1)
        features['return_5'] = close.pct_change(5)
        
        # Intraday return
        features['intraday_return'] = (close - open_price) / open_price
        
        # Cumulative return (rolling)
        features['cumulative_return_20'] = (close / close.shift(20)) - 1
        
        # High-Low range return
        features['hl_return'] = (high - low) / low
        
        return features
    
    def _trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate 6 trend strength features."""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']
        
        # ADX calculation
        tr = self._true_range(df)
        atr = tr.rolling(14).mean()
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        features['adx'] = dx.rolling(14).mean()
        features['plus_di'] = plus_di
        features['minus_di'] = minus_di
        
        # Aroon Indicator
        aroon_period = 14
        aroon_up = high.rolling(aroon_period + 1).apply(
            lambda x: (aroon_period - x.argmax()) / aroon_period * 100, raw=True
        )
        aroon_down = low.rolling(aroon_period + 1).apply(
            lambda x: (aroon_period - x.argmin()) / aroon_period * 100, raw=True
        )
        features['aroon_up'] = aroon_up
        features['aroon_down'] = aroon_down
        
        return features
    
    def _true_range(self, df: pd.DataFrame) -> pd.Series:
        """Calculate True Range."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr
    
    def _statistical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate 6 statistical features."""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        
        # Rolling statistics
        window = 20
        features['skewness'] = close.rolling(window).skew()
        features['kurtosis'] = close.rolling(window).kurtosis()
        
        # Z-score
        rolling_mean = close.rolling(window).mean()
        rolling_std = close.rolling(window).std()
        features['z_score'] = (close - rolling_mean) / (rolling_std + 1e-8)
        
        # Hurst Exponent (simplified)
        def hurst_approx(ts):
            if len(ts) < 10:
                return 0.5
            lags = range(2, min(10, len(ts)))
            tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        
        features['hurst'] = close.rolling(50).apply(hurst_approx, raw=True)
        
        # Price deviation from trend
        def linear_trend_dev(ts):
            if len(ts) < 5:
                return 0
            x = np.arange(len(ts))
            slope, intercept = np.polyfit(x, ts, 1)
            trend = slope * x + intercept
            return np.mean(np.abs(ts - trend)) / (np.mean(ts) + 1e-8)
        
        features['trend_deviation'] = close.rolling(20).apply(linear_trend_dev, raw=True)
        
        return features
    
    def _support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate 4 support/resistance features."""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Distance to recent highs/lows
        features['dist_to_high20'] = (high.rolling(20).max() - close) / close
        features['dist_to_high50'] = (high.rolling(50).max() - close) / close
        features['dist_to_low20'] = (close - low.rolling(20).min()) / close
        features['dist_to_low50'] = (close - low.rolling(50).min()) / close
        
        return features
    
    def _smc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate 6 Smart Money Concepts features."""
        features = pd.DataFrame(index=df.index)
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Swing High/Low
        window = 5
        features['swing_high'] = (high == high.rolling(window * 2 + 1, center=True).max()).astype(int)
        features['swing_low'] = (low == low.rolling(window * 2 + 1, center=True).min()).astype(int)
        
        # Break of Structure (BOS) - price breaks previous swing high
        swing_highs = high.rolling(window * 2 + 1, center=True).max()
        features['bos'] = (close > swing_highs.shift(window)).astype(int)
        
        # Change of Character (CHOCH) - trend change signal
        swing_lows = low.rolling(window * 2 + 1, center=True).min()
        features['choch'] = (close < swing_lows.shift(window)).astype(int)
        
        # Order blocks (simplified - strong moves)
        strong_up = (close > close.shift(3)) & (close.shift(3) > close.shift(6))
        strong_down = (close < close.shift(3)) & (close.shift(3) < close.shift(6))
        features['order_block_up'] = strong_up.astype(int)
        features['order_block_down'] = strong_down.astype(int)
        
        return features
    
    def _market_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate 10 market profile features."""
        features = pd.DataFrame(index=df.index)
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        # Simplified Market Profile (using price ranges)
        window = 20
        
        # POC (Point of Control) - price level with most volume
        # Simplified: use median price as proxy
        typical_price = (high + low + close) / 3
        features['poc'] = typical_price.rolling(window).median()
        
        # VAH/VAL (Value Area High/Low) - simplified
        price_range = high.rolling(window).max() - low.rolling(window).min()
        features['vah'] = high.rolling(window).max() - price_range * 0.3
        features['val'] = low.rolling(window).min() + price_range * 0.3
        
        # Price position relative to value area
        features['above_vah'] = (close > features['vah']).astype(int)
        features['below_val'] = (close < features['val']).astype(int)
        features['in_value_area'] = ((close >= features['val']) & (close <= features['vah'])).astype(int)
        
        # Volume profile features
        features['volume_price_corr'] = volume.rolling(window).corr(close)
        
        # Entropy (price distribution measure)
        def price_entropy(ts):
            if len(ts) < 5:
                return 0
            hist, _ = np.histogram(ts, bins=min(10, len(ts)))
            hist = hist[hist > 0]
            probs = hist / hist.sum()
            return -np.sum(probs * np.log(probs + 1e-8))
        
        features['entropy'] = close.rolling(window).apply(price_entropy, raw=True)
        
        # Time-based features
        features['price_vs_poc'] = (close - features['poc']) / features['poc']
        features['value_area_width'] = (features['vah'] - features['val']) / features['poc']
        
        return features
