"""
Natron V2 Feature Engineering Module
Generates ~100 technical features from OHLCV data
"""

import numpy as np
import pandas as pd
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


class FeatureEngine:
    """
    Generates 100+ technical features for financial market analysis.
    
    Feature Groups:
    - Moving Averages (13 features)
    - Momentum (13 features)
    - Volatility (15 features)
    - Volume (9 features)
    - Price Patterns (8 features)
    - Returns (8 features)
    - Trend Strength (6 features)
    - Statistical (6 features)
    - Support/Resistance (4 features)
    - Smart Money Concepts (6 features)
    - Market Profile (10 features)
    """
    
    def __init__(self):
        self.feature_names = []
        
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all technical features from OHLCV data.
        
        Args:
            df: DataFrame with columns [time, open, high, low, close, volume]
            
        Returns:
            DataFrame with ~100 features
        """
        features = df.copy()
        
        # Group 1: Moving Averages (13 features)
        features = self._add_moving_averages(features)
        
        # Group 2: Momentum Indicators (13 features)
        features = self._add_momentum(features)
        
        # Group 3: Volatility Indicators (15 features)
        features = self._add_volatility(features)
        
        # Group 4: Volume Indicators (9 features)
        features = self._add_volume(features)
        
        # Group 5: Price Patterns (8 features)
        features = self._add_price_patterns(features)
        
        # Group 6: Returns (8 features)
        features = self._add_returns(features)
        
        # Group 7: Trend Strength (6 features)
        features = self._add_trend_strength(features)
        
        # Group 8: Statistical Features (6 features)
        features = self._add_statistical(features)
        
        # Group 9: Support/Resistance (4 features)
        features = self._add_support_resistance(features)
        
        # Group 10: Smart Money Concepts (6 features)
        features = self._add_smc(features)
        
        # Group 11: Market Profile (10 features)
        features = self._add_market_profile(features)
        
        # Forward fill and backward fill NaN values
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Store feature names (excluding OHLCV and time)
        self.feature_names = [col for col in features.columns 
                             if col not in ['time', 'open', 'high', 'low', 'close', 'volume']]
        
        return features
    
    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Moving Average features (13 features)"""
        close = df['close']
        
        # Simple Moving Averages
        df['ma_5'] = close.rolling(5).mean()
        df['ma_10'] = close.rolling(10).mean()
        df['ma_20'] = close.rolling(20).mean()
        df['ma_50'] = close.rolling(50).mean()
        
        # Exponential Moving Averages
        df['ema_9'] = close.ewm(span=9).mean()
        df['ema_21'] = close.ewm(span=21).mean()
        
        # MA slopes
        df['ma_20_slope'] = df['ma_20'].diff(5)
        df['ma_50_slope'] = df['ma_50'].diff(5)
        
        # Price to MA ratios
        df['price_to_ma20'] = close / df['ma_20']
        df['price_to_ma50'] = close / df['ma_50']
        
        # MA crossovers
        df['ma_5_10_cross'] = (df['ma_5'] > df['ma_10']).astype(int)
        df['ma_10_20_cross'] = (df['ma_10'] > df['ma_20']).astype(int)
        df['ma_20_50_cross'] = (df['ma_20'] > df['ma_50']).astype(int)
        
        return df
    
    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum Indicators (13 features)"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # RSI
        df['rsi_14'] = self._calculate_rsi(close, 14)
        df['rsi_7'] = self._calculate_rsi(close, 7)
        
        # Rate of Change
        df['roc_10'] = ((close - close.shift(10)) / close.shift(10)) * 100
        df['roc_20'] = ((close - close.shift(20)) / close.shift(20)) * 100
        
        # CCI (Commodity Channel Index)
        df['cci_20'] = self._calculate_cci(df, 20)
        
        # Stochastic Oscillator
        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df, 14, 3)
        
        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_hist_change'] = df['macd_hist'].diff()
        
        # Momentum
        df['momentum_10'] = close - close.shift(10)
        
        return df
    
    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility Indicators (15 features)"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # ATR (Average True Range)
        df['atr_14'] = self._calculate_atr(df, 14)
        df['atr_20'] = self._calculate_atr(df, 20)
        
        # Bollinger Bands
        df['bb_mid_20'] = close.rolling(20).mean()
        df['bb_std_20'] = close.rolling(20).std()
        df['bb_upper'] = df['bb_mid_20'] + (2 * df['bb_std_20'])
        df['bb_lower'] = df['bb_mid_20'] - (2 * df['bb_std_20'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid_20']
        df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Keltner Channels
        df['kc_mid'] = close.ewm(span=20).mean()
        df['kc_upper'] = df['kc_mid'] + (2 * df['atr_20'])
        df['kc_lower'] = df['kc_mid'] - (2 * df['atr_20'])
        
        # Standard Deviation
        df['std_10'] = close.rolling(10).std()
        df['std_20'] = close.rolling(20).std()
        
        # Historical Volatility
        returns = np.log(close / close.shift(1))
        df['hist_vol_20'] = returns.rolling(20).std() * np.sqrt(252)
        
        return df
    
    def _add_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume Indicators (9 features)"""
        close = df['close']
        volume = df['volume']
        high = df['high']
        low = df['low']
        
        # Volume moving averages
        df['volume_ma_20'] = volume.rolling(20).mean()
        df['volume_ratio'] = volume / df['volume_ma_20']
        
        # OBV (On-Balance Volume)
        df['obv'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        df['obv_ema'] = df['obv'].ewm(span=20).mean()
        
        # VWAP (Volume Weighted Average Price)
        df['vwap'] = (volume * (high + low + close) / 3).cumsum() / volume.cumsum()
        
        # MFI (Money Flow Index)
        df['mfi_14'] = self._calculate_mfi(df, 14)
        
        # Volume-Price Trend
        df['vpt'] = (volume * ((close - close.shift(1)) / close.shift(1))).cumsum()
        
        # Force Index
        df['force_index'] = close.diff() * volume
        df['force_index_ema'] = df['force_index'].ewm(span=13).mean()
        
        return df
    
    def _add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price Pattern Features (8 features)"""
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Candle body and shadows
        df['body_size'] = abs(close - open_price)
        df['upper_shadow'] = high - np.maximum(open_price, close)
        df['lower_shadow'] = np.minimum(open_price, close) - low
        df['body_to_range'] = df['body_size'] / (high - low + 1e-10)
        
        # Doji detection
        df['is_doji'] = (df['body_size'] < (high - low) * 0.1).astype(int)
        
        # Gap detection
        df['gap_up'] = (low > high.shift(1)).astype(int)
        df['gap_down'] = (high < low.shift(1)).astype(int)
        
        # Price position in range
        df['close_position'] = (close - low) / (high - low + 1e-10)
        
        return df
    
    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return Features (8 features)"""
        close = df['close']
        open_price = df['open']
        high = df['high']
        low = df['low']
        
        # Log returns
        df['log_return'] = np.log(close / close.shift(1))
        df['log_return_5'] = np.log(close / close.shift(5))
        df['log_return_10'] = np.log(close / close.shift(10))
        
        # Simple returns
        df['simple_return'] = close.pct_change()
        
        # Intraday returns
        df['intraday_return'] = (close - open_price) / open_price
        df['high_low_range'] = (high - low) / low
        
        # Cumulative returns
        df['cum_return_20'] = (close / close.shift(20)) - 1
        df['cum_return_50'] = (close / close.shift(50)) - 1
        
        return df
    
    def _add_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trend Strength Indicators (6 features)"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # ADX (Average Directional Index)
        df['adx_14'], df['plus_di_14'], df['minus_di_14'] = self._calculate_adx(df, 14)
        
        # Aroon Indicator
        df['aroon_up'], df['aroon_down'] = self._calculate_aroon(df, 25)
        df['aroon_osc'] = df['aroon_up'] - df['aroon_down']
        
        return df
    
    def _add_statistical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Statistical Features (6 features)"""
        close = df['close']
        
        # Skewness and Kurtosis
        df['skewness_20'] = close.rolling(20).skew()
        df['kurtosis_20'] = close.rolling(20).kurt()
        
        # Z-Score
        df['zscore_20'] = (close - close.rolling(20).mean()) / close.rolling(20).std()
        
        # Percentile rank
        df['percentile_rank_20'] = close.rolling(20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        
        # Hurst Exponent (simplified)
        df['hurst_50'] = self._calculate_hurst(close, 50)
        
        # Autocorrelation
        df['autocorr_10'] = close.rolling(20).apply(
            lambda x: pd.Series(x).autocorr(lag=10), raw=False
        )
        
        return df
    
    def _add_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Support/Resistance Features (4 features)"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Distance to recent highs/lows
        df['dist_to_high_20'] = (high.rolling(20).max() - close) / close
        df['dist_to_low_20'] = (close - low.rolling(20).min()) / close
        df['dist_to_high_50'] = (high.rolling(50).max() - close) / close
        df['dist_to_low_50'] = (close - low.rolling(50).min()) / close
        
        return df
    
    def _add_smc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Smart Money Concepts (6 features)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Swing Highs and Lows
        df['swing_high'] = (high > high.shift(1)) & (high > high.shift(-1))
        df['swing_low'] = (low < low.shift(1)) & (low < low.shift(-1))
        
        # Break of Structure (BOS)
        df['bos_bull'] = (close > high.rolling(5).max().shift(1)).astype(int)
        df['bos_bear'] = (close < low.rolling(5).min().shift(1)).astype(int)
        
        # Change of Character (CHOCH)
        df['choch'] = ((df['bos_bull'].diff() > 0) | (df['bos_bear'].diff() > 0)).astype(int)
        
        # Fair Value Gap
        df['fvg'] = np.maximum(0, low.shift(2) - high) + np.maximum(0, low - high.shift(2))
        
        return df
    
    def _add_market_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market Profile Features (10 features)"""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Point of Control (POC) - simplified
        window = 20
        df['poc'] = close.rolling(window).apply(
            lambda x: pd.Series(x).mode()[0] if len(pd.Series(x).mode()) > 0 else x.median(),
            raw=False
        )
        
        # Value Area High/Low (VAH/VAL)
        df['vah'] = close.rolling(window).quantile(0.70)
        df['val'] = close.rolling(window).quantile(0.30)
        
        # Distance from POC
        df['dist_from_poc'] = (close - df['poc']) / close
        
        # Volume-at-Price distribution features
        df['vp_skew'] = close.rolling(window).apply(
            lambda x: pd.Series(x).skew(), raw=False
        )
        
        # Price acceptance
        df['above_vah'] = (close > df['vah']).astype(int)
        df['below_val'] = (close < df['val']).astype(int)
        df['in_value_area'] = ((close >= df['val']) & (close <= df['vah'])).astype(int)
        
        # Volume profile entropy (complexity measure)
        df['vp_entropy'] = close.rolling(window).apply(
            lambda x: -np.sum(np.histogram(x, bins=10)[0] / len(x) * 
                             np.log(np.histogram(x, bins=10)[0] / len(x) + 1e-10)),
            raw=False
        )
        
        # Balance/Imbalance
        df['price_imbalance'] = (df['vah'] - df['val']) / close
        
        return df
    
    # Helper calculation methods
    
    def _calculate_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp - sma) / (0.015 * mad + 1e-10)
        return cci
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, 
                             d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        low_min = df['low'].rolling(k_period).min()
        high_max = df['high'].rolling(k_period).max()
        k = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
        d = k.rolling(d_period).mean()
        return k, d
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr
    
    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        tp = (df['high'] + df['low'] + df['close']) / 3
        mf = tp * df['volume']
        
        mf_pos = mf.where(tp > tp.shift(1), 0).rolling(period).sum()
        mf_neg = mf.where(tp < tp.shift(1), 0).rolling(period).sum()
        
        mfi = 100 - (100 / (1 + mf_pos / (mf_neg + 1e-10)))
        return mfi
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Average Directional Index"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        atr = self._calculate_atr(df, period)
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()
        
        return adx, plus_di, minus_di
    
    def _calculate_aroon(self, df: pd.DataFrame, period: int = 25) -> Tuple[pd.Series, pd.Series]:
        """Calculate Aroon Indicator"""
        high = df['high']
        low = df['low']
        
        aroon_up = high.rolling(period + 1).apply(
            lambda x: (period - x.argmax()) / period * 100, raw=True
        )
        aroon_down = low.rolling(period + 1).apply(
            lambda x: (period - x.argmin()) / period * 100, raw=True
        )
        
        return aroon_up, aroon_down
    
    def _calculate_hurst(self, series: pd.Series, window: int = 50) -> pd.Series:
        """Calculate Hurst Exponent (simplified rolling version)"""
        def hurst_exp(ts):
            if len(ts) < 20:
                return 0.5
            lags = range(2, min(20, len(ts) // 2))
            tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
            if len(tau) < 2:
                return 0.5
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0]
        
        return series.rolling(window).apply(hurst_exp, raw=False)
    
    def get_feature_names(self) -> list:
        """Return list of generated feature names"""
        return self.feature_names
