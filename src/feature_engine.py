"""
Natron Transformer - Feature Engineering Module
Generates ~100 technical indicators from OHLCV data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class FeatureEngine:
    """
    Comprehensive feature engineering for financial time series.
    Generates approximately 100 technical indicators across multiple categories.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.feature_names = []
        
    def generate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all technical features from OHLCV data.
        
        Args:
            df: DataFrame with columns [time, open, high, low, close, volume]
            
        Returns:
            DataFrame with ~100 technical features
        """
        df = df.copy()
        
        # Ensure proper column names
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        features = pd.DataFrame(index=df.index)
        
        # 1. Moving Average Features (13)
        ma_features = self._moving_average_features(df)
        features = pd.concat([features, ma_features], axis=1)
        
        # 2. Momentum Features (13)
        momentum_features = self._momentum_features(df)
        features = pd.concat([features, momentum_features], axis=1)
        
        # 3. Volatility Features (15)
        volatility_features = self._volatility_features(df)
        features = pd.concat([features, volatility_features], axis=1)
        
        # 4. Volume Features (9)
        volume_features = self._volume_features(df)
        features = pd.concat([features, volume_features], axis=1)
        
        # 5. Price Pattern Features (8)
        pattern_features = self._price_pattern_features(df)
        features = pd.concat([features, pattern_features], axis=1)
        
        # 6. Returns Features (8)
        return_features = self._return_features(df)
        features = pd.concat([features, return_features], axis=1)
        
        # 7. Trend Strength Features (6)
        trend_features = self._trend_strength_features(df)
        features = pd.concat([features, trend_features], axis=1)
        
        # 8. Statistical Features (6)
        stat_features = self._statistical_features(df)
        features = pd.concat([features, stat_features], axis=1)
        
        # 9. Support/Resistance Features (4)
        sr_features = self._support_resistance_features(df)
        features = pd.concat([features, sr_features], axis=1)
        
        # 10. Smart Money Concepts (6)
        smc_features = self._smart_money_features(df)
        features = pd.concat([features, smc_features], axis=1)
        
        # 11. Market Profile Features (10)
        profile_features = self._market_profile_features(df)
        features = pd.concat([features, profile_features], axis=1)
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        # Fill NaN values (from calculations) with forward fill then 0
        features = features.fillna(method='ffill').fillna(0)
        
        print(f"✅ Generated {len(self.feature_names)} features")
        return features
    
    def _moving_average_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Moving Average group: 13 features"""
        close = df['close']
        features = pd.DataFrame(index=df.index)
        
        # Simple Moving Averages
        features['ma_5'] = close.rolling(5).mean()
        features['ma_10'] = close.rolling(10).mean()
        features['ma_20'] = close.rolling(20).mean()
        features['ma_50'] = close.rolling(50).mean()
        
        # Exponential Moving Averages
        features['ema_9'] = close.ewm(span=9, adjust=False).mean()
        features['ema_21'] = close.ewm(span=21, adjust=False).mean()
        
        # Price to MA ratios
        features['price_to_ma20'] = close / features['ma_20']
        features['price_to_ma50'] = close / features['ma_50']
        
        # MA slopes
        features['ma20_slope'] = features['ma_20'].diff(5) / features['ma_20']
        features['ma50_slope'] = features['ma_50'].diff(5) / features['ma_50']
        
        # MA crossovers
        features['ma5_cross_ma20'] = (features['ma_5'] > features['ma_20']).astype(int)
        features['ma20_cross_ma50'] = (features['ma_20'] > features['ma_50']).astype(int)
        
        # Distance from MA
        features['dist_from_ma20'] = (close - features['ma_20']) / features['ma_20']
        
        return features
    
    def _momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum group: 13 features"""
        close = df['close']
        high = df['high']
        low = df['low']
        features = pd.DataFrame(index=df.index)
        
        # RSI (Relative Strength Index)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi_14'] = 100 - (100 / (1 + rs))
        features['rsi_7'] = self._calculate_rsi(close, 7)
        
        # Rate of Change
        features['roc_5'] = close.pct_change(5) * 100
        features['roc_10'] = close.pct_change(10) * 100
        
        # CCI (Commodity Channel Index)
        tp = (high + low + close) / 3
        features['cci_20'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
        
        # Stochastic Oscillator
        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        features['stoch_k'] = 100 * (close - low_14) / (high_14 - low_14)
        features['stoch_d'] = features['stoch_k'].rolling(3).mean()
        
        # MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # Williams %R
        features['williams_r'] = -100 * (high_14 - close) / (high_14 - low_14)
        
        # Momentum
        features['momentum_10'] = close - close.shift(10)
        
        return features
    
    def _volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility group: 15 features"""
        close = df['close']
        high = df['high']
        low = df['low']
        features = pd.DataFrame(index=df.index)
        
        # ATR (Average True Range)
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        features['atr_14'] = tr.rolling(14).mean()
        features['atr_20'] = tr.rolling(20).mean()
        features['atr_pct'] = features['atr_14'] / close
        
        # Bollinger Bands
        ma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        features['bb_upper'] = ma_20 + (std_20 * 2)
        features['bb_lower'] = ma_20 - (std_20 * 2)
        features['bb_mid'] = ma_20
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_mid']
        features['bb_position'] = (close - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # Keltner Channels
        ema_20 = close.ewm(span=20, adjust=False).mean()
        features['keltner_upper'] = ema_20 + (features['atr_14'] * 2)
        features['keltner_lower'] = ema_20 - (features['atr_14'] * 2)
        
        # Standard Deviation
        features['std_10'] = close.rolling(10).std()
        features['std_20'] = close.rolling(20).std()
        
        # Historical Volatility
        features['hist_vol_10'] = close.pct_change().rolling(10).std() * np.sqrt(252)
        features['hist_vol_20'] = close.pct_change().rolling(20).std() * np.sqrt(252)
        
        # Normalized ATR
        features['natr'] = features['atr_14'] / close * 100
        
        return features
    
    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume group: 9 features"""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        features = pd.DataFrame(index=df.index)
        
        # Volume moving averages
        features['vol_ma_10'] = volume.rolling(10).mean()
        features['vol_ma_20'] = volume.rolling(20).mean()
        features['vol_ratio'] = volume / features['vol_ma_20']
        
        # On-Balance Volume
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        features['obv'] = obv
        features['obv_ema'] = obv.ewm(span=20, adjust=False).mean()
        
        # VWAP (Volume Weighted Average Price)
        tp = (high + low + close) / 3
        features['vwap'] = (tp * volume).cumsum() / volume.cumsum()
        
        # Money Flow Index
        tp_diff = tp.diff()
        pos_flow = (tp * volume).where(tp_diff > 0, 0).rolling(14).sum()
        neg_flow = (tp * volume).where(tp_diff < 0, 0).rolling(14).sum()
        mfi = 100 - (100 / (1 + pos_flow / neg_flow))
        features['mfi'] = mfi
        
        # Volume Rate of Change
        features['vol_roc'] = volume.pct_change(10) * 100
        
        return features
    
    def _price_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price Pattern group: 8 features"""
        open_ = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        features = pd.DataFrame(index=df.index)
        
        # Candle body and shadows
        body = abs(close - open_)
        range_ = high - low
        features['body_pct'] = body / range_
        features['upper_shadow'] = high - np.maximum(open_, close)
        features['lower_shadow'] = np.minimum(open_, close) - low
        features['shadow_ratio'] = (features['upper_shadow'] + features['lower_shadow']) / body
        
        # Doji detection (small body)
        features['is_doji'] = (body / range_ < 0.1).astype(int)
        
        # Price position in range
        features['close_position'] = (close - low) / range_
        
        # Gap detection
        features['gap_up'] = (low > high.shift()).astype(int)
        features['gap_down'] = (high < low.shift()).astype(int)
        
        return features
    
    def _return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns group: 8 features"""
        close = df['close']
        open_ = df['open']
        features = pd.DataFrame(index=df.index)
        
        # Simple returns
        features['return_1'] = close.pct_change(1)
        features['return_5'] = close.pct_change(5)
        features['return_10'] = close.pct_change(10)
        
        # Log returns
        features['log_return_1'] = np.log(close / close.shift(1))
        features['log_return_5'] = np.log(close / close.shift(5))
        
        # Intraday return
        features['intraday_return'] = (close - open_) / open_
        
        # Cumulative returns
        features['cum_return_20'] = (close / close.shift(20)) - 1
        features['cum_return_50'] = (close / close.shift(50)) - 1
        
        return features
    
    def _trend_strength_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trend Strength group: 6 features"""
        close = df['close']
        high = df['high']
        low = df['low']
        features = pd.DataFrame(index=df.index)
        
        # ADX (Average Directional Index)
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr_14 = tr.rolling(14).sum()
        plus_dm_14 = plus_dm.rolling(14).sum()
        minus_dm_14 = minus_dm.rolling(14).sum()
        
        plus_di = 100 * (plus_dm_14 / tr_14)
        minus_di = 100 * (minus_dm_14 / tr_14)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        features['adx'] = dx.rolling(14).mean()
        features['plus_di'] = plus_di
        features['minus_di'] = minus_di
        
        # Aroon Indicator
        aroon_up = 100 * close.rolling(25).apply(lambda x: x.argmax()) / 25
        aroon_down = 100 * close.rolling(25).apply(lambda x: x.argmin()) / 25
        features['aroon_up'] = aroon_up
        features['aroon_down'] = aroon_down
        features['aroon_oscillator'] = aroon_up - aroon_down
        
        return features
    
    def _statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Statistical group: 6 features"""
        close = df['close']
        features = pd.DataFrame(index=df.index)
        
        # Skewness and Kurtosis
        returns = close.pct_change()
        features['skew_20'] = returns.rolling(20).skew()
        features['kurt_20'] = returns.rolling(20).kurt()
        
        # Z-score
        ma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        features['zscore'] = (close - ma_20) / std_20
        
        # Hurst Exponent (simplified)
        features['hurst_50'] = close.rolling(50).apply(self._hurst_exponent, raw=False)
        
        # Autocorrelation
        features['autocorr_5'] = returns.rolling(20).apply(lambda x: x.autocorr(lag=5), raw=False)
        
        # Entropy
        features['entropy_20'] = returns.rolling(20).apply(self._calculate_entropy, raw=False)
        
        return features
    
    def _support_resistance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Support/Resistance group: 4 features"""
        close = df['close']
        high = df['high']
        low = df['low']
        features = pd.DataFrame(index=df.index)
        
        # Distance to recent highs/lows
        high_20 = high.rolling(20).max()
        low_20 = low.rolling(20).min()
        high_50 = high.rolling(50).max()
        low_50 = low.rolling(50).min()
        
        features['dist_to_high_20'] = (high_20 - close) / close
        features['dist_to_low_20'] = (close - low_20) / close
        features['dist_to_high_50'] = (high_50 - close) / close
        features['dist_to_low_50'] = (close - low_50) / close
        
        return features
    
    def _smart_money_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Smart Money Concepts group: 6 features"""
        high = df['high']
        low = df['low']
        close = df['close']
        features = pd.DataFrame(index=df.index)
        
        # Swing Highs and Lows
        swing_period = 5
        features['swing_high'] = high.rolling(swing_period * 2 + 1, center=True).max() == high
        features['swing_low'] = low.rolling(swing_period * 2 + 1, center=True).min() == low
        
        # Break of Structure (BOS)
        prev_high = high.rolling(20).max().shift(1)
        prev_low = low.rolling(20).min().shift(1)
        features['bos_bullish'] = (high > prev_high).astype(int)
        features['bos_bearish'] = (low < prev_low).astype(int)
        
        # Change of Character (CHOCH) - simplified
        features['choch'] = (features['bos_bullish'].diff().abs() + features['bos_bearish'].diff().abs()).clip(0, 1)
        
        # Fair Value Gap (simplified)
        gap_up = low.shift(-1) > high.shift(1)
        gap_down = high.shift(-1) < low.shift(1)
        features['fvg'] = (gap_up.astype(int) - gap_down.astype(int)).shift(1)
        
        return features
    
    def _market_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market Profile group: 10 features"""
        open_ = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        features = pd.DataFrame(index=df.index)
        
        # Value Area (simplified using percentiles)
        window = 20
        features['vah'] = high.rolling(window).quantile(0.70)  # Value Area High
        features['val'] = low.rolling(window).quantile(0.30)   # Value Area Low
        features['poc'] = close.rolling(window).median()        # Point of Control (approx)
        
        # Price position relative to value area
        features['above_vah'] = (close > features['vah']).astype(int)
        features['below_val'] = (close < features['val']).astype(int)
        features['in_value_area'] = ((close >= features['val']) & (close <= features['vah'])).astype(int)
        
        # Volume at price levels (simplified)
        features['vol_at_high'] = volume * (close == high).astype(int)
        features['vol_at_low'] = volume * (close == low).astype(int)
        
        # Price entropy (distribution)
        features['price_entropy'] = close.rolling(window).apply(self._calculate_entropy, raw=False)
        
        # Volume profile balance
        features['volume_balance'] = volume.rolling(window).apply(
            lambda x: (x.iloc[-window//2:].sum() / x.sum()) if x.sum() > 0 else 0.5,
            raw=False
        )
        
        return features
    
    @staticmethod
    def _calculate_rsi(series: pd.Series, period: int) -> pd.Series:
        """Calculate RSI for given period"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _hurst_exponent(ts):
        """Calculate Hurst Exponent (simplified)"""
        try:
            ts = np.array(ts)
            if len(ts) < 20:
                return 0.5
            
            lags = range(2, min(20, len(ts) // 2))
            tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
            
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        except:
            return 0.5
    
    @staticmethod
    def _calculate_entropy(series):
        """Calculate Shannon entropy"""
        try:
            series = np.array(series)
            series = series[~np.isnan(series)]
            if len(series) < 2:
                return 0
            
            # Discretize into bins
            hist, _ = np.histogram(series, bins=10)
            hist = hist / hist.sum()
            hist = hist[hist > 0]
            
            return -np.sum(hist * np.log2(hist))
        except:
            return 0


if __name__ == "__main__":
    # Test feature generation
    print("🧪 Testing Feature Engine...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='15min')
    
    df = pd.DataFrame({
        'time': dates,
        'open': 100 + np.cumsum(np.random.randn(n_samples) * 0.5),
        'high': 101 + np.cumsum(np.random.randn(n_samples) * 0.5),
        'low': 99 + np.cumsum(np.random.randn(n_samples) * 0.5),
        'close': 100 + np.cumsum(np.random.randn(n_samples) * 0.5),
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    # Ensure high >= low
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    engine = FeatureEngine()
    features = engine.generate_all_features(df)
    
    print(f"\n📊 Feature Shape: {features.shape}")
    print(f"📋 Feature Names ({len(engine.feature_names)}):")
    for i, name in enumerate(engine.feature_names, 1):
        print(f"  {i}. {name}")
    
    print(f"\n✅ Feature generation successful!")
    print(f"NaN values: {features.isna().sum().sum()}")
