"""
Natron Feature Engine - Generates 100+ Technical Features
Author: Natron AI System
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class FeatureEngine:
    """
    Generates comprehensive technical features for financial trading.
    Target: ~100 features across multiple categories.
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.feature_names = []
        
    def generate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all technical features from OHLCV data.
        
        Args:
            df: DataFrame with columns [time, open, high, low, close, volume]
            
        Returns:
            DataFrame with ~100 technical features
        """
        if self.verbose:
            print("🔧 Generating features...")
            
        df = df.copy()
        features = pd.DataFrame(index=df.index)
        
        # 1. Moving Averages (13 features)
        features = pd.concat([features, self._moving_averages(df)], axis=1)
        
        # 2. Momentum Indicators (13 features)
        features = pd.concat([features, self._momentum_indicators(df)], axis=1)
        
        # 3. Volatility Indicators (15 features)
        features = pd.concat([features, self._volatility_indicators(df)], axis=1)
        
        # 4. Volume Indicators (9 features)
        features = pd.concat([features, self._volume_indicators(df)], axis=1)
        
        # 5. Price Patterns (8 features)
        features = pd.concat([features, self._price_patterns(df)], axis=1)
        
        # 6. Returns (8 features)
        features = pd.concat([features, self._returns(df)], axis=1)
        
        # 7. Trend Strength (6 features)
        features = pd.concat([features, self._trend_strength(df)], axis=1)
        
        # 8. Statistical Features (6 features)
        features = pd.concat([features, self._statistical_features(df)], axis=1)
        
        # 9. Support/Resistance (4 features)
        features = pd.concat([features, self._support_resistance(df)], axis=1)
        
        # 10. Smart Money Concepts (6 features)
        features = pd.concat([features, self._smart_money_concepts(df)], axis=1)
        
        # 11. Market Profile (10 features)
        features = pd.concat([features, self._market_profile(df)], axis=1)
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        if self.verbose:
            print(f"✅ Generated {len(self.feature_names)} features")
            
        return features
    
    def _moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Moving Average features (13)"""
        close = df['close']
        feat = pd.DataFrame(index=df.index)
        
        # Simple and Exponential MAs
        for period in [5, 10, 20, 50]:
            feat[f'ma_{period}'] = close.rolling(period).mean()
            feat[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
        
        # MA slopes
        feat['ma20_slope'] = feat['ma_20'].diff(5) / feat['ma_20']
        
        # Price to MA ratios
        feat['price_to_ma20'] = close / feat['ma_20']
        feat['price_to_ma50'] = close / feat['ma_50']
        
        # MA crossovers
        feat['ma10_above_ma20'] = (feat['ma_10'] > feat['ma_20']).astype(int)
        feat['ma20_above_ma50'] = (feat['ma_20'] > feat['ma_50']).astype(int)
        
        return feat
    
    def _momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum indicators (13)"""
        close = df['close']
        high = df['high']
        low = df['low']
        feat = pd.DataFrame(index=df.index)
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        feat['rsi'] = 100 - (100 / (1 + rs))
        feat['rsi_oversold'] = (feat['rsi'] < 30).astype(int)
        feat['rsi_overbought'] = (feat['rsi'] > 70).astype(int)
        
        # Rate of Change
        feat['roc_5'] = close.pct_change(5)
        feat['roc_10'] = close.pct_change(10)
        
        # CCI (Commodity Channel Index)
        tp = (high + low + close) / 3
        ma_tp = tp.rolling(20).mean()
        md = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        feat['cci'] = (tp - ma_tp) / (0.015 * md + 1e-10)
        
        # Stochastic
        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        feat['stoch_k'] = 100 * (close - low_14) / (high_14 - low_14 + 1e-10)
        feat['stoch_d'] = feat['stoch_k'].rolling(3).mean()
        
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        feat['macd'] = ema12 - ema26
        feat['macd_signal'] = feat['macd'].ewm(span=9, adjust=False).mean()
        feat['macd_hist'] = feat['macd'] - feat['macd_signal']
        feat['macd_hist_increasing'] = (feat['macd_hist'].diff() > 0).astype(int)
        
        return feat
    
    def _volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility indicators (15)"""
        close = df['close']
        high = df['high']
        low = df['low']
        feat = pd.DataFrame(index=df.index)
        
        # ATR (Average True Range)
        tr1 = high - low
        tr2 = np.abs(high - close.shift())
        tr3 = np.abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        feat['atr_14'] = tr.rolling(14).mean()
        feat['atr_pct'] = feat['atr_14'] / close
        
        # Bollinger Bands
        for period in [20]:
            ma = close.rolling(period).mean()
            std = close.rolling(period).std()
            feat[f'bb_upper_{period}'] = ma + 2 * std
            feat[f'bb_lower_{period}'] = ma - 2 * std
            feat[f'bb_mid_{period}'] = ma
            feat[f'bb_width_{period}'] = (feat[f'bb_upper_{period}'] - feat[f'bb_lower_{period}']) / ma
            feat[f'bb_position_{period}'] = (close - feat[f'bb_lower_{period}']) / (feat[f'bb_upper_{period}'] - feat[f'bb_lower_{period}'] + 1e-10)
        
        # Keltner Channels
        ema20 = close.ewm(span=20, adjust=False).mean()
        feat['keltner_upper'] = ema20 + 2 * feat['atr_14']
        feat['keltner_lower'] = ema20 - 2 * feat['atr_14']
        feat['keltner_width'] = (feat['keltner_upper'] - feat['keltner_lower']) / ema20
        
        # Standard Deviation
        feat['std_20'] = close.rolling(20).std()
        feat['std_50'] = close.rolling(50).std()
        
        # Historical Volatility
        returns = np.log(close / close.shift())
        feat['hist_vol_20'] = returns.rolling(20).std() * np.sqrt(252)
        
        # Parkinson Volatility (uses high-low)
        feat['parkinson_vol'] = np.sqrt(1/(4*np.log(2)) * np.log(high/low)**2).rolling(20).mean()
        
        return feat
    
    def _volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume indicators (9)"""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        feat = pd.DataFrame(index=df.index)
        
        # On-Balance Volume
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        feat['obv'] = obv
        feat['obv_ema'] = obv.ewm(span=20, adjust=False).mean()
        
        # VWAP
        typical_price = (high + low + close) / 3
        feat['vwap'] = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
        
        # Money Flow Index
        tp = (high + low + close) / 3
        mf = tp * volume
        pos_mf = mf.where(tp > tp.shift(), 0).rolling(14).sum()
        neg_mf = mf.where(tp < tp.shift(), 0).rolling(14).sum()
        feat['mfi'] = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-10)))
        
        # Volume features
        feat['volume_ma_20'] = volume.rolling(20).mean()
        feat['volume_ratio'] = volume / (feat['volume_ma_20'] + 1e-10)
        feat['volume_spike'] = (feat['volume_ratio'] > 1.5).astype(int)
        
        # Accumulation/Distribution
        clv = ((close - low) - (high - close)) / (high - low + 1e-10)
        feat['ad'] = (clv * volume).cumsum()
        
        return feat
    
    def _price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price pattern features (8)"""
        open_ = df['open']
        close = df['close']
        high = df['high']
        low = df['low']
        feat = pd.DataFrame(index=df.index)
        
        # Candle body
        body = np.abs(close - open_)
        range_ = high - low
        feat['body_pct'] = body / (range_ + 1e-10)
        
        # Doji detection
        feat['is_doji'] = (feat['body_pct'] < 0.1).astype(int)
        
        # Upper and lower shadows
        feat['upper_shadow'] = (high - np.maximum(open_, close)) / (range_ + 1e-10)
        feat['lower_shadow'] = (np.minimum(open_, close) - low) / (range_ + 1e-10)
        
        # Gap detection
        feat['gap_up'] = (open_ > close.shift()).astype(int)
        feat['gap_down'] = (open_ < close.shift()).astype(int)
        
        # Price position in range
        feat['close_position'] = (close - low) / (range_ + 1e-10)
        feat['close_near_high'] = (feat['close_position'] >= 0.7).astype(int)
        
        return feat
    
    def _returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return features (8)"""
        close = df['close']
        open_ = df['open']
        high = df['high']
        low = df['low']
        feat = pd.DataFrame(index=df.index)
        
        # Log returns
        feat['log_return'] = np.log(close / close.shift())
        feat['log_return_5'] = np.log(close / close.shift(5))
        feat['log_return_10'] = np.log(close / close.shift(10))
        
        # Intraday return
        feat['intraday_return'] = (close - open_) / open_
        
        # Cumulative returns
        feat['cum_return_20'] = (close / close.shift(20)) - 1
        feat['cum_return_50'] = (close / close.shift(50)) - 1
        
        # Range returns
        feat['high_low_ratio'] = high / low
        feat['close_open_ratio'] = close / open_
        
        return feat
    
    def _trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trend strength indicators (6)"""
        close = df['close']
        high = df['high']
        low = df['low']
        feat = pd.DataFrame(index=df.index)
        
        # ADX (Average Directional Index)
        tr1 = high - low
        tr2 = np.abs(high - close.shift())
        tr3 = np.abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        
        plus_dm = (high - high.shift()).where((high - high.shift()) > (low.shift() - low), 0).where((high - high.shift()) > 0, 0)
        minus_dm = (low.shift() - low).where((low.shift() - low) > (high - high.shift()), 0).where((low.shift() - low) > 0, 0)
        
        plus_di = 100 * (plus_dm.rolling(14).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(14).mean() / (atr + 1e-10))
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        feat['adx'] = dx.rolling(14).mean()
        feat['plus_di'] = plus_di
        feat['minus_di'] = minus_di
        
        # Aroon
        aroon_up = 100 * close.rolling(25).apply(lambda x: x.argmax()) / 25
        aroon_down = 100 * close.rolling(25).apply(lambda x: x.argmin()) / 25
        feat['aroon_up'] = aroon_up
        feat['aroon_down'] = aroon_down
        feat['aroon_oscillator'] = aroon_up - aroon_down
        
        return feat
    
    def _statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Statistical features (6)"""
        close = df['close']
        feat = pd.DataFrame(index=df.index)
        
        # Skewness and Kurtosis
        returns = close.pct_change()
        feat['skewness_20'] = returns.rolling(20).skew()
        feat['kurtosis_20'] = returns.rolling(20).kurt()
        
        # Z-score
        ma = close.rolling(20).mean()
        std = close.rolling(20).std()
        feat['zscore'] = (close - ma) / (std + 1e-10)
        
        # Hurst Exponent approximation
        feat['hurst_approx'] = self._hurst_approx(close, window=100)
        
        # Autocorrelation
        feat['autocorr_1'] = returns.rolling(20).apply(lambda x: x.autocorr(lag=1))
        feat['autocorr_5'] = returns.rolling(20).apply(lambda x: x.autocorr(lag=5))
        
        return feat
    
    def _support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Support/Resistance features (4)"""
        close = df['close']
        high = df['high']
        low = df['low']
        feat = pd.DataFrame(index=df.index)
        
        # Distance to recent highs/lows
        high_20 = high.rolling(20).max()
        low_20 = low.rolling(20).min()
        high_50 = high.rolling(50).max()
        low_50 = low.rolling(50).min()
        
        feat['dist_to_high_20'] = (high_20 - close) / close
        feat['dist_to_low_20'] = (close - low_20) / close
        feat['dist_to_high_50'] = (high_50 - close) / close
        feat['dist_to_low_50'] = (close - low_50) / close
        
        return feat
    
    def _smart_money_concepts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Smart Money Concepts (6)"""
        close = df['close']
        high = df['high']
        low = df['low']
        feat = pd.DataFrame(index=df.index)
        
        # Swing highs and lows
        feat['swing_high'] = (high > high.shift(1)) & (high > high.shift(-1))
        feat['swing_low'] = (low < low.shift(1)) & (low < low.shift(-1))
        
        # Break of Structure (BOS)
        feat['bos_up'] = (close > high.rolling(20).max().shift(1)).astype(int)
        feat['bos_down'] = (close < low.rolling(20).min().shift(1)).astype(int)
        
        # Change of Character (CHOCH)
        prev_high = high.rolling(10).max()
        prev_low = low.rolling(10).min()
        feat['choch'] = ((close > prev_high.shift(1)) | (close < prev_low.shift(1))).astype(int)
        
        # Order blocks (simplified)
        feat['order_block'] = (df['volume'] > df['volume'].rolling(20).mean() * 2).astype(int)
        
        return feat
    
    def _market_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market Profile features (10)"""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        feat = pd.DataFrame(index=df.index)
        
        # Point of Control (simplified - highest volume price)
        feat['poc'] = close.rolling(20).apply(lambda x: x.value_counts().idxmax() if len(x.value_counts()) > 0 else x.mean())
        
        # Value Area High/Low (approximation)
        feat['vah'] = high.rolling(20).quantile(0.7)
        feat['val'] = low.rolling(20).quantile(0.3)
        
        # Price distribution
        feat['price_range_20'] = high.rolling(20).max() - low.rolling(20).min()
        feat['volume_at_price'] = volume.rolling(20).sum()
        
        # Volume profile metrics
        feat['vol_weighted_price'] = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
        
        # Price entropy (market uncertainty)
        returns = close.pct_change().abs()
        feat['price_entropy'] = -returns.rolling(20).apply(lambda x: np.sum(x * np.log(x + 1e-10)))
        
        # Balance metrics
        feat['price_above_poc'] = (close > feat['poc']).astype(int)
        feat['price_in_value_area'] = ((close >= feat['val']) & (close <= feat['vah'])).astype(int)
        feat['value_area_width'] = (feat['vah'] - feat['val']) / close
        
        return feat
    
    def _hurst_approx(self, series: pd.Series, window: int = 100) -> pd.Series:
        """Approximate Hurst exponent using rolling window"""
        def hurst(ts):
            if len(ts) < 20:
                return 0.5
            lags = range(2, min(20, len(ts)//2))
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        
        return series.rolling(window).apply(hurst, raw=False)
    
    def normalize_features(self, features: pd.DataFrame, method: str = 'robust') -> Tuple[pd.DataFrame, dict]:
        """
        Normalize features using specified method.
        
        Args:
            features: DataFrame with features
            method: 'robust', 'standard', or 'minmax'
            
        Returns:
            Normalized features and scaling parameters
        """
        from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
        
        if method == 'robust':
            scaler = RobustScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        
        # Handle NaN and inf
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(0)
        
        normalized = pd.DataFrame(
            scaler.fit_transform(features),
            index=features.index,
            columns=features.columns
        )
        
        scaling_params = {
            'scaler': scaler,
            'feature_names': features.columns.tolist()
        }
        
        return normalized, scaling_params


if __name__ == "__main__":
    # Test feature generation
    print("🧪 Testing FeatureEngine...")
    
    # Generate sample OHLCV data
    np.random.seed(42)
    n = 1000
    dates = pd.date_range('2023-01-01', periods=n, freq='15min')
    
    df = pd.DataFrame({
        'time': dates,
        'open': 100 + np.cumsum(np.random.randn(n) * 0.1),
        'high': 101 + np.cumsum(np.random.randn(n) * 0.1),
        'low': 99 + np.cumsum(np.random.randn(n) * 0.1),
        'close': 100 + np.cumsum(np.random.randn(n) * 0.1),
        'volume': np.random.randint(1000, 10000, n)
    })
    
    engine = FeatureEngine(verbose=True)
    features = engine.generate_all_features(df)
    
    print(f"\n✅ Generated {features.shape[1]} features")
    print(f"📊 Feature shape: {features.shape}")
    print(f"\n📋 First 5 features:\n{features.iloc[:5, :5]}")
    print(f"\n✅ Feature names ({len(engine.feature_names)}):\n{engine.feature_names[:10]}...")
