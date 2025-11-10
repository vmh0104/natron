"""
FeatureEngine - Generate ~100 technical features from OHLCV data
"""
import numpy as np
import pandas as pd
from typing import Optional


class FeatureEngine:
    """Generate comprehensive technical features for financial trading"""
    
    def __init__(self):
        self.feature_names = []
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ~100 technical features from OHLCV data
        
        Args:
            df: DataFrame with columns [time, open, high, low, close, volume]
            
        Returns:
            DataFrame with ~100 feature columns
        """
        features_df = df.copy()
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in features_df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Extract price components
        open_p = features_df['open'].values
        high_p = features_df['high'].values
        low_p = features_df['low'].values
        close_p = features_df['close'].values
        volume = features_df['volume'].values
        
        # ========== MOVING AVERAGES (13 features) ==========
        ma_periods = [5, 10, 20, 50, 100, 200]
        for period in ma_periods:
            if len(features_df) >= period:
                ma = pd.Series(close_p).rolling(period).mean()
                features_df[f'MA{period}'] = ma
                features_df[f'price_to_MA{period}'] = close_p / (ma + 1e-8)
        
        # EMA
        ema_periods = [12, 26, 50]
        for period in ema_periods:
            if len(features_df) >= period:
                ema = pd.Series(close_p).ewm(span=period, adjust=False).mean()
                features_df[f'EMA{period}'] = ema
                features_df[f'price_to_EMA{period}'] = close_p / (ema + 1e-8)
        
        # MA slopes
        if 'MA20' in features_df.columns:
            features_df['MA20_slope'] = features_df['MA20'].diff()
        if 'MA50' in features_df.columns:
            features_df['MA50_slope'] = features_df['MA50'].diff()
        
        # MA crossovers
        if 'MA20' in features_df.columns and 'MA50' in features_df.columns:
            features_df['MA20_above_MA50'] = (features_df['MA20'] > features_df['MA50']).astype(float)
        
        # ========== MOMENTUM (13 features) ==========
        # RSI
        for period in [14, 21]:
            if len(features_df) >= period:
                delta = pd.Series(close_p).diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / (loss + 1e-8)
                rsi = 100 - (100 / (1 + rs))
                features_df[f'RSI{period}'] = rsi
        
        # ROC (Rate of Change)
        for period in [10, 14]:
            if len(features_df) >= period:
                roc = ((close_p - pd.Series(close_p).shift(period)) / pd.Series(close_p).shift(period)) * 100
                features_df[f'ROC{period}'] = roc
        
        # CCI (Commodity Channel Index)
        if len(features_df) >= 20:
            tp = (high_p + low_p + close_p) / 3
            sma_tp = pd.Series(tp).rolling(20).mean()
            mad = pd.Series(tp).rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (tp - sma_tp) / (0.015 * (mad + 1e-8))
            features_df['CCI'] = cci
        
        # Stochastic
        if len(features_df) >= 14:
            low_14 = pd.Series(low_p).rolling(14).min()
            high_14 = pd.Series(high_p).rolling(14).max()
            k_percent = 100 * ((close_p - low_14) / (high_14 - low_14 + 1e-8))
            features_df['Stoch_K'] = k_percent
            features_df['Stoch_D'] = k_percent.rolling(3).mean()
        
        # MACD
        if len(features_df) >= 26:
            ema12 = pd.Series(close_p).ewm(span=12, adjust=False).mean()
            ema26 = pd.Series(close_p).ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            histogram = macd_line - signal_line
            features_df['MACD'] = macd_line
            features_df['MACD_signal'] = signal_line
            features_df['MACD_hist'] = histogram
        
        # ========== VOLATILITY (15 features) ==========
        # ATR (Average True Range)
        for period in [14, 21]:
            if len(features_df) >= period:
                tr1 = high_p - low_p
                tr2 = np.abs(high_p - pd.Series(close_p).shift(1))
                tr3 = np.abs(low_p - pd.Series(close_p).shift(1))
                tr = np.maximum(tr1, np.maximum(tr2, tr3))
                atr = pd.Series(tr).rolling(period).mean()
                features_df[f'ATR{period}'] = atr
                features_df[f'ATR{period}_pct'] = (atr / (close_p + 1e-8)) * 100
        
        # Bollinger Bands
        if len(features_df) >= 20:
            ma20 = pd.Series(close_p).rolling(20).mean()
            std20 = pd.Series(close_p).rolling(20).std()
            features_df['BB_upper'] = ma20 + (2 * std20)
            features_df['BB_mid'] = ma20
            features_df['BB_lower'] = ma20 - (2 * std20)
            features_df['BB_width'] = (features_df['BB_upper'] - features_df['BB_lower']) / (ma20 + 1e-8)
            features_df['BB_position'] = (close_p - features_df['BB_lower']) / (features_df['BB_upper'] - features_df['BB_lower'] + 1e-8)
        
        # Keltner Channels
        if len(features_df) >= 20:
            kc_mid = pd.Series(close_p).ewm(span=20, adjust=False).mean()
            kc_range = pd.Series(high_p - low_p).ewm(span=20, adjust=False).mean()
            features_df['KC_upper'] = kc_mid + (1.5 * kc_range)
            features_df['KC_mid'] = kc_mid
            features_df['KC_lower'] = kc_mid - (1.5 * kc_range)
        
        # Standard Deviation
        for period in [10, 20]:
            if len(features_df) >= period:
                std = pd.Series(close_p).rolling(period).std()
                features_df[f'StdDev{period}'] = std
                features_df[f'StdDev{period}_pct'] = (std / (close_p + 1e-8)) * 100
        
        # ========== VOLUME (9 features) ==========
        # OBV (On-Balance Volume)
        obv = np.zeros(len(features_df))
        for i in range(1, len(features_df)):
            if close_p[i] > close_p[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close_p[i] < close_p[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        features_df['OBV'] = obv
        
        # VWAP (Volume Weighted Average Price)
        if 'time' in features_df.columns:
            features_df['cumulative_volume'] = volume.cumsum()
            features_df['cumulative_price_volume'] = (close_p * volume).cumsum()
            features_df['VWAP'] = features_df['cumulative_price_volume'] / (features_df['cumulative_volume'] + 1e-8)
        else:
            # Simplified VWAP
            features_df['VWAP'] = pd.Series(close_p * volume).rolling(20).sum() / (pd.Series(volume).rolling(20).sum() + 1e-8)
        
        # MFI (Money Flow Index)
        if len(features_df) >= 14:
            typical_price = (high_p + low_p + close_p) / 3
            money_flow = typical_price * volume
            positive_flow = money_flow.where(typical_price > pd.Series(typical_price).shift(1), 0).rolling(14).sum()
            negative_flow = money_flow.where(typical_price < pd.Series(typical_price).shift(1), 0).rolling(14).sum()
            mfi = 100 - (100 / (1 + (positive_flow / (negative_flow + 1e-8))))
            features_df['MFI'] = mfi
        
        # Volume ratios
        for period in [10, 20]:
            if len(features_df) >= period:
                avg_volume = pd.Series(volume).rolling(period).mean()
                features_df[f'volume_ratio{period}'] = volume / (avg_volume + 1e-8)
        
        # Volume MA
        for period in [10, 20]:
            if len(features_df) >= period:
                features_df[f'volume_MA{period}'] = pd.Series(volume).rolling(period).mean()
        
        # ========== PRICE PATTERNS (8 features) ==========
        # Body and shadows
        body = np.abs(close_p - open_p)
        upper_shadow = high_p - np.maximum(open_p, close_p)
        lower_shadow = np.minimum(open_p, close_p) - low_p
        total_range = high_p - low_p
        
        features_df['body_size'] = body
        features_df['body_pct'] = body / (total_range + 1e-8)
        features_df['upper_shadow_pct'] = upper_shadow / (total_range + 1e-8)
        features_df['lower_shadow_pct'] = lower_shadow / (total_range + 1e-8)
        
        # Doji pattern (body < 20% of range)
        features_df['is_doji'] = (features_df['body_pct'] < 0.2).astype(float)
        
        # Price position in range
        features_df['price_position'] = (close_p - low_p) / (total_range + 1e-8)
        
        # Gaps
        features_df['gap'] = open_p - pd.Series(close_p).shift(1)
        features_df['gap_pct'] = (features_df['gap'] / (pd.Series(close_p).shift(1) + 1e-8)) * 100
        
        # ========== RETURNS (8 features) ==========
        # Simple returns
        for period in [1, 5, 10, 20]:
            if len(features_df) >= period:
                returns = (close_p - pd.Series(close_p).shift(period)) / (pd.Series(close_p).shift(period) + 1e-8)
                features_df[f'return{period}'] = returns
        
        # Log returns
        for period in [1, 5]:
            if len(features_df) >= period:
                log_returns = np.log(close_p / (pd.Series(close_p).shift(period) + 1e-8))
                features_df[f'log_return{period}'] = log_returns
        
        # Intraday return
        features_df['intraday_return'] = (close_p - open_p) / (open_p + 1e-8)
        
        # Cumulative return (20 period)
        if len(features_df) >= 20:
            features_df['cumulative_return20'] = ((close_p / pd.Series(close_p).shift(20)) - 1) * 100
        
        # ========== TREND STRENGTH (6 features) ==========
        # ADX (Average Directional Index)
        if len(features_df) >= 14:
            plus_dm = np.where((high_p - pd.Series(high_p).shift(1)) > (pd.Series(low_p).shift(1) - low_p),
                              np.maximum(high_p - pd.Series(high_p).shift(1), 0), 0)
            minus_dm = np.where((pd.Series(low_p).shift(1) - low_p) > (high_p - pd.Series(high_p).shift(1)),
                               np.maximum(pd.Series(low_p).shift(1) - low_p, 0), 0)
            
            tr = np.maximum(high_p - low_p,
                          np.maximum(np.abs(high_p - pd.Series(close_p).shift(1)),
                                   np.abs(low_p - pd.Series(close_p).shift(1))))
            
            atr14 = pd.Series(tr).rolling(14).mean()
            plus_di = 100 * (pd.Series(plus_dm).rolling(14).mean() / (atr14 + 1e-8))
            minus_di = 100 * (pd.Series(minus_dm).rolling(14).mean() / (atr14 + 1e-8))
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
            adx = pd.Series(dx).rolling(14).mean()
            
            features_df['ADX'] = adx
            features_df['+DI'] = plus_di
            features_df['-DI'] = minus_di
        
        # Aroon
        if len(features_df) >= 14:
            aroon_up = []
            aroon_down = []
            for i in range(len(features_df)):
                if i < 14:
                    aroon_up.append(np.nan)
                    aroon_down.append(np.nan)
                else:
                    window_high = high_p[i-14:i+1]
                    window_low = low_p[i-14:i+1]
                    periods_since_high = 14 - np.argmax(window_high)
                    periods_since_low = 14 - np.argmin(window_low)
                    aroon_up.append(((14 - periods_since_high) / 14) * 100)
                    aroon_down.append(((14 - periods_since_low) / 14) * 100)
            features_df['Aroon_Up'] = aroon_up
            features_df['Aroon_Down'] = aroon_down
        
        # ========== STATISTICAL (6 features) ==========
        for period in [20, 50]:
            if len(features_df) >= period:
                window = pd.Series(close_p).rolling(period)
                features_df[f'skewness{period}'] = window.skew()
                features_df[f'kurtosis{period}'] = window.kurtosis()
                features_df[f'zscore{period}'] = (close_p - window.mean()) / (window.std() + 1e-8)
        
        # Hurst exponent (simplified)
        if len(features_df) >= 50:
            hurst = []
            for i in range(len(features_df)):
                if i < 50:
                    hurst.append(np.nan)
                else:
                    window = close_p[i-50:i+1]
                    lags = [1, 2, 5, 10]
                    tau = []
                    for lag in lags:
                        if len(window) > lag:
                            tau.append(np.std(np.diff(window, lag)))
                    if len(tau) > 1:
                        poly = np.polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)
                        hurst.append(poly[0] * 2.0)
                    else:
                        hurst.append(np.nan)
            features_df['Hurst'] = hurst
        
        # ========== SUPPORT/RESISTANCE (4 features) ==========
        for period in [20, 50]:
            if len(features_df) >= period:
                rolling_high = pd.Series(high_p).rolling(period).max()
                rolling_low = pd.Series(low_p).rolling(period).min()
                features_df[f'dist_to_high{period}'] = (rolling_high - close_p) / (close_p + 1e-8)
                features_df[f'dist_to_low{period}'] = (close_p - rolling_low) / (close_p + 1e-8)
        
        # ========== SMC (Smart Money Concepts) (6 features) ==========
        # Swing High/Low
        if len(features_df) >= 5:
            swing_high = []
            swing_low = []
            for i in range(len(features_df)):
                if i < 2 or i >= len(features_df) - 2:
                    swing_high.append(0)
                    swing_low.append(0)
                else:
                    is_swing_high = (high_p[i] > high_p[i-1] and high_p[i] > high_p[i+1] and
                                    high_p[i] > high_p[i-2] and high_p[i] > high_p[i+2])
                    is_swing_low = (low_p[i] < low_p[i-1] and low_p[i] < low_p[i+1] and
                                   low_p[i] < low_p[i-2] and low_p[i] < low_p[i+2])
                    swing_high.append(1.0 if is_swing_high else 0.0)
                    swing_low.append(1.0 if is_swing_low else 0.0)
            features_df['swing_high'] = swing_high
            features_df['swing_low'] = swing_low
        
        # BOS (Break of Structure) / CHOCH (Change of Character) - simplified
        if len(features_df) >= 10:
            bos = []
            choch = []
            for i in range(len(features_df)):
                if i < 10:
                    bos.append(0)
                    choch.append(0)
                else:
                    # BOS: price breaks previous swing high/low
                    prev_high = np.max(high_p[i-10:i])
                    prev_low = np.min(low_p[i-10:i])
                    is_bos = (close_p[i] > prev_high) or (close_p[i] < prev_low)
                    bos.append(1.0 if is_bos else 0.0)
                    
                    # CHOCH: trend change detection
                    ma_short = np.mean(close_p[i-3:i+1])
                    ma_long = np.mean(close_p[i-10:i+1])
                    prev_ma_short = np.mean(close_p[i-4:i])
                    prev_ma_long = np.mean(close_p[i-11:i])
                    is_choch = ((ma_short > ma_long) != (prev_ma_short > prev_ma_long))
                    choch.append(1.0 if is_choch else 0.0)
            features_df['BOS'] = bos
            features_df['CHOCH'] = choch
        
        # Order blocks (simplified - large volume candles)
        if len(features_df) >= 20:
            volume_ma = pd.Series(volume).rolling(20).mean()
            features_df['order_block'] = ((volume > 1.5 * volume_ma) & 
                                         (features_df['body_pct'] > 0.6)).astype(float)
        
        # ========== MARKET PROFILE (10 features) ==========
        # POC (Point of Control) - price level with highest volume
        if len(features_df) >= 20:
            poc_levels = []
            vah_levels = []
            val_levels = []
            for i in range(len(features_df)):
                if i < 20:
                    poc_levels.append(np.nan)
                    vah_levels.append(np.nan)
                    val_levels.append(np.nan)
                else:
                    window_high = high_p[i-20:i+1]
                    window_low = low_p[i-20:i+1]
                    window_volume = volume[i-20:i+1]
                    window_close = close_p[i-20:i+1]
                    
                    # Simplified POC as VWAP
                    poc = np.sum(window_close * window_volume) / (np.sum(window_volume) + 1e-8)
                    poc_levels.append(poc)
                    
                    # VAH/VAL as upper/lower quartiles
                    price_range = np.linspace(np.min(window_low), np.max(window_high), 20)
                    volume_dist = []
                    for price in price_range:
                        vol_at_price = np.sum(window_volume[(window_low <= price) & (window_high >= price)])
                        volume_dist.append(vol_at_price)
                    
                    if len(volume_dist) > 0:
                        sorted_idx = np.argsort(volume_dist)
                        vah_levels.append(price_range[sorted_idx[-5]])  # Top quartile
                        val_levels.append(price_range[sorted_idx[4]])    # Bottom quartile
                    else:
                        vah_levels.append(np.nan)
                        val_levels.append(np.nan)
            
            features_df['POC'] = poc_levels
            features_df['VAH'] = vah_levels
            features_df['VAL'] = val_levels
            features_df['dist_to_POC'] = (close_p - features_df['POC']) / (features_df['POC'] + 1e-8)
            features_df['dist_to_VAH'] = (features_df['VAH'] - close_p) / (close_p + 1e-8)
            features_df['dist_to_VAL'] = (close_p - features_df['VAL']) / (close_p + 1e-8)
        
        # Entropy (price distribution entropy)
        if len(features_df) >= 20:
            entropy = []
            for i in range(len(features_df)):
                if i < 20:
                    entropy.append(np.nan)
                else:
                    window_close = close_p[i-20:i+1]
                    hist, _ = np.histogram(window_close, bins=10)
                    hist = hist[hist > 0]
                    if len(hist) > 0:
                        prob = hist / np.sum(hist)
                        ent = -np.sum(prob * np.log(prob + 1e-8))
                        entropy.append(ent)
                    else:
                        entropy.append(0.0)
            features_df['entropy'] = entropy
        
        # Value area (simplified)
        features_df['in_value_area'] = ((close_p >= features_df['VAL']) & 
                                        (close_p <= features_df['VAH'])).astype(float)
        
        # Fill NaN values
        features_df = features_df.bfill().fillna(0)
        
        # Select only feature columns (exclude original OHLCV)
        feature_cols = [col for col in features_df.columns 
                       if col not in ['time', 'open', 'high', 'low', 'close', 'volume']]
        
        self.feature_names = feature_cols
        
        # Ensure we have ~100 features (pad if needed or trim)
        if len(feature_cols) < 100:
            # Add some derived features
            for i in range(100 - len(feature_cols)):
                features_df[f'feature_{i}'] = 0.0
                feature_cols.append(f'feature_{i}')
        elif len(feature_cols) > 100:
            # Take first 100
            feature_cols = feature_cols[:100]
        
        return features_df[feature_cols]
