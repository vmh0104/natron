"""
Natron Label Generator - Creates buy/sell/direction/regime labels
"""
import numpy as np
import pandas as pd
from typing import Tuple


class LabelGenerator:
    """Generates trading labels based on institutional/technical rules"""
    
    def __init__(self):
        self.regime_names = [
            'BULL_STRONG', 'BULL_WEAK', 'RANGE', 
            'BEAR_WEAK', 'BEAR_STRONG', 'VOLATILE'
        ]
    
    def generate_labels(self, df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all labels: buy, sell, direction, regime
        Returns DataFrame with columns: buy, sell, direction, regime
        """
        labels = pd.DataFrame(index=df.index)
        
        # Generate buy/sell signals
        labels['buy'] = self._generate_buy_signals(df, features_df)
        labels['sell'] = self._generate_sell_signals(df, features_df)
        
        # Generate direction (0=down, 1=up)
        labels['direction'] = self._generate_direction(df)
        
        # Generate regime classes
        labels['regime'] = self._generate_regime(df, features_df)
        
        return labels
    
    def _generate_buy_signals(self, df: pd.DataFrame, features_df: pd.DataFrame) -> pd.Series:
        """Generate BUY signals (1 if ≥2 conditions true)"""
        buy_conditions = pd.DataFrame(index=df.index)
        
        # Condition 1: close > MA20 > MA50
        ma20 = features_df.get('MA20', df['close'].rolling(20).mean())
        ma50 = features_df.get('MA50', df['close'].rolling(50).mean())
        buy_conditions['cond1'] = ((df['close'] > ma20) & (ma20 > ma50)).astype(int)
        
        # Condition 2: RSI > 50 or recently left oversold (<30)
        rsi = features_df.get('RSI', pd.Series(50, index=df.index))
        rsi_oversold = (rsi < 30).rolling(5).max()  # Was oversold in last 5 periods
        buy_conditions['cond2'] = ((rsi > 50) | rsi_oversold).astype(int)
        
        # Condition 3: close > BB midband and MA20_slope > 0
        bb_mid = features_df.get('BB_mid', ma20)
        ma20_slope = features_df.get('MA20_slope', ma20.diff())
        buy_conditions['cond3'] = ((df['close'] > bb_mid) & (ma20_slope > 0)).astype(int)
        
        # Condition 4: volume > 1.5 × rolling20
        volume_ma20 = features_df.get('volume_MA20', df['volume'].rolling(20).mean())
        buy_conditions['cond4'] = (df['volume'] > 1.5 * volume_ma20).astype(int)
        
        # Condition 5: close near high (≥70%)
        position_in_range = features_df.get('position_in_range', 
            (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8))
        buy_conditions['cond5'] = (position_in_range >= 0.7).astype(int)
        
        # Condition 6: MACD_hist > 0 and increasing
        macd_hist = features_df.get('MACD_hist', pd.Series(0, index=df.index))
        macd_increasing = macd_hist > macd_hist.shift(1)
        buy_conditions['cond6'] = ((macd_hist > 0) & macd_increasing).astype(int)
        
        # BUY if ≥2 conditions true
        buy_signal = (buy_conditions.sum(axis=1) >= 2).astype(int)
        
        return buy_signal
    
    def _generate_sell_signals(self, df: pd.DataFrame, features_df: pd.DataFrame) -> pd.Series:
        """Generate SELL signals (1 if ≥2 conditions true)"""
        sell_conditions = pd.DataFrame(index=df.index)
        
        # Condition 1: close < MA20 < MA50
        ma20 = features_df.get('MA20', df['close'].rolling(20).mean())
        ma50 = features_df.get('MA50', df['close'].rolling(50).mean())
        sell_conditions['cond1'] = ((df['close'] < ma20) & (ma20 < ma50)).astype(int)
        
        # Condition 2: RSI < 50 or turning down from overbought (>70)
        rsi = features_df.get('RSI', pd.Series(50, index=df.index))
        rsi_overbought = (rsi > 70).rolling(5).max()
        rsi_turning_down = (rsi < rsi.shift(1)) & (rsi.shift(1) > 70)
        sell_conditions['cond2'] = ((rsi < 50) | rsi_overbought | rsi_turning_down).astype(int)
        
        # Condition 3: close < BB midband and MA20_slope < 0
        bb_mid = features_df.get('BB_mid', ma20)
        ma20_slope = features_df.get('MA20_slope', ma20.diff())
        sell_conditions['cond3'] = ((df['close'] < bb_mid) & (ma20_slope < 0)).astype(int)
        
        # Condition 4: volume > 1.5 × rolling20, close near low (≤30%)
        volume_ma20 = features_df.get('volume_MA20', df['volume'].rolling(20).mean())
        position_in_range = features_df.get('position_in_range',
            (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8))
        sell_conditions['cond4'] = ((df['volume'] > 1.5 * volume_ma20) & 
                                    (position_in_range <= 0.3)).astype(int)
        
        # Condition 5: MACD_hist < 0 and decreasing
        macd_hist = features_df.get('MACD_hist', pd.Series(0, index=df.index))
        macd_decreasing = macd_hist < macd_hist.shift(1)
        sell_conditions['cond5'] = ((macd_hist < 0) & macd_decreasing).astype(int)
        
        # SELL if ≥2 conditions true
        sell_signal = (sell_conditions.sum(axis=1) >= 2).astype(int)
        
        return sell_signal
    
    def _generate_direction(self, df: pd.DataFrame, lookforward: int = 5) -> pd.Series:
        """
        Generate direction label (0=down, 1=up)
        Based on future price movement
        """
        future_return = (df['close'].shift(-lookforward) - df['close']) / df['close']
        direction = (future_return > 0).astype(int)
        return direction.fillna(0)
    
    def _generate_regime(self, df: pd.DataFrame, features_df: pd.DataFrame) -> pd.Series:
        """
        Generate regime classification (6 classes)
        0: BULL_STRONG, 1: BULL_WEAK, 2: RANGE, 
        3: BEAR_WEAK, 4: BEAR_STRONG, 5: VOLATILE
        """
        regime = pd.Series(2, index=df.index, dtype=int)  # Default: RANGE
        
        # Calculate trend (20-period return)
        trend = df['close'].pct_change(20) * 100
        
        # ADX
        adx = features_df.get('ADX', pd.Series(0, index=df.index))
        
        # ATR for volatility
        atr = features_df.get('ATR_14', pd.Series(0, index=df.index))
        atr_percentile = atr.rolling(100).quantile(0.9)
        
        # Volume spike
        volume_ma = features_df.get('volume_MA20', df['volume'].rolling(20).mean())
        volume_spike = df['volume'] > 2.0 * volume_ma
        
        # Regime 5: VOLATILE (highest priority)
        volatile_mask = (atr > atr_percentile) | volume_spike
        regime[volatile_mask] = 5
        
        # Regime 0: BULL_STRONG (trend > +2%, ADX > 25)
        bull_strong_mask = (trend > 2) & (adx > 25) & ~volatile_mask
        regime[bull_strong_mask] = 0
        
        # Regime 1: BULL_WEAK (0 < trend ≤ 2%, ADX ≤ 25)
        bull_weak_mask = (trend > 0) & (trend <= 2) & (adx <= 25) & ~volatile_mask
        regime[bull_weak_mask] = 1
        
        # Regime 3: BEAR_WEAK (-2% ≤ trend < 0, ADX ≤ 25)
        bear_weak_mask = (trend >= -2) & (trend < 0) & (adx <= 25) & ~volatile_mask
        regime[bear_weak_mask] = 3
        
        # Regime 4: BEAR_STRONG (trend < -2%, ADX > 25)
        bear_strong_mask = (trend < -2) & (adx > 25) & ~volatile_mask
        regime[bear_strong_mask] = 4
        
        # Regime 2: RANGE (lateral market) - already default
        
        return regime.fillna(2)
