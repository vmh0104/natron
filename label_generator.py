"""
Label Generator - Institutional/Technical Rule-Based Labeling
Generates buy/sell signals, direction, and market regime labels.
"""

import numpy as np
import pandas as pd
from typing import Tuple


class LabelGenerator:
    """
    Generates multi-task labels for Natron Transformer:
    - Buy/Sell signals (binary)
    - Direction (up/down)
    - Market Regime (6 classes)
    """
    
    def __init__(self):
        pass
    
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all labels from feature-engineered DataFrame.
        
        Args:
            df: DataFrame with technical features
            
        Returns:
            DataFrame with added label columns: buy, sell, direction, regime
        """
        labels_df = df.copy()
        
        # Generate buy/sell signals
        labels_df['buy'] = self._generate_buy_signals(df)
        labels_df['sell'] = self._generate_sell_signals(df)
        
        # Generate direction labels
        labels_df['direction'] = self._generate_direction(df)
        
        # Generate regime labels
        labels_df['regime'] = self._generate_regime(df)
        
        return labels_df
    
    def _generate_buy_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        BUY signal: 1 if ≥2 conditions are true
        
        Conditions:
        1. close > MA20 > MA50
        2. RSI > 50 or recently left oversold (<30)
        3. close > BB midband and MA20_slope > 0
        4. volume > 1.5 × rolling20
        5. close near high (≥70%)
        6. MACD_hist > 0 and increasing
        """
        conditions = []
        
        # Condition 1: Trend alignment
        if 'MA20' in df.columns and 'MA50' in df.columns:
            cond1 = (df['close'] > df['MA20']) & (df['MA20'] > df['MA50'])
            conditions.append(cond1)
        
        # Condition 2: RSI momentum
        if 'RSI14' in df.columns:
            rsi_oversold_recovery = (df['RSI14'] < 30).rolling(5).max() & (df['RSI14'] > 30)
            cond2 = (df['RSI14'] > 50) | rsi_oversold_recovery
            conditions.append(cond2)
        
        # Condition 3: Bollinger Bands and trend
        if 'BB_midband20' in df.columns and 'MA20_slope' in df.columns:
            cond3 = (df['close'] > df['BB_midband20']) & (df['MA20_slope'] > 0)
            conditions.append(cond3)
        
        # Condition 4: Volume confirmation
        if 'volume_ratio' in df.columns:
            cond4 = df['volume_ratio'] > 1.5
            conditions.append(cond4)
        
        # Condition 5: Price position
        if 'position_in_range' in df.columns:
            cond5 = df['position_in_range'] >= 0.7
            conditions.append(cond5)
        
        # Condition 6: MACD momentum
        if 'MACD_hist' in df.columns and 'MACD_hist_change' in df.columns:
            cond6 = (df['MACD_hist'] > 0) & (df['MACD_hist_change'] > 0)
            conditions.append(cond6)
        
        # Buy signal: ≥2 conditions true
        if conditions:
            condition_count = sum(conditions)
            buy_signal = (condition_count >= 2).astype(int)
        else:
            buy_signal = pd.Series(0, index=df.index)
        
        return buy_signal
    
    def _generate_sell_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        SELL signal: 1 if ≥2 conditions are true
        
        Conditions:
        1. close < MA20 < MA50
        2. RSI < 50 or turning down from overbought (>70)
        3. close < BB midband and MA20_slope < 0
        4. volume > 1.5 × rolling20, close near low (≤30%)
        5. MACD_hist < 0 and decreasing
        """
        conditions = []
        
        # Condition 1: Trend alignment
        if 'MA20' in df.columns and 'MA50' in df.columns:
            cond1 = (df['close'] < df['MA20']) & (df['MA20'] < df['MA50'])
            conditions.append(cond1)
        
        # Condition 2: RSI momentum
        if 'RSI14' in df.columns:
            rsi_overbought_reversal = (df['RSI14'] > 70).rolling(5).max() & (df['RSI14'] < df['RSI14'].shift(1))
            cond2 = (df['RSI14'] < 50) | rsi_overbought_reversal
            conditions.append(cond2)
        
        # Condition 3: Bollinger Bands and trend
        if 'BB_midband20' in df.columns and 'MA20_slope' in df.columns:
            cond3 = (df['close'] < df['BB_midband20']) & (df['MA20_slope'] < 0)
            conditions.append(cond3)
        
        # Condition 4: Volume and price position
        if 'volume_ratio' in df.columns and 'position_in_range' in df.columns:
            cond4 = (df['volume_ratio'] > 1.5) & (df['position_in_range'] <= 0.3)
            conditions.append(cond4)
        
        # Condition 5: MACD momentum
        if 'MACD_hist' in df.columns and 'MACD_hist_change' in df.columns:
            cond5 = (df['MACD_hist'] < 0) & (df['MACD_hist_change'] < 0)
            conditions.append(cond5)
        
        # Sell signal: ≥2 conditions true
        if conditions:
            condition_count = sum(conditions)
            sell_signal = (condition_count >= 2).astype(int)
        else:
            sell_signal = pd.Series(0, index=df.index)
        
        return sell_signal
    
    def _generate_direction(self, df: pd.DataFrame, forward_period: int = 5) -> pd.Series:
        """
        Direction label: 1 for up, 0 for down
        Based on forward-looking price movement
        """
        if 'close' not in df.columns:
            return pd.Series(0, index=df.index)
        
        # Future return
        future_return = df['close'].shift(-forward_period) / df['close'] - 1
        
        # Direction: 1 if price goes up, 0 if down
        direction = (future_return > 0).astype(int)
        
        # Fill NaN at the end
        direction = direction.fillna(0)
        
        return direction
    
    def _generate_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Market Regime Classification (6 classes):
        0: BULL_STRONG - trend > +2%, ADX > 25
        1: BULL_WEAK - 0 < trend ≤ 2%, ADX ≤ 25
        2: RANGE - lateral market
        3: BEAR_WEAK - −2% ≤ trend < 0, ADX ≤ 25
        4: BEAR_STRONG - trend < −2%, ADX > 25
        5: VOLATILE - ATR > 90th percentile or volume spike
        """
        regime = pd.Series(2, index=df.index, dtype=int)  # Default: RANGE
        
        # Calculate trend (20-period return)
        if 'close' in df.columns:
            trend = df['close'].pct_change(20) * 100
        else:
            trend = pd.Series(0, index=df.index)
        
        # Get ADX
        if 'ADX' in df.columns:
            adx = df['ADX']
        else:
            adx = pd.Series(0, index=df.index)
        
        # Get ATR and volume metrics
        if 'ATR14_pct' in df.columns:
            atr_pct = df['ATR14_pct']
            atr_threshold = atr_pct.quantile(0.90)
        else:
            atr_pct = pd.Series(0, index=df.index)
            atr_threshold = 0
        
        if 'volume_ratio' in df.columns:
            volume_spike = df['volume_ratio'] > 2.0
        else:
            volume_spike = pd.Series(False, index=df.index)
        
        # Regime 5: VOLATILE (check first, highest priority)
        volatile_mask = (atr_pct > atr_threshold) | volume_spike
        regime[volatile_mask] = 5
        
        # Regime 0: BULL_STRONG
        bull_strong_mask = (trend > 2.0) & (adx > 25) & ~volatile_mask
        regime[bull_strong_mask] = 0
        
        # Regime 1: BULL_WEAK
        bull_weak_mask = (trend > 0) & (trend <= 2.0) & (adx <= 25) & ~volatile_mask
        regime[bull_weak_mask] = 1
        
        # Regime 3: BEAR_WEAK
        bear_weak_mask = (trend >= -2.0) & (trend < 0) & (adx <= 25) & ~volatile_mask
        regime[bear_weak_mask] = 3
        
        # Regime 4: BEAR_STRONG
        bear_strong_mask = (trend < -2.0) & (adx > 25) & ~volatile_mask
        regime[bear_strong_mask] = 4
        
        # Regime 2: RANGE (already default, but ensure it's set correctly)
        range_mask = (trend.abs() <= 0.5) & (adx <= 20) & ~volatile_mask
        regime[range_mask] = 2
        
        return regime
    
    def get_label_stats(self, labels_df: pd.DataFrame) -> dict:
        """Get statistics about generated labels"""
        stats = {
            'buy_rate': labels_df['buy'].mean(),
            'sell_rate': labels_df['sell'].mean(),
            'direction_up_rate': labels_df['direction'].mean(),
            'regime_distribution': labels_df['regime'].value_counts().to_dict()
        }
        return stats
