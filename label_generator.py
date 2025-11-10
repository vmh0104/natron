"""
Natron Label Generator
Creates Buy/Sell signals, Directional labels, and Market Regime classifications.
"""

import numpy as np
import pandas as pd
from typing import Tuple


class LabelGenerator:
    """
    Generates trading labels based on institutional/technical rules.
    """
    
    def __init__(self):
        pass
    
    def generate_all_labels(self, df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all label types: buy, sell, direction, regime.
        
        Args:
            df: Original OHLCV DataFrame
            features_df: DataFrame with technical features
        
        Returns:
            DataFrame with label columns: buy, sell, direction, regime
        """
        labels_df = pd.DataFrame(index=df.index)
        
        # Buy/Sell signals
        labels_df['buy'] = self._generate_buy_signals(df, features_df)
        labels_df['sell'] = self._generate_sell_signals(df, features_df)
        
        # Directional prediction (1 = up, 0 = down)
        labels_df['direction'] = self._generate_direction(df)
        
        # Market regime (0-5)
        labels_df['regime'] = self._generate_regime(df, features_df)
        
        return labels_df
    
    def _generate_buy_signals(self, df: pd.DataFrame, features_df: pd.DataFrame) -> pd.Series:
        """
        BUY signal (1) if ≥2 conditions are true:
        - close > MA20 > MA50
        - RSI > 50 or recently left oversold (<30)
        - close > BB midband and MA20_slope > 0
        - volume > 1.5 × rolling20
        - close near high (≥70%)
        - MACD_hist > 0 and increasing
        """
        close = df['close']
        buy_score = pd.Series(0, index=df.index)
        
        # Condition 1: Trend alignment
        ma20 = features_df.get('ma20', pd.Series(0, index=df.index))
        ma50 = features_df.get('ma50', pd.Series(0, index=df.index))
        cond1 = (close > ma20) & (ma20 > ma50)
        
        # Condition 2: RSI momentum
        rsi = features_df.get('rsi', pd.Series(50, index=df.index))
        rsi_oversold = (rsi < 30).rolling(5).max()  # Recently oversold
        cond2 = (rsi > 50) | (rsi_oversold == 1)
        
        # Condition 3: Bollinger + MA slope
        bb_mid = features_df.get('bb_mid', close)
        ma20_slope = features_df.get('ma20_slope', pd.Series(0, index=df.index))
        cond3 = (close > bb_mid) & (ma20_slope > 0)
        
        # Condition 4: Volume confirmation
        volume = df['volume']
        volume_ma20 = features_df.get('volume_ma20', volume)
        cond4 = volume > (1.5 * volume_ma20)
        
        # Condition 5: Price position
        position_in_range = features_df.get('position_in_range', pd.Series(0.5, index=df.index))
        cond5 = position_in_range >= 0.7
        
        # Condition 6: MACD momentum
        macd_hist = features_df.get('macd_hist', pd.Series(0, index=df.index))
        macd_hist_change = features_df.get('macd_hist_change', pd.Series(0, index=df.index))
        cond6 = (macd_hist > 0) & (macd_hist_change > 0)
        
        # Count conditions
        conditions = pd.concat([cond1, cond2, cond3, cond4, cond5, cond6], axis=1)
        buy_score = (conditions.sum(axis=1) >= 2).astype(int)
        
        return buy_score
    
    def _generate_sell_signals(self, df: pd.DataFrame, features_df: pd.DataFrame) -> pd.Series:
        """
        SELL signal (1) if ≥2 conditions are true:
        - close < MA20 < MA50
        - RSI < 50 or turning down from overbought (>70)
        - close < BB midband and MA20_slope < 0
        - volume > 1.5 × rolling20, close near low (≤30%)
        - MACD_hist < 0 and decreasing
        """
        close = df['close']
        sell_score = pd.Series(0, index=df.index)
        
        # Condition 1: Trend alignment
        ma20 = features_df.get('ma20', pd.Series(0, index=df.index))
        ma50 = features_df.get('ma50', pd.Series(0, index=df.index))
        cond1 = (close < ma20) & (ma20 < ma50)
        
        # Condition 2: RSI momentum
        rsi = features_df.get('rsi', pd.Series(50, index=df.index))
        rsi_overbought = (rsi > 70).rolling(5).max()  # Recently overbought
        rsi_turning_down = (rsi < rsi.shift(1)) if 'rsi' in features_df.columns else pd.Series(False, index=df.index)
        cond2 = (rsi < 50) | ((rsi_overbought == 1) & rsi_turning_down)
        
        # Condition 3: Bollinger + MA slope
        bb_mid = features_df.get('bb_mid', close)
        ma20_slope = features_df.get('ma20_slope', pd.Series(0, index=df.index))
        cond3 = (close < bb_mid) & (ma20_slope < 0)
        
        # Condition 4: Volume + price position
        volume = df['volume']
        volume_ma20 = features_df.get('volume_ma20', volume)
        position_in_range = features_df.get('position_in_range', pd.Series(0.5, index=df.index))
        cond4 = (volume > 1.5 * volume_ma20) & (position_in_range <= 0.3)
        
        # Condition 5: MACD momentum
        macd_hist = features_df.get('macd_hist', pd.Series(0, index=df.index))
        macd_hist_change = features_df.get('macd_hist_change', pd.Series(0, index=df.index))
        cond5 = (macd_hist < 0) & (macd_hist_change < 0)
        
        # Count conditions
        conditions = pd.concat([cond1, cond2, cond3, cond4, cond5], axis=1)
        sell_score = (conditions.sum(axis=1) >= 2).astype(int)
        
        return sell_score
    
    def _generate_direction(self, df: pd.DataFrame, horizon: int = 5) -> pd.Series:
        """
        Directional label: 1 if price goes up in next N candles, 0 otherwise.
        """
        close = df['close']
        future_close = close.shift(-horizon)
        direction = (future_close > close).astype(int)
        
        # Fill NaN with 0 (last N rows)
        direction = direction.fillna(0)
        
        return direction
    
    def _generate_regime(self, df: pd.DataFrame, features_df: pd.DataFrame) -> pd.Series:
        """
        Market Regime Classification (6 classes):
        0: BULL_STRONG - trend > +2%, ADX > 25
        1: BULL_WEAK - 0 < trend ≤ 2%, ADX ≤ 25
        2: RANGE - lateral market
        3: BEAR_WEAK - −2% ≤ trend < 0, ADX ≤ 25
        4: BEAR_STRONG - trend < −2%, ADX > 25
        5: VOLATILE - ATR > 90th percentile or volume spike
        """
        close = df['close']
        regime = pd.Series(2, index=df.index, dtype=int)  # Default: RANGE
        
        # Calculate trend (20-period return)
        trend_pct = close.pct_change(20) * 100
        
        # ADX
        adx = features_df.get('adx', pd.Series(25, index=df.index))
        
        # ATR percentile
        atr14 = features_df.get('atr14', pd.Series(0, index=df.index))
        atr_pct90 = atr14.rolling(100).quantile(0.9)
        
        # Volume spike
        volume = df['volume']
        volume_ma20 = features_df.get('volume_ma20', volume)
        volume_spike = volume > (2.0 * volume_ma20)
        
        # Regime classification
        # VOLATILE (5) - highest priority
        volatile_mask = (atr14 > atr_pct90) | (volume_spike == 1)
        regime[volatile_mask] = 5
        
        # BULL_STRONG (0)
        bull_strong_mask = (trend_pct > 2) & (adx > 25) & (~volatile_mask)
        regime[bull_strong_mask] = 0
        
        # BULL_WEAK (1)
        bull_weak_mask = (trend_pct > 0) & (trend_pct <= 2) & (adx <= 25) & (~volatile_mask)
        regime[bull_weak_mask] = 1
        
        # BEAR_WEAK (3)
        bear_weak_mask = (trend_pct >= -2) & (trend_pct < 0) & (adx <= 25) & (~volatile_mask)
        regime[bear_weak_mask] = 3
        
        # BEAR_STRONG (4)
        bear_strong_mask = (trend_pct < -2) & (adx > 25) & (~volatile_mask)
        regime[bear_strong_mask] = 4
        
        # RANGE (2) - default, already set
        
        return regime
