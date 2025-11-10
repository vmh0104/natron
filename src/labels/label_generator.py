"""
Natron V2 Label Generation Module
Generates trading signals based on institutional/technical rules
"""

import numpy as np
import pandas as pd
from typing import Tuple


class LabelGenerator:
    """
    Generates multi-task labels for trading:
    - Buy signals (binary)
    - Sell signals (binary)
    - Direction (0=down, 1=up)
    - Market Regime (6 classes)
    """
    
    REGIME_BULL_STRONG = 0
    REGIME_BULL_WEAK = 1
    REGIME_RANGE = 2
    REGIME_BEAR_WEAK = 3
    REGIME_BEAR_STRONG = 4
    REGIME_VOLATILE = 5
    
    def __init__(self):
        self.regime_names = {
            0: "BULL_STRONG",
            1: "BULL_WEAK",
            2: "RANGE",
            3: "BEAR_WEAK",
            4: "BEAR_STRONG",
            5: "VOLATILE"
        }
    
    def generate_labels(self, features_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Generate all labels from features.
        
        Args:
            features_df: DataFrame with features (must include required indicators)
            
        Returns:
            Tuple of (buy_labels, sell_labels, direction_labels, regime_labels)
        """
        buy_labels = self._generate_buy_signals(features_df)
        sell_labels = self._generate_sell_signals(features_df)
        direction_labels = self._generate_direction(features_df)
        regime_labels = self._generate_regime(features_df)
        
        return buy_labels, sell_labels, direction_labels, regime_labels
    
    def _generate_buy_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate BUY signals based on institutional rules.
        BUY = 1 if >= 2 conditions are true:
        1. close > MA20 > MA50
        2. RSI > 50 or recently left oversold (<30)
        3. close > BB midband and MA20_slope > 0
        4. volume > 1.5 × rolling20
        5. close near high (>=70%)
        6. MACD_hist > 0 and increasing
        """
        close = df['close']
        
        # Condition 1: Bullish MA alignment
        cond1 = (close > df['ma_20']) & (df['ma_20'] > df['ma_50'])
        
        # Condition 2: RSI momentum
        rsi_oversold_exit = (df['rsi_14'] > 30) & (df['rsi_14'].shift(1) <= 30)
        cond2 = (df['rsi_14'] > 50) | rsi_oversold_exit
        
        # Condition 3: Above BB mid with positive MA slope
        cond3 = (close > df['bb_mid_20']) & (df['ma_20_slope'] > 0)
        
        # Condition 4: Volume surge
        cond4 = df['volume'] > (1.5 * df['volume_ma_20'])
        
        # Condition 5: Close near high
        cond5 = df['close_position'] >= 0.7
        
        # Condition 6: MACD bullish and increasing
        cond6 = (df['macd_hist'] > 0) & (df['macd_hist_change'] > 0)
        
        # Count conditions
        condition_sum = cond1.astype(int) + cond2.astype(int) + cond3.astype(int) + \
                       cond4.astype(int) + cond5.astype(int) + cond6.astype(int)
        
        buy_signal = (condition_sum >= 2).astype(int)
        
        return buy_signal
    
    def _generate_sell_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate SELL signals based on institutional rules.
        SELL = 1 if >= 2 conditions are true:
        1. close < MA20 < MA50
        2. RSI < 50 or turning down from overbought (>70)
        3. close < BB midband and MA20_slope < 0
        4. volume > 1.5 × rolling20, close near low (<=30%)
        5. MACD_hist < 0 and decreasing
        """
        close = df['close']
        
        # Condition 1: Bearish MA alignment
        cond1 = (close < df['ma_20']) & (df['ma_20'] < df['ma_50'])
        
        # Condition 2: RSI weakness
        rsi_overbought_exit = (df['rsi_14'] < 70) & (df['rsi_14'].shift(1) >= 70)
        cond2 = (df['rsi_14'] < 50) | rsi_overbought_exit
        
        # Condition 3: Below BB mid with negative MA slope
        cond3 = (close < df['bb_mid_20']) & (df['ma_20_slope'] < 0)
        
        # Condition 4: Volume surge with close near low
        cond4 = (df['volume'] > (1.5 * df['volume_ma_20'])) & (df['close_position'] <= 0.3)
        
        # Condition 5: MACD bearish and decreasing
        cond5 = (df['macd_hist'] < 0) & (df['macd_hist_change'] < 0)
        
        # Count conditions
        condition_sum = cond1.astype(int) + cond2.astype(int) + cond3.astype(int) + \
                       cond4.astype(int) + cond5.astype(int)
        
        sell_signal = (condition_sum >= 2).astype(int)
        
        return sell_signal
    
    def _generate_direction(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate direction labels (future price movement).
        0 = down, 1 = up
        Based on next candle's close relative to current close
        """
        close = df['close']
        future_close = close.shift(-1)
        
        # Direction: 1 if price goes up, 0 if down
        direction = (future_close > close).astype(int)
        
        # For the last row, use current trend
        direction.iloc[-1] = (close.iloc[-1] > close.iloc[-2])
        
        return direction
    
    def _generate_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate market regime classification (6 classes).
        
        Regimes:
        0: BULL_STRONG - trend > +2%, ADX > 25
        1: BULL_WEAK - 0 < trend <= 2%, ADX <= 25
        2: RANGE - lateral market
        3: BEAR_WEAK - -2% <= trend < 0, ADX <= 25
        4: BEAR_STRONG - trend < -2%, ADX > 25
        5: VOLATILE - ATR > 90th percentile or volume spike
        """
        close = df['close']
        
        # Calculate trend (% change over 20 periods)
        trend = ((close - close.shift(20)) / close.shift(20)) * 100
        trend = trend.fillna(0)
        
        # Get ADX
        adx = df['adx_14']
        
        # Get ATR percentile
        atr_percentile_90 = df['atr_14'].rolling(100).quantile(0.9)
        
        # Volume spike detection
        volume_spike = df['volume'] > (2.0 * df['volume_ma_20'])
        
        # Initialize regime array
        regime = pd.Series(self.REGIME_RANGE, index=df.index)
        
        # Classify regimes (order matters - most specific first)
        
        # VOLATILE (highest priority)
        volatile_mask = (df['atr_14'] > atr_percentile_90) | volume_spike
        regime[volatile_mask] = self.REGIME_VOLATILE
        
        # BULL_STRONG
        bull_strong_mask = (trend > 2.0) & (adx > 25) & (~volatile_mask)
        regime[bull_strong_mask] = self.REGIME_BULL_STRONG
        
        # BULL_WEAK
        bull_weak_mask = (trend > 0) & (trend <= 2.0) & (~volatile_mask)
        regime[bull_weak_mask] = self.REGIME_BULL_WEAK
        
        # BEAR_STRONG
        bear_strong_mask = (trend < -2.0) & (adx > 25) & (~volatile_mask)
        regime[bear_strong_mask] = self.REGIME_BEAR_STRONG
        
        # BEAR_WEAK
        bear_weak_mask = (trend < 0) & (trend >= -2.0) & (~volatile_mask)
        regime[bear_weak_mask] = self.REGIME_BEAR_WEAK
        
        # RANGE (default, already initialized)
        range_mask = (abs(trend) < 0.5) & (adx < 20) & (~volatile_mask)
        regime[range_mask] = self.REGIME_RANGE
        
        return regime.astype(int)
    
    def get_regime_name(self, regime_id: int) -> str:
        """Convert regime ID to name"""
        return self.regime_names.get(regime_id, "UNKNOWN")
    
    def get_regime_distribution(self, regime_labels: pd.Series) -> dict:
        """Get distribution of regimes in dataset"""
        distribution = {}
        for regime_id in range(6):
            count = (regime_labels == regime_id).sum()
            percentage = (count / len(regime_labels)) * 100
            distribution[self.get_regime_name(regime_id)] = {
                'count': int(count),
                'percentage': float(percentage)
            }
        return distribution
