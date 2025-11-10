"""
Natron Label Generator - Creates buy/sell/direction/regime labels
"""
import numpy as np
import pandas as pd
from typing import Tuple


class LabelGenerator:
    """Generates trading labels based on institutional/technical rules."""
    
    def __init__(self):
        self.regime_names = {
            0: 'BULL_STRONG',
            1: 'BULL_WEAK',
            2: 'RANGE',
            3: 'BEAR_WEAK',
            4: 'BEAR_STRONG',
            5: 'VOLATILE'
        }
    
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy, sell, direction, and regime labels.
        
        Args:
            df: DataFrame with OHLCV + features
            
        Returns:
            DataFrame with added label columns: buy, sell, direction, regime
        """
        labels_df = df.copy()
        
        # Generate buy/sell signals
        labels_df['buy'] = self._generate_buy_signals(df)
        labels_df['sell'] = self._generate_sell_signals(df)
        
        # Generate direction (0=down, 1=up)
        labels_df['direction'] = self._generate_direction(df)
        
        # Generate regime classes
        labels_df['regime'] = self._generate_regime(df)
        
        return labels_df
    
    def _generate_buy_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate BUY signals (1 if ≥2 conditions true)."""
        conditions = []
        
        # Condition 1: close > MA20 > MA50
        cond1 = (df['close'] > df['MA20']) & (df['MA20'] > df['MA50'])
        conditions.append(cond1)
        
        # Condition 2: RSI > 50 or recently left oversold
        rsi_oversold_exit = (df['RSI'] > 30) & (df['RSI'].shift(1) <= 30)
        cond2 = (df['RSI'] > 50) | rsi_oversold_exit
        conditions.append(cond2)
        
        # Condition 3: close > BB midband and MA20_slope > 0
        cond3 = (df['close'] > df['BB_mid']) & (df['MA20_slope'] > 0)
        conditions.append(cond3)
        
        # Condition 4: volume > 1.5 × rolling20
        cond4 = df['volume'] > 1.5 * df['volume_MA20']
        conditions.append(cond4)
        
        # Condition 5: close near high (≥70%)
        cond5 = df['position_in_range'] >= 0.7
        conditions.append(cond5)
        
        # Condition 6: MACD_hist > 0 and increasing
        cond6 = (df['MACD_hist'] > 0) & (df['MACD_hist_change'] > 0)
        conditions.append(cond6)
        
        # Count conditions met
        condition_count = sum(conditions)
        
        # BUY = 1 if ≥2 conditions true
        buy_signal = (condition_count >= 2).astype(int)
        
        return buy_signal.fillna(0)
    
    def _generate_sell_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate SELL signals (1 if ≥2 conditions true)."""
        conditions = []
        
        # Condition 1: close < MA20 < MA50
        cond1 = (df['close'] < df['MA20']) & (df['MA20'] < df['MA50'])
        conditions.append(cond1)
        
        # Condition 2: RSI < 50 or turning down from overbought
        rsi_overbought_exit = (df['RSI'] < 70) & (df['RSI'].shift(1) >= 70)
        rsi_turning_down = (df['RSI'] < df['RSI'].shift(1)) & (df['RSI'] > 70)
        cond2 = (df['RSI'] < 50) | rsi_overbought_exit | rsi_turning_down
        conditions.append(cond2)
        
        # Condition 3: close < BB midband and MA20_slope < 0
        cond3 = (df['close'] < df['BB_mid']) & (df['MA20_slope'] < 0)
        conditions.append(cond3)
        
        # Condition 4: volume > 1.5 × rolling20, close near low (≤30%)
        cond4 = (df['volume'] > 1.5 * df['volume_MA20']) & (df['position_in_range'] <= 0.3)
        conditions.append(cond4)
        
        # Condition 5: MACD_hist < 0 and decreasing
        cond5 = (df['MACD_hist'] < 0) & (df['MACD_hist_change'] < 0)
        conditions.append(cond5)
        
        # Count conditions met
        condition_count = sum(conditions)
        
        # SELL = 1 if ≥2 conditions true
        sell_signal = (condition_count >= 2).astype(int)
        
        return sell_signal.fillna(0)
    
    def _generate_direction(self, df: pd.DataFrame) -> pd.Series:
        """Generate direction labels (0=down, 1=up) based on future returns."""
        # Look ahead 5 periods
        future_return = df['close'].shift(-5) / df['close'] - 1
        
        # Direction: 1 if price goes up, 0 if down
        direction = (future_return > 0).astype(int)
        
        return direction.fillna(0)
    
    def _generate_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate regime classification (6 classes).
        
        0: BULL_STRONG (trend > +2%, ADX > 25)
        1: BULL_WEAK (0 < trend ≤ 2%, ADX ≤ 25)
        2: RANGE (lateral market)
        3: BEAR_WEAK (-2% ≤ trend < 0, ADX ≤ 25)
        4: BEAR_STRONG (trend < -2%, ADX > 25)
        5: VOLATILE (ATR > 90th percentile or volume spike)
        """
        # Calculate trend (20-period return)
        trend = df['return_20'] * 100  # Convert to percentage
        
        # Get ADX
        adx = df['ADX'].fillna(0)
        
        # Calculate ATR percentile
        atr_pct_rank = df['ATR_pct_rank'].fillna(0.5)
        
        # Volume spike
        volume_spike = df['volume_spike'].fillna(0)
        
        # Initialize regime array
        regime = np.zeros(len(df), dtype=int)
        
        # BULL_STRONG: trend > +2%, ADX > 25
        regime[(trend > 2) & (adx > 25)] = 0
        
        # BULL_WEAK: 0 < trend ≤ 2%, ADX ≤ 25
        regime[(trend > 0) & (trend <= 2) & (adx <= 25)] = 1
        
        # RANGE: lateral market (trend between -1% and +1%, low ADX)
        regime[(trend >= -1) & (trend <= 1) & (adx <= 20)] = 2
        
        # BEAR_WEAK: -2% ≤ trend < 0, ADX ≤ 25
        regime[(trend >= -2) & (trend < 0) & (adx <= 25)] = 3
        
        # BEAR_STRONG: trend < -2%, ADX > 25
        regime[(trend < -2) & (adx > 25)] = 4
        
        # VOLATILE: ATR > 90th percentile or volume spike
        regime[(atr_pct_rank > 0.9) | (volume_spike == 1)] = 5
        
        # Default to RANGE if no condition met
        regime[regime == 0] = 2
        
        return pd.Series(regime, index=df.index)
    
    def get_regime_name(self, regime_id: int) -> str:
        """Get regime name from ID."""
        return self.regime_names.get(regime_id, 'UNKNOWN')
