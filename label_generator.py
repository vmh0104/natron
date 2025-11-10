"""
LabelGenerator: Create buy/sell/direction/regime labels from technical features
"""
import numpy as np
import pandas as pd
from typing import Tuple


class LabelGenerator:
    """Generate trading labels based on institutional/technical rules"""
    
    def __init__(self):
        self.regime_names = [
            'BULL_STRONG', 'BULL_WEAK', 'RANGE', 
            'BEAR_WEAK', 'BEAR_STRONG', 'VOLATILE'
        ]
    
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all labels: buy, sell, direction, regime
        
        Args:
            df: DataFrame with OHLCV + technical features
            
        Returns:
            DataFrame with added label columns: buy, sell, direction, regime
        """
        labels_df = df.copy()
        
        # Generate buy/sell signals
        labels_df['buy'] = self._generate_buy_signals(df)
        labels_df['sell'] = self._generate_sell_signals(df)
        
        # Generate direction (0=down, 1=up)
        labels_df['direction'] = self._generate_direction(df)
        
        # Generate regime classification
        labels_df['regime'] = self._generate_regime(df)
        
        return labels_df
    
    def _generate_buy_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate BUY signals (1 if ≥2 conditions true)"""
        conditions = []
        
        # Condition 1: close > MA20 > MA50
        cond1 = (df['close'] > df['MA20']) & (df['MA20'] > df['MA50'])
        conditions.append(cond1)
        
        # Condition 2: RSI > 50 or recently left oversold (<30)
        rsi_oversold_recovery = (df['RSI'] < 30).rolling(5).max() & (df['RSI'] > 30)
        cond2 = (df['RSI'] > 50) | rsi_oversold_recovery
        conditions.append(cond2)
        
        # Condition 3: close > BB midband and MA20_slope > 0
        cond3 = (df['close'] > df['BB_mid']) & (df['MA20_slope'] > 0)
        conditions.append(cond3)
        
        # Condition 4: volume > 1.5 × rolling20
        cond4 = df['volume'] > (df['Volume_MA20'] * 1.5)
        conditions.append(cond4)
        
        # Condition 5: close near high (≥70%)
        cond5 = df['Price_position'] >= 0.7
        conditions.append(cond5)
        
        # Condition 6: MACD_hist > 0 and increasing
        cond6 = (df['MACD_hist'] > 0) & (df['MACD_hist_diff'] > 0)
        conditions.append(cond6)
        
        # Count conditions met
        condition_count = sum(conditions)
        buy_signal = (condition_count >= 2).astype(int)
        
        return buy_signal
    
    def _generate_sell_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate SELL signals (1 if ≥2 conditions true)"""
        conditions = []
        
        # Condition 1: close < MA20 < MA50
        cond1 = (df['close'] < df['MA20']) & (df['MA20'] < df['MA50'])
        conditions.append(cond1)
        
        # Condition 2: RSI < 50 or turning down from overbought (>70)
        rsi_overbought_turn = (df['RSI'] > 70).rolling(5).max() & (df['RSI'] < 70)
        cond2 = (df['RSI'] < 50) | rsi_overbought_turn
        conditions.append(cond2)
        
        # Condition 3: close < BB midband and MA20_slope < 0
        cond3 = (df['close'] < df['BB_mid']) & (df['MA20_slope'] < 0)
        conditions.append(cond3)
        
        # Condition 4: volume > 1.5 × rolling20, close near low (≤30%)
        cond4 = (df['volume'] > (df['Volume_MA20'] * 1.5)) & (df['Price_position'] <= 0.3)
        conditions.append(cond4)
        
        # Condition 5: MACD_hist < 0 and decreasing
        cond5 = (df['MACD_hist'] < 0) & (df['MACD_hist_diff'] < 0)
        conditions.append(cond5)
        
        # Count conditions met
        condition_count = sum(conditions)
        sell_signal = (condition_count >= 2).astype(int)
        
        return sell_signal
    
    def _generate_direction(self, df: pd.DataFrame) -> pd.Series:
        """Generate direction label (0=down, 1=up) based on future return"""
        # Use forward-looking return over next 5 periods
        future_return = df['close'].shift(-5) / df['close'] - 1
        direction = (future_return > 0).astype(int)
        
        # Fill last values with current trend
        direction = direction.fillna((df['close'] > df['close'].shift(1)).astype(int))
        
        return direction
    
    def _generate_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate regime classification (6 classes)
        
        0: BULL_STRONG (trend > +2%, ADX > 25)
        1: BULL_WEAK (0 < trend ≤ 2%, ADX ≤ 25)
        2: RANGE (lateral market)
        3: BEAR_WEAK (-2% ≤ trend < 0, ADX ≤ 25)
        4: BEAR_STRONG (trend < -2%, ADX > 25)
        5: VOLATILE (ATR > 90th percentile or volume spike)
        """
        # Calculate trend (20-period return %)
        trend = df['Return_1'].rolling(20).sum() * 100  # Convert to percentage
        
        # ADX threshold
        adx_threshold = 25
        
        # ATR percentile for volatility
        atr_90th = df['ATR_pct'].rolling(100).quantile(0.9)
        
        # Volume spike
        volume_spike = df['Volume_spike']
        
        # Initialize regime array
        regime = pd.Series(index=df.index, dtype=int)
        
        # BULL_STRONG (0)
        regime.loc[(trend > 2) & (df['ADX'] > adx_threshold)] = 0
        
        # BULL_WEAK (1)
        regime.loc[(trend > 0) & (trend <= 2) & (df['ADX'] <= adx_threshold)] = 1
        
        # RANGE (2) - lateral market (low trend, low ADX)
        regime.loc[(abs(trend) <= 1) & (df['ADX'] <= 20)] = 2
        
        # BEAR_WEAK (3)
        regime.loc[(trend >= -2) & (trend < 0) & (df['ADX'] <= adx_threshold)] = 3
        
        # BEAR_STRONG (4)
        regime.loc[(trend < -2) & (df['ADX'] > adx_threshold)] = 4
        
        # VOLATILE (5) - overrides other regimes
        regime.loc[(df['ATR_pct'] > atr_90th) | (volume_spike == 1)] = 5
        
        # Fill NaN with RANGE (default)
        regime = regime.fillna(2)
        
        return regime.astype(int)
    
    def get_regime_name(self, regime_id: int) -> str:
        """Get regime name from ID"""
        if 0 <= regime_id < len(self.regime_names):
            return self.regime_names[regime_id]
        return 'UNKNOWN'
