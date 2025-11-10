"""
LabelGenerator - Creates Buy/Sell/Direction/Regime Labels
Part of Natron Transformer Multi-Task Trading System
"""

import numpy as np
import pandas as pd
from typing import Tuple


class LabelGenerator:
    """
    Generates multi-task labels for trading signals:
    - Buy/Sell classification (binary)
    - Direction prediction (up/down)
    - Market regime classification (6 classes)
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.labeling_config = config.get('labeling', {})
        
    def generate_labels(self, df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all labels based on institutional/technical rules.
        
        Args:
            df: Original OHLCV DataFrame
            features_df: DataFrame with engineered features
            
        Returns:
            DataFrame with columns: ['buy', 'sell', 'direction', 'regime']
        """
        labels = pd.DataFrame(index=df.index)
        
        # Generate buy/sell signals
        buy, sell = self._generate_buy_sell_signals(df, features_df)
        labels['buy'] = buy
        labels['sell'] = sell
        
        # Generate direction labels
        direction = self._generate_direction_labels(df)
        labels['direction'] = direction
        
        # Generate regime labels
        regime = self._generate_regime_labels(df, features_df)
        labels['regime'] = regime
        
        return labels
    
    def _generate_buy_sell_signals(self, df: pd.DataFrame, features_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate buy and sell signals based on multiple conditions.
        Signal = 1 if >= 2 conditions are true.
        """
        close = df['close']
        volume = df['volume']
        
        # Extract features
        ma20 = features_df.get('MA20', pd.Series(index=df.index, dtype=float))
        ma50 = features_df.get('MA50', pd.Series(index=df.index, dtype=float))
        rsi = features_df.get('RSI', pd.Series(index=df.index, dtype=float)) * 100  # Denormalize
        bb_mid = features_df.get('BB_mid', pd.Series(index=df.index, dtype=float))
        ma20_slope = features_df.get('MA20_slope', pd.Series(index=df.index, dtype=float))
        volume_ratio20 = features_df.get('volume_ratio20', pd.Series(index=df.index, dtype=float))
        price_position = features_df.get('price_position', pd.Series(index=df.index, dtype=float))
        macd_hist = features_df.get('MACD_hist', pd.Series(index=df.index, dtype=float))
        
        # Volume rolling average
        volume_rolling20 = volume.rolling(20).mean()
        
        # BUY Conditions
        buy_conditions = []
        
        # Condition 1: close > MA20 > MA50
        cond1 = (close > ma20) & (ma20 > ma50)
        buy_conditions.append(cond1)
        
        # Condition 2: RSI > 50 or recently left oversold
        rsi_recently_oversold = (rsi.shift(1) < self.labeling_config.get('rsi_oversold', 30)) & (rsi >= 30)
        cond2 = (rsi > 50) | rsi_recently_oversold
        buy_conditions.append(cond2)
        
        # Condition 3: close > BB midband and MA20_slope > 0
        cond3 = (close > close * bb_mid) & (ma20_slope > 0)
        buy_conditions.append(cond3)
        
        # Condition 4: volume > 1.5x rolling20
        cond4 = volume > (self.labeling_config.get('volume_multiplier', 1.5) * volume_rolling20)
        buy_conditions.append(cond4)
        
        # Condition 5: close near high (>=70%)
        cond5 = price_position >= self.labeling_config.get('price_position_threshold', 0.7)
        buy_conditions.append(cond5)
        
        # Condition 6: MACD_hist > 0 and increasing
        cond6 = (macd_hist > 0) & (macd_hist > macd_hist.shift(1))
        buy_conditions.append(cond6)
        
        # Count conditions met
        buy_score = sum(buy_conditions)
        buy_signal = (buy_score >= 2).astype(int)
        
        # SELL Conditions
        sell_conditions = []
        
        # Condition 1: close < MA20 < MA50
        cond1_sell = (close < ma20) & (ma20 < ma50)
        sell_conditions.append(cond1_sell)
        
        # Condition 2: RSI < 50 or turning down from overbought
        rsi_turning_down = (rsi.shift(1) > self.labeling_config.get('rsi_overbought', 70)) & (rsi < rsi.shift(1))
        cond2_sell = (rsi < 50) | rsi_turning_down
        sell_conditions.append(cond2_sell)
        
        # Condition 3: close < BB midband and MA20_slope < 0
        cond3_sell = (close < close * bb_mid) & (ma20_slope < 0)
        sell_conditions.append(cond3_sell)
        
        # Condition 4: volume > 1.5x rolling20, close near low (<=30%)
        cond4_sell = (volume > (self.labeling_config.get('volume_multiplier', 1.5) * volume_rolling20)) & \
                     (price_position <= (1 - self.labeling_config.get('price_position_threshold', 0.7)))
        sell_conditions.append(cond4_sell)
        
        # Condition 5: MACD_hist < 0 and decreasing
        cond5_sell = (macd_hist < 0) & (macd_hist < macd_hist.shift(1))
        sell_conditions.append(cond5_sell)
        
        # Count conditions met
        sell_score = sum(sell_conditions)
        sell_signal = (sell_score >= 2).astype(int)
        
        # Ensure mutual exclusivity (can't buy and sell at same time)
        # If both are true, prioritize the stronger signal
        both_true = buy_signal & sell_signal
        buy_signal[both_true] = (buy_score[both_true] > sell_score[both_true]).astype(int)
        sell_signal[both_true] = (sell_score[both_true] > buy_score[both_true]).astype(int)
        
        return buy_signal, sell_signal
    
    def _generate_direction_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate direction labels: 0 = down, 1 = up
        Based on future price movement (next 5 candles)
        """
        close = df['close']
        
        # Look ahead 5 periods
        future_return = (close.shift(-5) / close - 1)
        
        # 0 = down, 1 = up
        direction = (future_return > 0).astype(int)
        
        # Fill NaN at end with last known value
        direction = direction.ffill().fillna(0)
        
        return direction
    
    def _generate_regime_labels(self, df: pd.DataFrame, features_df: pd.DataFrame) -> pd.Series:
        """
        Generate market regime labels (6 classes):
        0: BULL_STRONG (trend > +2%, ADX > 25)
        1: BULL_WEAK (0 < trend <= 2%, ADX <= 25)
        2: RANGE (lateral market)
        3: BEAR_WEAK (-2% <= trend < 0, ADX <= 25)
        4: BEAR_STRONG (trend < -2%, ADX > 25)
        5: VOLATILE (ATR > 90th percentile or volume spike)
        """
        close = df['close']
        volume = df['volume']
        
        # Extract features
        adx = features_df.get('ADX', pd.Series(index=df.index, dtype=float)) * 100  # Denormalize
        atr14 = features_df.get('ATR14', pd.Series(index=df.index, dtype=float))
        volume_ratio20 = features_df.get('volume_ratio20', pd.Series(index=df.index, dtype=float))
        
        # Calculate trend (20-period return)
        trend = close.pct_change(20) * 100  # Percentage
        
        # Calculate ATR percentile threshold
        atr_percentile = self.labeling_config.get('atr_percentile', 90)
        atr_threshold = atr14.rolling(100).quantile(atr_percentile / 100)
        
        # ADX threshold
        adx_threshold = self.labeling_config.get('adx_threshold', 25)
        trend_threshold = self.labeling_config.get('trend_threshold', 0.02) * 100  # Convert to percentage
        
        # Initialize regime labels
        regime = pd.Series(index=df.index, dtype=int)
        
        # VOLATILE (5) - check first as it can override
        is_volatile = (atr14 > atr_threshold) | (volume_ratio20 > 2.0)
        regime[is_volatile] = 5
        
        # BULL_STRONG (0)
        bull_strong = (trend > trend_threshold) & (adx > adx_threshold) & ~is_volatile
        regime[bull_strong] = 0
        
        # BULL_WEAK (1)
        bull_weak = (trend > 0) & (trend <= trend_threshold) & (adx <= adx_threshold) & ~is_volatile
        regime[bull_weak] = 1
        
        # BEAR_STRONG (4)
        bear_strong = (trend < -trend_threshold) & (adx > adx_threshold) & ~is_volatile
        regime[bear_strong] = 4
        
        # BEAR_WEAK (3)
        bear_weak = (trend < 0) & (trend >= -trend_threshold) & (adx <= adx_threshold) & ~is_volatile
        regime[bear_weak] = 3
        
        # RANGE (2) - default for remaining cases
        regime[regime.isna()] = 2
        
        # Fill any remaining NaN
        regime = regime.fillna(2).astype(int)
        
        return regime
    
    def get_regime_name(self, regime_id: int) -> str:
        """Convert regime ID to name"""
        regime_names = {
            0: "BULL_STRONG",
            1: "BULL_WEAK",
            2: "RANGE",
            3: "BEAR_WEAK",
            4: "BEAR_STRONG",
            5: "VOLATILE"
        }
        return regime_names.get(regime_id, "UNKNOWN")
