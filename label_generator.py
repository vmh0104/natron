"""
Label Generator - Create buy/sell/direction/regime labels
"""
import numpy as np
import pandas as pd
from typing import Tuple


class LabelGenerator:
    """Generate trading labels based on institutional/technical rules"""
    
    def __init__(self):
        self.regime_names = [
            'BULL_STRONG',
            'BULL_WEAK',
            'RANGE',
            'BEAR_WEAK',
            'BEAR_STRONG',
            'VOLATILE'
        ]
    
    def generate_labels(self, df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy, sell, direction, and regime labels
        
        Args:
            df: Original OHLCV DataFrame
            features_df: DataFrame with technical features
            
        Returns:
            DataFrame with columns: buy, sell, direction, regime
        """
        labels = pd.DataFrame(index=df.index)
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # Get required features
        ma20 = features_df.get('MA20', pd.Series(np.nan, index=df.index))
        ma50 = features_df.get('MA50', pd.Series(np.nan, index=df.index))
        rsi14 = features_df.get('RSI14', pd.Series(50, index=df.index))
        bb_mid = features_df.get('BB_mid', pd.Series(close, index=df.index))
        ma20_slope = features_df.get('MA20_slope', pd.Series(0, index=df.index))
        macd_hist = features_df.get('MACD_hist', pd.Series(0, index=df.index))
        volume_ratio20 = features_df.get('volume_ratio20', pd.Series(1, index=df.index))
        price_position = features_df.get('price_position', pd.Series(0.5, index=df.index))
        
        # Calculate rolling volume average
        rolling_volume20 = pd.Series(volume).rolling(20).mean()
        
        # ========== BUY LABELS ==========
        buy_conditions = []
        
        # Condition 1: close > MA20 > MA50
        cond1 = (close > ma20) & (ma20 > ma50)
        buy_conditions.append(cond1)
        
        # Condition 2: RSI > 50 or recently left oversold
        rsi_oversold = (rsi14 < 30).rolling(5).max()  # Was oversold in last 5 periods
        cond2 = (rsi14 > 50) | rsi_oversold
        buy_conditions.append(cond2)
        
        # Condition 3: close > BB midband and MA20_slope > 0
        cond3 = (close > bb_mid) & (ma20_slope > 0)
        buy_conditions.append(cond3)
        
        # Condition 4: volume > 1.5 × rolling20
        cond4 = volume > (1.5 * rolling_volume20)
        buy_conditions.append(cond4)
        
        # Condition 5: close near high (≥70%)
        cond5 = price_position >= 0.7
        buy_conditions.append(cond5)
        
        # Condition 6: MACD_hist > 0 and increasing
        macd_increasing = macd_hist > pd.Series(macd_hist).shift(1)
        cond6 = (macd_hist > 0) & macd_increasing
        buy_conditions.append(cond6)
        
        # Buy if ≥2 conditions true
        buy_score = sum(buy_conditions)
        labels['buy'] = (buy_score >= 2).astype(int)
        
        # ========== SELL LABELS ==========
        sell_conditions = []
        
        # Condition 1: close < MA20 < MA50
        cond1_sell = (close < ma20) & (ma20 < ma50)
        sell_conditions.append(cond1_sell)
        
        # Condition 2: RSI < 50 or turning down from overbought
        rsi_overbought = (rsi14 > 70).rolling(5).max()
        rsi_turning_down = (rsi14 < pd.Series(rsi14).shift(1)) & (pd.Series(rsi14).shift(1) > 70)
        cond2_sell = (rsi14 < 50) | rsi_turning_down
        sell_conditions.append(cond2_sell)
        
        # Condition 3: close < BB midband and MA20_slope < 0
        cond3_sell = (close < bb_mid) & (ma20_slope < 0)
        sell_conditions.append(cond3_sell)
        
        # Condition 4: volume > 1.5 × rolling20, close near low (≤30%)
        cond4_sell = (volume > (1.5 * rolling_volume20)) & (price_position <= 0.3)
        sell_conditions.append(cond4_sell)
        
        # Condition 5: MACD_hist < 0 and decreasing
        macd_decreasing = macd_hist < pd.Series(macd_hist).shift(1)
        cond5_sell = (macd_hist < 0) & macd_decreasing
        sell_conditions.append(cond5_sell)
        
        # Sell if ≥2 conditions true
        sell_score = sum(sell_conditions)
        labels['sell'] = (sell_score >= 2).astype(int)
        
        # ========== DIRECTION LABELS ==========
        # Direction: 0 = down, 1 = up (based on future return)
        future_return = (pd.Series(close).shift(-5) - close) / (close + 1e-8)
        labels['direction'] = (future_return > 0).astype(int)
        
        # Fill NaN at end
        labels['direction'] = labels['direction'].ffill()
        
        # ========== REGIME LABELS ==========
        # Calculate trend (20-period return)
        trend_pct = ((close - pd.Series(close).shift(20)) / (pd.Series(close).shift(20) + 1e-8)) * 100
        trend_pct = trend_pct.fillna(0)
        
        # Get ADX
        adx = features_df.get('ADX', pd.Series(25, index=df.index))
        adx = adx.fillna(25)
        
        # Get ATR percentile
        atr14 = features_df.get('ATR14', pd.Series(0, index=df.index))
        atr_pct = (atr14 / (close + 1e-8)) * 100
        atr_pct = atr_pct.fillna(0)
        atr_90th = atr_pct.rolling(100).quantile(0.9).fillna(atr_pct.median())
        
        # Volume spike detection
        volume_spike = volume > (2.0 * rolling_volume20)
        
        # Regime classification
        regime = np.zeros(len(df), dtype=int)
        
        # BULL_STRONG: trend > +2%, ADX > 25
        regime[(trend_pct > 2) & (adx > 25)] = 0
        
        # BULL_WEAK: 0 < trend ≤ 2%, ADX ≤ 25
        regime[(trend_pct > 0) & (trend_pct <= 2) & (adx <= 25)] = 1
        
        # RANGE: lateral market (trend near 0, low ADX)
        regime[(np.abs(trend_pct) <= 1) & (adx <= 20)] = 2
        
        # BEAR_WEAK: -2% ≤ trend < 0, ADX ≤ 25
        regime[(trend_pct >= -2) & (trend_pct < 0) & (adx <= 25)] = 3
        
        # BEAR_STRONG: trend < -2%, ADX > 25
        regime[(trend_pct < -2) & (adx > 25)] = 4
        
        # VOLATILE: ATR > 90th percentile or volume spike
        regime[(atr_pct > atr_90th) | volume_spike] = 5
        
        labels['regime'] = regime
        
        return labels
