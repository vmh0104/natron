"""
Natron LabelGenerator - Generate Buy/Sell, Direction, and Regime Labels

Labeling Rules:
- BUY: ≥2 conditions true from institutional/technical rules
- SELL: ≥2 conditions true from institutional/technical rules
- Direction: Up (1) or Down (0) based on future price movement
- Regime: 6 classes (BULL_STRONG, BULL_WEAK, RANGE, BEAR_WEAK, BEAR_STRONG, VOLATILE)
"""

import numpy as np
import pandas as pd
from typing import Tuple


class LabelGenerator:
    """Generate trading labels from OHLCV and feature data."""
    
    def __init__(self, lookahead_periods: int = 5):
        """
        Args:
            lookahead_periods: Number of periods ahead to check for direction
        """
        self.lookahead_periods = lookahead_periods
        self.regime_names = [
            'BULL_STRONG', 'BULL_WEAK', 'RANGE', 
            'BEAR_WEAK', 'BEAR_STRONG', 'VOLATILE'
        ]
    
    def generate_labels(
        self, 
        df: pd.DataFrame, 
        features_df: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Generate all labels: buy, sell, direction, regime.
        
        Args:
            df: OHLCV DataFrame
            features_df: Feature DataFrame from FeatureEngine
        
        Returns:
            Tuple of (y_buy, y_sell, y_direction, y_regime) Series
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        n = len(df)
        
        # Extract required features
        ma20 = features_df['ma20'].values
        ma50 = features_df['ma50'].values
        rsi = features_df['rsi'].values
        bb_mid = features_df['bb_mid'].values
        ma20_slope = features_df['ma20_slope'].values
        volume_ma20 = features_df['volume_ma20'].values
        macd_hist = features_df['macd_hist'].values
        atr = features_df['atr'].values
        
        # Initialize label arrays
        y_buy = np.zeros(n, dtype=int)
        y_sell = np.zeros(n, dtype=int)
        y_direction = np.zeros(n, dtype=int)
        y_regime = np.zeros(n, dtype=int)
        
        # Calculate rolling statistics for regime classification
        returns_20 = features_df['return_1'].rolling(20).sum().values
        adx = features_df['adx'].values
        atr_pct = features_df['atr_pct'].values
        volume_ratio = features_df['volume_ratio'].values
        
        # Calculate percentiles for volatility detection
        atr_pct_90 = pd.Series(atr_pct).rolling(100).quantile(0.90).values
        
        for i in range(20, n - self.lookahead_periods):  # Start from 20 to have enough history
            # === BUY CONDITIONS ===
            buy_conditions = 0
            
            # Condition 1: close > MA20 > MA50
            if not np.isnan(ma20[i]) and not np.isnan(ma50[i]):
                if close[i] > ma20[i] > ma50[i]:
                    buy_conditions += 1
            
            # Condition 2: RSI > 50 or recently left oversold (<30)
            if not np.isnan(rsi[i]):
                if rsi[i] > 50 or (i > 5 and rsi[i-5] < 30 and rsi[i] > rsi[i-5]):
                    buy_conditions += 1
            
            # Condition 3: close > BB midband and MA20_slope > 0
            if not np.isnan(bb_mid[i]) and not np.isnan(ma20_slope[i]):
                if close[i] > bb_mid[i] and ma20_slope[i] > 0:
                    buy_conditions += 1
            
            # Condition 4: volume > 1.5 × rolling20
            if not np.isnan(volume_ma20[i]):
                if volume[i] > 1.5 * volume_ma20[i]:
                    buy_conditions += 1
            
            # Condition 5: close near high (≥70%)
            price_position = (close[i] - low[i]) / (high[i] - low[i] + 1e-8)
            if price_position >= 0.7:
                buy_conditions += 1
            
            # Condition 6: MACD_hist > 0 and increasing
            if not np.isnan(macd_hist[i]) and i > 0:
                if macd_hist[i] > 0 and macd_hist[i] > macd_hist[i-1]:
                    buy_conditions += 1
            
            if buy_conditions >= 2:
                y_buy[i] = 1
            
            # === SELL CONDITIONS ===
            sell_conditions = 0
            
            # Condition 1: close < MA20 < MA50
            if not np.isnan(ma20[i]) and not np.isnan(ma50[i]):
                if close[i] < ma20[i] < ma50[i]:
                    sell_conditions += 1
            
            # Condition 2: RSI < 50 or turning down from overbought (>70)
            if not np.isnan(rsi[i]):
                if rsi[i] < 50 or (i > 5 and rsi[i-5] > 70 and rsi[i] < rsi[i-5]):
                    sell_conditions += 1
            
            # Condition 3: close < BB midband and MA20_slope < 0
            if not np.isnan(bb_mid[i]) and not np.isnan(ma20_slope[i]):
                if close[i] < bb_mid[i] and ma20_slope[i] < 0:
                    sell_conditions += 1
            
            # Condition 4: volume > 1.5 × rolling20, close near low (≤30%)
            if not np.isnan(volume_ma20[i]):
                if volume[i] > 1.5 * volume_ma20[i] and price_position <= 0.3:
                    sell_conditions += 1
            
            # Condition 5: MACD_hist < 0 and decreasing
            if not np.isnan(macd_hist[i]) and i > 0:
                if macd_hist[i] < 0 and macd_hist[i] < macd_hist[i-1]:
                    sell_conditions += 1
            
            if sell_conditions >= 2:
                y_sell[i] = 1
            
            # === DIRECTION LABEL ===
            # Check if price goes up in next lookahead_periods
            future_return = (close[i + self.lookahead_periods] - close[i]) / close[i]
            y_direction[i] = 1 if future_return > 0 else 0
            
            # === REGIME CLASSIFICATION ===
            if not np.isnan(returns_20[i]) and not np.isnan(adx[i]):
                trend_pct = returns_20[i] * 100  # Convert to percentage
                
                # Check for volatility first
                if not np.isnan(atr_pct_90[i]) and (atr_pct[i] > atr_pct_90[i] or volume_ratio[i] > 2.0):
                    y_regime[i] = 5  # VOLATILE
                elif trend_pct > 2.0 and adx[i] > 25:
                    y_regime[i] = 0  # BULL_STRONG
                elif trend_pct > 0 and trend_pct <= 2.0 and adx[i] <= 25:
                    y_regime[i] = 1  # BULL_WEAK
                elif trend_pct >= -2.0 and trend_pct < 0 and adx[i] <= 25:
                    y_regime[i] = 3  # BEAR_WEAK
                elif trend_pct < -2.0 and adx[i] > 25:
                    y_regime[i] = 4  # BEAR_STRONG
                else:
                    y_regime[i] = 2  # RANGE
            else:
                y_regime[i] = 2  # Default to RANGE
        
        # Convert to Series
        y_buy = pd.Series(y_buy, index=df.index, name='buy')
        y_sell = pd.Series(y_sell, index=df.index, name='sell')
        y_direction = pd.Series(y_direction, index=df.index, name='direction')
        y_regime = pd.Series(y_regime, index=df.index, name='regime')
        
        return y_buy, y_sell, y_direction, y_regime
    
    def get_regime_name(self, regime_id: int) -> str:
        """Get regime name from ID."""
        if 0 <= regime_id < len(self.regime_names):
            return self.regime_names[regime_id]
        return 'UNKNOWN'
