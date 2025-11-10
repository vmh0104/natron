"""
Label Generator - Institutional Trading Rules
Generates multi-task labels: Buy, Sell, Direction, Regime

Label Types:
- Buy/Sell: Binary classification (0/1) based on technical conditions
- Direction: Binary (0=down, 1=up)
- Regime: 6-class classification (BULL_STRONG, BULL_WEAK, RANGE, BEAR_WEAK, BEAR_STRONG, VOLATILE)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict


class LabelGenerator:
    """Generate trading labels using institutional/technical rules"""
    
    # Regime class mapping
    REGIME_MAP = {
        0: "BULL_STRONG",
        1: "BULL_WEAK",
        2: "RANGE",
        3: "BEAR_WEAK",
        4: "BEAR_STRONG",
        5: "VOLATILE"
    }
    
    def __init__(self, config: Dict = None):
        """
        Initialize label generator
        
        Args:
            config: Dictionary with labeling thresholds
        """
        self.config = config or self._default_config()
        
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'buy_min_conditions': 2,
            'sell_min_conditions': 2,
            'regime': {
                'bull_strong_trend': 0.02,
                'bear_strong_trend': -0.02,
                'adx_threshold': 25,
                'volatility_percentile': 90
            }
        }
    
    def generate_labels(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Generate all labels from feature DataFrame
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with added label columns: buy, sell, direction, regime
        """
        if verbose:
            print("🏷️  Generating labels...")
            
        df = df.copy()
        
        # Generate buy/sell signals
        df['buy'] = self._generate_buy_signals(df)
        df['sell'] = self._generate_sell_signals(df)
        
        # Generate direction (1 = up, 0 = down)
        df['direction'] = self._generate_direction(df)
        
        # Generate regime classification (0-5)
        df['regime'] = self._generate_regime(df)
        
        if verbose:
            print(f"✅ Buy signals: {df['buy'].sum()} ({df['buy'].mean()*100:.1f}%)")
            print(f"✅ Sell signals: {df['sell'].sum()} ({df['sell'].mean()*100:.1f}%)")
            print(f"✅ Direction up: {df['direction'].sum()} ({df['direction'].mean()*100:.1f}%)")
            print(f"✅ Regime distribution:")
            for regime_id, regime_name in self.REGIME_MAP.items():
                count = (df['regime'] == regime_id).sum()
                pct = count / len(df) * 100
                print(f"   {regime_name}: {count} ({pct:.1f}%)")
        
        return df
    
    def _generate_buy_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate BUY signals (need ≥2 conditions)
        
        BUY Conditions:
        1. close > MA20 > MA50
        2. RSI > 50 or recently left oversold (<30)
        3. close > BB midband and MA20_slope > 0
        4. volume > 1.5 × rolling20
        5. close near high (≥70%)
        6. MACD_hist > 0 and increasing
        """
        conditions = []
        min_conditions = self.config['buy_min_conditions']
        
        # Condition 1: Bullish MA alignment
        if 'MA_20' in df.columns and 'MA_50' in df.columns:
            cond1 = (df['close'] > df['MA_20']) & (df['MA_20'] > df['MA_50'])
            conditions.append(cond1)
        
        # Condition 2: RSI bullish
        if 'RSI_14' in df.columns:
            rsi_oversold_exit = (df['RSI_14'] > 30) & (df['RSI_14'].shift(1) <= 30)
            cond2 = (df['RSI_14'] > 50) | rsi_oversold_exit
            conditions.append(cond2)
        
        # Condition 3: Price above BB midband with positive MA slope
        if 'MA_20' in df.columns and 'MA20_slope' in df.columns:
            bb_mid = df['MA_20']  # BB midband is typically MA20
            cond3 = (df['close'] > bb_mid) & (df['MA20_slope'] > 0)
            conditions.append(cond3)
        
        # Condition 4: Volume spike
        if 'Volume_MA_20' in df.columns:
            cond4 = df['volume'] > (1.5 * df['Volume_MA_20'])
            conditions.append(cond4)
        
        # Condition 5: Close near high
        if 'close_position' in df.columns:
            cond5 = df['close_position'] >= 0.7
            conditions.append(cond5)
        
        # Condition 6: MACD bullish
        if 'MACD_hist' in df.columns:
            macd_increasing = df['MACD_hist'] > df['MACD_hist'].shift(1)
            cond6 = (df['MACD_hist'] > 0) & macd_increasing
            conditions.append(cond6)
        
        # Count conditions met
        if len(conditions) > 0:
            condition_count = sum(conditions)
            buy_signal = (condition_count >= min_conditions).astype(int)
        else:
            buy_signal = pd.Series(0, index=df.index)
        
        return buy_signal
    
    def _generate_sell_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate SELL signals (need ≥2 conditions)
        
        SELL Conditions:
        1. close < MA20 < MA50
        2. RSI < 50 or turning down from overbought (>70)
        3. close < BB midband and MA20_slope < 0
        4. volume > 1.5 × rolling20 and close near low (≤30%)
        5. MACD_hist < 0 and decreasing
        6. Bearish divergence or momentum weakening
        """
        conditions = []
        min_conditions = self.config['sell_min_conditions']
        
        # Condition 1: Bearish MA alignment
        if 'MA_20' in df.columns and 'MA_50' in df.columns:
            cond1 = (df['close'] < df['MA_20']) & (df['MA_20'] < df['MA_50'])
            conditions.append(cond1)
        
        # Condition 2: RSI bearish
        if 'RSI_14' in df.columns:
            rsi_overbought_exit = (df['RSI_14'] < 70) & (df['RSI_14'].shift(1) >= 70)
            cond2 = (df['RSI_14'] < 50) | rsi_overbought_exit
            conditions.append(cond2)
        
        # Condition 3: Price below BB midband with negative MA slope
        if 'MA_20' in df.columns and 'MA20_slope' in df.columns:
            bb_mid = df['MA_20']
            cond3 = (df['close'] < bb_mid) & (df['MA20_slope'] < 0)
            conditions.append(cond3)
        
        # Condition 4: Volume spike with close near low
        if 'Volume_MA_20' in df.columns and 'close_position' in df.columns:
            volume_spike = df['volume'] > (1.5 * df['Volume_MA_20'])
            close_near_low = df['close_position'] <= 0.3
            cond4 = volume_spike & close_near_low
            conditions.append(cond4)
        
        # Condition 5: MACD bearish
        if 'MACD_hist' in df.columns:
            macd_decreasing = df['MACD_hist'] < df['MACD_hist'].shift(1)
            cond5 = (df['MACD_hist'] < 0) & macd_decreasing
            conditions.append(cond5)
        
        # Condition 6: Momentum weakening
        if 'ROC_12' in df.columns:
            cond6 = (df['ROC_12'] < 0) & (df['ROC_12'] < df['ROC_12'].shift(5))
            conditions.append(cond6)
        
        # Count conditions met
        if len(conditions) > 0:
            condition_count = sum(conditions)
            sell_signal = (condition_count >= min_conditions).astype(int)
        else:
            sell_signal = pd.Series(0, index=df.index)
        
        return sell_signal
    
    def _generate_direction(self, df: pd.DataFrame, lookahead: int = 5) -> pd.Series:
        """
        Generate direction labels (1 = up, 0 = down)
        
        Args:
            df: DataFrame with close prices
            lookahead: Number of periods to look ahead
            
        Returns:
            Series of direction labels
        """
        # Future return
        future_return = df['close'].shift(-lookahead) / df['close'] - 1
        
        # 1 if price goes up, 0 if goes down
        direction = (future_return > 0).astype(int)
        
        # Fill last values with current trend
        direction = direction.fillna(method='ffill')
        
        return direction
    
    def _generate_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate market regime classification (6 classes)
        
        Regimes:
        0. BULL_STRONG: trend > +2%, ADX > 25
        1. BULL_WEAK: 0 < trend ≤ 2%, ADX ≤ 25
        2. RANGE: lateral market
        3. BEAR_WEAK: −2% ≤ trend < 0, ADX ≤ 25
        4. BEAR_STRONG: trend < −2%, ADX > 25
        5. VOLATILE: ATR > 90th percentile or volume spike
        """
        regime = pd.Series(2, index=df.index)  # Default to RANGE
        
        # Calculate trend (20-period slope)
        if 'return_20' in df.columns:
            trend = df['return_20']
        elif 'MA20_slope' in df.columns:
            trend = df['MA20_slope']
        else:
            trend = df['close'].pct_change(20)
        
        # Get ADX
        adx = df.get('ADX_14', pd.Series(20, index=df.index))
        
        # Get volatility measure
        if 'ATR_14' in df.columns:
            atr_percentile = df['ATR_14'].rolling(100).quantile(
                self.config['regime']['volatility_percentile'] / 100
            )
            high_volatility = df['ATR_14'] > atr_percentile
        else:
            high_volatility = pd.Series(False, index=df.index)
        
        # Volume spike
        if 'Volume_ratio' in df.columns:
            volume_spike = df['Volume_ratio'] > 2.0
        else:
            volume_spike = pd.Series(False, index=df.index)
        
        # Classify regimes
        bull_strong_threshold = self.config['regime']['bull_strong_trend']
        bear_strong_threshold = self.config['regime']['bear_strong_trend']
        adx_threshold = self.config['regime']['adx_threshold']
        
        # 5. VOLATILE (check first, highest priority)
        regime[high_volatility | volume_spike] = 5
        
        # 0. BULL_STRONG
        regime[(trend > bull_strong_threshold) & (adx > adx_threshold)] = 0
        
        # 1. BULL_WEAK
        regime[(trend > 0) & (trend <= bull_strong_threshold) & (adx <= adx_threshold)] = 1
        
        # 3. BEAR_WEAK
        regime[(trend < 0) & (trend >= bear_strong_threshold) & (adx <= adx_threshold)] = 3
        
        # 4. BEAR_STRONG
        regime[(trend < bear_strong_threshold) & (adx > adx_threshold)] = 4
        
        # 2. RANGE (everything else - already set as default)
        
        return regime.astype(int)
    
    def get_regime_name(self, regime_id: int) -> str:
        """Get regime name from ID"""
        return self.REGIME_MAP.get(regime_id, "UNKNOWN")
    
    def get_label_distribution(self, df: pd.DataFrame) -> Dict:
        """Get distribution statistics of all labels"""
        stats = {
            'buy': {
                'count': int(df['buy'].sum()),
                'percentage': float(df['buy'].mean() * 100)
            },
            'sell': {
                'count': int(df['sell'].sum()),
                'percentage': float(df['sell'].mean() * 100)
            },
            'direction': {
                'up_count': int(df['direction'].sum()),
                'up_percentage': float(df['direction'].mean() * 100)
            },
            'regime': {}
        }
        
        for regime_id, regime_name in self.REGIME_MAP.items():
            count = int((df['regime'] == regime_id).sum())
            pct = float(count / len(df) * 100)
            stats['regime'][regime_name] = {
                'id': regime_id,
                'count': count,
                'percentage': pct
            }
        
        return stats


if __name__ == "__main__":
    # Test label generation
    print("Testing Label Generator...")
    
    # Create sample feature data
    np.random.seed(42)
    n = 1000
    
    sample_df = pd.DataFrame({
        'close': 100 + np.random.randn(n).cumsum() * 0.5,
        'volume': np.random.randint(1000, 10000, n),
        'MA_20': 100 + np.random.randn(n).cumsum() * 0.4,
        'MA_50': 100 + np.random.randn(n).cumsum() * 0.3,
        'MA20_slope': np.random.randn(n) * 0.001,
        'RSI_14': np.random.uniform(30, 70, n),
        'MACD_hist': np.random.randn(n) * 0.1,
        'Volume_MA_20': np.random.randint(800, 8000, n),
        'close_position': np.random.uniform(0, 1, n),
        'ADX_14': np.random.uniform(10, 40, n),
        'ATR_14': np.random.uniform(0.5, 2.0, n),
        'Volume_ratio': np.random.uniform(0.5, 2.5, n),
        'return_20': np.random.randn(n) * 0.03,
        'ROC_12': np.random.randn(n) * 2
    })
    
    generator = LabelGenerator()
    labeled_df = generator.generate_labels(sample_df, verbose=True)
    
    print(f"\n✅ Labeled data shape: {labeled_df.shape}")
    print(f"✅ Label columns: buy, sell, direction, regime")
    
    # Get distribution
    stats = generator.get_label_distribution(labeled_df)
    print("\n📊 Label Distribution:")
    print(f"Buy: {stats['buy']['percentage']:.1f}%")
    print(f"Sell: {stats['sell']['percentage']:.1f}%")
    print(f"Direction Up: {stats['direction']['up_percentage']:.1f}%")
