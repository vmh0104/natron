"""
Natron Label Generator - Institutional Trading Labels
Author: Natron AI System
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict


class LabelGenerator:
    """
    Generates trading labels based on institutional/technical rules:
    - Buy/Sell signals
    - Direction (up/down)
    - Market Regime (6 classes)
    """
    
    def __init__(self, 
                 buy_threshold: int = 2, 
                 sell_threshold: int = 2,
                 verbose: bool = False):
        """
        Args:
            buy_threshold: Minimum conditions for BUY signal
            sell_threshold: Minimum conditions for SELL signal
            verbose: Print debug information
        """
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.verbose = verbose
        
        self.regime_map = {
            0: 'BULL_STRONG',
            1: 'BULL_WEAK',
            2: 'RANGE',
            3: 'BEAR_WEAK',
            4: 'BEAR_STRONG',
            5: 'VOLATILE'
        }
    
    def generate_all_labels(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all labels from OHLCV data and features.
        
        Args:
            df: OHLCV DataFrame
            features: Technical features DataFrame
            
        Returns:
            DataFrame with columns: buy, sell, direction, regime
        """
        if self.verbose:
            print("🏷️ Generating labels...")
        
        labels = pd.DataFrame(index=df.index)
        
        # Generate individual labels
        labels['buy'] = self._generate_buy_signals(df, features)
        labels['sell'] = self._generate_sell_signals(df, features)
        labels['direction'] = self._generate_direction(df)
        labels['regime'] = self._generate_regime(df, features)
        
        if self.verbose:
            print(f"✅ Generated labels:")
            print(f"   Buy signals: {labels['buy'].sum()} ({labels['buy'].mean()*100:.2f}%)")
            print(f"   Sell signals: {labels['sell'].sum()} ({labels['sell'].mean()*100:.2f}%)")
            print(f"   Up direction: {(labels['direction']==1).sum()} ({(labels['direction']==1).mean()*100:.2f}%)")
            print(f"   Regime distribution:\n{labels['regime'].value_counts()}")
        
        return labels
    
    def _generate_buy_signals(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        """
        Generate BUY signals based on institutional rules.
        BUY if >= buy_threshold conditions are met:
        1. close > MA20 > MA50
        2. RSI > 50 or recently left oversold (<30)
        3. close > BB midband and MA20_slope > 0
        4. volume > 1.5 × rolling20
        5. close near high (≥70%)
        6. MACD_hist > 0 and increasing
        """
        conditions = pd.DataFrame(index=df.index)
        
        # Condition 1: Bullish MA alignment
        try:
            ma20 = features['ma_20']
            ma50 = features['ma_50']
            conditions['c1'] = (df['close'] > ma20) & (ma20 > ma50)
        except:
            conditions['c1'] = False
        
        # Condition 2: RSI bullish
        try:
            rsi = features['rsi']
            rsi_was_oversold = (rsi.shift(1) < 30) & (rsi > 30)
            conditions['c2'] = (rsi > 50) | rsi_was_oversold
        except:
            conditions['c2'] = False
        
        # Condition 3: Price above BB mid and MA slope positive
        try:
            bb_mid = features['bb_mid_20']
            ma20_slope = features['ma20_slope']
            conditions['c3'] = (df['close'] > bb_mid) & (ma20_slope > 0)
        except:
            conditions['c3'] = False
        
        # Condition 4: Volume spike
        try:
            volume_ratio = features['volume_ratio']
            conditions['c4'] = volume_ratio > 1.5
        except:
            conditions['c4'] = False
        
        # Condition 5: Close near high
        try:
            close_position = features['close_position']
            conditions['c5'] = close_position >= 0.7
        except:
            conditions['c5'] = False
        
        # Condition 6: MACD bullish and improving
        try:
            macd_hist = features['macd_hist']
            macd_increasing = features['macd_hist_increasing']
            conditions['c6'] = (macd_hist > 0) & (macd_increasing == 1)
        except:
            conditions['c6'] = False
        
        # Count conditions met
        buy_score = conditions.sum(axis=1)
        buy_signal = (buy_score >= self.buy_threshold).astype(int)
        
        return buy_signal
    
    def _generate_sell_signals(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        """
        Generate SELL signals based on institutional rules.
        SELL if >= sell_threshold conditions are met:
        1. close < MA20 < MA50
        2. RSI < 50 or turning down from overbought (>70)
        3. close < BB midband and MA20_slope < 0
        4. volume > 1.5 × rolling20, close near low (≤30%)
        5. MACD_hist < 0 and decreasing
        6. Bearish price patterns
        """
        conditions = pd.DataFrame(index=df.index)
        
        # Condition 1: Bearish MA alignment
        try:
            ma20 = features['ma_20']
            ma50 = features['ma_50']
            conditions['c1'] = (df['close'] < ma20) & (ma20 < ma50)
        except:
            conditions['c1'] = False
        
        # Condition 2: RSI bearish
        try:
            rsi = features['rsi']
            rsi_was_overbought = (rsi.shift(1) > 70) & (rsi < 70)
            conditions['c2'] = (rsi < 50) | rsi_was_overbought
        except:
            conditions['c2'] = False
        
        # Condition 3: Price below BB mid and MA slope negative
        try:
            bb_mid = features['bb_mid_20']
            ma20_slope = features['ma20_slope']
            conditions['c3'] = (df['close'] < bb_mid) & (ma20_slope < 0)
        except:
            conditions['c3'] = False
        
        # Condition 4: Volume spike with close near low
        try:
            volume_ratio = features['volume_ratio']
            close_position = features['close_position']
            conditions['c4'] = (volume_ratio > 1.5) & (close_position <= 0.3)
        except:
            conditions['c4'] = False
        
        # Condition 5: MACD bearish and worsening
        try:
            macd_hist = features['macd_hist']
            macd_increasing = features['macd_hist_increasing']
            conditions['c5'] = (macd_hist < 0) & (macd_increasing == 0)
        except:
            conditions['c5'] = False
        
        # Condition 6: Bearish patterns (breakdown)
        try:
            bos_down = features['bos_down']
            conditions['c6'] = bos_down == 1
        except:
            conditions['c6'] = False
        
        # Count conditions met
        sell_score = conditions.sum(axis=1)
        sell_signal = (sell_score >= self.sell_threshold).astype(int)
        
        return sell_signal
    
    def _generate_direction(self, df: pd.DataFrame, forward_periods: int = 5) -> pd.Series:
        """
        Generate direction labels (0=down, 1=up).
        Based on future price movement.
        
        Args:
            df: OHLCV DataFrame
            forward_periods: Look ahead periods
        """
        future_return = (df['close'].shift(-forward_periods) / df['close']) - 1
        direction = (future_return > 0).astype(int)
        
        return direction
    
    def _generate_regime(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        """
        Generate market regime labels (6 classes).
        
        Regimes:
        0: BULL_STRONG    - trend > +2%, ADX > 25
        1: BULL_WEAK      - 0 < trend ≤ 2%, ADX ≤ 25
        2: RANGE          - lateral market
        3: BEAR_WEAK      - -2% ≤ trend < 0, ADX ≤ 25
        4: BEAR_STRONG    - trend < -2%, ADX > 25
        5: VOLATILE       - ATR > 90th percentile or volume spike
        """
        regime = pd.Series(2, index=df.index)  # Default: RANGE
        
        # Calculate trend (20-period return)
        try:
            trend = features['cum_return_20'] * 100  # As percentage
        except:
            trend = (df['close'] / df['close'].shift(20) - 1) * 100
        
        # Get ADX
        try:
            adx = features['adx']
        except:
            adx = pd.Series(20, index=df.index)
        
        # Get ATR percentile
        try:
            atr = features['atr_14']
            atr_threshold = atr.quantile(0.9)
        except:
            atr_threshold = float('inf')
        
        # Get volume spike
        try:
            volume_spike = features['volume_spike']
        except:
            volume_spike = pd.Series(0, index=df.index)
        
        # Classify regimes
        # 5: VOLATILE (highest priority)
        try:
            regime[atr > atr_threshold] = 5
            regime[volume_spike == 1] = 5
        except:
            pass
        
        # 0: BULL_STRONG
        mask = (trend > 2) & (adx > 25) & (regime == 2)
        regime[mask] = 0
        
        # 1: BULL_WEAK
        mask = (trend > 0) & (trend <= 2) & (adx <= 25) & (regime == 2)
        regime[mask] = 1
        
        # 3: BEAR_WEAK
        mask = (trend < 0) & (trend >= -2) & (adx <= 25) & (regime == 2)
        regime[mask] = 3
        
        # 4: BEAR_STRONG
        mask = (trend < -2) & (adx > 25) & (regime == 2)
        regime[mask] = 4
        
        # 2: RANGE (already default for remaining)
        
        return regime
    
    def get_regime_name(self, regime_id: int) -> str:
        """Get regime name from ID"""
        return self.regime_map.get(regime_id, 'UNKNOWN')
    
    def get_label_statistics(self, labels: pd.DataFrame) -> Dict:
        """Get statistics about generated labels"""
        stats = {
            'total_samples': len(labels),
            'buy_signals': int(labels['buy'].sum()),
            'buy_pct': float(labels['buy'].mean() * 100),
            'sell_signals': int(labels['sell'].sum()),
            'sell_pct': float(labels['sell'].mean() * 100),
            'direction_up': int((labels['direction'] == 1).sum()),
            'direction_up_pct': float((labels['direction'] == 1).mean() * 100),
            'regime_distribution': labels['regime'].value_counts().to_dict()
        }
        
        return stats


if __name__ == "__main__":
    # Test label generation
    print("🧪 Testing LabelGenerator...")
    
    # Generate sample data
    np.random.seed(42)
    n = 1000
    dates = pd.date_range('2023-01-01', periods=n, freq='15min')
    
    df = pd.DataFrame({
        'time': dates,
        'open': 100 + np.cumsum(np.random.randn(n) * 0.1),
        'high': 101 + np.cumsum(np.random.randn(n) * 0.1),
        'low': 99 + np.cumsum(np.random.randn(n) * 0.1),
        'close': 100 + np.cumsum(np.random.randn(n) * 0.1),
        'volume': np.random.randint(1000, 10000, n)
    })
    
    # Generate features first
    from features.feature_engine import FeatureEngine
    engine = FeatureEngine(verbose=False)
    features = engine.generate_all_features(df)
    
    # Generate labels
    labeler = LabelGenerator(verbose=True)
    labels = labeler.generate_all_labels(df, features)
    
    print(f"\n✅ Generated labels shape: {labels.shape}")
    print(f"\n📊 Label statistics:")
    stats = labeler.get_label_statistics(labels)
    for key, value in stats.items():
        print(f"   {key}: {value}")
