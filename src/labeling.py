"""
Natron Transformer - Labeling Module
Generates buy/sell signals, direction, and market regime labels
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


class LabelGenerator:
    """
    Generate trading labels using institutional/technical rules:
    - Buy/Sell signals (binary)
    - Direction prediction (up/down)
    - Market regime classification (6 classes)
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.regime_names = [
            'BULL_STRONG',    # 0
            'BULL_WEAK',      # 1
            'RANGE',          # 2
            'BEAR_WEAK',      # 3
            'BEAR_STRONG',    # 4
            'VOLATILE'        # 5
        ]
    
    def generate_all_labels(
        self, 
        df: pd.DataFrame, 
        features: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Generate all labels from OHLCV and features.
        
        Args:
            df: Original OHLCV DataFrame
            features: Technical features DataFrame
            
        Returns:
            Tuple of (buy_labels, sell_labels, direction_labels, regime_labels)
        """
        # Generate buy signals
        buy_labels = self._generate_buy_signals(df, features)
        
        # Generate sell signals
        sell_labels = self._generate_sell_signals(df, features)
        
        # Generate direction labels (future price movement)
        direction_labels = self._generate_direction_labels(df)
        
        # Generate regime labels
        regime_labels = self._generate_regime_labels(df, features)
        
        print(f"✅ Labels generated:")
        print(f"  Buy signals: {buy_labels.sum()} / {len(buy_labels)} ({buy_labels.mean()*100:.2f}%)")
        print(f"  Sell signals: {sell_labels.sum()} / {len(sell_labels)} ({sell_labels.mean()*100:.2f}%)")
        print(f"  Direction Up: {(direction_labels==1).sum()} / {len(direction_labels)} ({(direction_labels==1).mean()*100:.2f}%)")
        print(f"  Regime distribution:")
        for i, name in enumerate(self.regime_names):
            count = (regime_labels == i).sum()
            pct = count / len(regime_labels) * 100
            print(f"    {name}: {count} ({pct:.2f}%)")
        
        return buy_labels, sell_labels, direction_labels, regime_labels
    
    def _generate_buy_signals(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        """
        Generate BUY signals (1 if ≥2 conditions true):
        1. close > MA20 > MA50
        2. RSI > 50 or recently left oversold (<30)
        3. close > BB midband and MA20_slope > 0
        4. volume > 1.5 × rolling20
        5. close near high (≥70%)
        6. MACD_hist > 0 and increasing
        """
        close = df['close']
        volume = df['volume']
        
        conditions = pd.DataFrame(index=df.index)
        
        # Condition 1: MA alignment
        if 'ma_20' in features.columns and 'ma_50' in features.columns:
            conditions['ma_align'] = (
                (close > features['ma_20']) & 
                (features['ma_20'] > features['ma_50'])
            )
        else:
            conditions['ma_align'] = False
        
        # Condition 2: RSI bullish
        if 'rsi_14' in features.columns:
            rsi = features['rsi_14']
            rsi_oversold = rsi < 30
            rsi_left_oversold = rsi_oversold.shift(1) & (rsi >= 30)
            conditions['rsi_bullish'] = (rsi > 50) | rsi_left_oversold
        else:
            conditions['rsi_bullish'] = False
        
        # Condition 3: BB position and MA slope
        if 'bb_mid' in features.columns and 'ma20_slope' in features.columns:
            conditions['bb_position'] = (
                (close > features['bb_mid']) & 
                (features['ma20_slope'] > 0)
            )
        else:
            conditions['bb_position'] = False
        
        # Condition 4: Volume spike
        if 'vol_ma_20' in features.columns:
            conditions['volume_spike'] = volume > (features['vol_ma_20'] * 1.5)
        else:
            conditions['volume_spike'] = False
        
        # Condition 5: Close near high
        if 'close_position' in features.columns:
            conditions['near_high'] = features['close_position'] >= 0.70
        else:
            conditions['near_high'] = False
        
        # Condition 6: MACD positive and increasing
        if 'macd_hist' in features.columns:
            macd_hist = features['macd_hist']
            conditions['macd_positive'] = (macd_hist > 0) & (macd_hist > macd_hist.shift(1))
        else:
            conditions['macd_positive'] = False
        
        # Buy signal = at least 2 conditions true
        buy_signal = conditions.sum(axis=1) >= 2
        
        return buy_signal.astype(int)
    
    def _generate_sell_signals(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        """
        Generate SELL signals (1 if ≥2 conditions true):
        1. close < MA20 < MA50
        2. RSI < 50 or turning down from overbought (>70)
        3. close < BB midband and MA20_slope < 0
        4. volume > 1.5 × rolling20, close near low (≤30%)
        5. close near low (≤30%)
        6. MACD_hist < 0 and decreasing
        """
        close = df['close']
        volume = df['volume']
        
        conditions = pd.DataFrame(index=df.index)
        
        # Condition 1: MA alignment (bearish)
        if 'ma_20' in features.columns and 'ma_50' in features.columns:
            conditions['ma_align'] = (
                (close < features['ma_20']) & 
                (features['ma_20'] < features['ma_50'])
            )
        else:
            conditions['ma_align'] = False
        
        # Condition 2: RSI bearish
        if 'rsi_14' in features.columns:
            rsi = features['rsi_14']
            rsi_overbought = rsi > 70
            rsi_left_overbought = rsi_overbought.shift(1) & (rsi <= 70)
            conditions['rsi_bearish'] = (rsi < 50) | rsi_left_overbought
        else:
            conditions['rsi_bearish'] = False
        
        # Condition 3: BB position and MA slope
        if 'bb_mid' in features.columns and 'ma20_slope' in features.columns:
            conditions['bb_position'] = (
                (close < features['bb_mid']) & 
                (features['ma20_slope'] < 0)
            )
        else:
            conditions['bb_position'] = False
        
        # Condition 4: Volume spike
        if 'vol_ma_20' in features.columns:
            conditions['volume_spike'] = volume > (features['vol_ma_20'] * 1.5)
        else:
            conditions['volume_spike'] = False
        
        # Condition 5: Close near low
        if 'close_position' in features.columns:
            conditions['near_low'] = features['close_position'] <= 0.30
        else:
            conditions['near_low'] = False
        
        # Condition 6: MACD negative and decreasing
        if 'macd_hist' in features.columns:
            macd_hist = features['macd_hist']
            conditions['macd_negative'] = (macd_hist < 0) & (macd_hist < macd_hist.shift(1))
        else:
            conditions['macd_negative'] = False
        
        # Sell signal = at least 2 conditions true
        sell_signal = conditions.sum(axis=1) >= 2
        
        return sell_signal.astype(int)
    
    def _generate_direction_labels(self, df: pd.DataFrame, horizon: int = 5) -> pd.Series:
        """
        Generate direction labels based on future price movement.
        
        Args:
            df: OHLCV DataFrame
            horizon: Look-ahead period (default 5 candles)
            
        Returns:
            Series with 0=down, 1=up
        """
        close = df['close']
        
        # Future return
        future_return = close.shift(-horizon) / close - 1
        
        # Label: 1 if price goes up, 0 if down
        direction = (future_return > 0).astype(int)
        
        # Fill last `horizon` values with the last valid label
        direction.iloc[-horizon:] = direction.iloc[-horizon-1]
        
        return direction
    
    def _generate_regime_labels(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        """
        Generate market regime labels (6 classes):
        
        ID | Regime        | Condition
        ---|---------------|------------------------------------------
        0  | BULL_STRONG   | trend > +2%, ADX > 25
        1  | BULL_WEAK     | 0 < trend ≤ 2%, ADX ≤ 25
        2  | RANGE         | lateral market
        3  | BEAR_WEAK     | −2% ≤ trend < 0, ADX ≤ 25
        4  | BEAR_STRONG   | trend < −2%, ADX > 25
        5  | VOLATILE      | ATR > 90th percentile or volume spike
        """
        close = df['close']
        volume = df['volume']
        
        # Calculate trend (20-period return)
        trend = close.pct_change(20) * 100  # percentage
        
        # Get ADX
        adx = features.get('adx', pd.Series(20, index=df.index))  # default 20 if missing
        
        # Get ATR percentile
        if 'atr_14' in features.columns:
            atr = features['atr_14']
            atr_90th = atr.rolling(200).quantile(0.90)
            is_high_volatility = atr > atr_90th
        else:
            is_high_volatility = pd.Series(False, index=df.index)
        
        # Volume spike
        vol_ma = volume.rolling(20).mean()
        is_volume_spike = volume > (vol_ma * 2)
        
        # Initialize regime array
        regime = pd.Series(2, index=df.index)  # default to RANGE
        
        # Classify regimes
        # 5: VOLATILE (highest priority)
        regime[is_high_volatility | is_volume_spike] = 5
        
        # 0: BULL_STRONG
        regime[(trend > 2) & (adx > 25)] = 0
        
        # 1: BULL_WEAK
        regime[(trend > 0) & (trend <= 2) & (adx <= 25)] = 1
        
        # 3: BEAR_WEAK
        regime[(trend < 0) & (trend >= -2) & (adx <= 25)] = 3
        
        # 4: BEAR_STRONG
        regime[(trend < -2) & (adx > 25)] = 4
        
        # 2: RANGE (everything else, already set as default)
        regime[(trend >= -1) & (trend <= 1) & ~is_high_volatility] = 2
        
        return regime.astype(int)
    
    def get_regime_name(self, regime_id: int) -> str:
        """Get regime name from ID"""
        if 0 <= regime_id < len(self.regime_names):
            return self.regime_names[regime_id]
        return "UNKNOWN"


if __name__ == "__main__":
    # Test label generation
    print("🧪 Testing Label Generator...")
    
    # Import feature engine
    import sys
    sys.path.append('/workspace/src')
    from feature_engine import FeatureEngine
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='15min')
    
    # Generate realistic price movements
    returns = np.random.randn(n_samples) * 0.01 + 0.0001  # slight upward drift
    prices = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'time': dates,
        'open': prices * (1 + np.random.randn(n_samples) * 0.001),
        'high': prices * (1 + abs(np.random.randn(n_samples)) * 0.002),
        'low': prices * (1 - abs(np.random.randn(n_samples)) * 0.002),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    # Ensure high >= low
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    # Generate features
    engine = FeatureEngine()
    features = engine.generate_all_features(df)
    
    # Generate labels
    labeler = LabelGenerator()
    buy, sell, direction, regime = labeler.generate_all_labels(df, features)
    
    # Show sample results
    print("\n📊 Sample Labels (last 10 rows):")
    results = pd.DataFrame({
        'close': df['close'].values[-10:],
        'buy': buy.values[-10:],
        'sell': sell.values[-10:],
        'direction': direction.values[-10:],
        'regime': regime.values[-10:],
        'regime_name': [labeler.get_regime_name(r) for r in regime.values[-10:]]
    })
    print(results.to_string(index=False))
    
    print("\n✅ Label generation successful!")
