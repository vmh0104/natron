"""
Natron Transformer - Inference Engine
Fast inference for real-time trading signals
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from model_natron import NatronTransformer
from feature_engine import FeatureEngine
from labeling import LabelGenerator


class NatronInference:
    """
    Inference engine for Natron Transformer
    Handles feature generation and model prediction
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: str = 'config.yaml',
        device: str = 'cuda'
    ):
        """
        Args:
            model_path: Path to trained model weights
            config_path: Path to configuration file
            device: Device to run inference on
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize feature engine
        self.feature_engine = FeatureEngine(self.config)
        self.label_generator = LabelGenerator(self.config)
        
        # Load model
        print(f"📂 Loading model from {model_path}...")
        self.model = NatronTransformer(self.config).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        print(f"✅ Model loaded successfully")
        print(f"   Device: {self.device}")
        
        # Sequence length
        self.sequence_length = self.config['data']['sequence_length']
        
        # Confidence threshold
        self.confidence_threshold = self.config['inference']['confidence_threshold']
    
    def preprocess_ohlcv(self, df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess OHLCV data to feature sequence
        
        Args:
            df: DataFrame with columns [time, open, high, low, close, volume]
               Must contain at least 96 rows
               
        Returns:
            Feature sequence (96, 100)
        """
        if len(df) < self.sequence_length:
            raise ValueError(
                f"Need at least {self.sequence_length} candles, got {len(df)}"
            )
        
        # Take last 96 candles
        df_recent = df.tail(self.sequence_length).copy()
        
        # Generate features
        features = self.feature_engine.generate_all_features(df_recent)
        
        # Convert to numpy array
        feature_array = features.values  # (96, 100)
        
        return feature_array
    
    @torch.no_grad()
    def predict(self, ohlcv_data: pd.DataFrame) -> Dict:
        """
        Predict trading signals from OHLCV data
        
        Args:
            ohlcv_data: DataFrame with OHLCV data (at least 96 rows)
            
        Returns:
            Dictionary with predictions:
            {
                'buy_prob': float,
                'sell_prob': float,
                'direction_up': float,
                'direction_down': float,
                'regime': str,
                'regime_id': int,
                'confidence': float,
                'timestamp': str
            }
        """
        # Preprocess
        features = self.preprocess_ohlcv(ohlcv_data)
        
        # Convert to tensor
        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)  # (1, 96, 100)
        
        # Predict
        outputs = self.model(x)
        
        # Extract predictions
        buy_prob = outputs['buy'].item()
        sell_prob = outputs['sell'].item()
        
        direction_probs = torch.softmax(outputs['direction'], dim=-1)[0]
        direction_down = direction_probs[0].item()
        direction_up = direction_probs[1].item()
        
        regime_probs = torch.softmax(outputs['regime'], dim=-1)[0]
        regime_id = regime_probs.argmax().item()
        regime_confidence = regime_probs.max().item()
        
        regime_name = self.label_generator.get_regime_name(regime_id)
        
        # Overall confidence (average of max probabilities)
        confidence = (
            max(buy_prob, 1 - buy_prob) +
            max(sell_prob, 1 - sell_prob) +
            max(direction_up, direction_down) +
            regime_confidence
        ) / 4
        
        # Get timestamp
        timestamp = ohlcv_data.iloc[-1]['time'] if 'time' in ohlcv_data.columns else None
        
        return {
            'buy_prob': float(buy_prob),
            'sell_prob': float(sell_prob),
            'direction_up': float(direction_up),
            'direction_down': float(direction_down),
            'regime': regime_name,
            'regime_id': int(regime_id),
            'confidence': float(confidence),
            'timestamp': str(timestamp) if timestamp is not None else None,
            'signal': self._get_trading_signal(buy_prob, sell_prob, confidence)
        }
    
    def _get_trading_signal(self, buy_prob: float, sell_prob: float, confidence: float) -> str:
        """
        Generate trading signal from probabilities
        
        Args:
            buy_prob: Buy probability
            sell_prob: Sell probability
            confidence: Overall confidence
            
        Returns:
            Signal: 'BUY', 'SELL', or 'HOLD'
        """
        if confidence < self.confidence_threshold:
            return 'HOLD'
        
        if buy_prob > 0.6 and buy_prob > sell_prob:
            return 'BUY'
        elif sell_prob > 0.6 and sell_prob > buy_prob:
            return 'SELL'
        else:
            return 'HOLD'
    
    def predict_batch(self, ohlcv_list: List[pd.DataFrame]) -> List[Dict]:
        """
        Predict for multiple OHLCV sequences (batch inference)
        
        Args:
            ohlcv_list: List of OHLCV DataFrames
            
        Returns:
            List of prediction dictionaries
        """
        # Preprocess all
        features_list = [self.preprocess_ohlcv(df) for df in ohlcv_list]
        features_batch = np.stack(features_list)  # (batch, 96, 100)
        
        # Convert to tensor
        x = torch.FloatTensor(features_batch).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(x)
        
        # Extract predictions for each sample
        results = []
        for i in range(len(ohlcv_list)):
            buy_prob = outputs['buy'][i].item()
            sell_prob = outputs['sell'][i].item()
            
            direction_probs = torch.softmax(outputs['direction'][i], dim=-1)
            direction_down = direction_probs[0].item()
            direction_up = direction_probs[1].item()
            
            regime_probs = torch.softmax(outputs['regime'][i], dim=-1)
            regime_id = regime_probs.argmax().item()
            regime_confidence = regime_probs.max().item()
            
            regime_name = self.label_generator.get_regime_name(regime_id)
            
            confidence = (
                max(buy_prob, 1 - buy_prob) +
                max(sell_prob, 1 - sell_prob) +
                max(direction_up, direction_down) +
                regime_confidence
            ) / 4
            
            timestamp = ohlcv_list[i].iloc[-1]['time'] if 'time' in ohlcv_list[i].columns else None
            
            results.append({
                'buy_prob': float(buy_prob),
                'sell_prob': float(sell_prob),
                'direction_up': float(direction_up),
                'direction_down': float(direction_down),
                'regime': regime_name,
                'regime_id': int(regime_id),
                'confidence': float(confidence),
                'timestamp': str(timestamp) if timestamp is not None else None,
                'signal': self._get_trading_signal(buy_prob, sell_prob, confidence)
            })
        
        return results


if __name__ == "__main__":
    # Test inference
    print("🧪 Testing Natron Inference...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='15min')
    
    returns = np.random.randn(n_samples) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'time': dates,
        'open': prices * (1 + np.random.randn(n_samples) * 0.001),
        'high': prices * (1 + abs(np.random.randn(n_samples)) * 0.002),
        'low': prices * (1 - abs(np.random.randn(n_samples)) * 0.002),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    # Note: This will fail without a trained model
    # Just showing the interface
    print("\n📋 Inference interface ready")
    print("  To use:")
    print("  1. Train a model first using train_natron.py")
    print("  2. Load model: engine = NatronInference('models/natron_v2.pt')")
    print("  3. Predict: result = engine.predict(ohlcv_df)")
