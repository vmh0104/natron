"""
Natron Server V5 - Realtime Trading Engine
Main server that processes predictions and executes trades via MQL5/MT5 API.
"""

import torch
import numpy as np
import pandas as pd
import yaml
import json
import socket
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple
import logging

from model_natron import NatronTransformer
from feature_engineering import FeatureEngineer
from data_pipeline import DataPipeline
from realtime_socket import RealtimeSocket, CandleBuffer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingEngine:
    """
    Trading engine that processes predictions and executes trades.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize trading engine.
        
        Args:
            config_path: Path to realtime config YAML
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model
        model_path = Path(self.config['model']['checkpoint_path'])
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize model
        model_config = self.config['model']
        self.model = NatronTransformer(
            n_features=model_config['n_features'],
            d_model=model_config.get('d_model', 128),
            nhead=model_config.get('nhead', 8),
            num_layers=model_config.get('num_layers', 6),
            dim_feedforward=model_config.get('dim_feedforward', 512),
            dropout=0.0,  # No dropout during inference
            max_seq_len=model_config.get('max_seq_len', 96),
            n_regime_classes=model_config.get('n_regime_classes', 6)
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        logger.info(f"Loaded model from {model_path}")
        
        # Feature engineering
        self.feature_engineer = FeatureEngineer()
        self.pipeline = DataPipeline()
        
        # Normalization stats (should be loaded from training)
        norm_stats_path = Path(self.config['model'].get('normalization_stats', 'models/natron_v6/norm_stats.npy'))
        if norm_stats_path.exists():
            norm_stats = np.load(norm_stats_path, allow_pickle=True)
            self.feature_mean = norm_stats[0]
            self.feature_std = norm_stats[1]
        else:
            logger.warning("Normalization stats not found, using defaults")
            self.feature_mean = None
            self.feature_std = None
        
        # Trading parameters
        self.trading_config = self.config['trading']
        self.entry_threshold = self.trading_config.get('entry_threshold', 0.6)
        self.cancel_threshold = self.trading_config.get('cancel_threshold', 0.3)
        self.atr_multiplier = self.trading_config.get('atr_multiplier', 2.0)
        
        # Event logging
        self.events_log = []
        self.events_file = Path(self.config.get('events_log', 'realtime_events.csv'))
    
    def preprocess_candle_sequence(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Preprocess candle sequence for model input.
        
        Args:
            df: DataFrame with OHLCV candles
            
        Returns:
            Preprocessed tensor (1, seq_len, n_features)
        """
        # Compute features
        feature_df = self.feature_engineer.compute_all_features(df)
        
        # Extract feature columns (exclude OHLCV)
        feature_cols = [col for col in feature_df.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume']]
        features = feature_df[feature_cols].values
        
        # Normalize
        if self.feature_mean is not None and self.feature_std is not None:
            features = (features - self.feature_mean) / self.feature_std
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Convert to tensor
        tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        return tensor
    
    def predict(self, df: pd.DataFrame) -> Dict:
        """
        Generate predictions from candle sequence.
        
        Args:
            df: DataFrame with OHLCV candles
            
        Returns:
            Dictionary with predictions
        """
        with torch.no_grad():
            # Preprocess
            X = self.preprocess_candle_sequence(df)
            
            # Forward pass
            regime_logits, context_score, forecast_logits = self.model(X)
            
            # Extract predictions
            regime_probs = torch.softmax(regime_logits, dim=1).cpu().numpy()[0]
            regime_pred = np.argmax(regime_probs)
            context_strength = context_score.cpu().numpy()[0, 0]
            forecast_probs = torch.softmax(forecast_logits, dim=1).cpu().numpy()[0]
            forecast_direction = np.argmax(forecast_probs)
            forecast_confidence = forecast_probs[forecast_direction]
        
        return {
            'regime': int(regime_pred),
            'regime_probs': regime_probs.tolist(),
            'context_strength': float(context_strength),
            'forecast_direction': int(forecast_direction),  # 0=down, 1=up
            'forecast_confidence': float(forecast_confidence),
            'timestamp': datetime.now().isoformat()
        }
    
    def determine_entry_zone(self, predictions: Dict, current_price: float, atr: float) -> Optional[Dict]:
        """
        Determine entry zone based on predictions.
        
        Args:
            predictions: Prediction dictionary
            current_price: Current price
            atr: ATR value
            
        Returns:
            Entry zone dictionary or None
        """
        forecast_dir = predictions['forecast_direction']
        confidence = predictions['forecast_confidence']
        context = predictions['context_strength']
        
        # Entry conditions
        if confidence < self.entry_threshold:
            return None
        
        if context < 0.3:  # Low context strength
            return None
        
        # Calculate entry zone
        entry_offset = self.atr_multiplier * atr
        
        if forecast_dir == 1:  # Up
            entry_price = current_price - entry_offset * 0.5  # Limit entry below current
            stop_loss = current_price - entry_offset
            take_profit = current_price + entry_offset * 2
            direction = 'BUY'
        else:  # Down
            entry_price = current_price + entry_offset * 0.5  # Limit entry above current
            stop_loss = current_price + entry_offset
            take_profit = current_price - entry_offset * 2
            direction = 'SELL'
        
        return {
            'direction': direction,
            'entry_price': float(entry_price),
            'stop_loss': float(stop_loss),
            'take_profit': float(take_profit),
            'confidence': float(confidence),
            'context_strength': float(context)
        }
    
    def check_cancel_conditions(self, predictions: Dict, atr: float, price_deviation: float) -> bool:
        """
        Check if trade should be cancelled.
        
        Args:
            predictions: Prediction dictionary
            atr: ATR value
            price_deviation: Current price deviation from entry
            
        Returns:
            True if should cancel
        """
        # ATR spike check
        atr_spike_threshold = atr * 1.5
        if price_deviation > atr_spike_threshold:
            return True
        
        # Confidence drop
        if predictions['forecast_confidence'] < self.cancel_threshold:
            return True
        
        # Context strength drop
        if predictions['context_strength'] < 0.2:
            return True
        
        return False
    
    def log_event(self, event_type: str, data: Dict):
        """
        Log trading event.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            **data
        }
        self.events_log.append(event)
        logger.info(f"Event: {event_type} - {data}")
    
    def save_events(self):
        """Save events to CSV."""
        if self.events_log:
            df = pd.DataFrame(self.events_log)
            df.to_csv(self.events_file, mode='a', header=not self.events_file.exists(), index=False)
            self.events_log.clear()


class NatronServer:
    """
    Main Natron server that coordinates realtime trading.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize Natron server.
        
        Args:
            config_path: Path to realtime config YAML
        """
        self.config_path = config_path
        self.trading_engine = TradingEngine(config_path)
        self.socket = RealtimeSocket(
            host=self.config['socket'].get('host', 'localhost'),
            port=self.config['socket'].get('port', 8888)
        )
        self.candle_buffer = CandleBuffer(sequence_length=96)
        self.running = False
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def on_new_candle(self, candle_data: Dict):
        """
        Callback for new candle data.
        
        Args:
            candle_data: Candle data dictionary
        """
        self.candle_buffer.add_candle(candle_data)
        
        # Check if we have enough candles
        sequence_df = self.candle_buffer.get_sequence()
        if sequence_df is not None:
            # Generate predictions
            predictions = self.trading_engine.predict(sequence_df)
            
            # Get current price and ATR
            current_price = sequence_df['close'].iloc[-1]
            atr = sequence_df.get('atr', pd.Series([0]))[-1] if 'atr' in sequence_df.columns else 0
            
            # Determine entry zone
            entry_zone = self.trading_engine.determine_entry_zone(
                predictions, current_price, atr
            )
            
            if entry_zone:
                self.trading_engine.log_event('ENTRY_SIGNAL', {
                    'predictions': predictions,
                    'entry_zone': entry_zone
                })
                
                # Send to MQL5 EA
                self.send_to_mql5(entry_zone)
    
    def send_to_mql5(self, entry_zone: Dict):
        """
        Send entry signal to MQL5 EA.
        
        Args:
            entry_zone: Entry zone dictionary
        """
        # This would send via named pipe or socket to MQL5
        # For now, log the signal
        logger.info(f"Sending entry signal to MQL5: {entry_zone}")
    
    def start(self):
        """Start Natron server."""
        logger.info("Starting Natron Server V5...")
        
        # Register callback
        self.socket.register_callback(self.on_new_candle)
        
        # Start listening
        if self.socket.start_listening():
            self.running = True
            
            # Main loop
            try:
                while self.running:
                    time.sleep(1)
                    # Save events periodically
                    self.trading_engine.save_events()
            except KeyboardInterrupt:
                logger.info("Shutting down...")
            finally:
                self.stop()
        else:
            logger.error("Failed to start socket listener")
    
    def stop(self):
        """Stop Natron server."""
        self.running = False
        self.socket.stop_listening()
        self.socket.disconnect()
        self.trading_engine.save_events()
        logger.info("Natron Server stopped")


def main():
    """Main server function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Natron Server V5')
    parser.add_argument('--config', type=str, default='realtime_config.yaml',
                       help='Path to realtime config YAML')
    args = parser.parse_args()
    
    server = NatronServer(args.config)
    server.start()


if __name__ == "__main__":
    main()
