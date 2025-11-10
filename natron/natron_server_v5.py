"""
Natron Realtime Trading Server V5
Python server that reads model predictions and executes trades via MQL5/MT5 API.
"""

import socket
import threading
import time
import pandas as pd
import numpy as np
import torch
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import queue

from model_natron import NatronTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NatronServer:
    """
    Natron realtime trading server.
    Monitors live data, generates predictions, and executes trades.
    """
    
    def __init__(self, config_path: str = "realtime_config.yaml"):
        """
        Initialize Natron server.
        
        Args:
            config_path: Path to realtime config YAML
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        self.model.eval()
        
        # Trading state
        self.current_position = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.trailing_stop = None
        
        # Data buffer
        self.data_buffer = []
        self.sequence_length = self.config.get('sequence_length', 96)
        
        # Event log
        self.event_log = []
        self.event_log_path = self.config.get('event_log_path', 'realtime_events.csv')
        
        # Socket for MQL5 communication
        self.socket_host = self.config.get('socket_host', 'localhost')
        self.socket_port = self.config.get('socket_port', 8888)
        self.socket = None
        self.socket_connected = False
        
        # Threading
        self.running = False
        self.prediction_queue = queue.Queue()
        
        logger.info("Natron Server V5 initialized")
    
    def _load_model(self) -> NatronTransformer:
        """Load trained model."""
        model_path = self.config.get('model_path', 'checkpoints/natron_v6.pt')
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get config from checkpoint
        model_config = checkpoint.get('config', {})
        
        # Initialize model (input_dim will be set from first prediction)
        model = NatronTransformer(
            input_dim=self.config.get('input_dim', 70),  # Will be updated
            d_model=model_config.get('d_model', 256),
            nhead=model_config.get('nhead', 8),
            num_layers=model_config.get('num_layers', 6),
            dim_feedforward=model_config.get('dim_feedforward', 1024),
            dropout=0.0,  # No dropout during inference
            max_seq_len=self.config.get('sequence_length', 96),
            num_regime_classes=model_config.get('num_regime_classes', 6),
            num_forecast_classes=model_config.get('num_forecast_classes', 2)
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from {model_path}")
        
        return model
    
    def connect_socket(self):
        """Connect to MQL5 socket."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.socket_host, self.socket_port))
            self.socket_connected = True
            logger.info(f"Connected to MQL5 socket at {self.socket_host}:{self.socket_port}")
        except Exception as e:
            logger.error(f"Failed to connect to socket: {e}")
            self.socket_connected = False
    
    def send_trade_signal(self, signal: Dict):
        """
        Send trade signal to MQL5 EA.
        
        Args:
            signal: Dictionary with trade signal data
        """
        if not self.socket_connected:
            logger.warning("Socket not connected, cannot send signal")
            return
        
        try:
            # Format: "ACTION|SYMBOL|PRICE|STOP_LOSS|TAKE_PROFIT|COMMENT"
            message = f"{signal['action']}|{signal['symbol']}|{signal['price']}|" \
                     f"{signal.get('stop_loss', 0)}|{signal.get('take_profit', 0)}|" \
                     f"{signal.get('comment', 'Natron')}"
            
            self.socket.sendall(message.encode('utf-8'))
            logger.info(f"Sent trade signal: {message}")
        except Exception as e:
            logger.error(f"Failed to send trade signal: {e}")
    
    def process_candle(self, candle_data: Dict) -> Optional[Dict]:
        """
        Process a new candle and generate prediction.
        
        Args:
            candle_data: Dictionary with OHLCV data
            
        Returns:
            Prediction dictionary or None
        """
        # Add to buffer
        self.data_buffer.append(candle_data)
        
        # Keep only last sequence_length candles
        if len(self.data_buffer) > self.sequence_length:
            self.data_buffer = self.data_buffer[-self.sequence_length:]
        
        # Need enough data for prediction
        if len(self.data_buffer) < self.sequence_length:
            return None
        
        # Prepare features (simplified - in production, use feature_engineering)
        features = self._prepare_features()
        
        if features is None:
            return None
        
        # Generate prediction
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            predictions = self.model(features_tensor)
            
            regime_pred = predictions['regime'].argmax(dim=1).item()
            context_score = predictions['context'].item()
            forecast_pred = predictions['forecast'].argmax(dim=1).item()
        
        prediction = {
            'timestamp': datetime.now().isoformat(),
            'regime': regime_pred,
            'context': context_score,
            'forecast': forecast_pred,
            'price': candle_data.get('close', 0)
        }
        
        return prediction
    
    def _prepare_features(self) -> Optional[np.ndarray]:
        """
        Prepare features from data buffer.
        In production, this should use the same feature engineering as training.
        """
        try:
            # Convert buffer to DataFrame
            df = pd.DataFrame(self.data_buffer)
            
            # Simple feature extraction (should match training pipeline)
            # This is simplified - use feature_engineering.py in production
            features = []
            
            for i in range(len(df)):
                row = df.iloc[i]
                feature_vec = [
                    row.get('open', 0),
                    row.get('high', 0),
                    row.get('low', 0),
                    row.get('close', 0),
                    row.get('volume', 0)
                ]
                
                # Add simple indicators
                if i > 0:
                    returns = (row['close'] - df.iloc[i-1]['close']) / df.iloc[i-1]['close']
                    feature_vec.append(returns)
                else:
                    feature_vec.append(0)
                
                features.append(feature_vec)
            
            # Pad or truncate to sequence_length
            if len(features) < self.sequence_length:
                return None
            
            features = features[-self.sequence_length:]
            
            # Normalize (should use training normalization stats)
            features = np.array(features, dtype=np.float32)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None
    
    def evaluate_trade_signal(self, prediction: Dict) -> Optional[Dict]:
        """
        Evaluate prediction and determine trade signal.
        
        Args:
            prediction: Prediction dictionary
            
        Returns:
            Trade signal dictionary or None
        """
        regime = prediction['regime']
        context = prediction['context']
        forecast = prediction['forecast']
        price = prediction['price']
        
        # Entry conditions
        entry_threshold = self.config.get('entry_threshold', 0.6)
        
        # Strong bullish regime + high context + forecast up
        if regime in [0, 1] and context > entry_threshold and forecast == 1:
            if self.current_position is None:
                # Calculate stop loss and take profit
                atr_multiplier = self.config.get('atr_multiplier', 2.0)
                # Simplified ATR calculation
                atr = price * 0.02  # Approximate
                
                stop_loss = price - (atr * atr_multiplier)
                take_profit = price + (atr * atr_multiplier * 2)
                
                signal = {
                    'action': 'BUY',
                    'symbol': self.config.get('symbol', 'EURUSD'),
                    'price': price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'comment': f'Natron_BULL_CTX{context:.2f}'
                }
                
                self.current_position = 'LONG'
                self.entry_price = price
                self.stop_loss = stop_loss
                self.take_profit = take_profit
                
                self.log_event('ENTRY', signal)
                return signal
        
        # Strong bearish regime + high context + forecast down
        elif regime in [2, 3] and context > entry_threshold and forecast == 0:
            if self.current_position is None:
                atr = price * 0.02
                stop_loss = price + (atr * 2.0)
                take_profit = price - (atr * 4.0)
                
                signal = {
                    'action': 'SELL',
                    'symbol': self.config.get('symbol', 'EURUSD'),
                    'price': price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'comment': f'Natron_BEAR_CTX{context:.2f}'
                }
                
                self.current_position = 'SHORT'
                self.entry_price = price
                self.stop_loss = stop_loss
                self.take_profit = take_profit
                
                self.log_event('ENTRY', signal)
                return signal
        
        # Exit conditions
        elif self.current_position is not None:
            # Cancel conditions: ATR spike or deviation
            cancel_threshold = self.config.get('cancel_threshold', 0.05)
            price_deviation = abs(price - self.entry_price) / self.entry_price
            
            if price_deviation > cancel_threshold:
                signal = {
                    'action': 'CLOSE',
                    'symbol': self.config.get('symbol', 'EURUSD'),
                    'price': price,
                    'comment': 'Natron_CANCEL'
                }
                
                self.log_event('EXIT', signal)
                self.current_position = None
                self.entry_price = None
                return signal
        
        return None
    
    def log_event(self, event_type: str, data: Dict):
        """
        Log trading event.
        
        Args:
            event_type: Type of event (ENTRY, EXIT, etc.)
            data: Event data
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            **data
        }
        
        self.event_log.append(event)
        
        # Save to CSV periodically
        if len(self.event_log) % 10 == 0:
            self.save_event_log()
    
    def save_event_log(self):
        """Save event log to CSV."""
        if self.event_log:
            df = pd.DataFrame(self.event_log)
            df.to_csv(self.event_log_path, index=False, mode='a', header=not Path(self.event_log_path).exists())
    
    def run(self):
        """Main server loop."""
        logger.info("Starting Natron Server...")
        self.running = True
        
        # Connect to socket
        self.connect_socket()
        
        # Main loop (in production, this would read from realtime data source)
        try:
            while self.running:
                # In production, read from realtime socket or data feed
                # For now, simulate with sleep
                time.sleep(1)
                
                # Process any pending predictions
                try:
                    prediction = self.prediction_queue.get_nowait()
                    signal = self.evaluate_trade_signal(prediction)
                    
                    if signal:
                        self.send_trade_signal(signal)
                except queue.Empty:
                    pass
        
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.running = False
            if self.socket:
                self.socket.close()
            self.save_event_log()


if __name__ == "__main__":
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "realtime_config.yaml"
    
    server = NatronServer(config_path)
    server.run()
