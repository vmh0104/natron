"""
Natron Server V5 - Realtime Trading Engine

This server reads model predictions and executes trades via MQL5/MT5 API.
Monitors live candle/tick stream and manages trade execution logic.
"""

import socket
import threading
import time
import json
import yaml
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import logging
from datetime import datetime
from typing import Optional, Dict
import queue

from ..training.model_natron import NatronTransformer
from ..data_pipeline.data_pipeline import DataPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NatronServer:
    """
    Main server class for realtime trading execution.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize Natron server.
        
        Args:
            config_path: Path to realtime configuration YAML
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = None
        self.load_model()
        
        # Initialize data pipeline
        self.data_pipeline = DataPipeline()
        
        # Trading state
        self.current_position = None
        self.pending_orders = []
        self.recent_candles = []
        self.prediction_buffer = queue.Queue(maxsize=100)
        
        # Socket for MQL5 communication
        self.socket = None
        self.socket_connected = False
        
        # Event log
        self.event_log = []
        self.log_file = Path(self.config['paths']['event_log'])
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Start socket listener
        self.start_socket_server()
    
    def load_model(self):
        """Load trained model."""
        model_path = self.config['model']['path']
        logger.info(f"Loading model from {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model config
        if 'config' in checkpoint:
            model_config = checkpoint['config']['model']
        else:
            model_config = self.config['model']
        
        # Load feature count from processed data
        processed_path = self.config['data']['processed_path']
        df = pd.read_csv(processed_path, index_col=0, nrows=1)
        exclude_cols = ['regime', 'regime_name', 'target', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        input_dim = len(feature_cols)
        
        # Build and load model
        self.model = NatronTransformer(
            input_dim=input_dim,
            d_model=model_config['d_model'],
            nhead=model_config['nhead'],
            num_layers=model_config['num_layers'],
            dim_feedforward=model_config['dim_feedforward'],
            dropout=0.0,  # No dropout during inference
            seq_len=model_config['seq_len'],
            num_regime_classes=model_config['num_regime_classes']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def start_socket_server(self):
        """Start socket server for MQL5 communication."""
        host = self.config['socket']['host']
        port = self.config['socket']['port']
        
        def socket_thread():
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.socket.bind((host, port))
                self.socket.listen(1)
                logger.info(f"Socket server listening on {host}:{port}")
                
                while True:
                    try:
                        client_socket, address = self.socket.accept()
                        logger.info(f"Connected to MQL5 client at {address}")
                        self.socket_connected = True
                        
                        # Handle client communication
                        self.handle_client(client_socket)
                    except Exception as e:
                        logger.error(f"Socket error: {e}")
                        self.socket_connected = False
                        time.sleep(1)
            except Exception as e:
                logger.error(f"Failed to start socket server: {e}")
        
        thread = threading.Thread(target=socket_thread, daemon=True)
        thread.start()
    
    def handle_client(self, client_socket):
        """Handle communication with MQL5 client."""
        try:
            while True:
                # Receive data from MQL5
                data = client_socket.recv(4096)
                if not data:
                    break
                
                try:
                    message = json.loads(data.decode('utf-8'))
                    self.process_mql5_message(message, client_socket)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON received from MQL5")
        
        except Exception as e:
            logger.error(f"Client handling error: {e}")
        finally:
            client_socket.close()
            self.socket_connected = False
    
    def process_mql5_message(self, message: dict, client_socket):
        """
        Process message from MQL5 EA.
        
        Args:
            message: Message dictionary from MQL5
            client_socket: Client socket for response
        """
        msg_type = message.get('type')
        
        if msg_type == 'candle':
            # New candle data received
            candle = message.get('data')
            self.process_new_candle(candle)
            
            # Send prediction response
            if not self.prediction_buffer.empty():
                prediction = self.prediction_buffer.get()
                response = {
                    'type': 'prediction',
                    'data': prediction
                }
                client_socket.send(json.dumps(response).encode('utf-8'))
        
        elif msg_type == 'tick':
            # New tick data received
            tick = message.get('data')
            self.process_new_tick(tick)
        
        elif msg_type == 'position_update':
            # Position update from MT5
            position = message.get('data')
            self.current_position = position
            self.log_event('position_update', position)
    
    def process_new_candle(self, candle: dict):
        """
        Process new candle and generate prediction.
        
        Args:
            candle: Candle dictionary with OHLCV data
        """
        # Convert to DataFrame format
        candle_df = pd.DataFrame([{
            'time': pd.to_datetime(candle['time']),
            'open': candle['open'],
            'high': candle['high'],
            'low': candle['low'],
            'close': candle['close'],
            'volume': candle.get('volume', 0)
        }])
        candle_df = candle_df.set_index('time')
        
        # Add to recent candles buffer
        self.recent_candles.append(candle_df)
        
        # Keep only last N candles (model sequence length)
        seq_len = self.config['model']['seq_len']
        if len(self.recent_candles) > seq_len:
            self.recent_candles = self.recent_candles[-seq_len:]
        
        # Generate prediction if we have enough candles
        if len(self.recent_candles) >= seq_len:
            prediction = self.generate_prediction()
            if prediction:
                self.prediction_buffer.put(prediction)
                self.execute_trading_logic(prediction)
    
    def generate_prediction(self) -> Optional[dict]:
        """
        Generate prediction from current candle sequence.
        
        Returns:
            Prediction dictionary or None
        """
        try:
            # Combine recent candles
            df = pd.concat(self.recent_candles)
            
            # Process through pipeline (without target creation)
            feature_df = self.data_pipeline.feature_engineer.engineer_features(df)
            processed_df = pd.concat([df, feature_df], axis=1)
            
            # Standardize features (using pre-computed stats)
            processed_df = self.data_pipeline.standardize_features(processed_df, fit=False)
            
            # Get feature columns
            exclude_cols = ['open', 'high', 'low', 'close', 'volume']
            feature_cols = [col for col in processed_df.columns if col not in exclude_cols]
            
            # Prepare tensor
            features = processed_df[feature_cols].values.astype(np.float32)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Model prediction
            with torch.no_grad():
                regime_pred, context_pred, forecast_pred = self.model(features_tensor)
            
            # Process predictions
            regime_probs = torch.softmax(regime_pred, dim=1).cpu().numpy()[0]
            regime_class = regime_pred.argmax(dim=1).cpu().numpy()[0]
            forecast_probs = torch.softmax(forecast_pred, dim=1).cpu().numpy()[0]
            forecast_class = forecast_pred.argmax(dim=1).cpu().numpy()[0]
            context_strength = context_pred.squeeze().cpu().numpy()[0]
            
            prediction = {
                'timestamp': datetime.now().isoformat(),
                'regime': int(regime_class),
                'regime_probs': regime_probs.tolist(),
                'forecast': int(forecast_class),
                'forecast_probs': forecast_probs.tolist(),
                'context_strength': float(context_strength)
            }
            
            return prediction
        
        except Exception as e:
            logger.error(f"Prediction generation error: {e}")
            return None
    
    def execute_trading_logic(self, prediction: dict):
        """
        Execute trading logic based on prediction.
        
        Args:
            prediction: Prediction dictionary
        """
        trading_config = self.config['trading']
        
        # Check entry conditions
        forecast_prob_up = prediction['forecast_probs'][1]
        context_strength = prediction['context_strength']
        regime = prediction['regime']
        
        # Entry conditions
        min_confidence = trading_config['min_confidence']
        min_context_strength = trading_config['min_context_strength']
        
        # Bullish entry
        if (forecast_prob_up > min_confidence and 
            context_strength > min_context_strength and
            regime in [0, 1]):  # BULL_STRONG or BULL_WEAK
            
            if self.current_position is None or self.current_position['type'] != 'BUY':
                self.enter_position('BUY', prediction)
        
        # Bearish entry
        elif (forecast_prob_up < (1 - min_confidence) and 
              context_strength > min_context_strength and
              regime in [2, 3]):  # BEAR_STRONG or BEAR_WEAK
            
            if self.current_position is None or self.current_position['type'] != 'SELL':
                self.enter_position('SELL', prediction)
        
        # Check exit conditions
        if self.current_position:
            self.check_exit_conditions(prediction)
    
    def enter_position(self, position_type: str, prediction: dict):
        """
        Enter a new position.
        
        Args:
            position_type: 'BUY' or 'SELL'
            prediction: Current prediction
        """
        trading_config = self.config['trading']
        
        # Calculate entry zone (limit orders)
        current_price = self.recent_candles[-1]['close'].iloc[0]
        atr = self.recent_candles[-1].get('atr', current_price * 0.01)  # Fallback ATR
        
        if position_type == 'BUY':
            entry_price = current_price - (atr * trading_config['entry_offset'])
            stop_loss = entry_price - (atr * trading_config['stop_loss_atr_multiple'])
            take_profit = entry_price + (atr * trading_config['take_profit_atr_multiple'])
        else:  # SELL
            entry_price = current_price + (atr * trading_config['entry_offset'])
            stop_loss = entry_price + (atr * trading_config['stop_loss_atr_multiple'])
            take_profit = entry_price - (atr * trading_config['take_profit_atr_multiple'])
        
        position = {
            'type': position_type,
            'entry_price': float(entry_price),
            'stop_loss': float(stop_loss),
            'take_profit': float(take_profit),
            'entry_time': datetime.now().isoformat(),
            'prediction': prediction
        }
        
        self.current_position = position
        self.log_event('position_entry', position)
        
        # Send order to MQL5
        if self.socket_connected:
            order = {
                'type': 'order',
                'action': 'OPEN',
                'symbol': trading_config['symbol'],
                'position_type': position_type,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'lot_size': trading_config['lot_size']
            }
            # In real implementation, send via socket
            logger.info(f"Order sent: {order}")
    
    def check_exit_conditions(self, prediction: dict):
        """
        Check if exit conditions are met.
        
        Args:
            prediction: Current prediction
        """
        if not self.current_position:
            return
        
        # Check for ATR spike / deviation (cancel condition)
        current_price = self.recent_candles[-1]['close'].iloc[0]
        atr = self.recent_candles[-1].get('atr', current_price * 0.01)
        
        trading_config = self.config['trading']
        max_deviation = atr * trading_config['max_deviation_atr_multiple']
        
        price_deviation = abs(current_price - self.current_position['entry_price'])
        
        if price_deviation > max_deviation:
            self.exit_position('CANCEL', 'Max deviation exceeded')
            return
        
        # Trailing stop logic
        if trading_config.get('use_trailing_stop', False):
            self.update_trailing_stop(current_price, atr)
        
        # Check stop loss / take profit
        if self.current_position['type'] == 'BUY':
            if current_price <= self.current_position['stop_loss']:
                self.exit_position('STOP_LOSS', f"Price: {current_price}")
            elif current_price >= self.current_position['take_profit']:
                self.exit_position('TAKE_PROFIT', f"Price: {current_price}")
        else:  # SELL
            if current_price >= self.current_position['stop_loss']:
                self.exit_position('STOP_LOSS', f"Price: {current_price}")
            elif current_price <= self.current_position['take_profit']:
                self.exit_position('TAKE_PROFIT', f"Price: {current_price}")
    
    def update_trailing_stop(self, current_price: float, atr: float):
        """Update trailing stop loss."""
        trading_config = self.config['trading']
        trailing_distance = atr * trading_config.get('trailing_stop_atr_multiple', 2.0)
        
        if self.current_position['type'] == 'BUY':
            new_stop = current_price - trailing_distance
            if new_stop > self.current_position['stop_loss']:
                self.current_position['stop_loss'] = new_stop
                self.log_event('trailing_stop_update', {'new_stop': new_stop})
        else:  # SELL
            new_stop = current_price + trailing_distance
            if new_stop < self.current_position['stop_loss']:
                self.current_position['stop_loss'] = new_stop
                self.log_event('trailing_stop_update', {'new_stop': new_stop})
    
    def exit_position(self, exit_reason: str, details: str):
        """
        Exit current position.
        
        Args:
            exit_reason: Reason for exit
            details: Additional details
        """
        if not self.current_position:
            return
        
        exit_info = {
            'exit_reason': exit_reason,
            'exit_time': datetime.now().isoformat(),
            'exit_price': self.recent_candles[-1]['close'].iloc[0],
            'details': details,
            'position': self.current_position.copy()
        }
        
        self.log_event('position_exit', exit_info)
        
        # Send close order to MQL5
        if self.socket_connected:
            order = {
                'type': 'order',
                'action': 'CLOSE',
                'position_id': self.current_position.get('id')
            }
            logger.info(f"Close order sent: {order}")
        
        self.current_position = None
    
    def log_event(self, event_type: str, data: dict):
        """
        Log trading event.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'data': data
        }
        
        self.event_log.append(event)
        
        # Save to CSV periodically
        if len(self.event_log) % 10 == 0:
            self.save_event_log()
    
    def save_event_log(self):
        """Save event log to CSV."""
        if not self.event_log:
            return
        
        # Convert to DataFrame
        events_df = pd.DataFrame(self.event_log)
        events_df.to_csv(self.log_file, index=False, mode='a', header=not self.log_file.exists())
        
        # Clear buffer (keep last 1000 in memory)
        if len(self.event_log) > 1000:
            self.event_log = self.event_log[-1000:]
    
    def run(self):
        """Main server loop."""
        logger.info("Natron Server V5 started")
        logger.info("Waiting for MQL5 connection...")
        
        try:
            while True:
                time.sleep(1)
                # Periodic tasks can be added here
        except KeyboardInterrupt:
            logger.info("Shutting down server...")
            self.save_event_log()
            if self.socket:
                self.socket.close()


if __name__ == "__main__":
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/realtime_config.yaml"
    
    server = NatronServer(config_path)
    server.run()
