"""
Natron Server V5 - Realtime Trading Engine
Main server that coordinates model inference and trading execution.
"""

import torch
import pandas as pd
import numpy as np
import yaml
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import MetaTrader5 as mt5

from ..training.model_natron import NatronTransformer
from ..training.dataset_loader import NatronDataset
from .realtime_socket import RealtimeSocket, CandleBuffer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NatronServer:
    """
    Main Natron trading server.
    """
    
    REGIME_NAMES = ['BULL_STRONG', 'BULL_WEAK', 'BEAR_STRONG', 'BEAR_WEAK', 'RANGE', 'VOLATILE']
    FORECAST_NAMES = ['DOWN', 'UP']
    
    def __init__(self, config_path: str):
        """
        Initialize Natron server.
        
        Args:
            config_path: Path to realtime config YAML
        """
        self.config = self._load_config(config_path)
        self.device = torch.device(self.config['inference']['device'] if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model()
        
        # Load scaler
        model_dir = Path(self.config['inference']['model_path']).parent
        scaler_path = model_dir / 'scaler.pkl'
        self.scaler = NatronDataset.load_scaler(str(scaler_path))
        
        # Initialize components
        self.socket = RealtimeSocket(
            host=self.config['socket']['host'],
            port=self.config['socket']['port'],
            buffer_size=self.config['socket']['buffer_size']
        )
        
        self.candle_buffer = CandleBuffer(
            sequence_length=self.config['inference']['sequence_length']
        )
        
        # Trading state
        self.current_position = None
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_trade_time = None
        
        # Initialize MT5 connection
        self.mt5_connected = False
        self._connect_mt5()
        
        # Event log
        self.events_log = []
        
        logger.info("Natron Server initialized")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_model(self) -> NatronTransformer:
        """Load trained model."""
        checkpoint = torch.load(self.config['inference']['model_path'], map_location=self.device)
        
        # Get config from checkpoint or use defaults
        model_config = checkpoint.get('config', {}).get('model', {})
        
        model = NatronTransformer(
            input_dim=checkpoint['input_dim'],
            d_model=model_config.get('d_model', 256),
            nhead=model_config.get('nhead', 8),
            num_layers=model_config.get('num_layers', 6),
            dim_feedforward=model_config.get('dim_feedforward', 1024),
            dropout=model_config.get('dropout', 0.1),
            activation=model_config.get('activation', 'gelu'),
            regime_classes=model_config.get('regime_classes', 6),
            context_output_dim=model_config.get('context_output_dim', 1),
            forecast_classes=model_config.get('forecast_classes', 2)
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        logger.info(f"Loaded model from {self.config['inference']['model_path']}")
        return model
    
    def _connect_mt5(self) -> bool:
        """Connect to MetaTrader 5."""
        mt5_config = self.config['metatrader']
        
        if not mt5.initialize(path=mt5_config.get('path'), login=mt5_config.get('login', 0),
                              password=mt5_config.get('password', ''),
                              server=mt5_config.get('server', '')):
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            self.mt5_connected = False
            return False
        
        self.mt5_connected = True
        logger.info("Connected to MetaTrader 5")
        return True
    
    def _log_event(self, event_type: str, data: dict):
        """
        Log trading event.
        
        Args:
            event_type: Type of event (prediction, trade, error, etc.)
            data: Event data
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'data': data
        }
        self.events_log.append(event)
        
        # Save to CSV periodically
        if len(self.events_log) % 10 == 0:
            self._save_events_log()
    
    def _save_events_log(self):
        """Save events log to CSV."""
        if not self.events_log:
            return
        
        log_file = self.config['logging']['events_file']
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame
        df_events = pd.DataFrame([
            {
                'timestamp': e['timestamp'],
                'type': e['type'],
                **e['data']
            }
            for e in self.events_log
        ])
        
        # Append to file
        if Path(log_file).exists():
            df_events.to_csv(log_file, mode='a', header=False, index=False)
        else:
            df_events.to_csv(log_file, index=False)
        
        # Clear log (keep in memory for now)
        self.events_log = []
    
    def _process_candle(self, candle_data: dict):
        """
        Process incoming candle data.
        
        Args:
            candle_data: Dictionary with candle OHLCV data
        """
        # Add to buffer
        self.candle_buffer.add_candle(candle_data)
        
        # Get sequence if ready
        sequence_df = self.candle_buffer.get_sequence()
        if sequence_df is None:
            return
        
        # Make prediction
        prediction = self._make_prediction(sequence_df)
        
        if prediction:
            self._log_event('prediction', prediction)
            self._evaluate_trading_signal(prediction, candle_data)
    
    def _make_prediction(self, sequence_df: pd.DataFrame) -> Optional[dict]:
        """
        Make model prediction on sequence.
        
        Args:
            sequence_df: DataFrame with sequence of candles
            
        Returns:
            Prediction dictionary or None
        """
        try:
            # Prepare features (simplified - in production, use full feature engineering)
            # For now, assume sequence_df already has features
            feature_cols = [col for col in sequence_df.columns 
                           if col not in ['time', 'open', 'high', 'low', 'close', 'volume', 'regime', 'regime_name']]
            
            if not feature_cols:
                # Fallback: use OHLCV only
                feature_cols = ['open', 'high', 'low', 'close']
                if 'volume' in sequence_df.columns:
                    feature_cols.append('volume')
            
            features = sequence_df[feature_cols].values.astype(np.float32)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(features_scaled).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                predictions = self.model.predict(features_tensor)
            
            # Extract predictions
            regime_pred = predictions['regime_pred'][0].cpu().item()
            regime_probs = predictions['regime_probs'][0].cpu().numpy()
            forecast_pred = predictions['forecast_pred'][0].cpu().item()
            forecast_probs = predictions['forecast_probs'][0].cpu().numpy()
            context_score = predictions['context_score'][0].cpu().item()
            
            prediction = {
                'regime': int(regime_pred),
                'regime_name': self.REGIME_NAMES[regime_pred],
                'regime_confidence': float(regime_probs.max()),
                'forecast': int(forecast_pred),
                'forecast_name': self.FORECAST_NAMES[forecast_pred],
                'forecast_confidence': float(forecast_probs.max()),
                'context_score': float(context_score),
                'timestamp': datetime.now().isoformat()
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None
    
    def _evaluate_trading_signal(self, prediction: dict, candle_data: dict):
        """
        Evaluate trading signal and execute if conditions met.
        
        Args:
            prediction: Model prediction
            candle_data: Current candle data
        """
        trading_config = self.config['trading']
        safety_config = self.config['safety']
        
        # Check safety limits
        if safety_config['max_daily_trades'] > 0 and self.daily_trades >= safety_config['max_daily_trades']:
            logger.warning("Daily trade limit reached")
            return
        
        if safety_config['max_daily_loss'] > 0 and self.daily_pnl <= -safety_config['max_daily_loss']:
            logger.warning("Daily loss limit reached")
            return
        
        if safety_config['emergency_stop']:
            logger.warning("Emergency stop activated")
            return
        
        # Check regime filter
        if prediction['regime_name'] not in trading_config['regime_filter']:
            return
        
        # Check confidence threshold
        min_confidence = max(
            prediction['regime_confidence'],
            prediction['forecast_confidence']
        )
        
        if min_confidence < trading_config['entry_confidence_threshold']:
            return
        
        # Determine trade direction
        if prediction['forecast_name'] == 'UP':
            trade_type = 'BUY'
        else:
            trade_type = 'SELL'
        
        # Execute trade
        self._execute_trade(trade_type, prediction, candle_data)
    
    def _execute_trade(self, trade_type: str, prediction: dict, candle_data: dict):
        """
        Execute a trade.
        
        Args:
            trade_type: 'BUY' or 'SELL'
            prediction: Model prediction
            candle_data: Current candle data
        """
        if not self.mt5_connected:
            logger.error("MT5 not connected, cannot execute trade")
            return
        
        trading_config = self.config['trading']
        symbol = self.config['metatrader']['symbol']
        
        # Get current price
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol {symbol} not found")
            return
        
        if trade_type == 'BUY':
            price = mt5.symbol_info_tick(symbol).ask
            order_type = mt5.ORDER_TYPE_BUY
        else:
            price = mt5.symbol_info_tick(symbol).bid
            order_type = mt5.ORDER_TYPE_SELL
        
        # Calculate position size
        account_info = mt5.account_info()
        balance = account_info.balance
        risk_amount = balance * trading_config['risk_per_trade']
        
        # Calculate stop loss and take profit
        atr = candle_data.get('atr', symbol_info.point * 100)  # Fallback ATR
        stop_loss_distance = atr * trading_config['stop_loss_atr_multiplier']
        take_profit_distance = atr * trading_config['take_profit_atr_multiplier']
        
        if trade_type == 'BUY':
            sl = price - stop_loss_distance
            tp = price + take_profit_distance
        else:
            sl = price + stop_loss_distance
            tp = price - take_profit_distance
        
        # Calculate lot size
        lot_size = min(
            risk_amount / stop_loss_distance / symbol_info.trade_contract_size,
            trading_config['max_position_size']
        )
        lot_size = round(lot_size, 2)
        
        # Place order
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 234000,
            "comment": f"Natron {prediction['regime_name']} {prediction['forecast_name']}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Trade failed: {result.retcode} - {result.comment}")
            self._log_event('trade_error', {
                'trade_type': trade_type,
                'error': result.comment,
                'retcode': result.retcode
            })
        else:
            logger.info(f"Trade executed: {trade_type} {lot_size} lots at {price}")
            self.daily_trades += 1
            self.last_trade_time = datetime.now()
            
            self._log_event('trade_executed', {
                'trade_type': trade_type,
                'lot_size': lot_size,
                'price': price,
                'stop_loss': sl,
                'take_profit': tp,
                'prediction': prediction
            })
    
    def start(self):
        """Start the server."""
        logger.info("=" * 60)
        logger.info("NATRON SERVER V5 - Starting")
        logger.info("=" * 60)
        
        # Set callback for incoming data
        self.socket.set_callback(self._process_candle)
        
        # Start listening
        if not self.socket.start_listening():
            logger.error("Failed to start socket listener")
            return
        
        logger.info("Server running. Press Ctrl+C to stop.")
        
        try:
            # Main loop
            while True:
                time.sleep(1)
                
                # Check for new data
                data = self.socket.get_latest_data(timeout=0.1)
                if data:
                    if data.get('type') == 'candle':
                        self._process_candle(data['data'])
                
                # Periodic tasks
                # Save events log
                if len(self.events_log) > 0:
                    self._save_events_log()
                
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.socket.disconnect()
            if self.mt5_connected:
                mt5.shutdown()
            self._save_events_log()
            logger.info("Server stopped")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Natron Server V5')
    parser.add_argument('--config', type=str, required=True, help='Path to realtime config YAML')
    
    args = parser.parse_args()
    
    server = NatronServer(args.config)
    server.start()


if __name__ == '__main__':
    main()
