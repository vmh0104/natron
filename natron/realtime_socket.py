"""
Realtime Socket Communication Module for Natron AI Trading System
Handles live candle/tick stream via socket communication.
"""

import socket
import json
import threading
import time
import pandas as pd
from datetime import datetime
from typing import Callable, Optional, Dict
from queue import Queue
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealtimeSocket:
    """
    Socket-based realtime data stream handler.
    """
    
    def __init__(self, host: str = 'localhost', port: int = 8888):
        """
        Initialize realtime socket.
        
        Args:
            host: Socket host
            port: Socket port
        """
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.data_queue = Queue()
        self.callbacks = []
        self.running = False
        self.thread = None
    
    def connect(self) -> bool:
        """
        Connect to socket server.
        
        Returns:
            True if connected successfully
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            self.socket.connect((self.host, self.port))
            self.connected = True
            logger.info(f"Connected to socket server at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to socket: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from socket server."""
        self.running = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.connected = False
        logger.info("Disconnected from socket server")
    
    def register_callback(self, callback: Callable):
        """
        Register callback function for new data.
        
        Args:
            callback: Function that takes (data_dict) as argument
        """
        self.callbacks.append(callback)
    
    def _receive_loop(self):
        """Internal receive loop running in separate thread."""
        buffer = ""
        
        while self.running and self.connected:
            try:
                data = self.socket.recv(4096).decode('utf-8')
                if not data:
                    break
                
                buffer += data
                
                # Process complete JSON messages
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        try:
                            data_dict = json.loads(line)
                            self.data_queue.put(data_dict)
                            
                            # Call registered callbacks
                            for callback in self.callbacks:
                                try:
                                    callback(data_dict)
                                except Exception as e:
                                    logger.error(f"Callback error: {e}")
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse JSON: {e}")
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"Receive error: {e}")
                break
        
        self.connected = False
    
    def start_listening(self):
        """Start listening for data in background thread."""
        if not self.connected:
            if not self.connect():
                return False
        
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        logger.info("Started listening for realtime data")
        return True
    
    def stop_listening(self):
        """Stop listening for data."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def get_latest_data(self, timeout: float = 1.0) -> Optional[Dict]:
        """
        Get latest data from queue (non-blocking).
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Data dictionary or None
        """
        try:
            return self.data_queue.get(timeout=timeout)
        except:
            return None
    
    def send_command(self, command: Dict) -> bool:
        """
        Send command to socket server.
        
        Args:
            command: Command dictionary
            
        Returns:
            True if sent successfully
        """
        if not self.connected:
            return False
        
        try:
            message = json.dumps(command) + '\n'
            self.socket.sendall(message.encode('utf-8'))
            return True
        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            return False


class CandleBuffer:
    """
    Buffer for accumulating candles and triggering predictions.
    """
    
    def __init__(self, sequence_length: int = 96):
        """
        Initialize candle buffer.
        
        Args:
            sequence_length: Required sequence length for model
        """
        self.sequence_length = sequence_length
        self.candles = []
        self.lock = threading.Lock()
    
    def add_candle(self, candle: Dict):
        """
        Add new candle to buffer.
        
        Args:
            candle: Candle dictionary with OHLCV data
        """
        with self.lock:
            # Convert to DataFrame row format
            candle_row = {
                'time': pd.to_datetime(candle.get('time', datetime.now())),
                'open': float(candle['open']),
                'high': float(candle['high']),
                'low': float(candle['low']),
                'close': float(candle['close']),
                'volume': float(candle.get('volume', 0))
            }
            
            self.candles.append(candle_row)
            
            # Keep only last sequence_length candles
            if len(self.candles) > self.sequence_length:
                self.candles.pop(0)
    
    def get_sequence(self) -> Optional[pd.DataFrame]:
        """
        Get current sequence if buffer is full.
        
        Returns:
            DataFrame with sequence_length candles or None
        """
        with self.lock:
            if len(self.candles) >= self.sequence_length:
                df = pd.DataFrame(self.candles)
                df = df.set_index('time')
                return df
            return None
    
    def clear(self):
        """Clear buffer."""
        with self.lock:
            self.candles.clear()
