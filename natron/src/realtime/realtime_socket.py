"""
Realtime Socket Communication Module
Handles realtime data streaming for Natron trading engine.
"""

import socket
import json
import threading
import time
import logging
from typing import Optional, Callable
from queue import Queue
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealtimeSocket:
    """
    Socket-based realtime data stream handler.
    """
    
    def __init__(self, host: str = 'localhost', port: int = 8888, buffer_size: int = 4096):
        """
        Initialize realtime socket.
        
        Args:
            host: Socket host
            port: Socket port
            buffer_size: Buffer size for receiving data
        """
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.socket = None
        self.connected = False
        self.running = False
        self.data_queue = Queue()
        self.callback: Optional[Callable] = None
        self.thread: Optional[threading.Thread] = None
    
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
            logger.info(f"Connected to socket at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to socket: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from socket."""
        self.running = False
        self.connected = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        logger.info("Disconnected from socket")
    
    def set_callback(self, callback: Callable):
        """
        Set callback function for received data.
        
        Args:
            callback: Function to call when data is received
        """
        self.callback = callback
    
    def _receive_loop(self):
        """Internal receive loop running in separate thread."""
        buffer = ""
        
        while self.running and self.connected:
            try:
                data = self.socket.recv(self.buffer_size).decode('utf-8')
                if not data:
                    break
                
                buffer += data
                
                # Process complete JSON messages
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        try:
                            message = json.loads(line)
                            self.data_queue.put(message)
                            
                            if self.callback:
                                self.callback(message)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse JSON: {e}")
                
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"Error receiving data: {e}")
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
    
    def send_message(self, message: dict) -> bool:
        """
        Send message through socket.
        
        Args:
            message: Dictionary to send as JSON
            
        Returns:
            True if sent successfully
        """
        if not self.connected:
            return False
        
        try:
            json_str = json.dumps(message) + '\n'
            self.socket.sendall(json_str.encode('utf-8'))
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    def get_latest_data(self, timeout: float = 1.0) -> Optional[dict]:
        """
        Get latest data from queue.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Latest message or None
        """
        try:
            return self.data_queue.get(timeout=timeout)
        except:
            return None


class CandleBuffer:
    """
    Buffer for accumulating candles into sequences.
    """
    
    def __init__(self, sequence_length: int = 96):
        """
        Initialize candle buffer.
        
        Args:
            sequence_length: Required sequence length for model input
        """
        self.sequence_length = sequence_length
        self.buffer = []
    
    def add_candle(self, candle: dict):
        """
        Add a new candle to buffer.
        
        Args:
            candle: Candle dictionary with OHLCV data
        """
        self.buffer.append(candle)
        
        # Keep only last sequence_length candles
        if len(self.buffer) > self.sequence_length:
            self.buffer = self.buffer[-self.sequence_length:]
    
    def get_sequence(self) -> Optional[pd.DataFrame]:
        """
        Get current sequence as DataFrame.
        
        Returns:
            DataFrame with sequence of candles or None if not enough data
        """
        if len(self.buffer) < self.sequence_length:
            return None
        
        df = pd.DataFrame(self.buffer)
        return df
    
    def clear(self):
        """Clear buffer."""
        self.buffer = []
