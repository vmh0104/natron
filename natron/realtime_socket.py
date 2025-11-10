"""
Natron Realtime Socket Client
Connects to data feed and streams candles to Natron Server.
"""

import socket
import json
import time
import logging
from typing import Dict, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealtimeSocketClient:
    """
    Client for receiving realtime market data via socket.
    """
    
    def __init__(self, host: str = 'localhost', port: int = 9999):
        """
        Initialize socket client.
        
        Args:
            host: Socket host
            port: Socket port
        """
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
    
    def connect(self):
        """Connect to data feed socket."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            logger.info(f"Connected to data feed at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self.connected = False
    
    def receive_candle(self) -> Optional[Dict]:
        """
        Receive a candle from the socket.
        
        Expected format: JSON string with OHLCV data
        
        Returns:
            Candle dictionary or None
        """
        if not self.connected:
            return None
        
        try:
            # Receive data (assuming JSON format)
            data = self.socket.recv(4096).decode('utf-8')
            
            if not data:
                return None
            
            # Parse JSON
            candle = json.loads(data)
            
            # Validate required fields
            required = ['time', 'open', 'high', 'low', 'close', 'volume']
            if all(key in candle for key in required):
                return candle
            else:
                logger.warning(f"Invalid candle format: {candle}")
                return None
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Error receiving candle: {e}")
            return None
    
    def close(self):
        """Close socket connection."""
        if self.socket:
            self.socket.close()
            self.connected = False
            logger.info("Socket closed")


if __name__ == "__main__":
    # Example usage
    client = RealtimeSocketClient()
    client.connect()
    
    if client.connected:
        try:
            while True:
                candle = client.receive_candle()
                if candle:
                    print(f"Received candle: {candle}")
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            client.close()
