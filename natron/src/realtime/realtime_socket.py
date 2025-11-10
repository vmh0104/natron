"""
Realtime Socket Communication Module

This module handles socket communication for realtime data streaming.
"""

import socket
import json
import threading
import time
import logging
from typing import Callable, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealtimeSocketClient:
    """
    Client for connecting to realtime data stream.
    """
    
    def __init__(self, host: str, port: int):
        """
        Initialize socket client.
        
        Args:
            host: Server host
            port: Server port
        """
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.callbacks = []
    
    def connect(self):
        """Connect to server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            logger.info(f"Connected to {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.connected = False
            return False
    
    def register_callback(self, callback: Callable):
        """
        Register callback for received data.
        
        Args:
            callback: Function to call with received data
        """
        self.callbacks.append(callback)
    
    def send(self, data: dict):
        """
        Send data to server.
        
        Args:
            data: Data dictionary to send
        """
        if not self.connected:
            logger.warning("Not connected, cannot send data")
            return False
        
        try:
            message = json.dumps(data).encode('utf-8')
            self.socket.send(message)
            return True
        except Exception as e:
            logger.error(f"Send error: {e}")
            self.connected = False
            return False
    
    def listen(self):
        """Listen for incoming data."""
        if not self.connected:
            return
        
        try:
            while self.connected:
                data = self.socket.recv(4096)
                if not data:
                    break
                
                try:
                    message = json.loads(data.decode('utf-8'))
                    for callback in self.callbacks:
                        callback(message)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON received")
        
        except Exception as e:
            logger.error(f"Listen error: {e}")
            self.connected = False
    
    def start_listening(self):
        """Start listening in background thread."""
        thread = threading.Thread(target=self.listen, daemon=True)
        thread.start()
    
    def disconnect(self):
        """Disconnect from server."""
        if self.socket:
            self.socket.close()
        self.connected = False
        logger.info("Disconnected")


class RealtimeSocketServer:
    """
    Server for broadcasting realtime data.
    """
    
    def __init__(self, host: str, port: int):
        """
        Initialize socket server.
        
        Args:
            host: Server host
            port: Server port
        """
        self.host = host
        self.port = port
        self.socket = None
        self.clients = []
        self.running = False
    
    def start(self):
        """Start server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(5)
            self.running = True
            logger.info(f"Socket server started on {self.host}:{self.port}")
            
            # Accept connections in background
            thread = threading.Thread(target=self._accept_connections, daemon=True)
            thread.start()
            
            return True
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False
    
    def _accept_connections(self):
        """Accept client connections."""
        while self.running:
            try:
                client_socket, address = self.socket.accept()
                logger.info(f"Client connected: {address}")
                self.clients.append(client_socket)
                
                # Handle client in background
                thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket,),
                    daemon=True
                )
                thread.start()
            except Exception as e:
                if self.running:
                    logger.error(f"Accept error: {e}")
    
    def _handle_client(self, client_socket):
        """Handle client communication."""
        try:
            while True:
                data = client_socket.recv(4096)
                if not data:
                    break
                # Process client data if needed
        except Exception as e:
            logger.error(f"Client handling error: {e}")
        finally:
            if client_socket in self.clients:
                self.clients.remove(client_socket)
            client_socket.close()
    
    def broadcast(self, data: dict):
        """
        Broadcast data to all connected clients.
        
        Args:
            data: Data dictionary to broadcast
        """
        message = json.dumps(data).encode('utf-8')
        disconnected_clients = []
        
        for client in self.clients:
            try:
                client.send(message)
            except Exception as e:
                logger.error(f"Broadcast error: {e}")
                disconnected_clients.append(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            if client in self.clients:
                self.clients.remove(client)
    
    def stop(self):
        """Stop server."""
        self.running = False
        if self.socket:
            self.socket.close()
        for client in self.clients:
            client.close()
        self.clients = []
        logger.info("Socket server stopped")
