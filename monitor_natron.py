"""
Monitoring Script for Natron Trading System
Tracks model performance, API health, and trading metrics
"""

import time
import json
import requests
import socket
import psutil
import torch
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NatronMonitor:
    """Monitor Natron system health and performance"""
    
    def __init__(self, api_url: str = "http://localhost:5000", socket_host: str = "localhost", socket_port: int = 8888):
        self.api_url = api_url
        self.socket_host = socket_host
        self.socket_port = socket_port
        self.metrics = {
            'api_health': False,
            'socket_health': False,
            'gpu_available': torch.cuda.is_available(),
            'gpu_memory': 0,
            'cpu_usage': 0,
            'memory_usage': 0,
            'last_check': None
        }
    
    def check_api_health(self) -> bool:
        """Check Flask API health"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.metrics['api_health'] = data.get('status') == 'healthy'
                self.metrics['model_loaded'] = data.get('model_loaded', False)
                return True
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            self.metrics['api_health'] = False
        return False
    
    def check_socket_health(self) -> bool:
        """Check socket server health"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((self.socket_host, self.socket_port))
            sock.close()
            self.metrics['socket_health'] = (result == 0)
            return result == 0
        except Exception as e:
            logger.error(f"Socket health check failed: {e}")
            self.metrics['socket_health'] = False
        return False
    
    def check_system_resources(self):
        """Check system resource usage"""
        # CPU usage
        self.metrics['cpu_usage'] = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.metrics['memory_usage'] = memory.percent
        self.metrics['memory_available_gb'] = memory.available / (1024**3)
        
        # GPU usage
        if torch.cuda.is_available():
            self.metrics['gpu_memory'] = torch.cuda.memory_allocated() / (1024**3)  # GB
            self.metrics['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.metrics['gpu_utilization'] = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
    
    def run_check(self):
        """Run all health checks"""
        logger.info("Running health checks...")
        
        self.check_api_health()
        self.check_socket_health()
        self.check_system_resources()
        
        self.metrics['last_check'] = datetime.now().isoformat()
        
        return self.metrics
    
    def print_status(self):
        """Print current status"""
        print("\n" + "="*60)
        print("NATRON SYSTEM STATUS")
        print("="*60)
        print(f"Timestamp: {self.metrics['last_check']}")
        print(f"\nAPI Server: {'✓ Healthy' if self.metrics['api_health'] else '✗ Unhealthy'}")
        print(f"Socket Server: {'✓ Connected' if self.metrics['socket_health'] else '✗ Disconnected'}")
        print(f"GPU Available: {'✓ Yes' if self.metrics['gpu_available'] else '✗ No'}")
        
        if self.metrics['gpu_available']:
            print(f"GPU Memory: {self.metrics.get('gpu_memory', 0):.2f} GB / {self.metrics.get('gpu_memory_total', 0):.2f} GB")
        
        print(f"\nCPU Usage: {self.metrics['cpu_usage']:.1f}%")
        print(f"Memory Usage: {self.metrics['memory_usage']:.1f}%")
        print(f"Memory Available: {self.metrics.get('memory_available_gb', 0):.2f} GB")
        print("="*60 + "\n")
    
    def monitor_loop(self, interval: int = 60):
        """Run monitoring loop"""
        logger.info(f"Starting monitoring loop (interval: {interval}s)")
        
        try:
            while True:
                self.run_check()
                self.print_status()
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Monitoring stopped")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Natron System Monitor')
    parser.add_argument('--api-url', type=str, default='http://localhost:5000', help='API URL')
    parser.add_argument('--socket-host', type=str, default='localhost', help='Socket host')
    parser.add_argument('--socket-port', type=int, default=8888, help='Socket port')
    parser.add_argument('--interval', type=int, default=60, help='Check interval (seconds)')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    
    args = parser.parse_args()
    
    monitor = NatronMonitor(
        api_url=args.api_url,
        socket_host=args.socket_host,
        socket_port=args.socket_port
    )
    
    if args.once:
        monitor.run_check()
        monitor.print_status()
    else:
        monitor.monitor_loop(interval=args.interval)


if __name__ == '__main__':
    main()
