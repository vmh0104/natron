"""
Monitoring Script for Natron Trading System
Tracks model performance, API health, and trading metrics
"""

import time
import requests
import psutil
import GPUtil
import json
from datetime import datetime
from typing import Dict, Optional


class NatronMonitor:
    """Monitor Natron system health and performance"""
    
    def __init__(self, api_url: str = "http://localhost:5000", socket_host: str = "127.0.0.1", socket_port: int = 8888):
        self.api_url = api_url
        self.socket_host = socket_host
        self.socket_port = socket_port
        
    def check_api_health(self) -> Dict:
        """Check Flask API health"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                return {
                    'status': 'healthy',
                    'response': response.json()
                }
            else:
                return {
                    'status': 'unhealthy',
                    'status_code': response.status_code
                }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def check_socket_server(self) -> Dict:
        """Check socket server connectivity"""
        import socket
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((self.socket_host, self.socket_port))
            sock.close()
            
            if result == 0:
                return {'status': 'connected'}
            else:
                return {'status': 'disconnected'}
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_system_metrics(self) -> Dict:
        """Get system resource metrics"""
        metrics = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'disk_percent': psutil.disk_usage('/').percent,
        }
        
        # GPU metrics if available
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                metrics['gpu'] = {
                    'name': gpu.name,
                    'load_percent': gpu.load * 100,
                    'memory_used_mb': gpu.memoryUsed,
                    'memory_total_mb': gpu.memoryTotal,
                    'temperature': gpu.temperature
                }
        except:
            metrics['gpu'] = None
        
        return metrics
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {}
        except:
            return {}
    
    def monitor_loop(self, interval: int = 60):
        """Run monitoring loop"""
        print("🔍 Natron Monitoring Started")
        print("=" * 60)
        
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{timestamp}] Monitoring Update")
            print("-" * 60)
            
            # API Health
            api_health = self.check_api_health()
            print(f"API Status: {api_health.get('status', 'unknown')}")
            if 'response' in api_health:
                model_loaded = api_health['response'].get('model_loaded', False)
                device = api_health['response'].get('device', 'unknown')
                print(f"  Model Loaded: {model_loaded}")
                print(f"  Device: {device}")
            
            # Socket Server
            socket_status = self.check_socket_server()
            print(f"Socket Server: {socket_status.get('status', 'unknown')}")
            
            # System Metrics
            system_metrics = self.get_system_metrics()
            print(f"\nSystem Resources:")
            print(f"  CPU: {system_metrics['cpu_percent']:.1f}%")
            print(f"  Memory: {system_metrics['memory_percent']:.1f}% "
                  f"({system_metrics['memory_used_gb']:.2f} GB / {system_metrics['memory_total_gb']:.2f} GB)")
            print(f"  Disk: {system_metrics['disk_percent']:.1f}%")
            
            if system_metrics.get('gpu'):
                gpu = system_metrics['gpu']
                print(f"\nGPU ({gpu['name']}):")
                print(f"  Load: {gpu['load_percent']:.1f}%")
                print(f"  Memory: {gpu['memory_used_mb']} MB / {gpu['memory_total_mb']} MB")
                print(f"  Temperature: {gpu['temperature']}°C")
            
            print("=" * 60)
            
            time.sleep(interval)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor Natron Trading System')
    parser.add_argument('--api_url', type=str, default='http://localhost:5000',
                       help='API server URL')
    parser.add_argument('--socket_host', type=str, default='127.0.0.1',
                       help='Socket server host')
    parser.add_argument('--socket_port', type=int, default=8888,
                       help='Socket server port')
    parser.add_argument('--interval', type=int, default=60,
                       help='Monitoring interval in seconds')
    
    args = parser.parse_args()
    
    monitor = NatronMonitor(
        api_url=args.api_url,
        socket_host=args.socket_host,
        socket_port=args.socket_port
    )
    
    try:
        monitor.monitor_loop(interval=args.interval)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


if __name__ == '__main__':
    main()
