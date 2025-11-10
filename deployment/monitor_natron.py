"""
Natron Transformer - System Monitoring
Real-time monitoring dashboard for all services
"""

import time
import psutil
import requests
import socket as sock
import json
from datetime import datetime
from typing import Dict, List
import os
import sys


class NatronMonitor:
    """
    System monitor for Natron Transformer
    Tracks GPU, CPU, memory, and service health
    """
    
    def __init__(self):
        self.api_url = "http://localhost:5000"
        self.socket_host = "localhost"
        self.socket_port = 9090
        
        # Try to import pynvml for GPU monitoring
        try:
            import pynvml
            pynvml.nvmlInit()
            self.gpu_available = True
            self.nvml = pynvml
        except:
            self.gpu_available = False
            self.nvml = None
    
    def get_system_info(self) -> Dict:
        """Get system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        info = {
            'cpu_percent': cpu_percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_total_gb': memory.total / (1024**3),
            'memory_percent': memory.percent,
            'disk_used_gb': disk.used / (1024**3),
            'disk_total_gb': disk.total / (1024**3),
            'disk_percent': disk.percent
        }
        
        return info
    
    def get_gpu_info(self) -> List[Dict]:
        """Get GPU information"""
        if not self.gpu_available:
            return []
        
        gpu_count = self.nvml.nvmlDeviceGetCount()
        gpus = []
        
        for i in range(gpu_count):
            handle = self.nvml.nvmlDeviceGetHandleByIndex(i)
            
            name = self.nvml.nvmlDeviceGetName(handle)
            memory = self.nvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = self.nvml.nvmlDeviceGetUtilizationRates(handle)
            temperature = self.nvml.nvmlDeviceGetTemperature(handle, self.nvml.NVML_TEMPERATURE_GPU)
            
            gpus.append({
                'index': i,
                'name': name.decode('utf-8') if isinstance(name, bytes) else name,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'memory_percent': (memory.used / memory.total) * 100,
                'utilization_percent': utilization.gpu,
                'temperature': temperature
            })
        
        return gpus
    
    def check_api_health(self) -> Dict:
        """Check API server health"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                return {
                    'status': 'healthy',
                    'data': response.json()
                }
            else:
                return {
                    'status': 'unhealthy',
                    'error': f'Status code: {response.status_code}'
                }
        except requests.exceptions.ConnectionError:
            return {
                'status': 'offline',
                'error': 'Connection refused'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def check_socket_health(self) -> Dict:
        """Check socket server health"""
        try:
            s = sock.socket(sock.AF_INET, sock.SOCK_STREAM)
            s.settimeout(5)
            s.connect((self.socket_host, self.socket_port))
            
            # Send ping
            ping = json.dumps({'type': 'ping'})
            s.send(ping.encode('utf-8'))
            
            # Receive pong
            response = s.recv(4096)
            s.close()
            
            data = json.loads(response.decode('utf-8'))
            
            if data.get('type') == 'pong':
                return {
                    'status': 'healthy',
                    'data': data
                }
            else:
                return {
                    'status': 'unhealthy',
                    'error': 'Invalid response'
                }
        except sock.timeout:
            return {
                'status': 'timeout',
                'error': 'Connection timeout'
            }
        except ConnectionRefusedError:
            return {
                'status': 'offline',
                'error': 'Connection refused'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def print_dashboard(self):
        """Print monitoring dashboard"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * 80)
        print("🧠 NATRON TRANSFORMER - SYSTEM MONITOR")
        print("=" * 80)
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("")
        
        # System info
        sys_info = self.get_system_info()
        print("🖥️  SYSTEM RESOURCES")
        print("-" * 80)
        print(f"CPU Usage:    {sys_info['cpu_percent']:.1f}%")
        print(f"Memory:       {sys_info['memory_used_gb']:.2f} GB / {sys_info['memory_total_gb']:.2f} GB ({sys_info['memory_percent']:.1f}%)")
        print(f"Disk:         {sys_info['disk_used_gb']:.2f} GB / {sys_info['disk_total_gb']:.2f} GB ({sys_info['disk_percent']:.1f}%)")
        print("")
        
        # GPU info
        gpus = self.get_gpu_info()
        if gpus:
            print("🎮 GPU STATUS")
            print("-" * 80)
            for gpu in gpus:
                print(f"GPU {gpu['index']}: {gpu['name']}")
                print(f"  Memory:       {gpu['memory_used_gb']:.2f} GB / {gpu['memory_total_gb']:.2f} GB ({gpu['memory_percent']:.1f}%)")
                print(f"  Utilization:  {gpu['utilization_percent']}%")
                print(f"  Temperature:  {gpu['temperature']}°C")
            print("")
        else:
            print("⚠️  No GPU detected or monitoring unavailable")
            print("")
        
        # API health
        api_health = self.check_api_health()
        print("🌐 API SERVER")
        print("-" * 80)
        status_symbol = "✅" if api_health['status'] == 'healthy' else "❌"
        print(f"{status_symbol} Status: {api_health['status'].upper()}")
        if 'error' in api_health:
            print(f"   Error: {api_health['error']}")
        print(f"   URL: {self.api_url}")
        print("")
        
        # Socket health
        socket_health = self.check_socket_health()
        print("🔌 SOCKET SERVER")
        print("-" * 80)
        status_symbol = "✅" if socket_health['status'] == 'healthy' else "❌"
        print(f"{status_symbol} Status: {socket_health['status'].upper()}")
        if 'error' in socket_health:
            print(f"   Error: {socket_health['error']}")
        print(f"   Address: {self.socket_host}:{self.socket_port}")
        print("")
        
        # Process info
        print("📊 PYTHON PROCESSES")
        print("-" * 80)
        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
            try:
                if 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    if 'natron' in cmdline.lower() or 'api_server' in cmdline or 'socket_server' in cmdline:
                        python_processes.append({
                            'pid': proc.info['pid'],
                            'cmdline': cmdline[:60] + '...' if len(cmdline) > 60 else cmdline,
                            'cpu': proc.info['cpu_percent'],
                            'memory_mb': proc.info['memory_info'].rss / (1024**2)
                        })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if python_processes:
            for p in python_processes:
                print(f"PID {p['pid']}: {p['cmdline']}")
                print(f"  CPU: {p['cpu']:.1f}%  Memory: {p['memory_mb']:.2f} MB")
        else:
            print("No Natron processes found")
        
        print("")
        print("=" * 80)
        print("Press Ctrl+C to exit")
        print("=" * 80)
    
    def run(self, interval: int = 5):
        """Run monitoring loop"""
        print("🚀 Starting Natron Monitor...")
        print(f"   Refresh interval: {interval} seconds")
        print("")
        
        try:
            while True:
                self.print_dashboard()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\n🛑 Monitor stopped")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Natron System Monitor')
    parser.add_argument('--interval', type=int, default=5, help='Refresh interval in seconds')
    parser.add_argument('--api-url', type=str, default='http://localhost:5000', help='API server URL')
    parser.add_argument('--socket-host', type=str, default='localhost', help='Socket server host')
    parser.add_argument('--socket-port', type=int, default=9090, help='Socket server port')
    
    args = parser.parse_args()
    
    monitor = NatronMonitor()
    monitor.api_url = args.api_url
    monitor.socket_host = args.socket_host
    monitor.socket_port = args.socket_port
    
    monitor.run(args.interval)


if __name__ == '__main__':
    main()
