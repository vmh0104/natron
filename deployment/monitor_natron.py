"""
Natron V2 Monitoring and Logging Utility
Real-time monitoring of model performance and trading metrics
"""

import os
import time
import yaml
import torch
import psutil
import json
from datetime import datetime
from typing import Dict, List
import argparse


class NatronMonitor:
    """
    Monitor Natron system health, performance, and trading metrics.
    """
    
    def __init__(self, log_dir: str = 'logs', config_path: str = 'config/config.yaml'):
        self.log_dir = log_dir
        self.config_path = config_path
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create logs directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Metrics storage
        self.metrics_history: List[Dict] = []
        
        print("Natron V2 Monitor initialized")
    
    def check_system_health(self) -> Dict:
        """
        Check system health: CPU, RAM, GPU, disk.
        """
        health = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'ram_percent': psutil.virtual_memory().percent,
            'ram_available_gb': psutil.virtual_memory().available / (1024**3),
            'disk_percent': psutil.disk_usage('/').percent,
            'disk_free_gb': psutil.disk_usage('/').free / (1024**3)
        }
        
        # GPU info (if available)
        if torch.cuda.is_available():
            health['gpu_available'] = True
            health['gpu_count'] = torch.cuda.device_count()
            health['gpu_name'] = torch.cuda.get_device_name(0)
            health['gpu_memory_allocated_gb'] = torch.cuda.memory_allocated(0) / (1024**3)
            health['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved(0) / (1024**3)
        else:
            health['gpu_available'] = False
        
        return health
    
    def check_model_files(self) -> Dict:
        """
        Check if model files exist and their sizes.
        """
        model_info = {}
        
        model_files = [
            'model/natron_v2.pt',
            'model/supervised_best.pt',
            'model/pretrain_best.pt',
            'model/rl_best.pt',
            'model/scaler.pkl'
        ]
        
        for filepath in model_files:
            if os.path.exists(filepath):
                size_mb = os.path.getsize(filepath) / (1024**2)
                mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                model_info[filepath] = {
                    'exists': True,
                    'size_mb': round(size_mb, 2),
                    'modified': mtime.isoformat()
                }
            else:
                model_info[filepath] = {'exists': False}
        
        return model_info
    
    def check_log_files(self) -> Dict:
        """
        Check tensorboard logs and their sizes.
        """
        log_info = {}
        
        if os.path.exists(self.log_dir):
            for subdir in os.listdir(self.log_dir):
                subdir_path = os.path.join(self.log_dir, subdir)
                if os.path.isdir(subdir_path):
                    total_size = sum(
                        os.path.getsize(os.path.join(subdir_path, f))
                        for f in os.listdir(subdir_path)
                        if os.path.isfile(os.path.join(subdir_path, f))
                    )
                    log_info[subdir] = {
                        'size_mb': round(total_size / (1024**2), 2),
                        'num_files': len(os.listdir(subdir_path))
                    }
        
        return log_info
    
    def monitor_loop(self, interval: int = 60):
        """
        Continuous monitoring loop.
        
        Args:
            interval: Monitoring interval in seconds
        """
        print(f"Starting monitoring loop (interval: {interval}s)")
        print("Press Ctrl+C to stop")
        print("=" * 80)
        
        try:
            while True:
                # Collect metrics
                metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'system_health': self.check_system_health(),
                    'model_files': self.check_model_files(),
                    'log_files': self.check_log_files()
                }
                
                # Display
                self.display_metrics(metrics)
                
                # Save to history
                self.metrics_history.append(metrics)
                
                # Save to file
                self.save_metrics(metrics)
                
                # Wait
                time.sleep(interval)
        
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped")
            self.generate_report()
    
    def display_metrics(self, metrics: Dict):
        """Display metrics in terminal"""
        print(f"\n[{metrics['timestamp']}]")
        print("-" * 80)
        
        health = metrics['system_health']
        print(f"CPU: {health['cpu_percent']:.1f}% | "
              f"RAM: {health['ram_percent']:.1f}% ({health['ram_available_gb']:.2f} GB free) | "
              f"Disk: {health['disk_percent']:.1f}%")
        
        if health.get('gpu_available'):
            print(f"GPU: {health['gpu_name']} | "
                  f"Memory: {health['gpu_memory_allocated_gb']:.2f} GB / "
                  f"{health['gpu_memory_reserved_gb']:.2f} GB")
        
        # Model files
        print("\nModel Files:")
        for filepath, info in metrics['model_files'].items():
            if info['exists']:
                print(f"  ✓ {filepath} ({info['size_mb']} MB)")
            else:
                print(f"  ✗ {filepath} (not found)")
        
        print("-" * 80)
    
    def save_metrics(self, metrics: Dict):
        """Save metrics to JSON file"""
        filepath = os.path.join(self.log_dir, 'monitoring.jsonl')
        
        with open(filepath, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
    
    def generate_report(self):
        """Generate summary report"""
        if not self.metrics_history:
            print("No metrics collected")
            return
        
        print("\n" + "=" * 80)
        print("MONITORING REPORT")
        print("=" * 80)
        
        # Calculate averages
        avg_cpu = sum(m['system_health']['cpu_percent'] for m in self.metrics_history) / len(self.metrics_history)
        avg_ram = sum(m['system_health']['ram_percent'] for m in self.metrics_history) / len(self.metrics_history)
        
        print(f"Total monitoring duration: {len(self.metrics_history)} samples")
        print(f"Average CPU usage: {avg_cpu:.1f}%")
        print(f"Average RAM usage: {avg_ram:.1f}%")
        
        if self.metrics_history[0]['system_health'].get('gpu_available'):
            avg_gpu_mem = sum(
                m['system_health']['gpu_memory_allocated_gb'] 
                for m in self.metrics_history
            ) / len(self.metrics_history)
            print(f"Average GPU memory: {avg_gpu_mem:.2f} GB")
        
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Natron V2 System Monitor')
    parser.add_argument('--interval', type=int, default=60,
                       help='Monitoring interval in seconds')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Log directory')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Config file path')
    parser.add_argument('--once', action='store_true',
                       help='Run once and exit (no loop)')
    
    args = parser.parse_args()
    
    monitor = NatronMonitor(args.log_dir, args.config)
    
    if args.once:
        # Single check
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'system_health': monitor.check_system_health(),
            'model_files': monitor.check_model_files(),
            'log_files': monitor.check_log_files()
        }
        monitor.display_metrics(metrics)
    else:
        # Continuous monitoring
        monitor.monitor_loop(args.interval)


if __name__ == '__main__':
    main()
