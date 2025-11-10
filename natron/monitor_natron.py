"""
Natron Monitoring Script
Monitors system health and sends alerts.
"""

import time
import psutil
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NatronMonitor:
    """
    Monitoring system for Natron.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize monitor.
        
        Args:
            config_path: Optional path to monitoring config
        """
        self.config = {
            'check_interval': 60,
            'telegram_enabled': False,
            'telegram_bot_token': None,
            'telegram_chat_id': None,
            'alert_thresholds': {
                'cpu_percent': 90,
                'memory_percent': 90,
                'disk_percent': 90
            }
        }
        
        if config_path and Path(config_path).exists():
            import yaml
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                self.config.update(user_config.get('monitoring', {}))
    
    def check_system_health(self) -> Dict:
        """
        Check system health metrics.
        
        Returns:
            Dictionary with health metrics
        """
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        health = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'disk_percent': disk.percent,
            'disk_free_gb': disk.free / (1024**3),
            'status': 'healthy'
        }
        
        # Check thresholds
        thresholds = self.config['alert_thresholds']
        alerts = []
        
        if cpu_percent > thresholds['cpu_percent']:
            alerts.append(f"High CPU usage: {cpu_percent:.1f}%")
            health['status'] = 'warning'
        
        if memory.percent > thresholds['memory_percent']:
            alerts.append(f"High memory usage: {memory.percent:.1f}%")
            health['status'] = 'warning'
        
        if disk.percent > thresholds['disk_percent']:
            alerts.append(f"High disk usage: {disk.percent:.1f}%")
            health['status'] = 'warning'
        
        health['alerts'] = alerts
        
        return health
    
    def check_natron_process(self) -> Dict:
        """
        Check if Natron process is running.
        
        Returns:
            Dictionary with process status
        """
        pid_file = Path('logs/natron.pid')
        status = {
            'timestamp': datetime.now().isoformat(),
            'running': False,
            'pid': None
        }
        
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                if psutil.pid_exists(pid):
                    process = psutil.Process(pid)
                    status['running'] = True
                    status['pid'] = pid
                    status['cpu_percent'] = process.cpu_percent()
                    status['memory_mb'] = process.memory_info().rss / (1024**2)
                    status['status'] = process.status()
                else:
                    status['status'] = 'process_not_found'
            except Exception as e:
                status['error'] = str(e)
        else:
            status['status'] = 'pid_file_not_found'
        
        return status
    
    def send_telegram_alert(self, message: str):
        """
        Send alert via Telegram.
        
        Args:
            message: Alert message
        """
        if not self.config.get('telegram_enabled'):
            return
        
        bot_token = self.config.get('telegram_bot_token')
        chat_id = self.config.get('telegram_chat_id')
        
        if not bot_token or not chat_id:
            return
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {
            'chat_id': chat_id,
            'text': f"🚨 Natron Alert\n\n{message}",
            'parse_mode': 'HTML'
        }
        
        try:
            response = requests.post(url, data=data, timeout=5)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
    
    def generate_daily_summary(self) -> str:
        """
        Generate daily summary report.
        
        Returns:
            Summary text
        """
        # Check log files
        events_file = Path('realtime_events.csv')
        summary_lines = [
            f"📊 Natron Daily Summary - {datetime.now().strftime('%Y-%m-%d')}",
            ""
        ]
        
        if events_file.exists():
            import pandas as pd
            try:
                events_df = pd.read_csv(events_file)
                entry_signals = len(events_df[events_df['event_type'] == 'ENTRY_SIGNAL'])
                summary_lines.append(f"Entry Signals: {entry_signals}")
            except:
                pass
        
        # System health
        health = self.check_system_health()
        summary_lines.extend([
            "",
            "System Health:",
            f"  CPU: {health['cpu_percent']:.1f}%",
            f"  Memory: {health['memory_percent']:.1f}%",
            f"  Disk: {health['disk_percent']:.1f}%",
            f"  Status: {health['status']}"
        ])
        
        # Process status
        process_status = self.check_natron_process()
        summary_lines.extend([
            "",
            "Process Status:",
            f"  Running: {process_status['running']}",
            f"  Status: {process_status.get('status', 'unknown')}"
        ])
        
        return "\n".join(summary_lines)
    
    def run_monitoring_loop(self):
        """Run continuous monitoring loop."""
        logger.info("Starting Natron monitoring...")
        
        last_daily_summary = None
        
        while True:
            try:
                # Check system health
                health = self.check_system_health()
                
                # Check process
                process_status = self.check_natron_process()
                
                # Send alerts if needed
                if health['alerts']:
                    alert_msg = "\n".join(health['alerts'])
                    logger.warning(f"Health alerts: {alert_msg}")
                    self.send_telegram_alert(alert_msg)
                
                if not process_status['running']:
                    logger.error("Natron process is not running!")
                    self.send_telegram_alert("⚠️ Natron process is not running!")
                
                # Daily summary
                current_date = datetime.now().date()
                if last_daily_summary != current_date:
                    summary = self.generate_daily_summary()
                    logger.info(f"\n{summary}")
                    self.send_telegram_alert(summary)
                    last_daily_summary = current_date
                
                # Wait before next check
                time.sleep(self.config['check_interval'])
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(self.config['check_interval'])


def main():
    """Main monitoring function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Natron Monitoring')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to monitoring config YAML')
    args = parser.parse_args()
    
    monitor = NatronMonitor(args.config)
    monitor.run_monitoring_loop()


if __name__ == "__main__":
    main()
