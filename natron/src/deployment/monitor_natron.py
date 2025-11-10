"""
Natron Monitoring Script

Monitors Natron server health and sends alerts.
"""

import time
import socket
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NatronMonitor:
    """
    Monitor class for Natron server.
    """
    
    def __init__(self, 
                 server_host: str = "localhost",
                 server_port: int = 8888,
                 check_interval: int = 60,
                 log_dir: str = "logs"):
        """
        Initialize monitor.
        
        Args:
            server_host: Server hostname
            server_port: Server port
            check_interval: Check interval in seconds
            log_dir: Log directory
        """
        self.server_host = server_host
        self.server_port = server_port
        self.check_interval = check_interval
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.status_log = self.log_dir / "monitor_status.json"
        self.alert_log = self.log_dir / "monitor_alerts.json"
    
    def check_server_health(self) -> dict:
        """
        Check server health.
        
        Returns:
            Dictionary with health status
        """
        status = {
            'timestamp': datetime.now().isoformat(),
            'server_reachable': False,
            'port_open': False,
            'response_time_ms': None
        }
        
        try:
            start_time = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((self.server_host, self.server_port))
            sock.close()
            
            response_time = (time.time() - start_time) * 1000
            
            if result == 0:
                status['server_reachable'] = True
                status['port_open'] = True
                status['response_time_ms'] = response_time
            else:
                status['port_open'] = False
        
        except Exception as e:
            logger.error(f"Health check error: {e}")
            status['error'] = str(e)
        
        return status
    
    def check_log_files(self) -> dict:
        """
        Check log file sizes and recent activity.
        
        Returns:
            Dictionary with log file status
        """
        log_status = {
            'timestamp': datetime.now().isoformat(),
            'log_files': {}
        }
        
        log_files = [
            'natron_server.log',
            'natron_server_error.log',
            'realtime_events.csv'
        ]
        
        for log_file in log_files:
            log_path = self.log_dir / log_file
            if log_path.exists():
                size_mb = log_path.stat().st_size / (1024 * 1024)
                mtime = datetime.fromtimestamp(log_path.stat().st_mtime)
                age_minutes = (datetime.now() - mtime).total_seconds() / 60
                
                log_status['log_files'][log_file] = {
                    'size_mb': round(size_mb, 2),
                    'last_modified': mtime.isoformat(),
                    'age_minutes': round(age_minutes, 2)
                }
        
        return log_status
    
    def send_alert(self, alert_type: str, message: str, data: Optional[dict] = None):
        """
        Send alert (can be extended to Telegram, email, etc.).
        
        Args:
            alert_type: Type of alert
            message: Alert message
            data: Additional alert data
        """
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message,
            'data': data or {}
        }
        
        # Log alert
        alerts = []
        if self.alert_log.exists():
            with open(self.alert_log, 'r') as f:
                alerts = json.load(f)
        
        alerts.append(alert)
        
        # Keep only last 100 alerts
        if len(alerts) > 100:
            alerts = alerts[-100:]
        
        with open(self.alert_log, 'w') as f:
            json.dump(alerts, f, indent=2)
        
        logger.warning(f"ALERT [{alert_type}]: {message}")
        
        # TODO: Integrate with Telegram bot or email service
        # Example:
        # if self.telegram_bot:
        #     self.telegram_bot.send_message(f"[{alert_type}] {message}")
    
    def generate_daily_summary(self) -> dict:
        """
        Generate daily summary report.
        
        Returns:
            Dictionary with daily summary
        """
        summary = {
            'date': datetime.now().date().isoformat(),
            'timestamp': datetime.now().isoformat(),
            'health_checks': [],
            'alerts': []
        }
        
        # Load today's status logs
        if self.status_log.exists():
            with open(self.status_log, 'r') as f:
                status_history = json.load(f)
            
            today = datetime.now().date().isoformat()
            today_statuses = [s for s in status_history if s.get('timestamp', '').startswith(today)]
            summary['health_checks'] = today_statuses
        
        # Load today's alerts
        if self.alert_log.exists():
            with open(self.alert_log, 'r') as f:
                alerts = json.load(f)
            
            today = datetime.now().date().isoformat()
            today_alerts = [a for a in alerts if a.get('timestamp', '').startswith(today)]
            summary['alerts'] = today_alerts
        
        return summary
    
    def monitor_loop(self):
        """Main monitoring loop."""
        logger.info("Starting Natron monitor...")
        logger.info(f"Monitoring server at {self.server_host}:{self.server_port}")
        logger.info(f"Check interval: {self.check_interval} seconds")
        
        consecutive_failures = 0
        max_failures = 3
        
        try:
            while True:
                # Health check
                health_status = self.check_server_health()
                
                # Log status
                status_history = []
                if self.status_log.exists():
                    with open(self.status_log, 'r') as f:
                        status_history = json.load(f)
                
                status_history.append(health_status)
                
                # Keep only last 1000 status checks
                if len(status_history) > 1000:
                    status_history = status_history[-1000:]
                
                with open(self.status_log, 'w') as f:
                    json.dump(status_history, f, indent=2)
                
                # Check for issues
                if not health_status['server_reachable']:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        self.send_alert(
                            'SERVER_DOWN',
                            f"Server unreachable for {consecutive_failures} consecutive checks",
                            health_status
                        )
                else:
                    consecutive_failures = 0
                
                # Check log files
                log_status = self.check_log_files()
                for log_file, file_info in log_status['log_files'].items():
                    if file_info['size_mb'] > 100:  # Alert if log > 100MB
                        self.send_alert(
                            'LOG_FILE_LARGE',
                            f"Log file {log_file} is {file_info['size_mb']} MB",
                            file_info
                        )
                
                # Sleep until next check
                time.sleep(self.check_interval)
        
        except KeyboardInterrupt:
            logger.info("Monitor stopped by user")
        except Exception as e:
            logger.error(f"Monitor error: {e}")
            self.send_alert('MONITOR_ERROR', str(e))


if __name__ == "__main__":
    import sys
    
    server_host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    server_port = int(sys.argv[2]) if len(sys.argv) > 2 else 8888
    check_interval = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    
    monitor = NatronMonitor(server_host, server_port, check_interval)
    monitor.monitor_loop()
