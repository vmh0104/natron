"""
Natron Monitoring Script
Monitors system health and sends alerts (optional Telegram integration).
"""

import os
import json
import time
import psutil
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NatronMonitor:
    """Monitor for Natron trading system."""
    
    def __init__(self, config_path: str = "monitor_config.yaml"):
        """
        Initialize monitor.
        
        Args:
            config_path: Path to monitor config
        """
        self.config = self._load_config(config_path)
        self.log_dir = Path(self.config.get('log_dir', 'logs'))
        self.event_log_path = Path(self.config.get('event_log_path', 'realtime_events.csv'))
        
    def _load_config(self, config_path: str) -> dict:
        """Load monitor configuration."""
        if Path(config_path).exists():
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            return {
                'log_dir': 'logs',
                'event_log_path': 'realtime_events.csv',
                'check_interval': 60,  # seconds
                'telegram_enabled': False,
                'telegram_bot_token': '',
                'telegram_chat_id': ''
            }
    
    def check_system_health(self) -> dict:
        """Check system health metrics."""
        health = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'process_running': False
        }
        
        # Check if Natron process is running
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and 'natron_server_v5.py' in ' '.join(cmdline):
                    health['process_running'] = True
                    health['process_pid'] = proc.info['pid']
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return health
    
    def check_trading_activity(self) -> dict:
        """Check recent trading activity."""
        activity = {
            'timestamp': datetime.now().isoformat(),
            'total_events': 0,
            'recent_trades': 0,
            'last_trade_time': None
        }
        
        if self.event_log_path.exists():
            try:
                df = pd.read_csv(self.event_log_path)
                activity['total_events'] = len(df)
                
                # Recent trades (last hour)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    recent = df[df['timestamp'] > datetime.now() - timedelta(hours=1)]
                    activity['recent_trades'] = len(recent[recent['event_type'] == 'ENTRY'])
                    
                    if len(df) > 0:
                        activity['last_trade_time'] = df['timestamp'].max().isoformat()
            except Exception as e:
                logger.error(f"Error reading event log: {e}")
        
        return activity
    
    def generate_daily_summary(self) -> dict:
        """Generate daily trading summary."""
        summary = {
            'date': datetime.now().date().isoformat(),
            'trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0
        }
        
        if self.event_log_path.exists():
            try:
                df = pd.read_csv(self.event_log_path)
                
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    today = df[df['timestamp'].dt.date == datetime.now().date()]
                    
                    entries = today[today['event_type'] == 'ENTRY']
                    exits = today[today['event_type'] == 'EXIT']
                    
                    summary['trades'] = len(entries)
                    
                    # Calculate P&L (simplified)
                    if len(entries) > 0 and len(exits) > 0:
                        # Match entries with exits
                        for idx, entry in entries.iterrows():
                            exit_row = exits[exits.index > idx]
                            if len(exit_row) > 0:
                                exit_price = exit_row.iloc[0].get('price', 0)
                                entry_price = entry.get('price', 0)
                                
                                if entry.get('action') == 'BUY':
                                    pnl = exit_price - entry_price
                                else:
                                    pnl = entry_price - exit_price
                                
                                summary['total_pnl'] += pnl
                                if pnl > 0:
                                    summary['winning_trades'] += 1
                                else:
                                    summary['losing_trades'] += 1
            except Exception as e:
                logger.error(f"Error generating summary: {e}")
        
        return summary
    
    def send_telegram_alert(self, message: str):
        """Send alert via Telegram (if configured)."""
        if not self.config.get('telegram_enabled', False):
            return
        
        bot_token = self.config.get('telegram_bot_token', '')
        chat_id = self.config.get('telegram_chat_id', '')
        
        if not bot_token or not chat_id:
            return
        
        try:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': f"🚨 Natron Alert\n\n{message}",
                'parse_mode': 'HTML'
            }
            requests.post(url, json=data, timeout=5)
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
    
    def run_monitoring_loop(self):
        """Run continuous monitoring loop."""
        logger.info("Starting Natron monitoring...")
        
        last_summary_date = None
        
        while True:
            try:
                # Check system health
                health = self.check_system_health()
                
                # Alert if process not running
                if not health['process_running']:
                    alert_msg = "⚠️ Natron server process is not running!"
                    logger.warning(alert_msg)
                    self.send_telegram_alert(alert_msg)
                
                # Alert if high resource usage
                if health['cpu_percent'] > 90:
                    alert_msg = f"⚠️ High CPU usage: {health['cpu_percent']:.1f}%"
                    logger.warning(alert_msg)
                    self.send_telegram_alert(alert_msg)
                
                if health['memory_percent'] > 90:
                    alert_msg = f"⚠️ High memory usage: {health['memory_percent']:.1f}%"
                    logger.warning(alert_msg)
                    self.send_telegram_alert(alert_msg)
                
                # Daily summary
                current_date = datetime.now().date()
                if last_summary_date != current_date:
                    summary = self.generate_daily_summary()
                    
                    summary_msg = (
                        f"📊 Daily Summary ({summary['date']})\n"
                        f"Trades: {summary['trades']}\n"
                        f"Wins: {summary['winning_trades']}\n"
                        f"Losses: {summary['losing_trades']}\n"
                        f"Total P&L: {summary['total_pnl']:.2f}"
                    )
                    
                    logger.info(summary_msg)
                    self.send_telegram_alert(summary_msg)
                    
                    last_summary_date = current_date
                
                # Sleep until next check
                time.sleep(self.config.get('check_interval', 60))
            
            except KeyboardInterrupt:
                logger.info("Monitoring stopped")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)


if __name__ == "__main__":
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "monitor_config.yaml"
    
    monitor = NatronMonitor(config_path)
    monitor.run_monitoring_loop()
