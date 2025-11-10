"""
Natron Monitoring Script
Monitors system health and sends alerts (optional Telegram integration).
"""

import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NatronMonitor:
    """
    Monitor for Natron trading system.
    """
    
    def __init__(self, log_dir: str = 'logs', telegram_config: Optional[dict] = None):
        """
        Initialize monitor.
        
        Args:
            log_dir: Directory containing log files
            telegram_config: Telegram bot configuration (optional)
        """
        self.log_dir = Path(log_dir)
        self.telegram_config = telegram_config
        self.telegram_bot = None
        
        if telegram_config:
            self._init_telegram()
    
    def _init_telegram(self):
        """Initialize Telegram bot."""
        try:
            from telegram import Bot
            self.telegram_bot = Bot(token=self.telegram_config['token'])
            self.telegram_chat_id = self.telegram_config['chat_id']
            logger.info("Telegram bot initialized")
        except ImportError:
            logger.warning("python-telegram-bot not installed, Telegram alerts disabled")
            self.telegram_bot = None
        except Exception as e:
            logger.warning(f"Failed to initialize Telegram: {e}")
            self.telegram_bot = None
    
    def send_alert(self, message: str, level: str = 'INFO'):
        """
        Send alert message.
        
        Args:
            message: Alert message
            level: Alert level (INFO, WARNING, ERROR)
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        full_message = f"[{timestamp}] [{level}] {message}"
        
        logger.log(
            logging.INFO if level == 'INFO' else logging.WARNING if level == 'WARNING' else logging.ERROR,
            message
        )
        
        if self.telegram_bot and level in ['WARNING', 'ERROR']:
            try:
                self.telegram_bot.send_message(
                    chat_id=self.telegram_chat_id,
                    text=f"🚨 *Natron Alert [{level}]*\n\n{message}",
                    parse_mode='Markdown'
                )
            except Exception as e:
                logger.error(f"Failed to send Telegram message: {e}")
    
    def check_server_health(self) -> dict:
        """
        Check server health status.
        
        Returns:
            Health status dictionary
        """
        health = {
            'status': 'unknown',
            'last_event': None,
            'recent_errors': 0,
            'recent_trades': 0
        }
        
        # Check events log
        events_file = self.log_dir / 'realtime_events.csv'
        if events_file.exists():
            try:
                df_events = pd.read_csv(events_file)
                if not df_events.empty:
                    df_events['timestamp'] = pd.to_datetime(df_events['timestamp'])
                    last_event = df_events['timestamp'].max()
                    health['last_event'] = last_event.isoformat()
                    
                    # Check if events are recent (within last 5 minutes)
                    time_since_last = datetime.now() - last_event.to_pydatetime()
                    if time_since_last < timedelta(minutes=5):
                        health['status'] = 'healthy'
                    else:
                        health['status'] = 'stale'
                        self.send_alert(f"No events in last {time_since_last}", 'WARNING')
                    
                    # Count recent errors
                    recent_errors = df_events[
                        (df_events['type'] == 'trade_error') &
                        (df_events['timestamp'] > datetime.now() - timedelta(hours=1))
                    ]
                    health['recent_errors'] = len(recent_errors)
                    
                    # Count recent trades
                    recent_trades = df_events[
                        (df_events['type'] == 'trade_executed') &
                        (df_events['timestamp'] > datetime.now() - timedelta(hours=1))
                    ]
                    health['recent_trades'] = len(recent_trades)
            except Exception as e:
                logger.error(f"Error checking events: {e}")
                health['status'] = 'error'
        
        return health
    
    def generate_daily_summary(self) -> str:
        """
        Generate daily trading summary.
        
        Returns:
            Summary string
        """
        events_file = self.log_dir / 'realtime_events.csv'
        
        if not events_file.exists():
            return "No events file found"
        
        try:
            df_events = pd.read_csv(events_file)
            if df_events.empty:
                return "No events recorded"
            
            df_events['timestamp'] = pd.to_datetime(df_events['timestamp'])
            today = datetime.now().date()
            today_events = df_events[df_events['timestamp'].dt.date == today]
            
            if today_events.empty:
                return "No events today"
            
            # Count events by type
            event_counts = today_events['type'].value_counts()
            
            trades = len(today_events[today_events['type'] == 'trade_executed'])
            predictions = len(today_events[today_events['type'] == 'prediction'])
            errors = len(today_events[today_events['type'] == 'trade_error'])
            
            summary = f"""
📊 Natron Daily Summary - {today}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Trades Executed: {trades}
Predictions Made: {predictions}
Errors: {errors}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            return summary
            
        except Exception as e:
            return f"Error generating summary: {e}"
    
    def monitor_loop(self, interval: int = 60):
        """
        Main monitoring loop.
        
        Args:
            interval: Check interval in seconds
        """
        logger.info("Starting Natron Monitor...")
        self.send_alert("Natron Monitor started", 'INFO')
        
        last_daily_summary = None
        
        try:
            while True:
                # Check health
                health = self.check_server_health()
                
                if health['status'] == 'stale':
                    self.send_alert("Server appears stale - no recent events", 'WARNING')
                
                if health['recent_errors'] > 5:
                    self.send_alert(f"High error rate: {health['recent_errors']} errors in last hour", 'ERROR')
                
                # Send daily summary (once per day)
                now = datetime.now()
                if last_daily_summary is None or (now - last_daily_summary).days >= 1:
                    summary = self.generate_daily_summary()
                    if self.telegram_bot:
                        try:
                            self.telegram_bot.send_message(
                                chat_id=self.telegram_chat_id,
                                text=summary,
                                parse_mode='Markdown'
                            )
                        except:
                            pass
                    last_daily_summary = now
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Monitor stopped by user")
        except Exception as e:
            logger.error(f"Monitor error: {e}")
            self.send_alert(f"Monitor error: {e}", 'ERROR')


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Natron Monitor')
    parser.add_argument('--log-dir', type=str, default='logs', help='Log directory')
    parser.add_argument('--telegram-token', type=str, default=None, help='Telegram bot token')
    parser.add_argument('--telegram-chat-id', type=str, default=None, help='Telegram chat ID')
    parser.add_argument('--interval', type=int, default=60, help='Check interval (seconds)')
    
    args = parser.parse_args()
    
    telegram_config = None
    if args.telegram_token and args.telegram_chat_id:
        telegram_config = {
            'token': args.telegram_token,
            'chat_id': args.telegram_chat_id
        }
    
    monitor = NatronMonitor(log_dir=args.log_dir, telegram_config=telegram_config)
    monitor.monitor_loop(interval=args.interval)


if __name__ == '__main__':
    main()
