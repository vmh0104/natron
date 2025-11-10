"""
Log Analyzer for Natron
Analyzes trading logs and predictions to compute performance metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogAnalyzer:
    """
    Analyzes trading logs and predictions to extract performance metrics.
    """
    
    def __init__(self, predictions_file: str, events_file: str):
        """
        Initialize log analyzer.
        
        Args:
            predictions_file: Path to predictions CSV
            events_file: Path to realtime events CSV
        """
        self.predictions_file = predictions_file
        self.events_file = events_file
        self.predictions_df = None
        self.events_df = None
    
    def load_data(self):
        """Load predictions and events data."""
        logger.info(f"Loading predictions from {self.predictions_file}")
        if Path(self.predictions_file).exists():
            self.predictions_df = pd.read_csv(self.predictions_file)
            logger.info(f"Loaded {len(self.predictions_df)} predictions")
        else:
            logger.warning(f"Predictions file not found: {self.predictions_file}")
            self.predictions_df = pd.DataFrame()
        
        logger.info(f"Loading events from {self.events_file}")
        if Path(self.events_file).exists():
            self.events_df = pd.read_csv(self.events_file)
            logger.info(f"Loaded {len(self.events_df)} events")
        else:
            logger.warning(f"Events file not found: {self.events_file}")
            self.events_df = pd.DataFrame()
    
    def compute_trade_performance(self) -> Dict:
        """
        Compute trade performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        if self.events_df.empty:
            return {}
        
        # Filter trade execution events
        trade_events = self.events_df[self.events_df['type'] == 'trade_executed'].copy()
        
        if trade_events.empty:
            return {}
        
        # Parse trade data
        trades = []
        for _, row in trade_events.iterrows():
            # Extract trade information from data column (if stored as JSON string)
            # For simplicity, assume columns are already extracted
            trade = {
                'timestamp': pd.to_datetime(row.get('timestamp', row.get('time', datetime.now()))),
                'trade_type': row.get('trade_type', 'UNKNOWN'),
                'lot_size': float(row.get('lot_size', 0)),
                'price': float(row.get('price', 0)),
                'stop_loss': float(row.get('stop_loss', 0)),
                'take_profit': float(row.get('take_profit', 0))
            }
            trades.append(trade)
        
        trades_df = pd.DataFrame(trades)
        
        if trades_df.empty:
            return {}
        
        # Compute metrics (simplified - in production, match with actual MT5 trade results)
        total_trades = len(trades_df)
        buy_trades = len(trades_df[trades_df['trade_type'] == 'BUY'])
        sell_trades = len(trades_df[trades_df['trade_type'] == 'SELL'])
        
        # Calculate win rate (simplified - would need actual P&L from MT5)
        # For now, use forecast accuracy as proxy
        if not self.predictions_df.empty:
            forecast_accuracy = (
                self.predictions_df['forecast_pred'] == self.predictions_df['forecast_true']
            ).mean()
        else:
            forecast_accuracy = 0.5
        
        metrics = {
            'total_trades': int(total_trades),
            'buy_trades': int(buy_trades),
            'sell_trades': int(sell_trades),
            'forecast_accuracy': float(forecast_accuracy),
            'win_rate': float(forecast_accuracy),  # Proxy
            'profit_factor': 1.0,  # Would need actual P&L
            'sharpe_ratio': 0.0,  # Would need returns series
            'max_drawdown': 0.0  # Would need equity curve
        }
        
        return metrics
    
    def compute_regime_performance(self) -> Dict:
        """
        Compute performance by regime.
        
        Returns:
            Dictionary with regime-specific metrics
        """
        if self.predictions_df.empty:
            return {}
        
        regime_performance = {}
        
        for regime_name in self.predictions_df['regime_name'].unique():
            regime_data = self.predictions_df[self.predictions_df['regime_name'] == regime_name]
            
            if len(regime_data) == 0:
                continue
            
            accuracy = (regime_data['regime_pred'] == regime_data['regime_true']).mean()
            forecast_accuracy = (regime_data['forecast_pred'] == regime_data['forecast_true']).mean()
            
            regime_performance[regime_name] = {
                'count': int(len(regime_data)),
                'regime_accuracy': float(accuracy),
                'forecast_accuracy': float(forecast_accuracy),
                'avg_confidence': float(regime_data['regime_confidence'].mean()),
                'avg_context_score': float(regime_data['context_score'].mean())
            }
        
        return regime_performance
    
    def get_recent_performance(self, days: int = 7) -> Dict:
        """
        Get performance metrics for recent period.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dictionary with recent performance metrics
        """
        if self.events_df.empty:
            return {}
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_events = self.events_df[
            pd.to_datetime(self.events_df['timestamp']) >= cutoff_date
        ]
        
        recent_trades = len(recent_events[recent_events['type'] == 'trade_executed'])
        
        return {
            'period_days': days,
            'recent_trades': int(recent_trades),
            'trades_per_day': float(recent_trades / days) if days > 0 else 0.0
        }
    
    def generate_report(self) -> Dict:
        """
        Generate comprehensive performance report.
        
        Returns:
            Dictionary with all performance metrics
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'trade_performance': self.compute_trade_performance(),
            'regime_performance': self.compute_regime_performance(),
            'recent_performance': self.get_recent_performance()
        }
        
        return report
