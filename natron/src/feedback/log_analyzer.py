"""
Log Analyzer for Natron Feedback Learning

This module analyzes trading logs and compares predicted vs actual outcomes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Tuple
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogAnalyzer:
    """
    Analyzes trading logs to extract performance metrics.
    """
    
    def __init__(self, predictions_path: str, events_path: str):
        """
        Initialize log analyzer.
        
        Args:
            predictions_path: Path to predictions CSV
            events_path: Path to realtime events CSV
        """
        self.predictions_path = predictions_path
        self.events_path = events_path
        
        self.predictions_df = None
        self.events_df = None
    
    def load_data(self):
        """Load predictions and events data."""
        logger.info(f"Loading predictions from {self.predictions_path}")
        self.predictions_df = pd.read_csv(self.predictions_path)
        
        if 'timestamp' in self.predictions_df.columns:
            self.predictions_df['timestamp'] = pd.to_datetime(self.predictions_df['timestamp'])
            self.predictions_df = self.predictions_df.set_index('timestamp')
        
        logger.info(f"Loading events from {self.events_path}")
        self.events_df = pd.read_csv(self.events_path)
        
        if 'timestamp' in self.events_df.columns:
            self.events_df['timestamp'] = pd.to_datetime(self.events_df['timestamp'])
    
    def match_predictions_to_trades(self) -> pd.DataFrame:
        """
        Match predictions to actual trade outcomes.
        
        Returns:
            DataFrame with matched predictions and outcomes
        """
        if self.predictions_df is None or self.events_df is None:
            self.load_data()
        
        # Extract position entries and exits
        entries = self.events_df[self.events_df['type'] == 'position_entry'].copy()
        exits = self.events_df[self.events_df['type'] == 'position_exit'].copy()
        
        matched_trades = []
        
        for _, entry in entries.iterrows():
            entry_time = pd.to_datetime(entry['timestamp'])
            
            # Find corresponding exit
            exit_candidates = exits[exits['timestamp'] > entry_time]
            if len(exit_candidates) > 0:
                exit = exit_candidates.iloc[0]
                
                # Find prediction closest to entry time
                pred_candidates = self.predictions_df[
                    (self.predictions_df.index >= entry_time - pd.Timedelta(minutes=5)) &
                    (self.predictions_df.index <= entry_time + pd.Timedelta(minutes=5))
                ]
                
                if len(pred_candidates) > 0:
                    prediction = pred_candidates.iloc[0]
                    
                    # Calculate trade outcome
                    entry_data = json.loads(entry['data']) if isinstance(entry['data'], str) else entry['data']
                    exit_data = json.loads(exit['data']) if isinstance(exit['data'], str) else exit['data']
                    
                    entry_price = entry_data.get('entry_price', 0)
                    exit_price = exit_data.get('exit_price', 0)
                    position_type = entry_data.get('type', 'BUY')
                    
                    # Calculate P&L
                    if position_type == 'BUY':
                        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                    else:  # SELL
                        pnl_pct = ((entry_price - exit_price) / entry_price) * 100
                    
                    # Determine if prediction was correct
                    forecast_pred = prediction.get('forecast_pred', 0)
                    if position_type == 'BUY':
                        correct = (forecast_pred == 1) and (pnl_pct > 0)
                    else:  # SELL
                        correct = (forecast_pred == 0) and (pnl_pct > 0)
                    
                    matched_trade = {
                        'entry_time': entry_time,
                        'exit_time': pd.to_datetime(exit['timestamp']),
                        'position_type': position_type,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl_pct': pnl_pct,
                        'exit_reason': exit_data.get('exit_reason', 'UNKNOWN'),
                        'forecast_pred': forecast_pred,
                        'forecast_prob_up': prediction.get('forecast_prob_up', 0),
                        'forecast_prob_down': prediction.get('forecast_prob_down', 0),
                        'regime_pred': prediction.get('regime_pred', 0),
                        'context_strength': prediction.get('context_strength', 0),
                        'prediction_correct': correct
                    }
                    
                    matched_trades.append(matched_trade)
        
        return pd.DataFrame(matched_trades)
    
    def compute_performance_metrics(self, matched_trades: pd.DataFrame) -> Dict:
        """
        Compute performance metrics from matched trades.
        
        Args:
            matched_trades: DataFrame with matched predictions and outcomes
            
        Returns:
            Dictionary with performance metrics
        """
        if len(matched_trades) == 0:
            return {}
        
        metrics = {}
        
        # Overall metrics
        metrics['total_trades'] = len(matched_trades)
        metrics['winning_trades'] = len(matched_trades[matched_trades['pnl_pct'] > 0])
        metrics['losing_trades'] = len(matched_trades[matched_trades['pnl_pct'] <= 0])
        metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades'] if metrics['total_trades'] > 0 else 0
        
        # P&L metrics
        metrics['total_pnl_pct'] = matched_trades['pnl_pct'].sum()
        metrics['avg_pnl_pct'] = matched_trades['pnl_pct'].mean()
        metrics['avg_win_pct'] = matched_trades[matched_trades['pnl_pct'] > 0]['pnl_pct'].mean() if metrics['winning_trades'] > 0 else 0
        metrics['avg_loss_pct'] = matched_trades[matched_trades['pnl_pct'] <= 0]['pnl_pct'].mean() if metrics['losing_trades'] > 0 else 0
        
        # Prediction accuracy
        metrics['prediction_accuracy'] = matched_trades['prediction_correct'].mean()
        
        # Metrics by regime
        regime_metrics = {}
        for regime in matched_trades['regime_pred'].unique():
            regime_trades = matched_trades[matched_trades['regime_pred'] == regime]
            regime_metrics[int(regime)] = {
                'count': len(regime_trades),
                'win_rate': len(regime_trades[regime_trades['pnl_pct'] > 0]) / len(regime_trades) if len(regime_trades) > 0 else 0,
                'avg_pnl': regime_trades['pnl_pct'].mean()
            }
        metrics['by_regime'] = regime_metrics
        
        # Metrics by confidence level
        matched_trades['confidence_bin'] = pd.cut(
            matched_trades['forecast_prob_up'],
            bins=[0, 0.5, 0.6, 0.7, 0.8, 1.0],
            labels=['Low', 'Medium', 'High', 'Very High', 'Extreme']
        )
        
        confidence_metrics = {}
        for conf_level in matched_trades['confidence_bin'].unique():
            conf_trades = matched_trades[matched_trades['confidence_bin'] == conf_level]
            confidence_metrics[str(conf_level)] = {
                'count': len(conf_trades),
                'win_rate': len(conf_trades[conf_trades['pnl_pct'] > 0]) / len(conf_trades) if len(conf_trades) > 0 else 0,
                'avg_pnl': conf_trades['pnl_pct'].mean()
            }
        metrics['by_confidence'] = confidence_metrics
        
        return metrics
    
    def analyze(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform complete analysis.
        
        Returns:
            Tuple of (matched_trades DataFrame, metrics dictionary)
        """
        logger.info("Analyzing trading logs...")
        
        matched_trades = self.match_predictions_to_trades()
        metrics = self.compute_performance_metrics(matched_trades)
        
        logger.info(f"Analyzed {len(matched_trades)} matched trades")
        logger.info(f"Win rate: {metrics.get('win_rate', 0):.2%}")
        logger.info(f"Prediction accuracy: {metrics.get('prediction_accuracy', 0):.2%}")
        
        return matched_trades, metrics
