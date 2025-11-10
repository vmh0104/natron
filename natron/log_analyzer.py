"""
Natron Log Analyzer
Analyzes trading logs and compares predictions vs actual outcomes.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogAnalyzer:
    """Analyzer for trading logs and predictions."""
    
    def __init__(self):
        """Initialize log analyzer."""
        logger.info("LogAnalyzer initialized")
    
    def analyze_trades(
        self,
        events_df: pd.DataFrame,
        predictions_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Analyze trade outcomes from events log.
        
        Args:
            events_df: DataFrame with trading events
            predictions_df: Optional DataFrame with predictions
            
        Returns:
            Performance metrics dictionary
        """
        if events_df is None or len(events_df) == 0:
            return {}
        
        # Filter entry and exit events
        entries = events_df[events_df['event_type'] == 'ENTRY'].copy()
        exits = events_df[events_df['event_type'] == 'EXIT'].copy()
        
        if len(entries) == 0:
            return {'total_trades': 0}
        
        # Match entries with exits
        trades = []
        for idx, entry in entries.iterrows():
            # Find corresponding exit
            exit_row = exits[exits.index > idx]
            if len(exit_row) > 0:
                exit_row = exit_row.iloc[0]
                
                # Calculate P&L
                entry_price = entry.get('price', 0)
                exit_price = exit_row.get('price', 0)
                action = entry.get('action', '')
                
                if action == 'BUY':
                    pnl = exit_price - entry_price
                    pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
                elif action == 'SELL':
                    pnl = entry_price - exit_price
                    pnl_pct = (entry_price - exit_price) / entry_price if entry_price > 0 else 0
                else:
                    pnl = 0
                    pnl_pct = 0
                
                trade = {
                    'entry_time': entry.get('timestamp', ''),
                    'exit_time': exit_row.get('timestamp', ''),
                    'action': action,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'is_win': pnl > 0,
                    'regime': entry.get('regime', ''),
                    'context': entry.get('context', 0)
                }
                
                trades.append(trade)
        
        if len(trades) == 0:
            return {'total_trades': 0}
        
        trades_df = pd.DataFrame(trades)
        
        # Calculate metrics
        total_trades = len(trades_df)
        winning_trades = trades_df['is_win'].sum()
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()
        avg_win = trades_df[trades_df['is_win']]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[~trades_df['is_win']]['pnl'].mean() if losing_trades > 0 else 0
        
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Regime performance
        regime_performance = {}
        if 'regime' in trades_df.columns:
            for regime in trades_df['regime'].unique():
                regime_trades = trades_df[trades_df['regime'] == regime]
                regime_wins = regime_trades['is_win'].sum()
                regime_total = len(regime_trades)
                
                regime_performance[regime] = {
                    'total_trades': regime_total,
                    'winning_trades': regime_wins,
                    'win_rate': regime_wins / regime_total if regime_total > 0 else 0,
                    'avg_pnl': regime_trades['pnl'].mean(),
                    'total_pnl': regime_trades['pnl'].sum()
                }
        
        # Compare with predictions if available
        prediction_accuracy = None
        if predictions_df is not None and len(predictions_df) > 0:
            # Match predictions with trades (simplified)
            prediction_accuracy = self._calculate_prediction_accuracy(trades_df, predictions_df)
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': float(total_pnl),
            'avg_pnl': float(avg_pnl),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'profit_factor': float(profit_factor),
            'regime_performance': regime_performance,
            'prediction_accuracy': prediction_accuracy
        }
        
        return metrics
    
    def _calculate_prediction_accuracy(
        self,
        trades_df: pd.DataFrame,
        predictions_df: pd.DataFrame
    ) -> Dict:
        """
        Calculate how well predictions matched actual outcomes.
        
        Args:
            trades_df: DataFrame with trade outcomes
            predictions_df: DataFrame with predictions
            
        Returns:
            Accuracy metrics dictionary
        """
        # Simplified matching (in production, match by timestamp)
        # For now, compare forecast direction with trade outcome
        
        correct_predictions = 0
        total_predictions = 0
        
        for _, trade in trades_df.iterrows():
            # Determine if trade was profitable (up direction)
            is_up = trade['pnl'] > 0
            
            # Find matching prediction (simplified - use first available)
            if len(predictions_df) > 0:
                # In production, match by timestamp
                pred = predictions_df.iloc[total_predictions % len(predictions_df)]
                pred_up = pred.get('forecast_pred', 0) == 1
                
                if pred_up == is_up:
                    correct_predictions += 1
                total_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        return {
            'accuracy': accuracy,
            'correct': correct_predictions,
            'total': total_predictions
        }


if __name__ == "__main__":
    import sys
    
    analyzer = LogAnalyzer()
    
    if len(sys.argv) < 2:
        print("Usage: python log_analyzer.py <events_csv> [predictions_csv]")
        sys.exit(1)
    
    events_path = sys.argv[1]
    predictions_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    events_df = pd.read_csv(events_path)
    predictions_df = pd.read_csv(predictions_path) if predictions_path else None
    
    performance = analyzer.analyze_trades(events_df, predictions_df)
    
    print("\n" + "="*50)
    print("TRADE ANALYSIS RESULTS")
    print("="*50)
    import json
    print(json.dumps(performance, indent=2, default=str))
    print("="*50)
