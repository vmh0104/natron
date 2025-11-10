"""
Log Analyzer for Natron AI Trading System
Analyzes trading logs and predictions to extract performance metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import json


class LogAnalyzer:
    """
    Analyzes trading logs and predictions.
    """
    
    def __init__(self, predictions_file: str, events_file: str):
        """
        Initialize log analyzer.
        
        Args:
            predictions_file: Path to predictions CSV
            events_file: Path to realtime events CSV
        """
        self.predictions_file = Path(predictions_file)
        self.events_file = Path(events_file)
        
        self.predictions_df = None
        self.events_df = None
    
    def load_data(self):
        """Load predictions and events data."""
        if self.predictions_file.exists():
            self.predictions_df = pd.read_csv(self.predictions_file)
            print(f"Loaded {len(self.predictions_df)} predictions")
        else:
            print(f"Warning: Predictions file not found: {self.predictions_file}")
        
        if self.events_file.exists():
            self.events_df = pd.read_csv(self.events_file)
            print(f"Loaded {len(self.events_df)} events")
        else:
            print(f"Warning: Events file not found: {self.events_file}")
    
    def analyze_prediction_accuracy(self) -> Dict:
        """
        Analyze prediction accuracy.
        
        Returns:
            Dictionary with accuracy metrics
        """
        if self.predictions_df is None:
            return {}
        
        metrics = {}
        
        # Regime accuracy
        if 'regime_pred' in self.predictions_df.columns and 'regime_target' in self.predictions_df.columns:
            regime_acc = (self.predictions_df['regime_pred'] == self.predictions_df['regime_target']).mean()
            metrics['regime_accuracy'] = float(regime_acc)
        
        # Forecast accuracy
        if 'forecast_pred' in self.predictions_df.columns and 'forecast_target' in self.predictions_df.columns:
            forecast_acc = (self.predictions_df['forecast_pred'] == self.predictions_df['forecast_target']).mean()
            metrics['forecast_accuracy'] = float(forecast_acc)
        
        # Context strength correlation
        if 'context_pred' in self.predictions_df.columns and 'context_target' in self.predictions_df.columns:
            context_corr = self.predictions_df[['context_pred', 'context_target']].corr().iloc[0, 1]
            metrics['context_correlation'] = float(context_corr)
        
        return metrics
    
    def analyze_trade_outcomes(self) -> Dict:
        """
        Analyze actual trade outcomes from events log.
        
        Returns:
            Dictionary with trade performance metrics
        """
        if self.events_df is None:
            return {}
        
        # Filter entry signals
        entry_signals = self.events_df[self.events_df['event_type'] == 'ENTRY_SIGNAL'].copy()
        
        if len(entry_signals) == 0:
            return {}
        
        metrics = {
            'total_signals': len(entry_signals),
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0
        }
        
        # Extract PnL from events (if available)
        if 'pnl' in self.events_df.columns:
            closed_trades = self.events_df[self.events_df['event_type'] == 'TRADE_CLOSED'].copy()
            
            if len(closed_trades) > 0:
                winning = closed_trades[closed_trades['pnl'] > 0]
                losing = closed_trades[closed_trades['pnl'] <= 0]
                
                metrics['winning_trades'] = len(winning)
                metrics['losing_trades'] = len(losing)
                metrics['total_pnl'] = float(closed_trades['pnl'].sum())
                
                if len(winning) > 0:
                    metrics['average_win'] = float(winning['pnl'].mean())
                if len(losing) > 0:
                    metrics['average_loss'] = float(abs(losing['pnl'].mean()))
                
                if metrics['winning_trades'] + metrics['losing_trades'] > 0:
                    metrics['win_rate'] = metrics['winning_trades'] / (metrics['winning_trades'] + metrics['losing_trades'])
                
                if metrics['average_loss'] > 0:
                    metrics['profit_factor'] = (metrics['average_win'] * metrics['winning_trades']) / \
                                              (metrics['average_loss'] * metrics['losing_trades'])
        
        return metrics
    
    def compare_predicted_vs_actual(self) -> Dict:
        """
        Compare predicted outcomes vs actual trade outcomes.
        
        Returns:
            Dictionary with comparison metrics
        """
        if self.predictions_df is None or self.events_df is None:
            return {}
        
        # Match predictions with actual outcomes
        # This is a simplified version - in production, you'd need proper timestamp matching
        
        comparison = {
            'matched_predictions': 0,
            'correct_forecasts': 0,
            'forecast_accuracy': 0.0
        }
        
        # Implementation would match timestamps and compare
        # For now, return structure
        
        return comparison
    
    def generate_report(self) -> Dict:
        """
        Generate comprehensive analysis report.
        
        Returns:
            Dictionary with full analysis report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'prediction_accuracy': self.analyze_prediction_accuracy(),
            'trade_outcomes': self.analyze_trade_outcomes(),
            'predicted_vs_actual': self.compare_predicted_vs_actual()
        }
        
        return report
    
    def save_report(self, output_path: str):
        """
        Save analysis report to JSON.
        
        Args:
            output_path: Output file path
        """
        report = self.generate_report()
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Saved analysis report to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Natron Trading Logs')
    parser.add_argument('--predictions', type=str, default='natron_predictions.csv',
                       help='Path to predictions CSV')
    parser.add_argument('--events', type=str, default='realtime_events.csv',
                       help='Path to events CSV')
    parser.add_argument('--output', type=str, default='log_analysis_report.json',
                       help='Output report file')
    args = parser.parse_args()
    
    analyzer = LogAnalyzer(args.predictions, args.events)
    analyzer.load_data()
    analyzer.save_report(args.output)
