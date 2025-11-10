"""
Natron Auto Feedback Engine
Learns from trading outcomes and adapts configuration dynamically.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json

from log_analyzer import LogAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoFeedbackEngine:
    """
    Automatic feedback learning engine.
    Analyzes trade outcomes and adjusts configuration weights.
    """
    
    def __init__(self, config_path: str = "adaptive_config.yaml"):
        """
        Initialize feedback engine.
        
        Args:
            config_path: Path to adaptive config YAML
        """
        self.config_path = config_path
        self.load_config()
        
        self.log_analyzer = LogAnalyzer()
        
        logger.info("AutoFeedbackEngine initialized")
    
    def load_config(self):
        """Load adaptive configuration."""
        if Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default config
            self.config = {
                'confidence_scaling': 1.0,
                'entry_threshold': 0.6,
                'regime_weights': {
                    'BULL_STRONG': 1.0,
                    'BULL_WEAK': 0.8,
                    'BEAR_STRONG': 1.0,
                    'BEAR_WEAK': 0.8,
                    'RANGE': 0.3,
                    'VOLATILE': 0.5
                },
                'learning_rate': 0.1,
                'min_samples': 50
            }
            self.save_config()
    
    def save_config(self):
        """Save adaptive configuration."""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def analyze_trading_performance(
        self,
        predictions_path: str = "natron_predictions.csv",
        events_path: str = "realtime_events.csv"
    ) -> Dict:
        """
        Analyze trading performance from logs.
        
        Args:
            predictions_path: Path to predictions CSV
            events_path: Path to trading events CSV
            
        Returns:
            Performance metrics dictionary
        """
        # Load data
        predictions_df = pd.read_csv(predictions_path) if Path(predictions_path).exists() else None
        events_df = pd.read_csv(events_path) if Path(events_path).exists() else None
        
        if events_df is None or len(events_df) == 0:
            logger.warning("No trading events found")
            return {}
        
        # Analyze trade outcomes
        performance = self.log_analyzer.analyze_trades(events_df, predictions_df)
        
        return performance
    
    def update_weights(self, performance: Dict):
        """
        Update configuration weights based on performance.
        
        Args:
            performance: Performance metrics dictionary
        """
        if not performance or performance.get('total_trades', 0) < self.config.get('min_samples', 50):
            logger.info("Insufficient data for weight update")
            return
        
        learning_rate = self.config.get('learning_rate', 0.1)
        
        # Update confidence scaling based on accuracy
        win_rate = performance.get('win_rate', 0.5)
        if win_rate > 0.55:
            # Increase confidence if performing well
            self.config['confidence_scaling'] = min(
                1.5,
                self.config.get('confidence_scaling', 1.0) + learning_rate * 0.1
            )
        elif win_rate < 0.45:
            # Decrease confidence if performing poorly
            self.config['confidence_scaling'] = max(
                0.5,
                self.config.get('confidence_scaling', 1.0) - learning_rate * 0.1
            )
        
        # Update entry threshold
        if win_rate > 0.6:
            # Be more selective
            self.config['entry_threshold'] = min(
                0.8,
                self.config.get('entry_threshold', 0.6) + learning_rate * 0.05
            )
        elif win_rate < 0.4:
            # Be less selective
            self.config['entry_threshold'] = max(
                0.4,
                self.config.get('entry_threshold', 0.6) - learning_rate * 0.05
            )
        
        # Update regime weights based on performance per regime
        regime_performance = performance.get('regime_performance', {})
        regime_weights = self.config.get('regime_weights', {})
        
        for regime, metrics in regime_performance.items():
            if regime in regime_weights:
                regime_win_rate = metrics.get('win_rate', 0.5)
                
                if regime_win_rate > 0.6:
                    regime_weights[regime] = min(
                        1.5,
                        regime_weights[regime] + learning_rate * 0.1
                    )
                elif regime_win_rate < 0.4:
                    regime_weights[regime] = max(
                        0.3,
                        regime_weights[regime] - learning_rate * 0.1
                    )
        
        self.config['regime_weights'] = regime_weights
        
        # Save updated config
        self.save_config()
        
        logger.info("Updated configuration weights")
        logger.info(f"Confidence scaling: {self.config['confidence_scaling']:.3f}")
        logger.info(f"Entry threshold: {self.config['entry_threshold']:.3f}")
    
    def generate_learning_curve(self, performance_history: List[Dict], output_path: str = "learning_curve.json"):
        """
        Generate learning curve from performance history.
        
        Args:
            performance_history: List of performance dictionaries over time
            output_path: Path to save learning curve
        """
        curve_data = {
            'timestamps': [],
            'win_rates': [],
            'total_trades': [],
            'confidence_scaling': [],
            'entry_threshold': []
        }
        
        for perf in performance_history:
            curve_data['timestamps'].append(perf.get('timestamp', datetime.now().isoformat()))
            curve_data['win_rates'].append(perf.get('win_rate', 0.5))
            curve_data['total_trades'].append(perf.get('total_trades', 0))
            curve_data['confidence_scaling'].append(perf.get('confidence_scaling', 1.0))
            curve_data['entry_threshold'].append(perf.get('entry_threshold', 0.6))
        
        with open(output_path, 'w') as f:
            json.dump(curve_data, f, indent=2)
        
        logger.info(f"Saved learning curve to {output_path}")
    
    def run_adaptation_cycle(self):
        """Run one adaptation cycle: analyze performance and update weights."""
        logger.info("Running adaptation cycle...")
        
        # Analyze performance
        performance = self.analyze_trading_performance()
        
        if performance:
            # Update weights
            self.update_weights(performance)
            
            # Log adaptation
            logger.info(f"Adaptation complete. Win rate: {performance.get('win_rate', 0):.2%}")
        else:
            logger.warning("No performance data available for adaptation")


if __name__ == "__main__":
    import sys
    
    engine = AutoFeedbackEngine()
    
    if len(sys.argv) > 1 and sys.argv[1] == 'adapt':
        engine.run_adaptation_cycle()
    else:
        # Analyze and print performance
        performance = engine.analyze_trading_performance()
        print("\n" + "="*50)
        print("TRADING PERFORMANCE ANALYSIS")
        print("="*50)
        print(json.dumps(performance, indent=2, default=str))
        print("="*50)
