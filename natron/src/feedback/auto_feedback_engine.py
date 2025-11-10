"""
Auto Feedback Engine for Natron

This module learns from trading outcomes and adjusts configuration dynamically.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import json
import logging
from typing import Dict
from datetime import datetime

from .log_analyzer import LogAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoFeedbackEngine:
    """
    Automatically adjusts trading parameters based on performance feedback.
    """
    
    def __init__(self, 
                 predictions_path: str,
                 events_path: str,
                 config_path: str,
                 output_config_path: str):
        """
        Initialize feedback engine.
        
        Args:
            predictions_path: Path to predictions CSV
            events_path: Path to realtime events CSV
            config_path: Path to current config YAML
            output_config_path: Path to save adjusted config
        """
        self.predictions_path = predictions_path
        self.events_path = events_path
        self.config_path = config_path
        self.output_config_path = output_config_path
        
        # Load current config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize analyzer
        self.analyzer = LogAnalyzer(predictions_path, events_path)
    
    def compute_adaptive_weights(self, metrics: Dict) -> Dict:
        """
        Compute adaptive weights based on performance metrics.
        
        Args:
            metrics: Performance metrics dictionary
            
        Returns:
            Dictionary with adjusted weights
        """
        adjustments = {}
        
        # Adjust confidence threshold based on win rate
        win_rate = metrics.get('win_rate', 0.5)
        current_confidence = self.config['trading'].get('min_confidence', 0.6)
        
        if win_rate > 0.6:
            # High win rate - can be more aggressive
            adjustments['min_confidence'] = max(0.5, current_confidence - 0.05)
        elif win_rate < 0.4:
            # Low win rate - be more conservative
            adjustments['min_confidence'] = min(0.8, current_confidence + 0.05)
        else:
            adjustments['min_confidence'] = current_confidence
        
        # Adjust context strength threshold
        prediction_accuracy = metrics.get('prediction_accuracy', 0.5)
        current_context = self.config['trading'].get('min_context_strength', 0.5)
        
        if prediction_accuracy > 0.6:
            # Good predictions - can lower threshold
            adjustments['min_context_strength'] = max(0.3, current_context - 0.05)
        elif prediction_accuracy < 0.4:
            # Poor predictions - raise threshold
            adjustments['min_context_strength'] = min(0.7, current_context + 0.05)
        else:
            adjustments['min_context_strength'] = current_context
        
        # Adjust stop loss / take profit based on avg win/loss ratio
        avg_win = metrics.get('avg_win_pct', 1.0)
        avg_loss = abs(metrics.get('avg_loss_pct', 1.0))
        
        if avg_loss > 0:
            win_loss_ratio = avg_win / avg_loss
            current_sl_mult = self.config['trading'].get('stop_loss_atr_multiple', 2.0)
            current_tp_mult = self.config['trading'].get('take_profit_atr_multiple', 3.0)
            
            if win_loss_ratio < 1.0:
                # Losses larger than wins - tighten stop loss
                adjustments['stop_loss_atr_multiple'] = max(1.0, current_sl_mult - 0.2)
            elif win_loss_ratio > 2.0:
                # Good win/loss ratio - can widen stop loss slightly
                adjustments['stop_loss_atr_multiple'] = min(3.0, current_sl_mult + 0.2)
            else:
                adjustments['stop_loss_atr_multiple'] = current_sl_mult
            
            # Adjust take profit to maintain risk/reward ratio
            target_rr = 1.5  # Target risk/reward ratio
            adjustments['take_profit_atr_multiple'] = adjustments['stop_loss_atr_multiple'] * target_rr
        
        # Regime-specific adjustments
        regime_metrics = metrics.get('by_regime', {})
        regime_adjustments = {}
        
        for regime_id, regime_data in regime_metrics.items():
            regime_win_rate = regime_data.get('win_rate', 0.5)
            if regime_win_rate < 0.3:
                # Poor performing regime - increase confidence requirement
                regime_adjustments[int(regime_id)] = {
                    'confidence_multiplier': 1.2,
                    'context_multiplier': 1.1
                }
            elif regime_win_rate > 0.7:
                # Good performing regime - can be more aggressive
                regime_adjustments[int(regime_id)] = {
                    'confidence_multiplier': 0.9,
                    'context_multiplier': 0.95
                }
        
        adjustments['regime_adjustments'] = regime_adjustments
        
        return adjustments
    
    def update_config(self, adjustments: Dict):
        """
        Update configuration with adjustments.
        
        Args:
            adjustments: Dictionary with adjustment values
        """
        # Update trading parameters
        trading_config = self.config['trading']
        
        for key, value in adjustments.items():
            if key not in ['regime_adjustments']:
                trading_config[key] = value
        
        # Store regime adjustments separately
        if 'regime_adjustments' in adjustments:
            self.config['trading']['regime_adjustments'] = adjustments['regime_adjustments']
        
        # Add metadata
        self.config['_metadata'] = {
            'last_updated': datetime.now().isoformat(),
            'version': 'adaptive_v1.0'
        }
    
    def learn_and_adapt(self) -> Dict:
        """
        Main method: analyze logs and adapt configuration.
        
        Returns:
            Dictionary with analysis results and adjustments
        """
        logger.info("Starting feedback learning...")
        
        # Analyze trading logs
        matched_trades, metrics = self.analyzer.analyze()
        
        if len(matched_trades) == 0:
            logger.warning("No matched trades found, skipping adaptation")
            return {'status': 'no_data', 'metrics': metrics}
        
        # Compute adaptive weights
        adjustments = self.compute_adaptive_weights(metrics)
        
        # Update configuration
        self.update_config(adjustments)
        
        # Save updated config
        output_path = Path(self.output_config_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Updated configuration saved to {output_path}")
        
        # Log learning curve
        learning_log = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'adjustments': adjustments,
            'config_snapshot': self.config['trading'].copy()
        }
        
        # Save learning log
        log_path = Path(self.config_path).parent / 'learning_curve.json'
        learning_history = []
        
        if log_path.exists():
            with open(log_path, 'r') as f:
                learning_history = json.load(f)
        
        learning_history.append(learning_log)
        
        # Keep only last 100 entries
        if len(learning_history) > 100:
            learning_history = learning_history[-100:]
        
        with open(log_path, 'w') as f:
            json.dump(learning_history, f, indent=2)
        
        logger.info("Feedback learning complete")
        
        return {
            'status': 'success',
            'metrics': metrics,
            'adjustments': adjustments,
            'matched_trades_count': len(matched_trades)
        }


if __name__ == "__main__":
    import sys
    
    predictions_path = sys.argv[1] if len(sys.argv) > 1 else "logs/evaluation/natron_predictions.csv"
    events_path = sys.argv[2] if len(sys.argv) > 2 else "logs/realtime_events.csv"
    config_path = sys.argv[3] if len(sys.argv) > 3 else "configs/realtime_config.yaml"
    output_config_path = sys.argv[4] if len(sys.argv) > 4 else "configs/adaptive_config.yaml"
    
    engine = AutoFeedbackEngine(predictions_path, events_path, config_path, output_config_path)
    result = engine.learn_and_adapt()
    
    print(f"Feedback learning result: {result['status']}")
    print(f"Win rate: {result['metrics'].get('win_rate', 0):.2%}")
    print(f"Prediction accuracy: {result['metrics'].get('prediction_accuracy', 0):.2%}")
