"""
Auto Feedback Engine for Natron
Learns from trading feedback and adapts configuration dynamically.
"""

import yaml
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Optional
from datetime import datetime

from log_analyzer import LogAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoFeedbackEngine:
    """
    Adaptive learning engine that adjusts configuration based on trading performance.
    """
    
    def __init__(self, config_path: str, predictions_file: str, events_file: str):
        """
        Initialize feedback engine.
        
        Args:
            config_path: Path to adaptive config YAML
            predictions_file: Path to predictions CSV
            events_file: Path to realtime events CSV
        """
        self.config = self._load_config(config_path)
        self.analyzer = LogAnalyzer(predictions_file, events_file)
        self.learning_curve = []
        
        # Load existing learning curve if available
        self._load_learning_curve()
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_learning_curve(self):
        """Load existing learning curve."""
        curve_path = self.config['learning_curve']['save_path']
        if Path(curve_path).exists():
            with open(curve_path, 'r') as f:
                self.learning_curve = json.load(f)
            logger.info(f"Loaded {len(self.learning_curve)} learning curve entries")
    
    def _save_learning_curve(self):
        """Save learning curve to file."""
        curve_path = self.config['learning_curve']['save_path']
        Path(curve_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(curve_path, 'w') as f:
            json.dump(self.learning_curve, f, indent=2)
    
    def analyze_performance(self) -> Dict:
        """
        Analyze current performance.
        
        Returns:
            Performance metrics dictionary
        """
        self.analyzer.load_data()
        report = self.analyzer.generate_report()
        return report
    
    def adapt_thresholds(self, performance: Dict) -> Dict:
        """
        Adapt confidence and entry thresholds based on performance.
        
        Args:
            performance: Performance metrics dictionary
            
        Returns:
            Updated threshold configuration
        """
        if not self.config['adaptive_thresholds']['confidence_scaling']['enabled']:
            return {}
        
        trade_perf = performance.get('trade_performance', {})
        win_rate = trade_perf.get('win_rate', 0.5)
        forecast_accuracy = trade_perf.get('forecast_accuracy', 0.5)
        
        # Adjust confidence threshold based on performance
        threshold_config = self.config['adaptive_thresholds']
        base_threshold = threshold_config['entry_threshold']['base_threshold']
        adjustment_rate = threshold_config['entry_threshold']['adjustment_rate']
        min_threshold = threshold_config['entry_threshold']['min_threshold']
        max_threshold = threshold_config['entry_threshold']['max_threshold']
        
        # If performance is good, can lower threshold slightly
        # If performance is poor, raise threshold
        if win_rate > 0.55 or forecast_accuracy > 0.55:
            new_threshold = base_threshold - adjustment_rate
        elif win_rate < 0.45 or forecast_accuracy < 0.45:
            new_threshold = base_threshold + adjustment_rate
        else:
            new_threshold = base_threshold
        
        new_threshold = np.clip(new_threshold, min_threshold, max_threshold)
        
        return {
            'entry_confidence_threshold': float(new_threshold)
        }
    
    def adapt_loss_weights(self, performance: Dict) -> Dict:
        """
        Adapt loss weights based on which head performs better.
        
        Args:
            performance: Performance metrics dictionary
            
        Returns:
            Updated loss weights
        """
        regime_perf = performance.get('regime_performance', {})
        
        if not regime_perf:
            return {}
        
        # Compute average regime accuracy
        regime_accuracies = [v.get('regime_accuracy', 0.5) for v in regime_perf.values()]
        avg_regime_accuracy = np.mean(regime_accuracies) if regime_accuracies else 0.5
        
        trade_perf = performance.get('trade_performance', {})
        forecast_accuracy = trade_perf.get('forecast_accuracy', 0.5)
        
        # Adjust weights based on relative performance
        weight_config = self.config['weight_adjustment']
        base_weights = {
            'regime': weight_config['regime_weight']['base'],
            'context': weight_config['context_weight']['base'],
            'forecast': weight_config['forecast_weight']['base']
        }
        
        # If forecast performs better, increase its weight
        if forecast_accuracy > avg_regime_accuracy + 0.1:
            base_weights['forecast'] *= 1.1
            base_weights['regime'] *= 0.95
        elif avg_regime_accuracy > forecast_accuracy + 0.1:
            base_weights['regime'] *= 1.1
            base_weights['forecast'] *= 0.95
        
        # Clip to adjustment ranges
        regime_range = weight_config['regime_weight']['adjustment_range']
        context_range = weight_config['context_weight']['adjustment_range']
        forecast_range = weight_config['forecast_weight']['adjustment_range']
        
        new_weights = {
            'regime': float(np.clip(base_weights['regime'], regime_range[0], regime_range[1])),
            'context': float(np.clip(base_weights['context'], context_range[0], context_range[1])),
            'forecast': float(np.clip(base_weights['forecast'], forecast_range[0], forecast_range[1]))
        }
        
        return new_weights
    
    def update_config(self, updates: Dict, output_path: Optional[str] = None):
        """
        Update configuration file with adaptations.
        
        Args:
            updates: Dictionary with configuration updates
            output_path: Path to save updated config (default: update original)
        """
        # This would update the realtime_config.yaml or create a new adapted config
        # For now, we'll create an adapted config file
        if output_path is None:
            output_path = 'configs/realtime_config_adapted.yaml'
        
        # Load base realtime config
        base_config_path = 'configs/realtime_config.yaml'
        if Path(base_config_path).exists():
            with open(base_config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {}
        
        # Apply updates
        if 'entry_confidence_threshold' in updates:
            config.setdefault('trading', {})['entry_confidence_threshold'] = updates['entry_confidence_threshold']
        
        # Save updated config
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Updated configuration saved to {output_path}")
    
    def run_adaptation(self) -> Dict:
        """
        Run one adaptation cycle.
        
        Returns:
            Dictionary with adaptation results
        """
        if not self.config['feedback']['enabled']:
            logger.info("Feedback learning is disabled")
            return {}
        
        logger.info("Running adaptation cycle...")
        
        # Analyze performance
        performance = self.analyze_performance()
        
        # Check if we have enough samples
        trade_perf = performance.get('trade_performance', {})
        total_trades = trade_perf.get('total_trades', 0)
        
        min_samples = self.config['feedback']['min_samples']
        if total_trades < min_samples:
            logger.info(f"Not enough samples ({total_trades} < {min_samples}), skipping adaptation")
            return performance
        
        # Adapt thresholds
        threshold_updates = self.adapt_thresholds(performance)
        
        # Adapt weights
        weight_updates = self.adapt_loss_weights(performance)
        
        # Combine updates
        updates = {**threshold_updates}
        if weight_updates:
            updates['loss_weights'] = weight_updates
        
        # Update configuration
        if updates:
            self.update_config(updates)
        
        # Record in learning curve
        curve_entry = {
            'timestamp': datetime.now().isoformat(),
            'performance': performance,
            'updates': updates
        }
        self.learning_curve.append(curve_entry)
        self._save_learning_curve()
        
        logger.info("Adaptation cycle complete")
        logger.info(f"  Entry threshold: {threshold_updates.get('entry_confidence_threshold', 'N/A')}")
        logger.info(f"  Loss weights: {weight_updates}")
        
        return {
            'performance': performance,
            'updates': updates,
            'learning_curve_length': len(self.learning_curve)
        }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Natron Auto Feedback Engine')
    parser.add_argument('--config', type=str, required=True, help='Path to adaptive config YAML')
    parser.add_argument('--predictions', type=str, required=True, help='Path to predictions CSV')
    parser.add_argument('--events', type=str, required=True, help='Path to events CSV')
    
    args = parser.parse_args()
    
    engine = AutoFeedbackEngine(args.config, args.predictions, args.events)
    result = engine.run_adaptation()
    
    print("\n" + "=" * 60)
    print("ADAPTATION RESULTS")
    print("=" * 60)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
