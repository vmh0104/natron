"""
Auto Feedback Learning Engine for Natron AI Trading System
Learns from trading feedback and adapts configuration dynamically.
"""

import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import logging

from log_analyzer import LogAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdaptiveConfig:
    """
    Adaptive configuration that adjusts based on performance feedback.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize adaptive config.
        
        Args:
            config_path: Path to base config YAML
        """
        self.config_path = Path(config_path)
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.learning_history = []
        self.adaptation_log = []
    
    def load_performance_metrics(self, report_path: str) -> Dict:
        """
        Load performance metrics from analysis report.
        
        Args:
            report_path: Path to analysis report JSON
            
        Returns:
            Performance metrics dictionary
        """
        with open(report_path, 'r') as f:
            report = json.load(f)
        return report
    
    def compute_confidence_scaling(self, metrics: Dict) -> float:
        """
        Compute confidence scaling factor based on prediction accuracy.
        
        Args:
            metrics: Performance metrics dictionary
            
        Returns:
            Confidence scaling factor (0.5 to 1.5)
        """
        forecast_acc = metrics.get('prediction_accuracy', {}).get('forecast_accuracy', 0.5)
        regime_acc = metrics.get('prediction_accuracy', {}).get('regime_accuracy', 0.5)
        
        # Average accuracy
        avg_acc = (forecast_acc + regime_acc) / 2.0
        
        # Scale confidence: if accuracy is high, increase threshold; if low, decrease
        # Base scaling: 1.0 when accuracy is 0.6
        scaling = 0.5 + (avg_acc - 0.5) * 2.0  # Maps 0.5->0.5, 0.6->0.7, 0.7->0.9, 0.8->1.1, etc.
        scaling = np.clip(scaling, 0.5, 1.5)
        
        return scaling
    
    def compute_threshold_adjustment(self, metrics: Dict) -> Dict[str, float]:
        """
        Compute threshold adjustments based on trade outcomes.
        
        Args:
            metrics: Performance metrics dictionary
            
        Returns:
            Dictionary with threshold adjustments
        """
        trade_outcomes = metrics.get('trade_outcomes', {})
        win_rate = trade_outcomes.get('win_rate', 0.5)
        profit_factor = trade_outcomes.get('profit_factor', 1.0)
        
        adjustments = {
            'entry_threshold': 0.0,
            'cancel_threshold': 0.0
        }
        
        # If win rate is low, increase entry threshold (be more selective)
        if win_rate < 0.4:
            adjustments['entry_threshold'] = 0.1
        elif win_rate > 0.6:
            adjustments['entry_threshold'] = -0.05  # Can be slightly less selective
        
        # If profit factor is low, increase cancel threshold (exit faster)
        if profit_factor < 1.0:
            adjustments['cancel_threshold'] = 0.1
        elif profit_factor > 2.0:
            adjustments['cancel_threshold'] = -0.05
        
        return adjustments
    
    def adapt_config(self, metrics: Dict) -> Dict:
        """
        Adapt configuration based on performance metrics.
        
        Args:
            metrics: Performance metrics dictionary
            
        Returns:
            Updated configuration dictionary
        """
        # Compute adjustments
        confidence_scaling = self.compute_confidence_scaling(metrics)
        threshold_adjustments = self.compute_threshold_adjustment(metrics)
        
        # Get current trading config
        trading_config = self.config.get('trading', {})
        
        # Apply confidence scaling
        current_entry_threshold = trading_config.get('entry_threshold', 0.6)
        current_cancel_threshold = trading_config.get('cancel_threshold', 0.3)
        
        # Scale thresholds
        new_entry_threshold = current_entry_threshold * confidence_scaling
        new_cancel_threshold = current_cancel_threshold * confidence_scaling
        
        # Apply threshold adjustments
        new_entry_threshold += threshold_adjustments['entry_threshold']
        new_cancel_threshold += threshold_adjustments['cancel_threshold']
        
        # Clip to reasonable ranges
        new_entry_threshold = np.clip(new_entry_threshold, 0.3, 0.9)
        new_cancel_threshold = np.clip(new_cancel_threshold, 0.1, 0.5)
        
        # Update config
        updated_config = self.config.copy()
        updated_config['trading']['entry_threshold'] = float(new_entry_threshold)
        updated_config['trading']['cancel_threshold'] = float(new_cancel_threshold)
        
        # Log adaptation
        adaptation = {
            'timestamp': datetime.now().isoformat(),
            'confidence_scaling': float(confidence_scaling),
            'threshold_adjustments': threshold_adjustments,
            'old_entry_threshold': float(current_entry_threshold),
            'new_entry_threshold': float(new_entry_threshold),
            'old_cancel_threshold': float(current_cancel_threshold),
            'new_cancel_threshold': float(new_cancel_threshold),
            'metrics': metrics
        }
        
        self.adaptation_log.append(adaptation)
        self.learning_history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'adaptations': adaptation
        })
        
        logger.info(f"Adapted config: Entry threshold {current_entry_threshold:.3f} -> {new_entry_threshold:.3f}, "
                   f"Cancel threshold {current_cancel_threshold:.3f} -> {new_cancel_threshold:.3f}")
        
        return updated_config
    
    def save_adapted_config(self, output_path: str):
        """
        Save adapted configuration to file.
        
        Args:
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved adapted config to {output_path}")
    
    def save_learning_history(self, output_path: str):
        """
        Save learning history to JSON.
        
        Args:
            output_path: Output file path
        """
        with open(output_path, 'w') as f:
            json.dump(self.learning_history, f, indent=2)
        
        logger.info(f"Saved learning history to {output_path}")


class AutoFeedbackEngine:
    """
    Main feedback learning engine.
    """
    
    def __init__(self, 
                 config_path: str,
                 predictions_file: str,
                 events_file: str):
        """
        Initialize feedback engine.
        
        Args:
            config_path: Path to realtime config YAML
            predictions_file: Path to predictions CSV
            events_file: Path to events CSV
        """
        self.adaptive_config = AdaptiveConfig(config_path)
        self.log_analyzer = LogAnalyzer(predictions_file, events_file)
    
    def run_feedback_cycle(self) -> Dict:
        """
        Run one feedback learning cycle.
        
        Returns:
            Updated configuration dictionary
        """
        logger.info("Starting feedback learning cycle...")
        
        # Load and analyze data
        self.log_analyzer.load_data()
        report = self.log_analyzer.generate_report()
        
        # Adapt configuration
        updated_config = self.adaptive_config.adapt_config(report)
        
        # Update internal config
        self.adaptive_config.config = updated_config
        
        logger.info("Feedback learning cycle complete")
        
        return updated_config
    
    def save_results(self, output_dir: str):
        """
        Save feedback learning results.
        
        Args:
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save adapted config
        config_output = output_dir / 'adaptive_config.yaml'
        self.adaptive_config.save_adapted_config(config_output)
        
        # Save learning history
        history_output = output_dir / 'learning_history.json'
        self.adaptive_config.save_learning_history(history_output)
        
        # Save analysis report
        report_output = output_dir / 'feedback_analysis_report.json'
        self.log_analyzer.save_report(report_output)


def main():
    """Main feedback learning function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Natron Auto Feedback Learning Engine')
    parser.add_argument('--config', type=str, default='realtime_config.yaml',
                       help='Path to realtime config YAML')
    parser.add_argument('--predictions', type=str, default='natron_predictions.csv',
                       help='Path to predictions CSV')
    parser.add_argument('--events', type=str, default='realtime_events.csv',
                       help='Path to events CSV')
    parser.add_argument('--output', type=str, default='feedback_output',
                       help='Output directory')
    args = parser.parse_args()
    
    engine = AutoFeedbackEngine(args.config, args.predictions, args.events)
    engine.run_feedback_cycle()
    engine.save_results(args.output)


if __name__ == "__main__":
    main()
