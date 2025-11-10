"""
Natron Model Evaluation Script
Evaluates model performance and generates reports.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
import logging
from tqdm import tqdm

from model_natron import NatronTransformer
from dataset_loader import create_dataloaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluator for Natron model."""
    
    def __init__(self, config_path: str, model_path: str):
        """
        Initialize evaluator.
        
        Args:
            config_path: Path to training config YAML
            model_path: Path to saved model checkpoint
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load data
        df = pd.read_csv(self.config['data_path'])
        
        # Create dataloaders
        _, _, self.test_loader = create_dataloaders(
            df=df,
            train_ratio=self.config.get('train_ratio', 0.7),
            val_ratio=self.config.get('val_ratio', 0.15),
            sequence_length=self.config.get('sequence_length', 96),
            forecast_horizon=self.config.get('forecast_horizon', 5),
            batch_size=self.config.get('batch_size', 32),
            num_workers=self.config.get('num_workers', 4)
        )
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Regime class names
        self.regime_classes = [
            'BULL_STRONG', 'BULL_WEAK', 'BEAR_STRONG',
            'BEAR_WEAK', 'RANGE', 'VOLATILE'
        ]
    
    def _load_model(self, model_path: str) -> NatronTransformer:
        """Load model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get input dimension from first batch
        sample_batch = next(iter(self.test_loader))
        input_dim = sample_batch['features'].shape[2]
        
        # Initialize model
        model = NatronTransformer(
            input_dim=input_dim,
            d_model=self.config.get('d_model', 256),
            nhead=self.config.get('nhead', 8),
            num_layers=self.config.get('num_layers', 6),
            dim_feedforward=self.config.get('dim_feedforward', 1024),
            dropout=self.config.get('dropout', 0.1),
            max_seq_len=self.config.get('sequence_length', 96),
            num_regime_classes=self.config.get('num_regime_classes', 6),
            num_forecast_classes=self.config.get('num_forecast_classes', 2)
        ).to(self.device)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from {model_path}")
        
        return model
    
    def evaluate(self) -> dict:
        """Evaluate model on test set."""
        logger.info("Evaluating model...")
        
        all_predictions = {
            'regime': [],
            'context': [],
            'forecast': []
        }
        all_targets = {
            'regime': [],
            'context': [],
            'forecast': []
        }
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                features = batch['features'].to(self.device)
                
                predictions = self.model(features)
                
                # Regime predictions
                regime_pred = predictions['regime'].argmax(dim=1).cpu().numpy()
                all_predictions['regime'].extend(regime_pred)
                all_targets['regime'].extend(batch['regime'].numpy())
                
                # Context predictions
                context_pred = predictions['context'].squeeze().cpu().numpy()
                all_predictions['context'].extend(context_pred)
                all_targets['context'].extend(batch['context'].numpy())
                
                # Forecast predictions
                forecast_pred = predictions['forecast'].argmax(dim=1).cpu().numpy()
                all_predictions['forecast'].extend(forecast_pred)
                all_targets['forecast'].extend(batch['forecast'].numpy())
        
        # Convert to numpy arrays
        for key in all_predictions:
            all_predictions[key] = np.array(all_predictions[key])
            all_targets[key] = np.array(all_targets[key])
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_predictions, all_targets)
        
        return metrics, all_predictions, all_targets
    
    def _calculate_metrics(self, predictions: dict, targets: dict) -> dict:
        """Calculate evaluation metrics."""
        metrics = {}
        
        # Regime classification metrics
        regime_acc = accuracy_score(targets['regime'], predictions['regime'])
        regime_f1 = f1_score(targets['regime'], predictions['regime'], average='weighted')
        regime_f1_macro = f1_score(targets['regime'], predictions['regime'], average='macro')
        precision, recall, f1, support = precision_recall_fscore_support(
            targets['regime'], predictions['regime'], average=None, zero_division=0
        )
        
        metrics['regime'] = {
            'accuracy': float(regime_acc),
            'f1_weighted': float(regime_f1),
            'f1_macro': float(regime_f1_macro),
            'per_class': {
                self.regime_classes[i]: {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1': float(f1[i]),
                    'support': int(support[i])
                }
                for i in range(len(self.regime_classes))
            }
        }
        
        # Context strength metrics (MSE, MAE)
        context_mse = np.mean((predictions['context'] - targets['context']) ** 2)
        context_mae = np.mean(np.abs(predictions['context'] - targets['context']))
        context_corr = np.corrcoef(predictions['context'], targets['context'])[0, 1]
        
        metrics['context'] = {
            'mse': float(context_mse),
            'mae': float(context_mae),
            'correlation': float(context_corr)
        }
        
        # Forecast classification metrics
        forecast_acc = accuracy_score(targets['forecast'], predictions['forecast'])
        forecast_f1 = f1_score(targets['forecast'], predictions['forecast'], average='weighted')
        precision, recall, f1, support = precision_recall_fscore_support(
            targets['forecast'], predictions['forecast'], average=None, zero_division=0
        )
        
        metrics['forecast'] = {
            'accuracy': float(forecast_acc),
            'f1_weighted': float(forecast_f1),
            'per_class': {
                'down': {
                    'precision': float(precision[0]),
                    'recall': float(recall[0]),
                    'f1': float(f1[0]),
                    'support': int(support[0])
                },
                'up': {
                    'precision': float(precision[1]),
                    'recall': float(recall[1]),
                    'f1': float(f1[1]),
                    'support': int(support[1])
                }
            }
        }
        
        return metrics
    
    def save_predictions(self, predictions: dict, targets: dict, output_path: str = "natron_predictions.csv"):
        """Save predictions to CSV."""
        df = pd.DataFrame({
            'regime_pred': predictions['regime'],
            'regime_true': targets['regime'],
            'regime_pred_name': [self.regime_classes[p] for p in predictions['regime']],
            'regime_true_name': [self.regime_classes[t] for t in targets['regime']],
            'context_pred': predictions['context'],
            'context_true': targets['context'],
            'forecast_pred': predictions['forecast'],
            'forecast_true': targets['forecast'],
            'forecast_pred_name': ['up' if p == 1 else 'down' for p in predictions['forecast']],
            'forecast_true_name': ['up' if t == 1 else 'down' for t in targets['forecast']]
        })
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved predictions to {output_path}")
        return df
    
    def generate_report(self, metrics: dict, output_path: str = "metrics_report.json"):
        """Generate evaluation report."""
        report = {
            'model': 'Natron V6',
            'evaluation_date': pd.Timestamp.now().isoformat(),
            'metrics': metrics,
            'summary': {
                'regime_accuracy': metrics['regime']['accuracy'],
                'forecast_accuracy': metrics['forecast']['accuracy'],
                'context_correlation': metrics['context']['correlation']
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved evaluation report to {output_path}")
        return report


if __name__ == "__main__":
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config_train.yaml"
    model_path = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/natron_v6.pt"
    
    evaluator = ModelEvaluator(config_path, model_path)
    metrics, predictions, targets = evaluator.evaluate()
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Regime Accuracy: {metrics['regime']['accuracy']:.4f}")
    print(f"Regime F1 (weighted): {metrics['regime']['f1_weighted']:.4f}")
    print(f"Forecast Accuracy: {metrics['forecast']['accuracy']:.4f}")
    print(f"Context Correlation: {metrics['context']['correlation']:.4f}")
    print("="*50)
    
    # Save predictions and report
    evaluator.save_predictions(predictions, targets)
    evaluator.generate_report(metrics)
