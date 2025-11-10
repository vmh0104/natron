"""
Model Evaluation Script for Natron
Evaluates model performance and generates reports.
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import yaml
import json
from pathlib import Path
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from typing import Dict, Optional

from ..training.model_natron import NatronTransformer
from ..training.dataset_loader import NatronDataset, create_data_loaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NatronEvaluator:
    """Evaluator for Natron model."""
    
    REGIME_NAMES = ['BULL_STRONG', 'BULL_WEAK', 'BEAR_STRONG', 'BEAR_WEAK', 'RANGE', 'VOLATILE']
    FORECAST_NAMES = ['DOWN', 'UP']
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to config YAML (optional, will load from checkpoint if not provided)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = checkpoint.get('config', {})
        
        # Load model
        input_dim = checkpoint['input_dim']
        model_config = self.config.get('model', {})
        
        self.model = NatronTransformer(
            input_dim=input_dim,
            d_model=model_config.get('d_model', 256),
            nhead=model_config.get('nhead', 8),
            num_layers=model_config.get('num_layers', 6),
            dim_feedforward=model_config.get('dim_feedforward', 1024),
            dropout=model_config.get('dropout', 0.1),
            activation=model_config.get('activation', 'gelu'),
            regime_classes=model_config.get('regime_classes', 6),
            context_output_dim=model_config.get('context_output_dim', 1),
            forecast_classes=model_config.get('forecast_classes', 2)
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Model trained for {checkpoint.get('epoch', 'unknown')} epochs")
        logger.info(f"Best validation loss: {checkpoint.get('val_loss', 'unknown'):.4f}")
    
    def evaluate(self, data_loader, split_name: str = 'test') -> Dict:
        """
        Evaluate model on a dataset.
        
        Args:
            data_loader: DataLoader for evaluation
            split_name: Name of the split (for logging)
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating on {split_name} set...")
        
        all_regime_preds = []
        all_regime_probs = []
        all_regime_true = []
        
        all_forecast_preds = []
        all_forecast_probs = []
        all_forecast_true = []
        
        all_context_scores = []
        all_context_true = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {split_name}"):
                features = batch['features'].to(self.device)
                
                # Get predictions
                predictions = self.model.predict(features)
                
                # Regime
                all_regime_preds.extend(predictions['regime_pred'].cpu().numpy())
                all_regime_probs.extend(predictions['regime_probs'].cpu().numpy())
                all_regime_true.extend(batch['regime'].numpy())
                
                # Forecast
                all_forecast_preds.extend(predictions['forecast_pred'].cpu().numpy())
                all_forecast_probs.extend(predictions['forecast_probs'].cpu().numpy())
                all_forecast_true.extend(batch['forecast'].numpy())
                
                # Context
                all_context_scores.extend(predictions['context_score'].cpu().numpy().flatten())
                all_context_true.extend(batch['context'].numpy())
        
        # Convert to numpy arrays
        regime_preds = np.array(all_regime_preds)
        regime_true = np.array(all_regime_true)
        forecast_preds = np.array(all_forecast_preds)
        forecast_true = np.array(all_forecast_true)
        context_scores = np.array(all_context_scores)
        context_true = np.array(all_context_true)
        
        # Compute metrics
        metrics = {}
        
        # Regime metrics
        metrics['regime'] = {
            'accuracy': float(accuracy_score(regime_true, regime_preds)),
            'f1_macro': float(f1_score(regime_true, regime_preds, average='macro')),
            'f1_weighted': float(f1_score(regime_true, regime_preds, average='weighted')),
            'precision_macro': float(precision_score(regime_true, regime_preds, average='macro', zero_division=0)),
            'recall_macro': float(recall_score(regime_true, regime_preds, average='macro', zero_division=0))
        }
        
        # Per-class regime metrics
        regime_report = classification_report(
            regime_true, regime_preds,
            target_names=self.REGIME_NAMES,
            output_dict=True,
            zero_division=0
        )
        metrics['regime']['per_class'] = regime_report
        
        # Forecast metrics
        metrics['forecast'] = {
            'accuracy': float(accuracy_score(forecast_true, forecast_preds)),
            'f1': float(f1_score(forecast_true, forecast_preds)),
            'precision': float(precision_score(forecast_true, forecast_preds, zero_division=0)),
            'recall': float(recall_score(forecast_true, forecast_preds, zero_division=0))
        }
        
        # Context metrics (MSE, MAE, correlation)
        context_mse = np.mean((context_scores - context_true) ** 2)
        context_mae = np.mean(np.abs(context_scores - context_true))
        context_corr = np.corrcoef(context_scores, context_true)[0, 1]
        
        metrics['context'] = {
            'mse': float(context_mse),
            'mae': float(context_mae),
            'correlation': float(context_corr)
        }
        
        # Store predictions for export
        metrics['predictions'] = {
            'regime_preds': regime_preds.tolist(),
            'regime_probs': np.array(all_regime_probs).tolist(),
            'regime_true': regime_true.tolist(),
            'forecast_preds': forecast_preds.tolist(),
            'forecast_probs': np.array(all_forecast_probs).tolist(),
            'forecast_true': forecast_true.tolist(),
            'context_scores': context_scores.tolist(),
            'context_true': context_true.tolist()
        }
        
        return metrics
    
    def export_predictions(self, data_loader, output_file: str):
        """
        Export predictions to CSV.
        
        Args:
            data_loader: DataLoader for evaluation
            output_file: Path to output CSV file
        """
        logger.info(f"Exporting predictions to {output_file}")
        
        predictions_list = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Exporting predictions"):
                features = batch['features'].to(self.device)
                predictions = self.model.predict(features)
                
                batch_size = features.size(0)
                for i in range(batch_size):
                    regime_probs = predictions['regime_probs'][i].cpu().numpy()
                    forecast_probs = predictions['forecast_probs'][i].cpu().numpy()
                    
                    pred_dict = {
                        'regime_pred': int(predictions['regime_pred'][i].cpu().item()),
                        'regime_name': self.REGIME_NAMES[predictions['regime_pred'][i].cpu().item()],
                        'regime_true': int(batch['regime'][i].item()),
                        'regime_true_name': self.REGIME_NAMES[batch['regime'][i].item()],
                        'regime_confidence': float(regime_probs.max()),
                        'forecast_pred': int(predictions['forecast_pred'][i].cpu().item()),
                        'forecast_name': self.FORECAST_NAMES[predictions['forecast_pred'][i].cpu().item()],
                        'forecast_true': int(batch['forecast'][i].item()),
                        'forecast_confidence': float(forecast_probs.max()),
                        'context_score': float(predictions['context_score'][i].cpu().item()),
                        'context_true': float(batch['context'][i].item())
                    }
                    
                    # Add individual regime probabilities
                    for j, regime_name in enumerate(self.REGIME_NAMES):
                        pred_dict[f'prob_{regime_name}'] = float(regime_probs[j])
                    
                    predictions_list.append(pred_dict)
        
        df_predictions = pd.DataFrame(predictions_list)
        
        # Ensure directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        df_predictions.to_csv(output_file, index=False)
        logger.info(f"Exported {len(df_predictions)} predictions to {output_file}")
    
    def generate_report(self, metrics: Dict, output_file: str):
        """
        Generate evaluation report JSON.
        
        Args:
            metrics: Evaluation metrics dictionary
            output_file: Path to output JSON file
        """
        # Remove predictions from metrics for report (too large)
        report_metrics = {k: v for k, v in metrics.items() if k != 'predictions'}
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(report_metrics, f, indent=2)
        
        logger.info(f"Saved evaluation report to {output_file}")
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"\nRegime Classification:")
        logger.info(f"  Accuracy: {metrics['regime']['accuracy']:.4f}")
        logger.info(f"  F1 (macro): {metrics['regime']['f1_macro']:.4f}")
        logger.info(f"  F1 (weighted): {metrics['regime']['f1_weighted']:.4f}")
        
        logger.info(f"\nForecast Direction:")
        logger.info(f"  Accuracy: {metrics['forecast']['accuracy']:.4f}")
        logger.info(f"  F1: {metrics['forecast']['f1']:.4f}")
        logger.info(f"  Precision: {metrics['forecast']['precision']:.4f}")
        logger.info(f"  Recall: {metrics['forecast']['recall']:.4f}")
        
        logger.info(f"\nContext Strength:")
        logger.info(f"  MSE: {metrics['context']['mse']:.4f}")
        logger.info(f"  MAE: {metrics['context']['mae']:.4f}")
        logger.info(f"  Correlation: {metrics['context']['correlation']:.4f}")
        logger.info("=" * 60)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Natron Model')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to processed data CSV')
    parser.add_argument('--config', type=str, default=None, help='Path to config YAML')
    parser.add_argument('--output', type=str, default='logs/natron_predictions.csv', help='Output predictions CSV')
    parser.add_argument('--report', type=str, default='logs/metrics_report.json', help='Output metrics JSON')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='Data split to evaluate')
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.data)
    
    # Split data (same logic as training)
    # For simplicity, use same splits as training
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)
    
    if args.split == 'train':
        eval_data = df[:train_size]
    elif args.split == 'val':
        eval_data = df[train_size:train_size + val_size]
    else:
        eval_data = df[train_size + val_size:]
    
    # Create evaluator
    evaluator = NatronEvaluator(args.model, args.config)
    
    # Create data loader
    # Load scaler
    model_dir = Path(args.model).parent
    scaler_path = model_dir / 'scaler.pkl'
    scaler = NatronDataset.load_scaler(str(scaler_path))
    
    eval_dataset = NatronDataset(
        eval_data,
        sequence_length=96,  # Should match training
        scaler=scaler,
        fit_scaler=False
    )
    
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=32,
        shuffle=False
    )
    
    # Evaluate
    metrics = evaluator.evaluate(eval_loader, split_name=args.split)
    
    # Export predictions
    evaluator.export_predictions(eval_loader, args.output)
    
    # Generate report
    evaluator.generate_report(metrics, args.report)


if __name__ == '__main__':
    main()
