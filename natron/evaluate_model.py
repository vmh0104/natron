"""
Model Evaluation Script for Natron AI Trading System
Evaluates model performance and generates reports.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from tqdm import tqdm
import yaml

from model_natron import NatronTransformer
from dataset_loader import NatronDataset, DataLoader


class ModelEvaluator:
    """
    Evaluator for Natron model.
    """
    
    def __init__(self, model_path: str, config_path: str, device: str = 'cuda'):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to training config YAML
            device: Device to run evaluation on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        model_config = self.config['model']
        
        # Determine n_features from checkpoint or config
        # We'll need to pass this or infer from data
        self.model = None  # Will be initialized in evaluate()
        self.checkpoint = checkpoint
    
    def load_model(self, n_features: int):
        """
        Load model architecture.
        
        Args:
            n_features: Number of input features
        """
        model_config = self.config['model']
        self.model = NatronTransformer(
            n_features=n_features,
            d_model=model_config.get('d_model', 128),
            nhead=model_config.get('nhead', 8),
            num_layers=model_config.get('num_layers', 6),
            dim_feedforward=model_config.get('dim_feedforward', 512),
            dropout=model_config.get('dropout', 0.1),
            max_seq_len=model_config.get('max_seq_len', 96),
            n_regime_classes=model_config.get('n_regime_classes', 6)
        ).to(self.device)
        
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
    
    def evaluate(self, data_loader: DataLoader) -> dict:
        """
        Evaluate model on dataset.
        
        Args:
            data_loader: Data loader for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        all_regime_preds = []
        all_regime_targets = []
        all_context_preds = []
        all_context_targets = []
        all_forecast_preds = []
        all_forecast_targets = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                X, y_regime, y_context, y_forecast = [b.to(self.device) for b in batch]
                
                regime_logits, context_score, forecast_logits = self.model(X)
                
                # Regime predictions
                regime_preds = torch.argmax(regime_logits, dim=1).cpu().numpy()
                all_regime_preds.extend(regime_preds)
                all_regime_targets.extend(y_regime.cpu().numpy())
                
                # Context predictions
                context_preds = context_score.squeeze().cpu().numpy()
                all_context_preds.extend(context_preds)
                all_context_targets.extend(y_context.cpu().numpy())
                
                # Forecast predictions
                forecast_preds = torch.argmax(forecast_logits, dim=1).cpu().numpy()
                all_forecast_preds.extend(forecast_preds)
                all_forecast_targets.extend(y_forecast.cpu().numpy())
        
        # Convert to numpy arrays
        all_regime_preds = np.array(all_regime_preds)
        all_regime_targets = np.array(all_regime_targets)
        all_context_preds = np.array(all_context_preds)
        all_context_targets = np.array(all_context_targets)
        all_forecast_preds = np.array(all_forecast_preds)
        all_forecast_targets = np.array(all_forecast_targets)
        
        # Compute metrics
        metrics = {}
        
        # Regime classification metrics
        metrics['regime'] = {
            'accuracy': accuracy_score(all_regime_targets, all_regime_preds),
            'f1_macro': f1_score(all_regime_targets, all_regime_preds, average='macro'),
            'f1_weighted': f1_score(all_regime_targets, all_regime_preds, average='weighted'),
            'precision': precision_score(all_regime_targets, all_regime_preds, average='macro', zero_division=0),
            'recall': recall_score(all_regime_targets, all_regime_preds, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(all_regime_targets, all_regime_preds).tolist(),
            'classification_report': classification_report(
                all_regime_targets, all_regime_preds,
                target_names=['BULL_STRONG', 'BULL_WEAK', 'BEAR_STRONG', 'BEAR_WEAK', 'RANGE', 'VOLATILE'],
                output_dict=True,
                zero_division=0
            )
        }
        
        # Context strength metrics (MSE, MAE, R2)
        context_mse = np.mean((all_context_preds - all_context_targets) ** 2)
        context_mae = np.mean(np.abs(all_context_preds - all_context_targets))
        context_r2 = 1 - (np.sum((all_context_targets - all_context_preds) ** 2) /
                         np.sum((all_context_targets - np.mean(all_context_targets)) ** 2))
        
        metrics['context'] = {
            'mse': float(context_mse),
            'mae': float(context_mae),
            'r2': float(context_r2),
            'correlation': float(np.corrcoef(all_context_preds, all_context_targets)[0, 1])
        }
        
        # Forecast direction metrics
        metrics['forecast'] = {
            'accuracy': accuracy_score(all_forecast_targets, all_forecast_preds),
            'f1': f1_score(all_forecast_targets, all_forecast_preds),
            'precision': precision_score(all_forecast_targets, all_forecast_preds, zero_division=0),
            'recall': recall_score(all_forecast_targets, all_forecast_preds, zero_division=0)
        }
        
        # Store predictions for export
        metrics['predictions'] = {
            'regime_pred': all_regime_preds.tolist(),
            'regime_target': all_regime_targets.tolist(),
            'context_pred': all_context_preds.tolist(),
            'context_target': all_context_targets.tolist(),
            'forecast_pred': all_forecast_preds.tolist(),
            'forecast_target': all_forecast_targets.tolist()
        }
        
        return metrics
    
    def print_metrics(self, metrics: dict):
        """
        Print evaluation metrics.
        
        Args:
            metrics: Metrics dictionary
        """
        print("\n" + "="*60)
        print("NATRON MODEL EVALUATION RESULTS")
        print("="*60)
        
        print("\n[1] Regime Classification:")
        print(f"  Accuracy:  {metrics['regime']['accuracy']:.4f}")
        print(f"  F1 (macro): {metrics['regime']['f1_macro']:.4f}")
        print(f"  F1 (weighted): {metrics['regime']['f1_weighted']:.4f}")
        print(f"  Precision: {metrics['regime']['precision']:.4f}")
        print(f"  Recall:    {metrics['regime']['recall']:.4f}")
        
        print("\n[2] Context Strength:")
        print(f"  MSE:         {metrics['context']['mse']:.6f}")
        print(f"  MAE:         {metrics['context']['mae']:.6f}")
        print(f"  R²:          {metrics['context']['r2']:.4f}")
        print(f"  Correlation: {metrics['context']['correlation']:.4f}")
        
        print("\n[3] Forecast Direction:")
        print(f"  Accuracy:  {metrics['forecast']['accuracy']:.4f}")
        print(f"  F1:        {metrics['forecast']['f1']:.4f}")
        print(f"  Precision: {metrics['forecast']['precision']:.4f}")
        print(f"  Recall:    {metrics['forecast']['recall']:.4f}")
        
        print("\n" + "="*60)
    
    def export_predictions(self, metrics: dict, output_path: str):
        """
        Export predictions to CSV.
        
        Args:
            metrics: Metrics dictionary with predictions
            output_path: Output CSV file path
        """
        df = pd.DataFrame({
            'regime_pred': metrics['predictions']['regime_pred'],
            'regime_target': metrics['predictions']['regime_target'],
            'context_pred': metrics['predictions']['context_pred'],
            'context_target': metrics['predictions']['context_target'],
            'forecast_pred': metrics['predictions']['forecast_pred'],
            'forecast_target': metrics['predictions']['forecast_target']
        })
        
        df.to_csv(output_path, index=False)
        print(f"\n✓ Exported predictions to {output_path}")


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Natron Model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config_train.yaml',
                       help='Path to training config YAML')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to processed data CSV')
    parser.add_argument('--output', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load processed data
    print("Loading processed data...")
    processed_df = pd.read_csv(args.data, index_col=0)
    
    # Extract features and labels
    feature_cols = [col for col in processed_df.columns 
                   if col not in ['open', 'high', 'low', 'close', 'volume', 'regime']]
    feature_df = processed_df[feature_cols]
    labels = processed_df['regime'].astype(int)
    
    # Create sequences
    from data_pipeline import DataPipeline
    pipeline = DataPipeline()
    X, y_regime, y_context, y_forecast = pipeline.create_sequences(
        feature_df, labels,
        sequence_length=96,
        forecast_horizon=5
    )
    
    # Split data
    n_samples = len(X)
    n_train = int(n_samples * 0.7)
    n_val = int(n_samples * 0.15)
    
    if args.split == 'train':
        indices = slice(0, n_train)
    elif args.split == 'val':
        indices = slice(n_train, n_train + n_val)
    else:  # test
        indices = slice(n_train + n_val, None)
    
    X_split = X[indices]
    y_regime_split = y_regime[indices]
    y_context_split = y_context[indices]
    y_forecast_split = y_forecast[indices]
    
    # Create dataset (need normalization stats from training)
    # For simplicity, we'll use the full dataset to get normalization stats
    from dataset_loader import create_dataloaders
    _, _, _, norm_stats = create_dataloaders(
        X, y_regime, y_context, y_forecast,
        train_ratio=0.7,
        val_ratio=0.15,
        batch_size=32,
        shuffle=False
    )
    
    # Create dataset for split
    from dataset_loader import NatronDataset
    dataset = NatronDataset(
        X_split, y_regime_split, y_context_split, y_forecast_split,
        normalize=True,
        feature_mean=norm_stats[0],
        feature_std=norm_stats[1]
    )
    
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model, args.config)
    evaluator.load_model(n_features=X.shape[-1])
    
    # Evaluate
    metrics = evaluator.evaluate(data_loader)
    
    # Print metrics
    evaluator.print_metrics(metrics)
    
    # Export results
    metrics_path = output_dir / 'metrics_report.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Saved metrics to {metrics_path}")
    
    # Export predictions
    predictions_path = output_dir / 'natron_predictions.csv'
    evaluator.export_predictions(metrics, predictions_path)


if __name__ == "__main__":
    main()
