"""
Model Evaluation Module for Natron

This module evaluates the trained model and generates reports.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
import yaml
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm

from ..training.model_natron import NatronTransformer
from ..training.dataset_loader import NatronDataset, create_dataloaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluator class for Natron model.
    """
    
    def __init__(self, model_path: str, config_path: str):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to training config YAML
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = None
        self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load trained model from checkpoint."""
        logger.info(f"Loading model from {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get input dimension from config or checkpoint
        if 'config' in checkpoint:
            model_config = checkpoint['config']['model']
        else:
            model_config = self.config['model']
        
        # Load processed data to get feature count
        processed_path = self.config['data']['processed_path']
        df = pd.read_csv(processed_path, index_col=0, nrows=1)
        exclude_cols = ['regime', 'regime_name', 'target', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        input_dim = len(feature_cols)
        
        # Build model
        self.model = NatronTransformer(
            input_dim=input_dim,
            d_model=model_config['d_model'],
            nhead=model_config['nhead'],
            num_layers=model_config['num_layers'],
            dim_feedforward=model_config['dim_feedforward'],
            dropout=model_config['dropout'],
            seq_len=model_config['seq_len'],
            num_regime_classes=model_config['num_regime_classes']
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def evaluate(self, data_loader, split_name: str = "test") -> dict:
        """
        Evaluate model on a dataset.
        
        Args:
            data_loader: DataLoader for evaluation
            split_name: Name of data split
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating on {split_name} set...")
        
        all_regime_preds = []
        all_regime_targets = []
        all_forecast_preds = []
        all_forecast_targets = []
        all_context_preds = []
        all_context_targets = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader):
                features = batch['features'].to(self.device)
                regime_target = batch['regime_target'].cpu().numpy()
                forecast_target = batch['forecast_target'].cpu().numpy()
                context_target = batch['context_target'].cpu().numpy()
                
                # Forward pass
                regime_pred, context_pred, forecast_pred = self.model(features)
                
                # Get predictions
                regime_pred_class = regime_pred.argmax(dim=1).cpu().numpy()
                forecast_pred_class = forecast_pred.argmax(dim=1).cpu().numpy()
                context_pred_value = context_pred.squeeze().cpu().numpy()
                
                # Store
                all_regime_preds.extend(regime_pred_class)
                all_regime_targets.extend(regime_target)
                all_forecast_preds.extend(forecast_pred_class)
                all_forecast_targets.extend(forecast_target)
                all_context_preds.extend(context_pred_value)
                all_context_targets.extend(context_target)
        
        # Convert to numpy arrays
        regime_preds = np.array(all_regime_preds)
        regime_targets = np.array(all_regime_targets)
        forecast_preds = np.array(all_forecast_preds)
        forecast_targets = np.array(all_forecast_targets)
        context_preds = np.array(all_context_preds)
        context_targets = np.array(all_context_targets)
        
        # Compute metrics
        metrics = {}
        
        # Regime classification metrics
        regime_acc = accuracy_score(regime_targets, regime_preds)
        regime_precision, regime_recall, regime_f1, _ = precision_recall_fscore_support(
            regime_targets, regime_preds, average=None, zero_division=0
        )
        regime_precision_macro, regime_recall_macro, regime_f1_macro, _ = precision_recall_fscore_support(
            regime_targets, regime_preds, average='macro', zero_division=0
        )
        
        metrics['regime'] = {
            'accuracy': float(regime_acc),
            'precision_macro': float(regime_precision_macro),
            'recall_macro': float(regime_recall_macro),
            'f1_macro': float(regime_f1_macro),
            'precision_per_class': regime_precision.tolist(),
            'recall_per_class': regime_recall.tolist(),
            'f1_per_class': regime_f1.tolist(),
            'confusion_matrix': confusion_matrix(regime_targets, regime_preds).tolist()
        }
        
        # Forecast direction metrics
        forecast_acc = accuracy_score(forecast_targets, forecast_preds)
        forecast_precision, forecast_recall, forecast_f1, _ = precision_recall_fscore_support(
            forecast_targets, forecast_preds, average=None, zero_division=0
        )
        
        metrics['forecast'] = {
            'accuracy': float(forecast_acc),
            'precision': forecast_precision.tolist(),
            'recall': forecast_recall.tolist(),
            'f1': forecast_f1.tolist(),
            'confusion_matrix': confusion_matrix(forecast_targets, forecast_preds).tolist()
        }
        
        # Context strength metrics (MSE, MAE, R2)
        context_mse = np.mean((context_preds - context_targets) ** 2)
        context_mae = np.mean(np.abs(context_preds - context_targets))
        context_r2 = 1 - (np.sum((context_targets - context_preds) ** 2) /
                         (np.sum((context_targets - np.mean(context_targets)) ** 2) + 1e-8))
        
        metrics['context'] = {
            'mse': float(context_mse),
            'mae': float(context_mae),
            'r2': float(context_r2),
            'mean_predicted': float(np.mean(context_preds)),
            'mean_actual': float(np.mean(context_targets))
        }
        
        logger.info(f"{split_name} Results:")
        logger.info(f"  Regime Accuracy: {regime_acc:.4f}")
        logger.info(f"  Regime F1 (macro): {regime_f1_macro:.4f}")
        logger.info(f"  Forecast Accuracy: {forecast_acc:.4f}")
        logger.info(f"  Context MSE: {context_mse:.4f}, R2: {context_r2:.4f}")
        
        return metrics, {
            'regime_preds': regime_preds,
            'regime_targets': regime_targets,
            'forecast_preds': forecast_preds,
            'forecast_targets': forecast_targets,
            'context_preds': context_preds,
            'context_targets': context_targets
        }
    
    def generate_predictions_csv(self, data_loader, output_path: str, df_source: pd.DataFrame):
        """
        Generate CSV file with predictions for all samples.
        
        Args:
            data_loader: DataLoader for evaluation
            output_path: Path to save predictions CSV
            df_source: Source DataFrame with timestamps
        """
        logger.info("Generating predictions CSV...")
        
        predictions = []
        
        with torch.no_grad():
            idx = 0
            for batch in tqdm(data_loader):
                features = batch['features'].to(self.device)
                
                # Forward pass
                regime_pred, context_pred, forecast_pred = self.model(features)
                
                # Get probabilities and predictions
                regime_probs = torch.softmax(regime_pred, dim=1).cpu().numpy()
                regime_pred_class = regime_pred.argmax(dim=1).cpu().numpy()
                forecast_probs = torch.softmax(forecast_pred, dim=1).cpu().numpy()
                forecast_pred_class = forecast_pred.argmax(dim=1).cpu().numpy()
                context_value = context_pred.squeeze().cpu().numpy()
                
                batch_size = len(regime_pred_class)
                for i in range(batch_size):
                    seq_start_idx = idx + i
                    if seq_start_idx < len(df_source):
                        pred_dict = {
                            'timestamp': df_source.index[seq_start_idx + self.config['model']['seq_len'] - 1],
                            'regime_pred': int(regime_pred_class[i]),
                            'regime_prob_bull_strong': float(regime_probs[i][0]),
                            'regime_prob_bull_weak': float(regime_probs[i][1]),
                            'regime_prob_bear_strong': float(regime_probs[i][2]),
                            'regime_prob_bear_weak': float(regime_probs[i][3]),
                            'regime_prob_range': float(regime_probs[i][4]),
                            'regime_prob_volatile': float(regime_probs[i][5]),
                            'forecast_pred': int(forecast_pred_class[i]),
                            'forecast_prob_down': float(forecast_probs[i][0]),
                            'forecast_prob_up': float(forecast_probs[i][1]),
                            'context_strength': float(context_value[i])
                        }
                        predictions.append(pred_dict)
                
                idx += batch_size
        
        # Create DataFrame and save
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(output_path, index=False)
        logger.info(f"Saved predictions to {output_path}")
        
        return pred_df
    
    def evaluate_all_splits(self, output_dir: str):
        """
        Evaluate on train, validation, and test sets.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        processed_path = self.config['data']['processed_path']
        df = pd.read_csv(processed_path, index_col=0, parse_dates=True)
        
        exclude_cols = ['regime', 'regime_name', 'target', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Create dataloaders
        train_config = self.config['training']
        train_loader, val_loader, test_loader = create_dataloaders(
            df=df,
            feature_cols=feature_cols,
            train_ratio=train_config['train_ratio'],
            val_ratio=train_config['val_ratio'],
            seq_len=self.config['model']['seq_len'],
            batch_size=train_config['batch_size'],
            num_workers=train_config.get('num_workers', 4)
        )
        
        # Evaluate each split
        all_metrics = {}
        
        for split_name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
            metrics, predictions = self.evaluate(loader, split_name)
            all_metrics[split_name] = metrics
        
        # Save metrics
        metrics_path = output_dir / 'metrics_report.json'
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")
        
        # Generate predictions CSV for test set
        test_df = df.iloc[int(len(df) * (train_config['train_ratio'] + train_config['val_ratio'])):].copy()
        pred_df = self.generate_predictions_csv(test_loader, output_dir / 'natron_predictions.csv', test_df)
        
        return all_metrics, pred_df


if __name__ == "__main__":
    import sys
    
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/natron_v6.pt"
    config_path = sys.argv[2] if len(sys.argv) > 2 else "configs/config_train.yaml"
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "logs/evaluation"
    
    evaluator = ModelEvaluator(model_path, config_path)
    metrics, pred_df = evaluator.evaluate_all_splits(output_dir)
