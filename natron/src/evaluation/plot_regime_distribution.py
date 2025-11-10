"""
Visualization Script for Regime Distribution
Plots regime distributions over time.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_regime_distribution(
    predictions_file: str,
    output_dir: str = 'logs',
    time_col: Optional[str] = None
):
    """
    Plot regime distribution over time.
    
    Args:
        predictions_file: Path to predictions CSV
        output_dir: Output directory for plots
        time_col: Name of time column (if available)
    """
    logger.info(f"Loading predictions from {predictions_file}")
    df = pd.read_csv(predictions_file)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("darkgrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # 1. Regime distribution pie chart
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Predicted regime distribution
    regime_counts = df['regime_name'].value_counts()
    axes[0, 0].pie(regime_counts.values, labels=regime_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Predicted Regime Distribution', fontsize=14, fontweight='bold')
    
    # True regime distribution
    regime_true_counts = df['regime_true_name'].value_counts()
    axes[0, 1].pie(regime_true_counts.values, labels=regime_true_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('True Regime Distribution', fontsize=14, fontweight='bold')
    
    # Regime over time (if time column available)
    if time_col and time_col in df.columns:
        df_time = df.copy()
        df_time[time_col] = pd.to_datetime(df_time[time_col])
        df_time = df_time.sort_values(time_col)
        
        # Count regimes per time period (e.g., daily)
        df_time['date'] = df_time[time_col].dt.date
        regime_by_date = df_time.groupby(['date', 'regime_name']).size().unstack(fill_value=0)
        
        regime_by_date.plot(kind='area', stacked=True, ax=axes[1, 0], alpha=0.7)
        axes[1, 0].set_title('Predicted Regime Distribution Over Time', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend(title='Regime', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        # Bar chart comparison
        comparison_df = pd.DataFrame({
            'Predicted': regime_counts,
            'True': regime_true_counts
        }).fillna(0)
        comparison_df.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Regime Distribution Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Regime')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend()
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Confusion matrix for regimes
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(df['regime_true'], df['regime_pred'])
    regime_names = df['regime_name'].unique()
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
                xticklabels=regime_names, yticklabels=regime_names)
    axes[1, 1].set_title('Regime Confusion Matrix', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('True')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'regime_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved regime distribution plot to {output_path}")
    plt.close()
    
    # 2. Forecast accuracy over time
    if time_col and time_col in df.columns:
        fig, ax = plt.subplots(figsize=(15, 6))
        
        df_time = df.copy()
        df_time[time_col] = pd.to_datetime(df_time[time_col])
        df_time = df_time.sort_values(time_col)
        df_time['date'] = df_time[time_col].dt.date
        
        # Compute rolling accuracy
        df_time['forecast_correct'] = (df_time['forecast_pred'] == df_time['forecast_true']).astype(int)
        rolling_accuracy = df_time.groupby('date')['forecast_correct'].mean()
        
        ax.plot(rolling_accuracy.index, rolling_accuracy.values, linewidth=2)
        ax.axhline(y=0.5, color='r', linestyle='--', label='Random (50%)')
        ax.set_title('Forecast Accuracy Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_path = Path(output_dir) / 'forecast_accuracy_over_time.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved forecast accuracy plot to {output_path}")
        plt.close()
    
    # 3. Context score distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].hist(df['context_score'], bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_title('Context Score Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Context Score')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(df['context_true'], df['context_score'], alpha=0.5, s=10)
    axes[1].plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
    axes[1].set_title('Context Score: Predicted vs True', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('True Context Score')
    axes[1].set_ylabel('Predicted Context Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'context_score_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved context score analysis to {output_path}")
    plt.close()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot Regime Distribution')
    parser.add_argument('--predictions', type=str, required=True, help='Path to predictions CSV')
    parser.add_argument('--output', type=str, default='logs', help='Output directory')
    parser.add_argument('--time-col', type=str, default=None, help='Name of time column')
    
    args = parser.parse_args()
    
    plot_regime_distribution(args.predictions, args.output, args.time_col)


if __name__ == '__main__':
    main()
