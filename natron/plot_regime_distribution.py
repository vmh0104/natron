"""
Natron Regime Distribution Visualization
Plots regime distributions over time.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_regime_distribution(
    predictions_path: str = "natron_predictions.csv",
    output_path: str = "regime_distribution.png"
):
    """
    Plot regime distribution over time.
    
    Args:
        predictions_path: Path to predictions CSV
        output_path: Path to save plot
    """
    df = pd.read_csv(predictions_path)
    
    # Regime class names
    regime_classes = [
        'BULL_STRONG', 'BULL_WEAK', 'BEAR_STRONG',
        'BEAR_WEAK', 'RANGE', 'VOLATILE'
    ]
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: Regime distribution over time (stacked area)
    ax1 = axes[0]
    regime_counts = pd.crosstab(
        pd.Series(range(len(df))), 
        df['regime_pred_name'],
        normalize='index'
    )
    
    # Ensure all regime classes are present
    for regime in regime_classes:
        if regime not in regime_counts.columns:
            regime_counts[regime] = 0
    
    regime_counts = regime_counts[regime_classes]
    
    ax1.stackplot(
        regime_counts.index,
        *[regime_counts[regime] for regime in regime_classes],
        labels=regime_classes,
        alpha=0.7
    )
    ax1.set_title('Predicted Regime Distribution Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Proportion')
    ax1.legend(loc='upper right', ncol=3)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Context strength over time
    ax2 = axes[1]
    ax2.plot(df['context_pred'], label='Predicted', alpha=0.7, linewidth=1)
    ax2.plot(df['context_true'], label='True', alpha=0.7, linewidth=1)
    ax2.set_title('Context Strength Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Context Strength')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Forecast accuracy over time (rolling)
    ax3 = axes[2]
    forecast_correct = (df['forecast_pred'] == df['forecast_true']).astype(int)
    rolling_acc = forecast_correct.rolling(window=100, min_periods=1).mean()
    
    ax3.plot(rolling_acc, label='Rolling Accuracy (100 samples)', linewidth=2)
    ax3.axhline(y=0.5, color='r', linestyle='--', label='Random Baseline')
    ax3.set_title('Forecast Accuracy Over Time (Rolling)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved regime distribution plot to {output_path}")
    
    # Print statistics
    print("\n" + "="*50)
    print("REGIME DISTRIBUTION STATISTICS")
    print("="*50)
    print("\nPredicted Regime Distribution:")
    print(df['regime_pred_name'].value_counts().sort_index())
    print("\nTrue Regime Distribution:")
    print(df['regime_true_name'].value_counts().sort_index())
    print("\nForecast Accuracy:")
    print(f"Overall: {(df['forecast_pred'] == df['forecast_true']).mean():.4f}")
    print("="*50)


if __name__ == "__main__":
    import sys
    
    predictions_path = sys.argv[1] if len(sys.argv) > 1 else "natron_predictions.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "regime_distribution.png"
    
    plot_regime_distribution(predictions_path, output_path)
