"""
Visualization Module for Regime Distribution

This module creates plots showing regime distributions over time.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)


def plot_regime_distribution(predictions_path: str, output_path: str):
    """
    Plot regime distribution over time.
    
    Args:
        predictions_path: Path to predictions CSV
        output_path: Path to save plot
    """
    logger.info(f"Loading predictions from {predictions_path}")
    df = pd.read_csv(predictions_path)
    
    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    
    # Regime names
    regime_names = [
        'BULL_STRONG', 'BULL_WEAK', 'BEAR_STRONG',
        'BEAR_WEAK', 'RANGE', 'VOLATILE'
    ]
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Plot 1: Regime probabilities over time
    ax1 = axes[0]
    regime_cols = [f'regime_prob_{name.lower().replace("_", "_")}' for name in regime_names]
    regime_cols = [col for col in regime_cols if col in df.columns]
    
    if regime_cols:
        for col in regime_cols:
            ax1.plot(df.index, df[col], label=col.replace('regime_prob_', ''), alpha=0.7, linewidth=1.5)
        ax1.set_title('Regime Probabilities Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Probability')
        ax1.legend(loc='upper right', ncol=3)
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Predicted regime over time
    ax2 = axes[1]
    if 'regime_pred' in df.columns:
        regime_pred = df['regime_pred']
        ax2.scatter(df.index, regime_pred, c=regime_pred, cmap='tab10', s=10, alpha=0.6)
        ax2.set_title('Predicted Regime Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Regime ID')
        ax2.set_yticks(range(6))
        ax2.set_yticklabels(regime_names)
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Context strength and forecast probabilities
    ax3 = axes[2]
    if 'context_strength' in df.columns:
        ax3_twin = ax3.twinx()
        
        # Context strength
        ax3.plot(df.index, df['context_strength'], label='Context Strength', color='purple', linewidth=2)
        ax3.set_ylabel('Context Strength', color='purple')
        ax3.tick_params(axis='y', labelcolor='purple')
        
        # Forecast probabilities
        if 'forecast_prob_up' in df.columns:
            ax3_twin.plot(df.index, df['forecast_prob_up'], label='Forecast Prob (Up)', 
                         color='green', linewidth=1.5, alpha=0.7)
            ax3_twin.plot(df.index, df['forecast_prob_down'], label='Forecast Prob (Down)', 
                         color='red', linewidth=1.5, alpha=0.7)
            ax3_twin.set_ylabel('Forecast Probability', color='black')
            ax3_twin.tick_params(axis='y', labelcolor='black')
        
        ax3.set_title('Context Strength and Forecast Probabilities', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Time')
        ax3.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {output_path}")
    plt.close()


def plot_regime_statistics(predictions_path: str, output_path: str):
    """
    Plot regime statistics (distribution, transitions, etc.).
    
    Args:
        predictions_path: Path to predictions CSV
        output_path: Path to save plot
    """
    logger.info(f"Loading predictions from {predictions_path}")
    df = pd.read_csv(predictions_path)
    
    regime_names = [
        'BULL_STRONG', 'BULL_WEAK', 'BEAR_STRONG',
        'BEAR_WEAK', 'RANGE', 'VOLATILE'
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Regime distribution (bar chart)
    ax1 = axes[0, 0]
    if 'regime_pred' in df.columns:
        regime_counts = df['regime_pred'].value_counts().sort_index()
        regime_labels = [regime_names[int(i)] for i in regime_counts.index]
        ax1.bar(regime_labels, regime_counts.values, color=sns.color_palette("husl", 6))
        ax1.set_title('Regime Distribution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Context strength distribution
    ax2 = axes[0, 1]
    if 'context_strength' in df.columns:
        ax2.hist(df['context_strength'], bins=50, color='purple', alpha=0.7, edgecolor='black')
        ax2.set_title('Context Strength Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Context Strength')
        ax2.set_ylabel('Frequency')
        ax2.axvline(df['context_strength'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["context_strength"].mean():.3f}')
        ax2.legend()
    
    # Plot 3: Forecast probability distribution
    ax3 = axes[1, 0]
    if 'forecast_prob_up' in df.columns:
        ax3.hist(df['forecast_prob_up'], bins=50, color='green', alpha=0.7, label='Up', edgecolor='black')
        ax3.hist(df['forecast_prob_down'], bins=50, color='red', alpha=0.7, label='Down', edgecolor='black')
        ax3.set_title('Forecast Probability Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Probability')
        ax3.set_ylabel('Frequency')
        ax3.legend()
    
    # Plot 4: Regime transition matrix (heatmap)
    ax4 = axes[1, 1]
    if 'regime_pred' in df.columns:
        transitions = []
        for i in range(len(df) - 1):
            transitions.append((df['regime_pred'].iloc[i], df['regime_pred'].iloc[i+1]))
        
        transition_df = pd.DataFrame(transitions, columns=['From', 'To'])
        transition_matrix = pd.crosstab(transition_df['From'], transition_df['To'], normalize='index')
        
        sns.heatmap(transition_matrix, annot=True, fmt='.2f', cmap='YlOrRd', 
                   ax=ax4, cbar_kws={'label': 'Transition Probability'})
        ax4.set_title('Regime Transition Matrix', fontsize=14, fontweight='bold')
        ax4.set_xlabel('To Regime')
        ax4.set_ylabel('From Regime')
        ax4.set_xticklabels([regime_names[int(i)] for i in transition_matrix.columns], rotation=45, ha='right')
        ax4.set_yticklabels([regime_names[int(i)] for i in transition_matrix.index], rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved statistics plot to {output_path}")
    plt.close()


if __name__ == "__main__":
    import sys
    
    predictions_path = sys.argv[1] if len(sys.argv) > 1 else "logs/evaluation/natron_predictions.csv"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "logs/evaluation"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_regime_distribution(predictions_path, output_dir / "regime_distribution.png")
    plot_regime_statistics(predictions_path, output_dir / "regime_statistics.png")
