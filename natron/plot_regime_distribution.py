"""
Regime Distribution Visualization for Natron AI Trading System
Plots regime distributions over time.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

from regime_labeler import RegimeLabeler


def plot_regime_distribution_over_time(df: pd.DataFrame, 
                                      labels: pd.Series,
                                      output_path: str = "regime_distribution.png"):
    """
    Plot regime distribution over time.
    
    Args:
        df: DataFrame with time index and price data
        labels: Regime labels series
        output_path: Output file path
    """
    regime_labeler = RegimeLabeler()
    regime_names = regime_labeler.get_regime_names()
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: Price chart with regime colors
    ax1 = axes[0]
    ax1.plot(df.index, df['close'], label='Close Price', linewidth=1, alpha=0.7)
    
    # Color background by regime
    colors = {
        0: '#00FF00',  # BULL_STRONG - bright green
        1: '#90EE90',  # BULL_WEAK - light green
        2: '#FF0000',  # BEAR_STRONG - bright red
        3: '#FFB6C1',  # BEAR_WEAK - light pink
        4: '#FFFF00',  # RANGE - yellow
        5: '#FFA500'   # VOLATILE - orange
    }
    
    prev_regime = None
    start_idx = None
    
    for i, (idx, regime) in enumerate(labels.items()):
        if regime != prev_regime:
            if prev_regime is not None and start_idx is not None:
                ax1.axvspan(df.index[start_idx], df.index[i-1], 
                           alpha=0.2, color=colors.get(prev_regime, 'gray'))
            start_idx = i
            prev_regime = regime
    
    # Last regime
    if start_idx is not None:
        ax1.axvspan(df.index[start_idx], df.index[-1], 
                   alpha=0.2, color=colors.get(prev_regime, 'gray'))
    
    ax1.set_title('Price Chart with Regime Overlay', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Regime distribution over time (stacked area)
    ax2 = axes[1]
    
    # Create regime matrix
    regime_matrix = pd.DataFrame(index=df.index)
    for regime_id, regime_name in regime_names.items():
        regime_matrix[regime_name] = (labels == regime_id).astype(int)
    
    # Compute rolling distribution (30-period window)
    window = min(30, len(regime_matrix) // 10)
    regime_rolling = regime_matrix.rolling(window=window, center=True).mean()
    
    # Plot stacked area
    regime_rolling.plot(kind='area', ax=ax2, alpha=0.6, 
                       color=[colors.get(i, 'gray') for i in range(6)])
    ax2.set_title(f'Regime Distribution (Rolling {window}-period Average)', 
                 fontsize=14, fontweight='bold')
    ax2.set_ylabel('Proportion', fontsize=12)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Regime counts bar chart
    ax3 = axes[2]
    regime_counts = labels.value_counts().sort_index()
    regime_labels = [regime_names.get(i, f'UNKNOWN_{i}') for i in regime_counts.index]
    colors_list = [colors.get(i, 'gray') for i in regime_counts.index]
    
    bars = ax3.bar(regime_labels, regime_counts.values, color=colors_list, alpha=0.7)
    ax3.set_title('Regime Distribution (Total Counts)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_xlabel('Regime Type', fontsize=12)
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    ax3.grid(True, alpha=0.3, axis='y')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved regime distribution plot to {output_path}")
    plt.close()


def plot_regime_transitions(labels: pd.Series, output_path: str = "regime_transitions.png"):
    """
    Plot regime transition matrix.
    
    Args:
        labels: Regime labels series
        output_path: Output file path
    """
    regime_labeler = RegimeLabeler()
    regime_names = regime_labeler.get_regime_names()
    
    # Compute transition matrix
    transitions = []
    for i in range(len(labels) - 1):
        transitions.append((labels.iloc[i], labels.iloc[i+1]))
    
    transition_matrix = pd.crosstab(
        pd.Series([t[0] for t in transitions]),
        pd.Series([t[1] for t in transitions]),
        normalize='index'
    )
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(transition_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
               xticklabels=[regime_names.get(i, f'R{i}') for i in transition_matrix.columns],
               yticklabels=[regime_names.get(i, f'R{i}') for i in transition_matrix.index])
    plt.title('Regime Transition Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Next Regime', fontsize=12)
    plt.ylabel('Current Regime', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved regime transition matrix to {output_path}")
    plt.close()


def main():
    """Main plotting function."""
    parser = argparse.ArgumentParser(description='Plot Regime Distribution')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to processed data CSV')
    parser.add_argument('--output', type=str, default='plots',
                       help='Output directory for plots')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load processed data
    print("Loading processed data...")
    df = pd.read_csv(args.data, index_col=0)
    df.index = pd.to_datetime(df.index)
    
    # Extract regime labels
    labels = df['regime'].astype(int)
    
    # Plot regime distribution over time
    plot_regime_distribution_over_time(
        df, labels,
        output_path=str(output_dir / 'regime_distribution.png')
    )
    
    # Plot regime transitions
    plot_regime_transitions(
        labels,
        output_path=str(output_dir / 'regime_transitions.png')
    )


if __name__ == "__main__":
    main()
