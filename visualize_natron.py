"""
Visualization helpers for Natron predictions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REGIME_ORDER = ["BULL_STRONG", "BULL_WEAK", "RANGE", "BEAR_WEAK", "BEAR_STRONG", "VOLATILE"]


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"])
    required = {"time", "buy_prob", "sell_prob", "direction_up", "regime", "confidence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in predictions file: {missing}")
    df.sort_values("time", inplace=True)
    df["regime"] = pd.Categorical(df["regime"], categories=REGIME_ORDER, ordered=True)
    return df


def plot_signals(df: pd.DataFrame, output: Path | None = None) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(df["time"], df["buy_prob"], label="Buy", color="green")
    axes[0].plot(df["time"], df["sell_prob"], label="Sell", color="red")
    axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Probability")
    axes[0].legend(loc="upper left")
    axes[0].set_title("Natron Buy/Sell Probabilities")

    axes[1].plot(df["time"], df["direction_up"], label="Direction Up", color="blue")
    axes[1].fill_between(df["time"], 0, df["direction_up"], alpha=0.2, color="blue")
    axes[1].set_ylabel("Probability")
    axes[1].set_title("Directional Probability (Up)")

    regime_pivot = pd.crosstab(df["time"], df["regime"])
    sns.heatmap(regime_pivot.T, ax=axes[2], cbar=True, cmap="viridis")
    axes[2].set_ylabel("Regime")
    axes[2].set_title("Regime Heatmap")

    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=200)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Natron prediction visualiser")
    parser.add_argument("--predictions", type=Path, required=True, help="CSV file with prediction history")
    parser.add_argument("--output", type=Path, help="Optional output PNG path")
    args = parser.parse_args()

    df = load_predictions(args.predictions)
    plot_signals(df, args.output)


if __name__ == "__main__":
    main()
