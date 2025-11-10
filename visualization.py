"""
Natron visualization utilities for regime maps and signal heatmaps.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REGIME_COLORS = {
    "BULL_STRONG": "#2ecc71",
    "BULL_WEAK": "#27ae60",
    "RANGE": "#95a5a6",
    "BEAR_WEAK": "#e67e22",
    "BEAR_STRONG": "#e74c3c",
    "VOLATILE": "#9b59b6",
}


def plot_regime_map(df: pd.DataFrame, time_col: str = "time", regime_col: str = "regime") -> None:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    plt.figure(figsize=(12, 2))
    cmap = [REGIME_COLORS.get(reg, "#34495e") for reg in df[regime_col]]
    plt.bar(df[time_col], np.ones(len(df)), color=cmap, width=0.02)
    plt.title("Natron Regime Map")
    plt.yticks([])
    plt.xlabel("Time")
    plt.tight_layout()
    plt.show()


def plot_signal_heatmap(df: pd.DataFrame, columns: List[str]) -> None:
    plt.figure(figsize=(10, len(columns)))
    sns.heatmap(df[columns].T, cmap="mako", cbar=True)
    plt.title("Natron Signal Heatmap")
    plt.xlabel("Time Index")
    plt.tight_layout()
    plt.show()


def load_predictions(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".json", ".jsonl"}:
        return pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    raise ValueError(f"Unsupported file format: {path}")


def main():
    parser = argparse.ArgumentParser(description="Natron visualization utilities")
    parser.add_argument("--predictions", type=str, required=True, help="CSV/JSON predictions file")
    parser.add_argument("--plot", type=str, choices=["regime", "heatmap"], default="regime")
    parser.add_argument("--columns", type=str, default="buy_prob,sell_prob,direction_up")
    args = parser.parse_args()

    df = load_predictions(Path(args.predictions))
    if args.plot == "regime":
        plot_regime_map(df)
    else:
        cols = [c.strip() for c in args.columns.split(",") if c.strip() in df.columns]
        if not cols:
            raise ValueError("No valid columns provided for heatmap")
        plot_signal_heatmap(df, cols)


if __name__ == "__main__":
    main()
