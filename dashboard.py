"""
Terminal dashboard for Natron predictions using Rich.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests
from rich.live import Live
from rich.table import Table


def load_latest_row(path: Path) -> Optional[pd.Series]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df.iloc[-1]


def fetch_from_api(host: str, port: int, candles_path: Path) -> Optional[Dict]:
    df = pd.read_csv(candles_path)
    payload = {"candles": df.to_dict(orient="records")}
    resp = requests.post(f"http://{host}:{port}/predict", json=payload, timeout=5)
    resp.raise_for_status()
    return resp.json()


def render_table(payload: Dict, title: str) -> Table:
    table = Table(title=title)
    table.add_column("Metric", justify="left", style="bold cyan")
    table.add_column("Value", justify="right", style="bold white")

    for key, value in payload.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.3f}")
        elif isinstance(value, list):
            table.add_row(key, ", ".join(f"{v:.2f}" for v in value))
        else:
            table.add_row(key, str(value))
    return table


def main():
    parser = argparse.ArgumentParser(description="Natron live terminal dashboard")
    parser.add_argument("--predictions", type=str, help="CSV file with prediction history")
    parser.add_argument("--candles", type=str, help="CSV file with latest candles for API polling")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Inference host")
    parser.add_argument("--port", type=int, default=8080, help="Inference HTTP port")
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh interval in seconds")
    args = parser.parse_args()

    if not args.predictions and not args.candles:
        parser.error("Provide either --predictions or --candles to stream data.")

    with Live(auto_refresh=False) as live:
        while True:
            payload = {}
            title = "Natron Dashboard"
            if args.predictions:
                row = load_latest_row(Path(args.predictions))
                if row is not None:
                    payload = row.to_dict()
                    title = f"Natron Dashboard (CSV: {args.predictions})"
            if args.candles:
                try:
                    payload = fetch_from_api(args.host, args.port, Path(args.candles))
                    title = f"Natron Dashboard (API: {args.host}:{args.port})"
                except Exception as exc:
                    payload = {"error": str(exc)}

            table = render_table(payload, title)
            live.update(table, refresh=True)
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
