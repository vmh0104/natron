"""
Natron runtime monitor for latency and system health.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import psutil
import requests
import yaml

from utils import create_logger


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def format_bytes(num: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024:
            return f"{num:.1f}{unit}"
        num /= 1024
    return f"{num:.1f}PB"


def gather_system_stats() -> dict:
    vm = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=None)
    return {
        "cpu_percent": cpu,
        "memory_percent": vm.percent,
        "memory_used": format_bytes(vm.used),
        "memory_total": format_bytes(vm.total),
    }


def monitor_loop(config_path: str) -> None:
    config = load_config(config_path)
    logger = create_logger("natron-monitor", config["experiment"]["log_dir"])
    interval = config["monitoring"]["poll_interval_s"]
    host = config["inference"]["host"]
    port = config["inference"]["flask_port"]

    url = f"http://{host}:{port}/health"
    thresholds = config["monitoring"]["alert_thresholds"]

    logger.info(f"Starting Natron monitor. Polling {url} every {interval}s")

    while True:
        start = time.time()
        status = {"healthy": False}
        try:
            response = requests.get(url, timeout=thresholds["latency_ms"] / 1000.0)
            latency_ms = (time.time() - start) * 1000
            status = response.json()
            status["latency_ms"] = latency_ms
            status["healthy"] = response.status_code == 200
        except Exception as exc:
            latency_ms = (time.time() - start) * 1000
            status["error"] = str(exc)
            status["latency_ms"] = latency_ms

        system_stats = gather_system_stats()
        report = {"status": status, "system": system_stats}
        logger.info(json.dumps(report))

        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Natron monitoring agent")
    parser.add_argument("--config", type=str, default="configs/natron_config.yaml")
    args = parser.parse_args()
    monitor_loop(args.config)


if __name__ == "__main__":
    main()
