"""
Monitoring utility for the Natron server and GPU statistics.
"""

from __future__ import annotations

import argparse
import datetime as dt
import time
from typing import Tuple

import psutil
import requests
import torch


def gpu_metrics() -> Tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    device = torch.cuda.current_device()
    total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    free_mem, total_mem_bytes = torch.cuda.mem_get_info(device)
    used = (total_mem_bytes - free_mem) / (1024**3)
    return used, total_mem


def ping_server(url: str, timeout: float) -> bool:
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False


def format_bytes(num: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}PB"


def monitor_loop(host: str, port: int, interval: float, timeout: float) -> None:
    url = f"http://{host}:{port}/health"
    print(f"Monitoring Natron server at {url}")
    try:
        while True:
            timestamp = dt.datetime.utcnow().isoformat()
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory()
            gpu_used, gpu_total = gpu_metrics()
            healthy = ping_server(url, timeout)

            print(
                f"[{timestamp}] health={'OK' if healthy else 'DOWN'} | "
                f"CPU={cpu:.1f}% | RAM={ram.percent:.1f}% ({format_bytes(ram.used)}/{format_bytes(ram.total)}) | "
                f"GPU={gpu_used:.2f}/{gpu_total:.2f} GB"
            )
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Stopping monitor.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor Natron inference server.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=8000, help="Server port.")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between checks.")
    parser.add_argument("--timeout", type=float, default=1.5, help="Health request timeout.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    monitor_loop(args.host, args.port, args.interval, args.timeout)


if __name__ == "__main__":
    main()
