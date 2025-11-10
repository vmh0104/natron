"""
Real-time monitoring utility for the Natron inference stack.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Optional

import psutil
import requests
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor Natron inference server.")
    parser.add_argument("--host", type=str, default="http://127.0.0.1", help="API host (with protocol)")
    parser.add_argument("--port", type=int, default=8081, help="API port")
    parser.add_argument("--interval", type=float, default=5.0, help="Polling interval in seconds")
    return parser.parse_args()


def gpu_metrics() -> Optional[dict]:
    if not torch.cuda.is_available():
        return None
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    total_mem = props.total_memory / (1024 ** 3)
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    utilization = None
    if hasattr(torch.cuda, "utilization"):
        utilization = torch.cuda.utilization(device)
    return {
        "name": torch.cuda.get_device_name(device),
        "memory_total_gb": total_mem,
        "memory_allocated_gb": allocated,
        "memory_reserved_gb": reserved,
        "utilization_pct": utilization,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    url = f"{args.host}:{args.port}/health"
    logging.info("Starting Natron monitor for %s", url)

    try:
        while True:
            try:
                start = time.perf_counter()
                resp = requests.get(url, timeout=3.0)
                latency_ms = (time.perf_counter() - start) * 1000
                if resp.ok:
                    payload = resp.json()
                    gpu_info = gpu_metrics()
                    cpu_percent = psutil.cpu_percent(interval=None)
                    mem = psutil.virtual_memory()
                    logging.info(
                        "STATUS OK | Model=%s | Latency=%.2f ms | CPU=%.1f%% | RAM=%.1f%%%s",
                        payload.get("model", "unknown"),
                        latency_ms,
                        cpu_percent,
                        mem.percent,
                        (
                            " | GPU={name} {util}% {alloc:.2f}/{total:.2f} GB".format(
                                name=gpu_info["name"],
                                util=f"{gpu_info['utilization_pct']:.1f}" if gpu_info["utilization_pct"] is not None else "NA",
                                alloc=gpu_info["memory_allocated_gb"],
                                total=gpu_info["memory_total_gb"],
                            )
                            if gpu_info
                            else ""
                        ),
                    )
                else:
                    logging.warning("Health check failed (%s): %s", resp.status_code, resp.text)
            except Exception as exc:
                logging.exception("Monitoring error: %s", exc)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        logging.info("Monitor stopped by user.")


if __name__ == "__main__":
    main()
