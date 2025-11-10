"""
Natron monitoring utility for latency and health checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import socket
import time
from pathlib import Path
from typing import Dict

import requests


def load_yaml(path: Path) -> Dict:
    import yaml

    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def check_http_health(host: str, port: int, timeout: float = 2.0) -> Dict:
    url = f"http://{host}:{port}/health"
    start = time.time()
    resp = requests.get(url, timeout=timeout)
    latency_ms = (time.time() - start) * 1000.0
    resp.raise_for_status()
    payload = resp.json()
    return {"latency_ms": latency_ms, "payload": payload}


def check_tcp_ping(host: str, port: int, timeout: float = 2.0) -> float:
    sample = json.dumps({"ping": True}).encode("utf-8")
    start = time.time()
    with socket.create_connection((host, port), timeout=timeout) as conn:
        conn.sendall(sample)
        conn.recv(4096)
    latency_ms = (time.time() - start) * 1000.0
    return latency_ms


def monitor_loop(config_path: Path, interval: float, log_path: Path, once: bool = False) -> None:
    config = load_yaml(config_path)
    inference_cfg = config.get("inference", {})
    http_host = inference_cfg.get("flask_host", "127.0.0.1")
    http_port = inference_cfg.get("flask_port", 8080)
    tcp_host = inference_cfg.get("server_host", "127.0.0.1")
    tcp_port = inference_cfg.get("server_port", 7600)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", newline="") as fp:
        writer = csv.writer(fp)
        if fp.tell() == 0:
            writer.writerow(["timestamp", "http_latency_ms", "tcp_latency_ms", "http_status"])

        while True:
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            try:
                http_result = check_http_health(http_host, http_port)
                http_latency = http_result["latency_ms"]
                http_status = http_result["payload"].get("status", "unknown")
            except Exception as exc:  # pragma: no cover - monitoring fallback
                logging.error("HTTP health check failed: %s", exc)
                http_latency = float("nan")
                http_status = "error"

            try:
                tcp_latency = check_tcp_ping(tcp_host, tcp_port)
            except Exception as exc:  # pragma: no cover
                logging.error("TCP ping failed: %s", exc)
                tcp_latency = float("nan")

            logging.info(
                "timestamp=%s http_latency=%.2fms tcp_latency=%.2fms status=%s",
                timestamp,
                http_latency,
                tcp_latency,
                http_status,
            )
            writer.writerow([timestamp, f"{http_latency:.2f}", f"{tcp_latency:.2f}", http_status])
            fp.flush()

            if once:
                break
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Natron monitoring loop")
    parser.add_argument("--config", type=str, default="configs/natron_config.yaml", help="Config file")
    parser.add_argument("--interval", type=float, default=30.0, help="Seconds between checks")
    parser.add_argument("--log-path", type=str, default="logs/monitor.csv", help="CSV output")
    parser.add_argument("--once", action="store_true", help="Run a single probe then exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    monitor_loop(Path(args.config), interval=args.interval, log_path=Path(args.log_path), once=args.once)


if __name__ == "__main__":
    main()
