"""
Natron inference server and TCP bridge for MetaTrader 5.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, request

from natron.dataset_loader import NatronDatasetConfig, NatronFeatureEngineer
from natron.model_natron import NatronInferenceWrapper, NatronModelConfig, NatronTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Natron inference server")
    parser.add_argument("--config", type=str, required=True, help="Path to natron_config.yaml")
    parser.add_argument("--model", type=str, default=None, help="Override model path")
    parser.add_argument("--host", type=str, default=None, help="Override API host")
    parser.add_argument("--port", type=int, default=None, help="Override API port")
    parser.add_argument("--socket-port", type=int, default=None, help="Override TCP socket port")
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def load_model(checkpoint_path: Path, device: torch.device) -> NatronInferenceWrapper:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = NatronModelConfig(**checkpoint["model_config"])
    model = NatronTransformer(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    feature_mean = torch.tensor(checkpoint["feature_mean"], dtype=torch.float32)
    feature_std = torch.tensor(checkpoint["feature_std"], dtype=torch.float32)
    wrapper = NatronInferenceWrapper(model, feature_mean, feature_std, device)
    return wrapper


def prepare_feature_engineer(sequence_length: int) -> NatronFeatureEngineer:
    config = NatronDatasetConfig(
        sequence_length=sequence_length,
        cache_features=False,
        feature_cache_path=None,
        normalization=False,
        min_samples=max(sequence_length + 16, 120),
    )
    return NatronFeatureEngineer(config, use_cache=False)


def features_from_payload(
    payload: Dict[str, Any],
    feature_engineer: NatronFeatureEngineer,
    sequence_length: int,
) -> torch.Tensor:
    if "features" in payload:
        features = np.array(payload["features"], dtype=np.float32)
        if features.shape != (sequence_length, feature_engineer.config.feature_count):
            raise ValueError(f"Expected features shape ({sequence_length}, {feature_engineer.config.feature_count})")
        return torch.from_numpy(features)

    if "ohlcv" in payload:
        df = pd.DataFrame(payload["ohlcv"])
        required_cols = {"time", "open", "high", "low", "close", "volume"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"OHLCV payload must include columns: {required_cols}")
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time")
        features_df = feature_engineer.fit_transform(df)
        if len(features_df) < sequence_length:
            raise ValueError("Not enough OHLCV candles to compute features.")
        features_array = features_df.tail(sequence_length).to_numpy(dtype=np.float32)
        return torch.from_numpy(features_array)

    raise ValueError("Payload must contain either 'features' or 'ohlcv'.")


class NatronServer:
    def __init__(
        self,
        config: Dict[str, Any],
        inference_wrapper: NatronInferenceWrapper,
        feature_engineer: NatronFeatureEngineer,
        api_host: str,
        api_port: int,
        socket_port: int,
    ) -> None:
        self.config = config
        self.inference = inference_wrapper
        self.feature_engineer = feature_engineer
        self.sequence_length = config["data"]["sequence_length"]
        self.api_host = api_host
        self.api_port = api_port
        self.socket_port = socket_port
        self._lock = threading.Lock()
        self._app = Flask("natron")
        self._setup_routes()
        self._tcp_server: Optional[asyncio.AbstractServer] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _setup_routes(self) -> None:
        @self._app.route("/health", methods=["GET"])
        def health() -> Any:
            return jsonify({"status": "ok", "model": self.config["experiment"]["name"]})

        @self._app.route("/metadata", methods=["GET"])
        def metadata() -> Any:
            return jsonify(
                {
                    "model": self.config["experiment"]["name"],
                    "sequence_length": self.sequence_length,
                    "feature_dim": self.feature_engineer.config.feature_count,
                    "tasks": ["buy", "sell", "direction", "regime"],
                }
            )

        @self._app.route("/predict", methods=["POST"])
        def predict() -> Any:
            start_time = time.perf_counter()
            payload = request.get_json(force=True)
            try:
                features = features_from_payload(payload, self.feature_engineer, self.sequence_length)
                with self._lock:
                    result = self.inference(features)
            except Exception as exc:
                logging.exception("Prediction error")
                return jsonify({"error": str(exc)}), 400
            latency_ms = (time.perf_counter() - start_time) * 1000
            result["latency_ms"] = latency_ms
            return jsonify(result)

    def run_flask(self) -> None:
        logging.info("Starting Flask API on %s:%d", self.api_host, self.api_port)
        self._app.run(host=self.api_host, port=self.api_port, debug=False, use_reloader=False)

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        addr = writer.get_extra_info("peername")
        logging.info("Socket connection from %s", addr)
        try:
            while True:
                data = await reader.readline()
                if not data:
                    break
                message = data.decode().strip()
                if not message:
                    continue
                try:
                    payload = json.loads(message)
                    features = features_from_payload(payload, self.feature_engineer, self.sequence_length)
                    with self._lock:
                        result = self.inference(features)
                    response = json.dumps(result) + "\n"
                    writer.write(response.encode())
                    await writer.drain()
                except Exception as exc:
                    logging.exception("Socket handling error")
                    error_msg = json.dumps({"error": str(exc)}) + "\n"
                    writer.write(error_msg.encode())
                    await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            logging.info("Socket connection closed: %s", addr)

    def run_tcp_server(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        server_coro = asyncio.start_server(self._handle_client, host=self.api_host, port=self.socket_port)
        self._tcp_server = self._loop.run_until_complete(server_coro)
        logging.info("TCP socket server listening on %s:%d", self.api_host, self.socket_port)
        try:
            self._loop.run_forever()
        finally:
            self._tcp_server.close()
            self._loop.run_until_complete(self._tcp_server.wait_closed())
            self._loop.close()

    def start(self) -> None:
        tcp_thread = threading.Thread(target=self.run_tcp_server, daemon=True)
        tcp_thread.start()

        def shutdown_handler(signum, frame) -> None:  # type: ignore[override]
            logging.info("Received signal %s; shutting down.", signum)
            if self._loop:
                self._loop.call_soon_threadsafe(self._loop.stop)

        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)
        self.run_flask()


def main() -> None:
    args = parse_args()
    config = load_yaml(Path(args.config))
    api_host = args.host or config["deployment"]["api_host"]
    api_port = args.port or config["deployment"]["api_port"]
    socket_port = args.socket_port or config["deployment"]["socket_port"]

    model_path = Path(args.model or config["deployment"]["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    log_level = config["deployment"].get("log_level", "INFO")
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))

    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    inference_wrapper = load_model(model_path, device)
    feature_engineer = prepare_feature_engineer(config["data"]["sequence_length"])

    server = NatronServer(config, inference_wrapper, feature_engineer, api_host, api_port, socket_port)
    server.start()


if __name__ == "__main__":
    main()
