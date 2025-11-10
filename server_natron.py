"""
Flask inference server for the Natron Transformer model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import json
import socketserver
import threading
import joblib
import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, request

from dataset_loader import FeatureEngineer, FeatureSpec, load_config
from model_natron import init_model_from_config


LOGGER = logging.getLogger("natron.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


class NatronService:
    REGIME_MAP = {
        0: "BULL_STRONG",
        1: "BULL_WEAK",
        2: "RANGE",
        3: "BEAR_WEAK",
        4: "BEAR_STRONG",
        5: "VOLATILE",
    }

    def __init__(self, config_path: Path):
        self.config = load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config.get("server", {}).get("use_gpu", True) else "cpu")
        LOGGER.info("Natron service running on device %s", self.device)

        self.model = init_model_from_config(self.config).to(self.device)
        self.model.eval()

        model_dir = Path(self.config.get("paths", {}).get("model_dir", "model"))
        checkpoint_path = model_dir / "natron_v2_best.pt"
        if not checkpoint_path.exists():
            checkpoint_path = model_dir / "natron_v2_final.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found in {model_dir}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        LOGGER.info("Loaded model weights from %s", checkpoint_path)

        scaler_path = model_dir / "feature_scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Feature scaler not found at {scaler_path}")
        self.scaler = joblib.load(scaler_path)

        seq_len = self.config.get("data", {}).get("sequence_length", 96)
        self.feature_engineer = FeatureEngineer(
            FeatureSpec(sequence_length=seq_len, cache_path=None, scaler_type=self.config.get("data", {}).get("scaler", "robust"))
        )
        self.sequence_length = seq_len

    def preprocess(self, candles: List[Dict]) -> torch.Tensor:
        df = pd.DataFrame(candles)
        if len(df) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} candles, received {len(df)}.")
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.sort_values("time")
        features = self.feature_engineer._build_features(df)  # pylint: disable=protected-access
        scaled = self.scaler.transform(features)
        seq = torch.tensor(scaled[-self.sequence_length :], dtype=torch.float32, device=self.device).unsqueeze(0)
        return seq

    def postprocess(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        buy_prob = torch.sigmoid(outputs["buy_logits"]).item()
        sell_prob = torch.sigmoid(outputs["sell_logits"]).item()
        direction_probs = torch.softmax(outputs["direction_logits"], dim=-1).squeeze(0)
        regime_probs = torch.softmax(outputs["regime_logits"], dim=-1).squeeze(0)

        direction_up = direction_probs[1].item()
        regime_idx = int(torch.argmax(regime_probs).item())
        regime_confidence = regime_probs[regime_idx].item()

        confidence = np.mean(
            [
                max(buy_prob, sell_prob),
                direction_probs.max().item(),
                regime_confidence,
            ]
        )

        return {
            "buy_prob": float(buy_prob),
            "sell_prob": float(sell_prob),
            "direction_up": float(direction_up),
            "regime": self.REGIME_MAP.get(regime_idx, "UNKNOWN"),
            "confidence": float(confidence),
        }

    @torch.no_grad()
    def predict_from_candles(self, candles: List[Dict]) -> Dict[str, float]:
        sequence = self.preprocess(candles)
        outputs = self.model(sequence)
        return self.postprocess(outputs)


def create_app(config_path: Path = Path("configs/natron_config.yaml"), service: NatronService | None = None) -> Flask:
    service = service or NatronService(config_path)
    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health() -> tuple:
        return jsonify({"status": "ok"}), 200

    @app.route("/predict", methods=["POST"])
    def predict() -> tuple:
        try:
            payload = request.get_json(force=True)
            candles = payload.get("candles")
            if not candles:
                return jsonify({"error": "Missing 'candles' in payload."}), 400
            predictions = service.predict_from_candles(candles)
            return jsonify(predictions), 200
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("Prediction error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    return app


def start_socket_server(service: NatronService, host: str, port: int) -> socketserver.ThreadingTCPServer:
    LOGGER.info("Starting TCP bridge on %s:%d", host, port)

    class NatronTCPHandler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            buffer = b""
            try:
                while True:
                    chunk = self.request.recv(4096)
                    if not chunk:
                        break
                    buffer += chunk
                    if buffer.endswith(b"\n"):
                        break
                if not buffer:
                    return
                payload = json.loads(buffer.decode("utf-8").strip())
                candles = payload.get("candles")
                if not candles:
                    response = {"error": "Missing candles"}
                else:
                    response = service.predict_from_candles(candles)
                self.request.sendall((json.dumps(response) + "\n").encode("utf-8"))
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.exception("Socket handler error: %s", exc)
                self.request.sendall((json.dumps({"error": str(exc)}) + "\n").encode("utf-8"))

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    server = socketserver.ThreadingTCPServer((host, port), NatronTCPHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


if __name__ == "__main__":
    import os

    cfg_env = os.environ.get("NATRON_CONFIG", "configs/natron_config.yaml")
    cfg_path = Path(cfg_env)
    service = NatronService(cfg_path)
    app = create_app(cfg_path, service=service)
    server_cfg = load_config(cfg_path).get("server", {})
    socket_cfg = server_cfg.get("socket", {})
    tcp_server = None
    if socket_cfg.get("enabled", False):
        tcp_server = start_socket_server(
            service,
            socket_cfg.get("host", server_cfg.get("host", "0.0.0.0")),
            socket_cfg.get("port", 9000),
        )
    app.run(
        host=server_cfg.get("host", "0.0.0.0"),
        port=server_cfg.get("port", 8000),
        debug=server_cfg.get("reload", False),
    )
