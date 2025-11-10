"""
Natron inference server combining REST API and TCP socket bridge for MetaTrader 5.
"""

from __future__ import annotations

import argparse
import json
import logging
import socketserver
import threading
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

from feature_engineering import FeatureConfig, FeatureEngineer
from model_natron import ModelConfig, NatronTransformer

LOGGER = logging.getLogger("natron.server")


def load_yaml(path: Path) -> Dict:
    import yaml

    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def filter_kwargs(target_cls, cfg_dict: Dict) -> Dict:
    valid_keys = target_cls.__annotations__.keys()
    return {k: v for k, v in cfg_dict.items() if k in valid_keys}


class NatronInferenceEngine:
    """Thread-safe inference engine wrapping feature engineering and model inference."""

    def __init__(self, config_path: Path):
        self.config = load_yaml(config_path)
        project_cfg = self.config.get("project", {})
        inference_cfg = self.config.get("inference", {})
        model_cfg = self.config.get("model", {})
        features_cfg = self.config.get("features", {})
        data_cfg = self.config.get("data", {})

        self.device = torch.device(project_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.sequence_length = data_cfg.get("sequence_length", 96)
        self.lock = threading.Lock()

        self.feature_engineer = FeatureEngineer(FeatureConfig(**filter_kwargs(FeatureConfig, features_cfg)))

        checkpoint_dir = Path(project_cfg.get("checkpoint_dir", "models"))
        model_path = Path(inference_cfg.get("model_path", checkpoint_dir / "natron_v2.pt"))
        scaler_path = Path(inference_cfg.get("scaler_path", checkpoint_dir / "scaler.pkl"))
        feature_columns_path = checkpoint_dir / "feature_columns.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler artifact not found: {scaler_path}")
        if not feature_columns_path.exists():
            raise FileNotFoundError(f"Feature columns metadata not found: {feature_columns_path}")

        self.scaler = joblib.load(scaler_path)
        with open(feature_columns_path, "r", encoding="utf-8") as fp:
            metadata = json.load(fp)
        self.feature_columns: List[str] = metadata["feature_columns"]
        model_cfg["input_dim"] = len(self.feature_columns)
        self.model = NatronTransformer(ModelConfig(**filter_kwargs(ModelConfig, model_cfg))).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"], strict=False)
        self.model.eval()

        self.regime_labels = inference_cfg.get(
            "regime_labels",
            ["BULL_STRONG", "BULL_WEAK", "RANGE", "BEAR_WEAK", "BEAR_STRONG", "VOLATILE"],
        )
        LOGGER.info("NatronInferenceEngine initialized (model: %s)", model_path)

    def predict(self, candles: pd.DataFrame) -> Dict:
        if candles.shape[0] < self.sequence_length:
            raise ValueError(f"Require at least {self.sequence_length} candles, received {candles.shape[0]}")

        with self.lock, torch.no_grad():
            features_df = self.feature_engineer.transform(candles)
            features_df = features_df[self.feature_columns]
            feature_arr = self.scaler.transform(features_df.values)

            seq = feature_arr[-self.sequence_length :]
            if seq.shape[0] < self.sequence_length:
                pad = np.repeat(seq[:1], self.sequence_length - seq.shape[0], axis=0)
                seq = np.concatenate([pad, seq], axis=0)

            tensor = torch.from_numpy(seq).unsqueeze(0).to(self.device).float()
            outputs = self.model(tensor)

            buy_prob = torch.sigmoid(outputs["buy_logits"]).item()
            sell_prob = torch.sigmoid(outputs["sell_logits"]).item()
            direction_probs = torch.softmax(outputs["direction_logits"], dim=-1).squeeze().cpu().numpy()
            regime_probs = torch.softmax(outputs["regime_logits"], dim=-1).squeeze().cpu().numpy()

            regime_idx = int(np.argmax(regime_probs))
            confidence = float(np.mean([direction_probs.max(), regime_probs.max(), buy_prob, 1 - sell_prob]))

            return {
                "buy_prob": float(buy_prob),
                "sell_prob": float(sell_prob),
                "direction_up": float(direction_probs[1]),
                "direction_down": float(direction_probs[0]),
                "regime": self.regime_labels[regime_idx] if regime_idx < len(self.regime_labels) else regime_idx,
                "regime_probs": regime_probs.tolist(),
                "confidence": confidence,
            }


class NatronTCPHandler(socketserver.BaseRequestHandler):
    engine: NatronInferenceEngine = None  # type: ignore[assignment]

    def handle(self):
        data = self.request.recv(1_048_576).decode("utf-8").strip()
        if not data:
            return
        try:
            payload = json.loads(data)
            candles = pd.DataFrame(payload["candles"])
            result = self.engine.predict(candles)
            self.request.sendall((json.dumps(result) + "\n").encode("utf-8"))
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("TCP handler error: %s", exc)
            error_payload = json.dumps({"error": str(exc)})
            self.request.sendall((error_payload + "\n").encode("utf-8"))


def start_tcp_server(host: str, port: int, engine: NatronInferenceEngine) -> socketserver.ThreadingTCPServer:
    NatronTCPHandler.engine = engine
    server = socketserver.ThreadingTCPServer((host, port), NatronTCPHandler)
    server.daemon_threads = True

    thread = threading.Thread(target=server.serve_forever, name="NatronTCPServer", daemon=True)
    thread.start()
    LOGGER.info("TCP server listening on %s:%d", host, port)
    return server


def create_app(engine: NatronInferenceEngine) -> Flask:
    app = Flask(__name__)
    CORS(app)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/predict", methods=["POST"])
    def predict():
        payload = request.get_json(force=True)
        candles = pd.DataFrame(payload["candles"])
        result = engine.predict(candles)
        return jsonify(result)

    return app


def main():
    parser = argparse.ArgumentParser(description="Natron inference server")
    parser.add_argument("--config", type=str, default="configs/natron_config.yaml", help="Path to YAML config")
    parser.add_argument("--no-tcp", action="store_true", help="Disable TCP bridge")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    engine = NatronInferenceEngine(Path(args.config))
    inference_cfg = engine.config.get("inference", {})

    if not args.no_tcp:
        start_tcp_server(
            inference_cfg.get("server_host", "0.0.0.0"),
            inference_cfg.get("server_port", 7600),
            engine,
        )

    app = create_app(engine)
    app.run(
        host=inference_cfg.get("flask_host", "0.0.0.0"),
        port=inference_cfg.get("flask_port", 8080),
        debug=False,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
