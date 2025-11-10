"""
Natron inference and socket bridge server.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from flask import Flask, jsonify, request
from sklearn.preprocessing import RobustScaler, StandardScaler

from feature_engine import FeatureEngineer, FeatureEngineerConfig
from model_natron import NatronTransformer, NatronTransformerConfig

REGIME_MAP = {
    0: "BULL_STRONG",
    1: "BULL_WEAK",
    2: "RANGE",
    3: "BEAR_WEAK",
    4: "BEAR_STRONG",
    5: "VOLATILE",
}


@dataclass
class InferenceState:
    config: Dict[str, Any]
    metadata: Dict[str, Any]
    device: torch.device
    model: NatronTransformer
    scaler: Any
    feature_engineer: FeatureEngineer
    sequence_length: int
    feature_names: List[str]


class NatronInferenceEngine:
    def __init__(self, config_path: str, metadata_path: str, model_path: str, device: str | None = None) -> None:
        self.config = self._load_yaml(config_path)
        self.metadata = self._load_json(metadata_path)
        self.device = torch.device(device or self.config["inference"]["device"])
        self.sequence_length = self.config["data"]["sequence_length"]

        fe_conf = FeatureEngineerConfig(
            rolling_windows=tuple(self.config["features"]["rolling_windows"]),
            atr_windows=tuple(self.config["features"]["atr_windows"]),
            bollinger_window=self.config["features"]["bollinger_windows"]["window"],
            bollinger_num_std=self.config["features"]["bollinger_windows"]["num_std"],
            regime_trend_window=self.config["features"]["regime_trend_window"],
            volume_spike_threshold=self.config["features"]["volume_spike_threshold"],
            enable_market_profile=self.config["features"]["enable_market_profile"],
            enable_smc=self.config["features"]["enable_smc"],
        )
        self.feature_engineer = FeatureEngineer(fe_conf)
        self.feature_names = self.metadata["feature_names"]
        self.scaler = self._restore_scaler(self.metadata.get("scaler_state"), self.config["data"]["feature_normalization"])

        model_conf = NatronTransformerConfig(
            input_dim=len(self.feature_names),
            d_model=self.config["model"]["d_model"],
            n_heads=self.config["model"]["n_heads"],
            num_layers=self.config["model"]["num_layers"],
            mlp_ratio=self.config["model"]["mlp_ratio"],
            dropout=self.config["model"]["dropout"],
            activation=self.config["model"]["activation"],
            max_seq_len=max(self.config["model"]["max_seq_len"], self.sequence_length + 1),
            projection_dim=self.config["model"]["projection_dim"],
        )
        self.model = NatronTransformer(model_conf)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _load_yaml(path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _load_json(path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return json.load(f)

    def _restore_scaler(self, state_json: str | None, mode: str):
        if not state_json or mode == "none":
            return None
        scaler_cls = RobustScaler if mode == "robust" else StandardScaler
        scaler = scaler_cls()
        state = json.loads(state_json)
        for attr, values in state.items():
            setattr(scaler, attr, np.array(values))
        return scaler

    def preprocess(self, candles: List[Dict[str, Any]]) -> torch.Tensor:
        df = pd.DataFrame(candles)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.sort_values("time").reset_index(drop=True)
        features_df = self.feature_engineer.transform(df)
        features_df = features_df.reindex(columns=self.feature_names, fill_value=0.0)
        if self.scaler is not None:
            features_df[self.feature_names] = self.scaler.transform(features_df[self.feature_names].values)
        features_array = features_df.to_numpy(dtype=np.float32)
        if len(features_array) < self.sequence_length:
            pad = np.repeat(features_array[:1], self.sequence_length - len(features_array), axis=0)
            features_array = np.concatenate([pad, features_array], axis=0)
        else:
            features_array = features_array[-self.sequence_length :]
        tensor = torch.from_numpy(features_array).unsqueeze(0).to(self.device)
        return tensor

    @torch.no_grad()
    def predict(self, candles: List[Dict[str, Any]]) -> Dict[str, Any]:
        sequence = self.preprocess(candles)
        outputs = self.model.forward_supervised(sequence)
        buy_prob = torch.sigmoid(outputs["buy_logits"]).item()
        sell_prob = torch.sigmoid(outputs["sell_logits"]).item()
        direction_probs = torch.softmax(outputs["direction_logits"], dim=-1).squeeze(0)
        regime_probs = torch.softmax(outputs["regime_logits"], dim=-1).squeeze(0)

        direction_up = direction_probs[1].item()
        regime_idx = torch.argmax(regime_probs).item()
        regime_label = REGIME_MAP.get(regime_idx, "UNKNOWN")
        confidence = float(
            max(
                buy_prob,
                sell_prob,
                direction_probs.max().item(),
                regime_probs.max().item(),
            )
        )
        return {
            "buy_prob": float(buy_prob),
            "sell_prob": float(sell_prob),
            "direction_up": float(direction_up),
            "direction_down": float(direction_probs[0].item()),
            "regime": regime_label,
            "regime_probs": {REGIME_MAP[i]: float(regime_probs[i].item()) for i in range(len(regime_probs))},
            "confidence": confidence,
        }


def create_flask_app(engine: NatronInferenceEngine) -> Flask:
    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/predict", methods=["POST"])
    def predict():
        payload = request.get_json(force=True)
        candles = payload.get("candles")
        if not candles:
            return jsonify({"error": "Missing candles"}), 400
        try:
            prediction = engine.predict(candles)
            return jsonify(prediction)
        except Exception as exc:  # pragma: no cover
            return jsonify({"error": str(exc)}), 500

    return app


async def handle_tcp_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, engine: NatronInferenceEngine):
    addr = writer.get_extra_info("peername")
    try:
        data = await reader.readline()
        message = data.decode().strip()
        payload = json.loads(message)
        candles = payload.get("candles", [])
        result = engine.predict(candles)
        writer.write((json.dumps(result) + "\n").encode())
        await writer.drain()
    except Exception as exc:  # pragma: no cover
        error = json.dumps({"error": str(exc)})
        writer.write((error + "\n").encode())
        await writer.drain()
    finally:
        writer.close()
        await writer.wait_closed()


def run_flask(app: Flask, host: str, port: int) -> None:
    app.run(host=host, port=port, debug=False, use_reloader=False)


async def main_async(engine: NatronInferenceEngine) -> None:
    host = engine.config["inference"]["host"]
    port = engine.config["inference"]["port"]
    flask_port = engine.config["inference"]["flask_port"]

    app = create_flask_app(engine)
    flask_thread = threading.Thread(target=run_flask, args=(app, host, flask_port), daemon=True)
    flask_thread.start()

    server = await asyncio.start_server(lambda r, w: handle_tcp_client(r, w, engine), host, port)
    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
    print(f"[Natron] TCP server listening on {addrs}")
    async with server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Natron inference server")
    parser.add_argument("--config", type=str, default="configs/natron_config.yaml")
    parser.add_argument("--metadata", type=str, default="outputs/data_metadata.json")
    parser.add_argument("--model", type=str, default="model/natron_v2.pt")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    engine = NatronInferenceEngine(
        config_path=args.config,
        metadata_path=args.metadata,
        model_path=args.model,
        device=args.device,
    )
    asyncio.run(main_async(engine))


if __name__ == "__main__":
    main()
