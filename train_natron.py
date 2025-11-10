"""
Natron training entry point covering pretraining, supervised fine-tuning, and optional PPO.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from dataset_loader import NatronDataConfig, NatronDatasetBuilder, create_dataloaders
from feature_engine import FeatureEngineerConfig
from labeling import LabelGeneratorConfig
from losses import InfoNCELoss, MaskedMSELoss, MultiTaskLoss
from model_natron import NatronTransformer, NatronTransformerConfig
from reinforcement import PPOConfig, run_ppo_training
from utils import augment_batch, compute_multitask_metrics, create_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Natron V2 Training")
    parser.add_argument("--config", type=str, default="configs/natron_config.yaml")
    parser.add_argument("--phase", type=str, default="all", choices=["all", "pretrain", "supervised", "rl"])
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_output_dirs(config: Dict) -> Tuple[Path, Path, Path]:
    output_dir = Path(config["experiment"]["output_dir"])
    checkpoint_dir = Path(config["experiment"]["checkpoint_dir"])
    log_dir = Path(config["experiment"]["log_dir"])
    for directory in [output_dir, checkpoint_dir, log_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    return output_dir, checkpoint_dir, log_dir


def build_dataset(config: Dict) -> Tuple:
    data_conf = NatronDataConfig(**config["data"])
    feature_conf = FeatureEngineerConfig(
        rolling_windows=tuple(config["features"]["rolling_windows"]),
        atr_windows=tuple(config["features"]["atr_windows"]),
        bollinger_window=config["features"]["bollinger_windows"]["window"],
        bollinger_num_std=config["features"]["bollinger_windows"]["num_std"],
        regime_trend_window=config["features"]["regime_trend_window"],
        volume_spike_threshold=config["features"]["volume_spike_threshold"],
        enable_market_profile=config["features"]["enable_market_profile"],
        enable_smc=config["features"]["enable_smc"],
    )
    label_conf = LabelGeneratorConfig(
        regime_trend_window=config["features"]["regime_trend_window"],
        volume_spike_threshold=config["features"]["volume_spike_threshold"],
    )
    builder = NatronDatasetBuilder(data_conf, feature_conf, label_conf)
    train_set, val_set, test_set, metadata = builder.build()
    train_loader, val_loader, test_loader = create_dataloaders(
        train_set, val_set, test_set, batch_size=data_conf.batch_size, num_workers=data_conf.num_workers
    )
    return builder, train_set, val_set, test_set, train_loader, val_loader, test_loader, metadata


def create_model(config: Dict, feature_dim: int) -> NatronTransformer:
    model_conf = config["model"]
    natron_conf = NatronTransformerConfig(
        input_dim=feature_dim,
        d_model=model_conf["d_model"],
        n_heads=model_conf["n_heads"],
        num_layers=model_conf["num_layers"],
        mlp_ratio=model_conf["mlp_ratio"],
        dropout=model_conf["dropout"],
        activation=model_conf["activation"],
        max_seq_len=max(model_conf["max_seq_len"], config["data"]["sequence_length"] + 1),
        projection_dim=model_conf["projection_dim"],
    )
    return NatronTransformer(natron_conf)


def train_pretraining(
    model: NatronTransformer,
    train_loader,
    val_loader,
    config: Dict,
    device: torch.device,
    logger,
    writer: SummaryWriter,
    checkpoint_dir: Path,
) -> None:
    model.to(device)
    model.train()

    criterion_recon = MaskedMSELoss()
    criterion_contrast = InfoNCELoss(temperature=config["pretraining"]["contrastive"]["temperature"])

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["pretraining"]["lr"],
        weight_decay=config["pretraining"]["weight_decay"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    epochs = config["pretraining"]["epochs"]
    mask_ratio = config["pretraining"]["masking_ratio"]

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Pretrain Epoch {epoch}/{epochs}")
        for batch in pbar:
            sequences = batch["sequences"].to(device)
            aug1 = augment_batch(sequences)
            aug2 = augment_batch(sequences)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                out1 = model.forward_pretrain(aug1, mask_ratio=mask_ratio)
                out2 = model.forward_pretrain(aug2, mask_ratio=mask_ratio)
                recon_loss = criterion_recon(out1["reconstruction"], sequences, out1["mask"])
                recon_loss += criterion_recon(out2["reconstruction"], sequences, out2["mask"])
                contrastive_loss = criterion_contrast(out1["projection"], out2["projection"])
                loss = recon_loss + contrastive_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = epoch_loss / len(train_loader)
        val_loss = evaluate_reconstruction(model, val_loader, criterion_recon, mask_ratio, device)
        writer.add_scalar("pretrain/train_loss", avg_train_loss, epoch)
        writer.add_scalar("pretrain/val_loss", val_loss, epoch)
        logger.info(f"[Pretrain] Epoch {epoch}: train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f}")

        torch.save(model.state_dict(), checkpoint_dir / f"natron_pretrain_epoch{epoch}.pt")


@torch.no_grad()
def evaluate_reconstruction(model, data_loader, criterion, mask_ratio, device) -> float:
    model.eval()
    losses = []
    for batch in data_loader:
        sequences = batch["sequences"].to(device)
        out = model.forward_pretrain(sequences, mask_ratio=mask_ratio)
        loss = criterion(out["reconstruction"], sequences, out["mask"])
        losses.append(loss.item())
    return sum(losses) / max(len(losses), 1)


def train_supervised(
    model: NatronTransformer,
    train_loader,
    val_loader,
    config: Dict,
    device: torch.device,
    logger,
    writer: SummaryWriter,
    checkpoint_dir: Path,
) -> str:
    model.to(device)
    if config["supervised"]["freeze_encoder"]:
        for name, param in model.named_parameters():
            if "buy_head" in name or "sell_head" in name or "direction_head" in name or "regime_head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["supervised"]["lr"],
        weight_decay=config["supervised"]["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=config["supervised"]["scheduler"]["factor"],
        patience=config["supervised"]["scheduler"]["patience"],
        mode="max",
        verbose=False,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    loss_fn = MultiTaskLoss(config["supervised"]["loss_weights"])
    best_score = -float("inf")
    best_path = checkpoint_dir / "natron_supervised_best.pt"

    for epoch in range(1, config["supervised"]["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Supervised Epoch {epoch}/{config['supervised']['epochs']}")
        for batch in pbar:
            sequences = batch["sequences"].to(device)
            targets = {
                "buy": batch["buy"].to(device),
                "sell": batch["sell"].to(device),
                "direction": batch["direction"].to(device),
                "regime": batch["regime"].to(device),
            }

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                outputs = model.forward_supervised(sequences)
                loss_out = loss_fn(outputs, targets)
                loss = loss_out.total
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = epoch_loss / len(train_loader)
        val_metrics = evaluate_supervised(model, val_loader, loss_fn, device)
        scheduler.step(val_metrics["macro_f1"])

        writer.add_scalar("supervised/train_loss", avg_train_loss, epoch)
        writer.add_scalar("supervised/val_loss", val_metrics["loss"], epoch)
        writer.add_scalar("supervised/macro_f1", val_metrics["macro_f1"], epoch)
        logger.info(
            f"[Supervised] Epoch {epoch}: train_loss={avg_train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_score:
            best_score = val_metrics["macro_f1"]
            torch.save(model.state_dict(), best_path)

    return str(best_path)


@torch.no_grad()
def evaluate_supervised(model, data_loader, loss_fn, device) -> Dict[str, float]:
    model.eval()
    losses = []
    metrics_accumulator: Dict[str, float] = {}
    count = 0
    for batch in data_loader:
        sequences = batch["sequences"].to(device)
        targets = {
            "buy": batch["buy"].to(device),
            "sell": batch["sell"].to(device),
            "direction": batch["direction"].to(device),
            "regime": batch["regime"].to(device),
        }
        outputs = model.forward_supervised(sequences)
        loss_out = loss_fn(outputs, targets)
        losses.append(loss_out.total.item())
        metrics_dict = compute_multitask_metrics(outputs, targets)
        for k, v in metrics_dict.items():
            metrics_accumulator[k] = metrics_accumulator.get(k, 0.0) + v
        count += 1

    averaged_metrics = {k: v / max(count, 1) for k, v in metrics_accumulator.items()}
    averaged_metrics["loss"] = sum(losses) / max(len(losses), 1)
    averaged_metrics["macro_f1"] = (
        averaged_metrics.get("buy_f1", 0.0) + averaged_metrics.get("sell_f1", 0.0)
    ) / 2.0
    return averaged_metrics


def run_reinforcement_learning(
    model: NatronTransformer,
    train_loader,
    config: Dict,
    device: torch.device,
    logger,
) -> None:
    sequences = []
    direction_scores = []
    for batch in train_loader:
        sequences.append(batch["sequences"].numpy())
        direction_scores.append(batch["direction_score"].numpy())
    sequences_np = np.concatenate(sequences, axis=0)
    direction_scores_np = np.concatenate(direction_scores, axis=0)

    ppo_conf = PPOConfig(
        gamma=config["reinforcement"]["gamma"],
        gae_lambda=config["reinforcement"]["gae_lambda"],
        clip_ratio=config["reinforcement"]["clip_ratio"],
        lr=config["reinforcement"]["lr"],
        entropy_coef=config["reinforcement"]["entropy_coef"],
        value_coef=config["reinforcement"]["value_coef"],
        steps_per_epoch=config["reinforcement"]["steps_per_epoch"],
        update_epochs=config["reinforcement"]["epochs"],
        turnover_penalty=config["reinforcement"]["turnover_penalty"],
        drawdown_penalty=config["reinforcement"]["drawdown_penalty"],
        device=device.type,
    )
    metrics = run_ppo_training(model.to(device), sequences_np, direction_scores_np, ppo_conf)
    logger.info(f"[RL] Completed PPO training: {metrics}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])
    device = torch.device(args.device or config["inference"]["device"])

    output_dir, checkpoint_dir, log_dir = prepare_output_dirs(config)
    logger = create_logger(log_dir=str(log_dir))
    writer = SummaryWriter(log_dir=log_dir / "tensorboard")

    builder, train_set, val_set, test_set, train_loader, val_loader, test_loader, metadata = build_dataset(config)
    feature_dim = train_set.sequences.shape[-1]

    model = create_model(config, feature_dim)
    metadata_payload = {
        "feature_names": metadata["feature_names"].tolist(),
        "scaler_state": metadata["scaler_state"],
        "config": metadata["config"],
    }
    metadata_path = output_dir / "data_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata_payload, f, indent=2)
    logger.info(f"Wrote dataset metadata to {metadata_path}")

    if config["pretraining"]["run"] and args.phase in ("all", "pretrain"):
        train_pretraining(model, train_loader, val_loader, config, device, logger, writer, checkpoint_dir)
        logger.info("Pretraining phase complete.")

    if config["supervised"]["run"] and args.phase in ("all", "supervised"):
        if config["pretraining"]["run"]:
            latest_pretrain = checkpoint_dir / f"natron_pretrain_epoch{config['pretraining']['epochs']}.pt"
            if latest_pretrain.exists():
                model.load_state_dict(torch.load(latest_pretrain, map_location=device), strict=False)
                logger.info(f"Loaded pretrained weights from {latest_pretrain}")
        best_path = train_supervised(model, train_loader, val_loader, config, device, logger, writer, checkpoint_dir)
        model.load_state_dict(torch.load(best_path, map_location=device))
        model_path = Path(config["inference"]["model_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        logger.info(f"Supervised training complete. Saved best model to {config['inference']['model_path']}")

    if config["reinforcement"]["run"] and args.phase in ("all", "rl"):
        run_reinforcement_learning(model, train_loader, config, device, logger)

    writer.close()
    logger.info("Natron training pipeline finished.")


if __name__ == "__main__":
    main()
