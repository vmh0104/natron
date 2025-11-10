"""
Natron Transformer training entrypoint.

Pipeline:
    1. Data preparation with engineered features and institutional labels.
    2. Phase 1: Self-supervised pretraining (masked reconstruction + InfoNCE).
    3. Phase 2: Supervised multi-task fine-tuning.
    4. Optional Phase 3: Reinforcement learning via PPO (delegated to rl_trainer.py).
"""

from __future__ import annotations

import argparse
import logging
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm
import yaml

from dataset_loader import DataConfig, NatronDataModule
from feature_engineering import FeatureConfig
from labeling import LabelConfig
from losses import LossConfig, LossWeights, MultiTaskLoss, info_nce_loss, masked_mse_loss
from model_natron import ModelConfig, NatronTransformer
from rl_trainer import maybe_train_rl

LOGGER = logging.getLogger("natron.train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Natron Transformer training script")
    parser.add_argument("--config", type=str, default="configs/natron_config.yaml", help="YAML config path")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda)")
    parser.add_argument("--skip-pretrain", action="store_true", help="Skip self-supervised pretraining phase")
    parser.add_argument("--skip-finetune", action="store_true", help="Skip supervised fine-tuning phase")
    parser.add_argument("--force-cache-refresh", action="store_true", help="Recompute dataset cache")
    parser.add_argument("--output-dir", type=str, default=None, help="Override checkpoint directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint for warm start")
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "train.log"),
        ],
    )


def filter_kwargs(target_cls, cfg_dict: Dict) -> Dict:
    valid_keys = target_cls.__annotations__.keys()
    return {k: v for k, v in cfg_dict.items() if k in valid_keys}


def create_masks(inputs: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = torch.rand_like(inputs) < mask_ratio
    masked_inputs = inputs.clone()
    masked_inputs[mask] = 0.0
    return masked_inputs, mask


def augment_batch(inputs: torch.Tensor, jitter: float = 0.01, scaling: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
    noise1 = torch.randn_like(inputs) * jitter
    noise2 = torch.randn_like(inputs) * jitter
    scale1 = torch.randn(inputs.size(0), inputs.size(2), device=inputs.device) * scaling + 1.0
    scale2 = torch.randn(inputs.size(0), inputs.size(2), device=inputs.device) * scaling + 1.0
    aug1 = inputs * scale1.unsqueeze(1) + noise1
    aug2 = inputs * scale2.unsqueeze(1) + noise2
    return aug1, aug2


def save_checkpoint(path: Path, model: NatronTransformer, metadata: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "config": model.config.__dict__,
    }
    payload.update(metadata)
    torch.save(payload, path)
    LOGGER.info("Checkpoint saved to %s", path)


def run_pretraining(
    model: NatronTransformer,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    training_cfg: Dict,
    device: torch.device,
) -> None:
    epochs = training_cfg.get("max_epochs_pretrain", 0)
    if epochs <= 0:
        LOGGER.info("Skipping pretraining (max_epochs_pretrain <= 0)")
        return

    optimizer_cfg = training_cfg.get("optimizer", {})
    lr = optimizer_cfg.get("lr", 1e-4)
    weight_decay = optimizer_cfg.get("weight_decay", 1e-5)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=optimizer_cfg.get("betas", (0.9, 0.999)))

    scheduler_cfg = training_cfg.get("scheduler", {})
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_cfg.get("factor", 0.5),
        patience=scheduler_cfg.get("patience", 5),
    )

    loss_weights = training_cfg.get("loss_weights", {})
    recon_w = loss_weights.get("reconstruction", 1.0)
    contrast_w = loss_weights.get("contrastive", 1.0)
    contrast_cfg = training_cfg.get("contrastive", {})
    temperature = contrast_cfg.get("temperature", 0.5)
    mask_ratio = training_cfg.get("masking_ratio", model.config.masking_ratio)
    grad_clip = training_cfg.get("grad_clip_norm", 1.0)

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in tqdm(dataloaders["train"], desc=f"[Pretrain] Epoch {epoch}", leave=False):
            inputs = batch["inputs"].to(device)
            masked_inputs, mask = create_masks(inputs, mask_ratio)
            mask = mask.to(device)

            masked_outputs = model.masked_reconstruct(masked_inputs, mask)
            loss_recon = masked_mse_loss(masked_outputs["pred"], masked_outputs["target"])

            aug1, aug2 = augment_batch(inputs, jitter=training_cfg.get("augmentations", {}).get("jitter", 0.01), scaling=training_cfg.get("augmentations", {}).get("scaling", 0.05))
            proj1 = model.contrastive_projection(aug1)
            proj2 = model.contrastive_projection(aug2)
            loss_contrast = info_nce_loss(proj1, proj2, temperature=temperature)

            loss = recon_w * loss_recon + contrast_w * loss_contrast
            optimizer.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / max(len(dataloaders["train"]), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in dataloaders["val"]:
                inputs = batch["inputs"].to(device)
                masked_inputs, mask = create_masks(inputs, mask_ratio)
                mask = mask.to(device)

                masked_outputs = model.masked_reconstruct(masked_inputs, mask)
                loss_recon = masked_mse_loss(masked_outputs["pred"], masked_outputs["target"])
                aug1, aug2 = augment_batch(inputs)
                proj1 = model.contrastive_projection(aug1)
                proj2 = model.contrastive_projection(aug2)
                loss_contrast = info_nce_loss(proj1, proj2, temperature=temperature)
                loss = recon_w * loss_recon + contrast_w * loss_contrast
                val_loss += loss.item()

        avg_val_loss = val_loss / max(len(dataloaders["val"]), 1)
        scheduler.step(avg_val_loss)
        LOGGER.info(
            "[Pretrain] Epoch %d | train_loss=%.5f | val_loss=%.5f | lr=%.6f",
            epoch,
            avg_train_loss,
            avg_val_loss,
            optimizer.param_groups[0]["lr"],
        )

        if avg_val_loss < best_val:
            best_val = avg_val_loss
            save_checkpoint(Path(training_cfg["checkpoint_dir"]) / "natron_encoder.pt", model, {"stage": "pretrain", "epoch": epoch})


def run_supervised(
    model: NatronTransformer,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    training_cfg: Dict,
    device: torch.device,
) -> None:
    epochs = training_cfg.get("max_epochs_supervised", 0)
    if epochs <= 0:
        LOGGER.info("Skipping supervised training (max_epochs_supervised <= 0)")
        return

    optimizer_cfg = training_cfg.get("optimizer", {})
    lr = optimizer_cfg.get("lr", 1e-4)
    weight_decay = optimizer_cfg.get("weight_decay", 1e-5)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=optimizer_cfg.get("betas", (0.9, 0.999)))

    scheduler_cfg = training_cfg.get("scheduler", {})
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_cfg.get("factor", 0.5),
        patience=scheduler_cfg.get("patience", 5),
    )

    loss_weights_cfg = training_cfg.get("loss_weights", {})
    weights = LossWeights(
        buy=loss_weights_cfg.get("buy", 0.8),
        sell=loss_weights_cfg.get("sell", 0.8),
        direction=loss_weights_cfg.get("direction", 1.0),
        regime=loss_weights_cfg.get("regime", 1.2),
        reconstruction=loss_weights_cfg.get("reconstruction", 1.0),
        contrastive=loss_weights_cfg.get("contrastive", 1.0),
    )
    multitask_loss = MultiTaskLoss(LossConfig(weights=weights))
    grad_clip = training_cfg.get("grad_clip_norm", 1.0)
    freeze_epochs = training_cfg.get("freeze_encoder_epochs", 0)

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        if epoch <= freeze_epochs:
            for param in model.encoder_layers.parameters():
                param.requires_grad = False
        else:
            for param in model.encoder_layers.parameters():
                param.requires_grad = True

        model.train()
        train_loss = 0.0
        for batch in tqdm(dataloaders["train"], desc=f"[Supervised] Epoch {epoch}", leave=False):
            inputs = batch["inputs"].to(device)
            buy = batch["buy"].to(device)
            sell = batch["sell"].to(device)
            direction = batch["direction"].to(device)
            regime = batch["regime"].to(device)

            outputs = model(inputs)
            targets = {"buy": buy, "sell": sell, "direction": direction, "regime": regime}
            loss_dict = multitask_loss(outputs, targets)
            loss = loss_dict["total_supervised"]

            optimizer.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / max(len(dataloaders["train"]), 1)

        model.eval()
        val_loss = 0.0
        hit_rate = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloaders["val"]:
                inputs = batch["inputs"].to(device)
                buy = batch["buy"].to(device)
                sell = batch["sell"].to(device)
                direction = batch["direction"].to(device)
                regime = batch["regime"].to(device)

                outputs = model(inputs)
                targets = {"buy": buy, "sell": sell, "direction": direction, "regime": regime}
                loss_dict = multitask_loss(outputs, targets)
                val_loss += loss_dict["total_supervised"].item()

                direction_pred = outputs["direction_logits"].argmax(dim=1)
                hit_rate += (direction_pred == direction).sum().item()
                total_samples += direction.size(0)

        avg_val_loss = val_loss / max(len(dataloaders["val"]), 1)
        hit_ratio = hit_rate / max(total_samples, 1)
        scheduler.step(avg_val_loss)

        LOGGER.info(
            "[Supervised] Epoch %d | train_loss=%.5f | val_loss=%.5f | hit_rate=%.3f | lr=%.6f",
            epoch,
            avg_train_loss,
            avg_val_loss,
            hit_ratio,
            optimizer.param_groups[0]["lr"],
        )

        if avg_val_loss < best_val:
            best_val = avg_val_loss
            save_checkpoint(Path(training_cfg["checkpoint_dir"]) / "natron_v2.pt", model, {"stage": "supervised", "epoch": epoch})


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    project_cfg = config.get("project", {})
    output_dir = Path(args.output_dir or project_cfg.get("checkpoint_dir", "models"))
    log_dir = Path(project_cfg.get("log_dir", "logs"))
    setup_logging(log_dir)
    seed_everything(project_cfg.get("seed", 1337))

    device = torch.device(args.device or project_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    LOGGER.info("Using device: %s", device)

    data_cfg = DataConfig(**filter_kwargs(DataConfig, config.get("data", {})))
    feature_cfg = FeatureConfig(**filter_kwargs(FeatureConfig, config.get("features", {})))
    label_cfg = LabelConfig(**filter_kwargs(LabelConfig, config.get("labels", {}))) if "labels" in config else LabelConfig()
    data_module = NatronDataModule(data_cfg, feature_cfg, label_cfg)
    data_module.prepare(force=args.force_cache_refresh)
    dataloaders = data_module.dataloaders()

    model_cfg_dict = config.get("model", {})
    model_cfg_dict["input_dim"] = len(data_module.feature_columns)
    model_cfg = ModelConfig(**filter_kwargs(ModelConfig, model_cfg_dict))
    model = NatronTransformer(model_cfg).to(device)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        LOGGER.info("Loaded checkpoint from %s", args.resume)

    training_cfg = config.get("training", {})
    training_cfg["checkpoint_dir"] = str(output_dir)

    if not args.skip_pretrain:
        run_pretraining(model, dataloaders, training_cfg, device)
    else:
        LOGGER.info("Skipping pretraining as requested.")

    if not args.skip_finetune:
        run_supervised(model, dataloaders, training_cfg, device)
    else:
        LOGGER.info("Skipping supervised fine-tuning as requested.")

    scaler_path = Path(config.get("inference", {}).get("scaler_path", output_dir / "scaler.pkl"))
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    if data_module.scaler is not None:
        joblib.dump(data_module.scaler, scaler_path)
        LOGGER.info("Saved scaler to %s", scaler_path)

    metadata_path = Path(output_dir) / "feature_columns.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as fp:
        json.dump({"feature_columns": data_module.feature_columns}, fp, indent=2)
    LOGGER.info("Saved feature column order to %s", metadata_path)

    if config.get("reinforcement", {}).get("enabled", False):
        maybe_train_rl(model, config, device=device, data_module=data_module)

    LOGGER.info("Natron training pipeline complete.")


if __name__ == "__main__":
    main()
