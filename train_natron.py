"""
Natron training entrypoint.

Pipeline:
1. Feature engineering + dataset preparation
2. Self-supervised pretraining (masked modeling + contrastive)
3. Supervised multi-task fine-tuning
4. Persist model checkpoint with normalization statistics
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from natron.dataset_loader import NatronDatasetConfig, prepare_natron_datasets
from natron.losses import LossWeights, MaskedModelingLoss, MultiTaskLoss, NTXentContrastiveLoss
from natron.model_natron import NatronModelConfig, NatronTransformer


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Natron Transformer pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Override device placement.",
    )
    return parser.parse_args()


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_dataloaders(
    datasets: Dict[str, Any],
    batch_size: int,
    num_workers: int,
) -> Dict[str, DataLoader]:
    return {
        "supervised_train": DataLoader(
            datasets["supervised_train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        "supervised_val": DataLoader(
            datasets["supervised_val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "supervised_test": DataLoader(
            datasets["supervised_test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "masked_pretrain": DataLoader(
            datasets["masked_pretrain"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        "contrastive_pretrain": DataLoader(
            datasets["contrastive_pretrain"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
    }


def pretrain_encoder(
    model: NatronTransformer,
    dataloaders: Dict[str, DataLoader],
    device: torch.device,
    config: Dict[str, Any],
) -> None:
    logging.info("Starting pretraining phase...")
    training_cfg = config["training"]
    optimizer = AdamW(
        model.parameters(),
        lr=training_cfg["optimizer"]["lr"],
        weight_decay=training_cfg["optimizer"]["weight_decay"],
        betas=tuple(training_cfg["optimizer"].get("betas", (0.9, 0.999))),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=training_cfg["scheduler"]["factor"],
        patience=training_cfg["scheduler"]["patience"],
        verbose=True,
    )
    mask_loss_fn = MaskedModelingLoss()
    contrastive_loss_fn = NTXentContrastiveLoss()
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    best_loss = math.inf
    patience_counter = 0
    patience_limit = training_cfg.get("early_stopping_patience", 15)

    for epoch in range(1, training_cfg["epochs_pretrain"] + 1):
        model.train()
        epoch_mask_loss = 0.0
        epoch_contrastive_loss = 0.0
        steps = 0

        for batch in tqdm(dataloaders["masked_pretrain"], desc=f"[Pretrain] Masked Epoch {epoch}", leave=False):
            optimizer.zero_grad(set_to_none=True)
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            mask = batch["mask"].to(device)

            outputs = model(inputs)
            reconstruction = outputs["reconstruction"]
            loss = mask_loss_fn(reconstruction, targets, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_cfg.get("gradient_clip_norm", 1.0))
            optimizer.step()

            epoch_mask_loss += loss.item()
            steps += 1

        for batch in tqdm(dataloaders["contrastive_pretrain"], desc=f"[Pretrain] Contrastive Epoch {epoch}", leave=False):
            optimizer.zero_grad(set_to_none=True)
            view1 = batch["view_1"].to(device)
            view2 = batch["view_2"].to(device)

            proj1 = model(view1)["projection"]
            proj2 = model(view2)["projection"]
            loss = contrastive_loss_fn(proj1, proj2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_cfg.get("gradient_clip_norm", 1.0))
            optimizer.step()

            epoch_contrastive_loss += loss.item()
            steps += 1

        avg_mask = epoch_mask_loss / max(steps, 1)
        avg_contrastive = epoch_contrastive_loss / max(steps, 1)
        scheduler.step(avg_mask + avg_contrastive)

        logging.info(
            "Pretrain Epoch %d | Mask %.4f | Contrastive %.4f | LR %.6f",
            epoch,
            avg_mask,
            avg_contrastive,
            optimizer.param_groups[0]["lr"],
        )

        if avg_mask + avg_contrastive < best_loss:
            best_loss = avg_mask + avg_contrastive
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "pretrain_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                logging.info("Pretraining early-stopped at epoch %d", epoch)
                break

    if output_dir.joinpath("pretrain_best.pt").exists():
        model.load_state_dict(
            torch.load(
                output_dir / "pretrain_best.pt",
                map_location=device,
            )
        )


def evaluate_supervised(
    model: NatronTransformer,
    dataloader: DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    loss_meter = 0.0
    count = 0
    metrics = {
        "buy_acc": 0.0,
        "sell_acc": 0.0,
        "direction_acc": 0.0,
        "regime_acc": 0.0,
    }

    with torch.inference_mode():
        for batch in dataloader:
            sequence = batch["sequence"].to(device)
            targets = {
                "buy": batch["buy"].to(device),
                "sell": batch["sell"].to(device),
                "direction": batch["direction"].to(device),
                "regime": batch["regime"].to(device),
            }
            outputs = model(sequence)
            loss_dict = criterion(outputs, targets)
            loss_meter += loss_dict["total_loss"].item()
            count += 1

            buy_pred = (torch.sigmoid(outputs["buy_logits"]) > 0.5).float()
            sell_pred = (torch.sigmoid(outputs["sell_logits"]) > 0.5).float()
            dir_pred = torch.argmax(outputs["direction_logits"], dim=-1)
            regime_pred = torch.argmax(outputs["regime_logits"], dim=-1)

            metrics["buy_acc"] += (buy_pred == targets["buy"]).float().mean().item()
            metrics["sell_acc"] += (sell_pred == targets["sell"]).float().mean().item()
            metrics["direction_acc"] += (dir_pred == targets["direction"]).float().mean().item()
            metrics["regime_acc"] += (regime_pred == targets["regime"]).float().mean().item()

    for key in metrics:
        metrics[key] /= max(count, 1)
    metrics["loss"] = loss_meter / max(count, 1)
    return metrics


def supervised_training(
    model: NatronTransformer,
    dataloaders: Dict[str, DataLoader],
    device: torch.device,
    config: Dict[str, Any],
) -> Dict[str, float]:
    training_cfg = config["training"]
    loss_weights = LossWeights(**config["loss_weights"])
    criterion = MultiTaskLoss(loss_weights)
    optimizer = AdamW(
        model.parameters(),
        lr=training_cfg["optimizer"]["lr"],
        weight_decay=training_cfg["optimizer"]["weight_decay"],
        betas=tuple(training_cfg["optimizer"].get("betas", (0.9, 0.999))),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=training_cfg["scheduler"]["factor"],
        patience=training_cfg["scheduler"]["patience"],
        verbose=True,
    )

    best_val_loss = math.inf
    best_metrics: Dict[str, float] = {}
    patience_counter = 0
    patience_limit = training_cfg.get("early_stopping_patience", 15)
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, training_cfg["epochs_supervised"] + 1):
        model.train()
        progress = tqdm(dataloaders["supervised_train"], desc=f"[Supervised] Epoch {epoch}")
        for step, batch in enumerate(progress, start=1):
            sequence = batch["sequence"].to(device)
            targets = {
                "buy": batch["buy"].to(device),
                "sell": batch["sell"].to(device),
                "direction": batch["direction"].to(device),
                "regime": batch["regime"].to(device),
            }
            outputs = model(sequence)
            losses = criterion(outputs, targets)
            loss = losses["total_loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_cfg.get("gradient_clip_norm", 1.0))
            optimizer.step()

            progress.set_postfix({"loss": loss.item()})

        val_metrics = evaluate_supervised(model, dataloaders["supervised_val"], criterion, device)
        scheduler.step(val_metrics["loss"])

        logging.info(
            "Epoch %d | Val Loss %.4f | Buy %.3f | Sell %.3f | Dir %.3f | Regime %.3f",
            epoch,
            val_metrics["loss"],
            val_metrics["buy_acc"],
            val_metrics["sell_acc"],
            val_metrics["direction_acc"],
            val_metrics["regime_acc"],
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_metrics = val_metrics
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "supervised_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                logging.info("Supervised training early-stopped at epoch %d", epoch)
                break

    if output_dir.joinpath("supervised_best.pt").exists():
        model.load_state_dict(
            torch.load(
                output_dir / "supervised_best.pt",
                map_location=device,
            )
        )

    test_metrics = evaluate_supervised(model, dataloaders["supervised_test"], criterion, device)
    logging.info(
        "Test Metrics | Loss %.4f | Buy %.3f | Sell %.3f | Dir %.3f | Regime %.3f",
        test_metrics["loss"],
        test_metrics["buy_acc"],
        test_metrics["sell_acc"],
        test_metrics["direction_acc"],
        test_metrics["regime_acc"],
    )
    return test_metrics


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_yaml_config(config_path)
    output_dir = Path(config["experiment"]["output_dir"])
    setup_logging(output_dir)
    set_seed(config["experiment"]["seed"])

    device = torch.device(args.device)
    logging.info("Using device: %s", device)

    dataset_config = NatronDatasetConfig(
        sequence_length=config["data"]["sequence_length"],
        train_split=config["data"]["train_split"],
        val_split=config["data"]["val_split"],
        test_split=config["data"]["test_split"],
        masking_probability=config["data"]["augmentations"]["masking_probability"],
        normalization=config["data"]["normalize"],
        feature_cache_path=Path(config["data"]["feature_cache"]),
    )

    logging.info("Loading datasets...")
    datasets = prepare_natron_datasets(
        csv_path=Path(config["data"]["csv_path"]),
        config=dataset_config,
    )

    dataloaders = build_dataloaders(
        datasets,
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    model_config = NatronModelConfig(
        feature_dim=dataset_config.feature_count,
        d_model=config["model"]["d_model"],
        nhead=config["model"]["nhead"],
        num_layers=config["model"]["num_layers"],
        dim_feedforward=config["model"]["dim_feedforward"],
        dropout=config["model"]["dropout"],
        activation=config["model"]["activation"],
        max_sequence_length=dataset_config.sequence_length + 8,
    )
    model = NatronTransformer(model_config).to(device)

    if config["training"]["epochs_pretrain"] > 0:
        pretrain_encoder(model, dataloaders, device, config)

    test_metrics = supervised_training(model, dataloaders, device, config)

    model_path = Path(config["deployment"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "model_config": model_config.__dict__,
            "feature_mean": datasets["feature_mean"],
            "feature_std": datasets["feature_std"],
            "metrics": test_metrics,
        },
        model_path,
    )

    logging.info("Training complete. Model saved to %s", model_path)


if __name__ == "__main__":
    main()
