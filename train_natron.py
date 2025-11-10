"""
Natron training pipeline covering pretraining, supervised fine-tuning, and optional RL adaptation.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import joblib
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataset_loader import create_data_module_from_config, load_config
from losses import LossWeights, info_nce_loss, masked_reconstruction_loss, multi_task_loss
from model_natron import NatronModel, init_model_from_config


LOGGER = logging.getLogger("natron.train")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def mask_inputs(batch: torch.Tensor, mask_ratio: float) -> Dict[str, torch.Tensor]:
    batch_size, seq_len, _ = batch.shape
    mask = torch.rand(batch_size, seq_len, device=batch.device) < mask_ratio
    masked = batch.clone()
    masked[mask] = 0.0
    return {"masked": masked, "mask": mask}


def augment_sequence(
    batch: torch.Tensor,
    jitter_std: float = 0.02,
    dropout_prob: float = 0.1,
    time_warp_sigma: float = 0.2,
) -> torch.Tensor:
    augmented = batch.clone()
    if jitter_std > 0:
        augmented = augmented + torch.randn_like(augmented) * jitter_std
    if dropout_prob > 0:
        drop_mask = torch.rand_like(augmented) < dropout_prob
        augmented = augmented.masked_fill(drop_mask, 0.0)
    if time_warp_sigma > 0:
        seq_len = augmented.size(1)
        max_shift = max(1, int(seq_len * time_warp_sigma))
        shifts = torch.randint(-max_shift, max_shift + 1, (augmented.size(0),), device=augmented.device)
        warped = torch.zeros_like(augmented)
        for idx, shift in enumerate(shifts):
            warped[idx] = torch.roll(augmented[idx], shifts=int(shift.item()), dims=0)
        augmented = warped
    return augmented


class NatronTrainer:
    def __init__(self, config: Dict, device: Optional[torch.device] = None):
        self.config = config
        seed = config.get("project", {}).get("seed", 42)
        set_seed(seed)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info("Using device: %s", self.device)

        self.data_module = create_data_module_from_config(config)
        self.data_module.prepare()

        self.model: NatronModel = init_model_from_config(config).to(self.device)
        self.loss_weights = LossWeights(**config.get("loss_weights", {}))

        sup_cfg = config.get("supervised", {})
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=sup_cfg.get("optimizer", {}).get("lr", 1e-4),
            weight_decay=sup_cfg.get("optimizer", {}).get("weight_decay", 1e-5),
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode=sup_cfg.get("scheduler", {}).get("mode", "max"),
            patience=sup_cfg.get("scheduler", {}).get("patience", 5),
            factor=sup_cfg.get("scheduler", {}).get("factor", 0.5),
            verbose=True,
        )

        self.best_metric = -float("inf")
        self.early_stop_counter = 0
        self.early_stop_patience = sup_cfg.get("early_stopping_patience", 10)

    def run(self) -> None:
        if self.config.get("pretraining", {}).get("enabled", True):
            self.run_pretraining()
        if self.config.get("supervised", {}).get("enabled", True):
            self.run_supervised()
        if self.config.get("rl", {}).get("enabled", False):
            self.run_reinforcement()
        self.save_model()

    def run_pretraining(self) -> None:
        LOGGER.info("Starting pretraining phase.")
        loaders = self.data_module.dataloaders(
            batch_size_sup=self.config.get("supervised", {}).get("batch_size", 48),
            batch_size_unsup=self.config.get("pretraining", {}).get("batch_size", 64),
        )
        unsup_loader = loaders.get("unsupervised")
        if unsup_loader is None:
            LOGGER.warning("Pretraining requested but unsupervised loader is not available.")
            return

        pre_cfg = self.config.get("pretraining", {})
        epochs = pre_cfg.get("epochs", 10)
        mask_ratio = pre_cfg.get("mask_ratio", 0.15)
        temp = pre_cfg.get("contrastive_temperature", 0.07)
        aug_cfg = self.config.get("data", {}).get("unsupervised_augmentations", {})

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=pre_cfg.get("optimizer", {}).get("lr", 3e-4),
            weight_decay=pre_cfg.get("optimizer", {}).get("weight_decay", 1e-5),
        )

        self.model.train()
        for epoch in range(1, epochs + 1):
            pbar = tqdm(unsup_loader, desc=f"[Pretrain] Epoch {epoch}/{epochs}")
            epoch_loss = 0.0
            for batch in pbar:
                batch = batch.to(self.device)
                masked_payload = mask_inputs(batch, mask_ratio)
                masked_inputs = masked_payload["masked"]
                mask = masked_payload["mask"]

                outputs = self.model(masked_inputs, return_token_embeddings=True)
                recon_loss = masked_reconstruction_loss(outputs["reconstruction"], batch, mask)

                view_a = augment_sequence(
                    batch,
                    jitter_std=aug_cfg.get("jitter_std", 0.02),
                    dropout_prob=aug_cfg.get("dropout_prob", 0.1),
                    time_warp_sigma=aug_cfg.get("time_warp_sigma", 0.2),
                )
                view_b = augment_sequence(
                    batch,
                    jitter_std=aug_cfg.get("jitter_std", 0.02),
                    dropout_prob=aug_cfg.get("dropout_prob", 0.1),
                    time_warp_sigma=aug_cfg.get("time_warp_sigma", 0.2),
                )
                proj_a = self.model(view_a)["projection"]
                proj_b = self.model(view_b)["projection"]
                contrastive_loss = info_nce_loss(proj_a, proj_b, temperature=temp)

                loss = self.loss_weights.masked * recon_loss + self.loss_weights.contrastive * contrastive_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), pre_cfg.get("grad_clip", 1.0))
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item(), recon=recon_loss.item(), contrast=contrastive_loss.item())

            LOGGER.info(
                "Pretraining epoch %d/%d complete. Avg loss: %.6f",
                epoch,
                epochs,
                epoch_loss / max(len(unsup_loader), 1),
            )

    def run_supervised(self) -> None:
        LOGGER.info("Starting supervised fine-tuning phase.")
        loaders = self.data_module.dataloaders(
            batch_size_sup=self.config.get("supervised", {}).get("batch_size", 48),
            batch_size_unsup=self.config.get("pretraining", {}).get("batch_size", 64)
            if self.config.get("pretraining", {}).get("enabled", True)
            else None,
        )
        train_loader = loaders["train"]
        val_loader = loaders["val"]

        sup_cfg = self.config.get("supervised", {})
        epochs = sup_cfg.get("epochs", 20)
        freeze_epochs = sup_cfg.get("freeze_encoder_epochs", 0)

        for epoch in range(1, epochs + 1):
            if epoch <= freeze_epochs:
                for param in self.model.encoder.parameters():
                    param.requires_grad = False
            elif epoch == freeze_epochs + 1:
                for param in self.model.encoder.parameters():
                    param.requires_grad = True

            train_metrics = self._run_supervised_epoch(train_loader, train=True)
            val_metrics = self._run_supervised_epoch(val_loader, train=False)
            self.scheduler.step(val_metrics["direction_acc"])

            LOGGER.info(
                "Epoch %d/%d | Train loss %.4f | Val loss %.4f | Val dir acc %.4f | Val regime acc %.4f",
                epoch,
                epochs,
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["direction_acc"],
                val_metrics["regime_acc"],
            )

            if val_metrics["direction_acc"] > self.best_metric:
                self.best_metric = val_metrics["direction_acc"]
                self.early_stop_counter = 0
                self.save_model(suffix="best")
            else:
                self.early_stop_counter += 1

            if self.early_stop_counter >= self.early_stop_patience:
                LOGGER.info("Early stopping triggered.")
                break

    def _run_supervised_epoch(self, loader, train: bool) -> Dict[str, float]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_samples = 0
        correct_direction = 0
        correct_regime = 0

        for batch in loader:
            sequence = batch["sequence"].to(self.device)
            targets = {
                "buy": batch["buy"].to(self.device),
                "sell": batch["sell"].to(self.device),
                "direction": batch["direction"].to(self.device),
                "regime": batch["regime"].to(self.device),
            }

            if train:
                self.optimizer.zero_grad()

            outputs = self.model(sequence)
            losses = multi_task_loss(outputs, targets, self.loss_weights)
            loss = losses["total"]

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            total_loss += loss.item() * sequence.size(0)
            total_samples += sequence.size(0)

            direction_pred = outputs["direction_logits"].argmax(dim=-1)
            regime_pred = outputs["regime_logits"].argmax(dim=-1)
            correct_direction += (direction_pred == targets["direction"]).sum().item()
            correct_regime += (regime_pred == targets["regime"]).sum().item()

        avg_loss = total_loss / max(total_samples, 1)
        direction_acc = correct_direction / max(total_samples, 1)
        regime_acc = correct_regime / max(total_samples, 1)

        return {
            "loss": avg_loss,
            "direction_acc": direction_acc,
            "regime_acc": regime_acc,
        }

    def run_reinforcement(self) -> None:
        from rl_trainer import PPOTrainer  # Lazy import to avoid dependency if not needed

        LOGGER.info("Starting reinforcement learning phase.")
        rl_cfg = self.config.get("rl", {})
        trainer = PPOTrainer(self.model, self.data_module, rl_cfg, device=self.device)
        trainer.train()

    def save_model(self, suffix: str = "final") -> None:
        model_dir = Path(self.config.get("paths", {}).get("model_dir", "model"))
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / f"natron_v2_{suffix}.pt"
        payload = {"model_state": self.model.state_dict(), "config": self.config}
        torch.save(payload, path)
        if self.data_module.scaler is not None:
            scaler_path = model_dir / "feature_scaler.pkl"
            joblib.dump(self.data_module.scaler, scaler_path)
        LOGGER.info("Saved model checkpoint to %s", path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Natron Transformer pipeline.")
    parser.add_argument("--config", type=str, default="configs/natron_config.yaml", help="Path to YAML configuration.")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu or cuda).")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    args = parse_args()
    config = load_config(Path(args.config))
    trainer = NatronTrainer(config, device=torch.device(args.device) if args.device else None)
    trainer.run()


if __name__ == "__main__":
    main()
