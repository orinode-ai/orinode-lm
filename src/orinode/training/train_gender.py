"""Stage aux-G: gender classifier training.

Backbone: wav2vec2-base (frozen)
Head:     mean-pool → Linear(768, 2) → softmax
Data:     NaijaVoices filtered to speakers with gender metadata
          ~10 h balanced (5 h male, 5 h female), pooled across all languages
Eval:     accuracy + F1 on held-out speakers (no speaker overlap)
Target:   >95% clean speech, >90% 8 kHz telephony

Entry point: python -m orinode.training.train_gender [--smoke-test] [--config ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

from orinode.models.gender_classifier import GenderClassifier
from orinode.training.trainer import BaseTrainer, TrainingState
from orinode.utils.logging import get_logger

log = get_logger(__name__)

_SAMPLE_RATE = 16_000


class GenderTrainer(BaseTrainer):
    """Train the gender classifier head (encoder frozen)."""

    stage_name = "aux_gender"

    def _build_model(self, cfg: DictConfig) -> nn.Module:
        if self.smoke_test:
            return GenderClassifier.for_smoke_test()
        return GenderClassifier.from_config(cfg.model)

    def _build_dataloaders(self, cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
        if self.smoke_test:
            return self._smoke_loaders()
        raise NotImplementedError("Gender dataloader not yet implemented — use --smoke-test")

    @staticmethod
    def _smoke_loaders() -> tuple[DataLoader, DataLoader]:
        B, T = 2, 3200  # 0.2 s at 16 kHz
        waveforms = torch.randn(B, T)
        labels = torch.tensor([0, 1])  # male, female
        ds = TensorDataset(waveforms, labels)
        loader = DataLoader(ds, batch_size=B)
        return loader, loader

    def _training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        model: nn.Module,
        state: TrainingState,
    ) -> torch.Tensor:
        waveforms, labels = batch
        log_probs = model(input_values=waveforms)
        return F.nll_loss(log_probs, labels)

    def _validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        model: nn.Module,
        state: TrainingState,
    ) -> dict[str, float]:
        waveforms, labels = batch
        log_probs = model(input_values=waveforms)
        loss = F.nll_loss(log_probs, labels).item()
        preds = log_probs.argmax(dim=-1)
        acc = (preds == labels).float().mean().item()
        return {"val_loss": loss, "accuracy": acc}


def main() -> None:
    parser = argparse.ArgumentParser(description="Gender classifier training")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    if args.smoke_test:
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "training": {
                    "lr": 1e-3,
                    "num_epochs": 1,
                    "grad_accum_steps": 1,
                    "max_grad_norm": 1.0,
                    "warmup_steps": 0,
                    "checkpoint_every_steps": 9999,
                },
                "run_id": "smoke_aux_gender",
            }
        )
    else:
        from orinode.utils.config import load_config

        cfg = load_config(args.config or Path("configs/training/aux_gender_classifier.yaml"))

    from orinode.paths import ensure_workspace

    ensure_workspace()
    trainer = GenderTrainer(cfg, smoke_test=args.smoke_test)
    state = trainer.train()
    log.info(
        f"Gender training done  step={state.global_step}  " f"train_loss={state.train_loss:.4f}"
    )


if __name__ == "__main__":
    main()
