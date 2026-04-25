"""Stage aux-E: emotion classifier training.

Backbone: wav2vec2-large-xlsr-53 (frozen, multilingual)
Head:     mean-pool → Linear(1024, 4) → softmax
Labels:   [happy, angry, sad, neutral]

Data strategy (see scripts/data/prepare_emotion_labels.py for detail):
  Strategy 1 (default): IEMOCAP + RAVDESS English transfer — auto-executed.
  Strategy 2: Nollywood scene-labeled audio — documented, manual curation needed.
  Strategy 3: Crowdsource emotion labels on NaijaVoices — future work.

The UI always shows a "Preview" badge until strategy 2 or 3 data lands.
See docs/AUX_MODELS.md for the full methodology and accuracy numbers.

Entry point: python -m orinode.training.train_emotion [--smoke-test] [--config ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

from orinode.models.emotion_classifier import EmotionClassifier
from orinode.training.trainer import BaseTrainer, TrainingState
from orinode.utils.logging import get_logger

log = get_logger(__name__)


class EmotionTrainer(BaseTrainer):
    """Train the emotion classifier head (encoder frozen)."""

    stage_name = "aux_emotion"

    def _build_model(self, cfg: DictConfig) -> nn.Module:
        if self.smoke_test:
            return EmotionClassifier.for_smoke_test()
        return EmotionClassifier.from_config(cfg.model)

    def _build_dataloaders(self, cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
        if self.smoke_test:
            return self._smoke_loaders()
        raise NotImplementedError("Emotion dataloader not yet implemented — use --smoke-test")

    @staticmethod
    def _smoke_loaders() -> tuple[DataLoader, DataLoader]:
        B, T = 2, 3200
        waveforms = torch.randn(B, T)
        labels = torch.tensor([0, 3])  # happy, neutral
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
    parser = argparse.ArgumentParser(description="Emotion classifier training")
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
                "run_id": "smoke_aux_emotion",
            }
        )
    else:
        from orinode.utils.config import load_config

        cfg = load_config(args.config or Path("configs/training/aux_emotion_classifier.yaml"))

    from orinode.paths import ensure_workspace

    ensure_workspace()
    trainer = EmotionTrainer(cfg, smoke_test=args.smoke_test)
    state = trainer.train()
    log.info(
        f"Emotion training done  step={state.global_step}  " f"train_loss={state.train_loss:.4f}"
    )


if __name__ == "__main__":
    main()
