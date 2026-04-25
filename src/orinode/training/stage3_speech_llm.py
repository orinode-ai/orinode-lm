"""Stage 3: Speech-LLM bridge training.

Trains only the AudioLLMAdapter (frozen encoder, frozen LLM decoder).
Goal: teach the adapter to compress Whisper tokens into a representation
the LLM can consume.

Entry point: ``python -m orinode.training.stage3_speech_llm [--smoke-test]``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

from orinode.models.speech_llm import SpeechLLM
from orinode.training.trainer import BaseTrainer, TrainingState
from orinode.utils.logging import get_logger

log = get_logger(__name__)


class Stage3Trainer(BaseTrainer):
    """Train only the adapter; encoder and decoder are frozen."""

    stage_name = "stage3_speech_llm"

    def _build_model(self, cfg: DictConfig) -> nn.Module:
        if self.smoke_test:
            model = SpeechLLM.for_smoke_test()
            # Freeze encoder and decoder; only adapter trains
            for p in model.encoder.parameters():
                p.requires_grad_(False)
            for p in model.decoder.parameters():
                p.requires_grad_(False)
            return model

        from transformers import AutoModelForCausalLM

        from orinode.models.adapter import AudioLLMAdapter
        from orinode.models.whisper_encoder import WhisperEncoder

        encoder = WhisperEncoder.from_config(cfg.model.whisper_encoder)
        for p in encoder.parameters():
            p.requires_grad_(False)

        decoder = AutoModelForCausalLM.from_pretrained(
            cfg.model.decoder.model_id, torch_dtype=torch.bfloat16
        )
        for p in decoder.parameters():
            p.requires_grad_(False)

        adapter = AudioLLMAdapter.from_config(
            cfg.model.adapter,
            decoder_hidden_size=decoder.config.hidden_size,
        )
        return SpeechLLM(encoder, adapter, decoder)

    def _build_dataloaders(self, cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
        if self.smoke_test:
            return self._smoke_loaders()
        raise NotImplementedError("Stage 3 real dataloader not yet implemented — use --smoke-test")

    @staticmethod
    def _smoke_loaders() -> tuple[DataLoader, DataLoader]:
        B, n_mels, T, S = 2, 128, 64, 6
        features = torch.randn(B, n_mels, T)
        input_ids = torch.randint(0, 256, (B, S))
        labels = input_ids.clone()
        labels[:, :2] = -100  # ignore first 2 tokens
        ds = TensorDataset(features, input_ids, labels)
        loader = DataLoader(ds, batch_size=B)
        return loader, loader

    def _training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        model: nn.Module,
        state: TrainingState,
    ) -> torch.Tensor:
        features, input_ids, labels = batch
        out = model(input_features=features, input_ids=input_ids, labels=labels)
        return out.loss

    def _validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        model: nn.Module,
        state: TrainingState,
    ) -> dict[str, float]:
        features, input_ids, labels = batch
        out = model(input_features=features, input_ids=input_ids, labels=labels)
        return {"val_loss": out.loss.item()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: Speech-LLM bridge")
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
                "run_id": "smoke_stage3",
            }
        )
    else:
        from orinode.utils.config import load_config

        cfg = load_config(args.config or Path("configs/training/stage3.yaml"))

    from orinode.paths import ensure_workspace

    ensure_workspace()

    trainer = Stage3Trainer(cfg, smoke_test=args.smoke_test)
    state = trainer.train()
    log.info(f"Stage 3 done  step={state.global_step}  " f"train_loss={state.train_loss:.4f}")


if __name__ == "__main__":
    main()
