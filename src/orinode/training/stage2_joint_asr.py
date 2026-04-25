"""Stage 2: Joint multilingual ASR fine-tuning.

Encoder (with DoRA) + decoder jointly trained on all five Nigerian languages
using temperature-weighted sampling.  Still uses
``WhisperForConditionalGeneration``; the LLM decoder is not involved yet.

Entry point: ``python -m orinode.training.stage2_joint_asr [--smoke-test]``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

from orinode.models.losses import whisper_seq2seq_loss
from orinode.training.trainer import BaseTrainer, TrainingState
from orinode.utils.logging import get_logger

log = get_logger(__name__)


class Stage2Trainer(BaseTrainer):
    """Joint multilingual Whisper fine-tuning with temperature sampling."""

    stage_name = "stage2_joint_asr"

    def _build_model(self, cfg: DictConfig) -> nn.Module:
        if self.smoke_test:
            from transformers import WhisperConfig, WhisperForConditionalGeneration

            wcfg = WhisperConfig(
                d_model=64,
                encoder_layers=2,
                encoder_attention_heads=4,
                encoder_ffn_dim=256,
                num_mel_bins=128,
                max_source_positions=32,
                decoder_layers=2,
                decoder_attention_heads=4,
                decoder_ffn_dim=256,
                max_target_positions=32,
                vocab_size=100,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
                decoder_start_token_id=1,
            )
            return WhisperForConditionalGeneration(wcfg)

        from transformers import WhisperForConditionalGeneration

        from orinode.models.lora_utils import LoRAConfig, apply_lora

        model = WhisperForConditionalGeneration.from_pretrained(
            cfg.model.whisper_encoder.model_id,
            torch_dtype=torch.bfloat16,
        )
        dora_cfg = cfg.model.whisper_encoder.get("dora", {})
        if dora_cfg.get("enabled", False):
            lora_cfg = LoRAConfig(
                r=dora_cfg.r,
                alpha=dora_cfg.get("alpha", dora_cfg.r * 2),
                target_modules=list(dora_cfg.target_modules),
                dropout=dora_cfg.get("dropout", 0.05),
                use_dora=True,
            )
            model.model.encoder = apply_lora(model.model.encoder, lora_cfg)
        return model

    def _build_dataloaders(self, cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
        if self.smoke_test:
            return self._smoke_loaders()
        raise NotImplementedError("Stage 2 real dataloader not yet implemented — use --smoke-test")

    @staticmethod
    def _smoke_loaders() -> tuple[DataLoader, DataLoader]:
        B, n_mels, T, S = 2, 128, 64, 8
        features = torch.randn(B, n_mels, T)
        labels = torch.randint(0, 100, (B, S))
        labels[:, -2:] = -100
        ds = TensorDataset(features, labels)
        loader = DataLoader(ds, batch_size=B)
        return loader, loader

    def _training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        model: nn.Module,
        state: TrainingState,
    ) -> torch.Tensor:
        features, labels = batch
        out = model(input_features=features, labels=labels)
        return whisper_seq2seq_loss(out)

    def _validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        model: nn.Module,
        state: TrainingState,
    ) -> dict[str, float]:
        features, labels = batch
        out = model(input_features=features, labels=labels)
        return {"val_loss": whisper_seq2seq_loss(out).item()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: Joint multilingual ASR")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    if args.smoke_test:
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "training": {
                    "lr": 5e-5,
                    "num_epochs": 1,
                    "grad_accum_steps": 1,
                    "max_grad_norm": 1.0,
                    "warmup_steps": 0,
                    "checkpoint_every_steps": 9999,
                },
                "run_id": "smoke_stage2",
            }
        )
    else:
        from orinode.utils.config import load_config

        cfg = load_config(args.config or Path("configs/training/stage2.yaml"))

    from orinode.paths import ensure_workspace

    ensure_workspace()

    trainer = Stage2Trainer(cfg, smoke_test=args.smoke_test)
    state = trainer.train()
    log.info(f"Stage 2 done  step={state.global_step}  " f"train_loss={state.train_loss:.4f}")


if __name__ == "__main__":
    main()
