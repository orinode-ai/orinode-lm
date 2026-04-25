"""Stage 1: Whisper encoder adaptation via seq2seq ASR fine-tuning.

Only the Whisper encoder is trained (with DoRA adapters).
The decoder, LLM adapter, and LLM are not involved.

Entry points:
    python -m orinode.training.stage1_encoder --smoke-test       # tiny synthetic model
    python -m orinode.training.stage1_encoder --smoke-test-real  # real model, 8 clips
    python -m orinode.training.stage1_encoder --config <path>    # full training run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset, TensorDataset

from orinode.models.losses import whisper_seq2seq_loss
from orinode.training.trainer import BaseTrainer, TrainingState
from orinode.utils.logging import get_logger

log = get_logger(__name__)


class Stage1Trainer(BaseTrainer):
    """Seq2seq ASR fine-tuning of the Whisper encoder.

    Supports three modes:
    - smoke_test=True          — tiny synthetic Whisper (2 steps, no GPU needed)
    - smoke_test_real=True     — real Whisper large-v3, 8 real clips, 2 steps
    - normal run               — full training from config
    """

    stage_name = "stage1_encoder"

    def __init__(
        self, cfg: DictConfig, smoke_test: bool = False, smoke_test_real: bool = False
    ) -> None:
        self.smoke_test_real = smoke_test_real
        # Pass smoke_test=True to BaseTrainer for both smoke modes so _smoke_train is used
        super().__init__(cfg, smoke_test=smoke_test or smoke_test_real)

    # ── model ──────────────────────────────────────────────────────────────────

    def _build_model(self, cfg: DictConfig) -> nn.Module:
        if self.smoke_test and not self.smoke_test_real:
            return self._smoke_model()

        from transformers import WhisperForConditionalGeneration

        from orinode.models.lora_utils import LoRAConfig, apply_lora, log_parameter_counts

        # attn_implementation lives under model.encoder in stage1 config
        model_enc_cfg = cfg.get("model", {}).get("encoder", {})
        attn_impl = model_enc_cfg.get("attn_implementation", "sdpa")

        model = WhisperForConditionalGeneration.from_pretrained(
            cfg.encoder.model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
        log.info(f"Loaded {cfg.encoder.model_id}  attn={attn_impl}")

        # Freeze decoder — stage 1 trains encoder only
        for p in model.model.decoder.parameters():
            p.requires_grad_(False)
        model.proj_out.requires_grad_(False)
        log.info("Decoder frozen")

        # Apply DoRA to encoder
        dora_cfg = cfg.encoder.get("dora", {})
        if dora_cfg.get("enabled", False):
            lora_config = LoRAConfig(
                r=int(dora_cfg.r),
                alpha=int(dora_cfg.get("alpha", dora_cfg.r * 2)),
                target_modules=list(dora_cfg.target_modules),
                dropout=float(dora_cfg.get("dropout", 0.05)),
                use_dora=True,
            )
            model.model.encoder = apply_lora(model.model.encoder, lora_config)
            log.info("DoRA applied to Whisper encoder")

        # Gradient checkpointing
        if cfg.training.get("gradient_checkpointing", False):
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
            log.info("Gradient checkpointing enabled")

        log_parameter_counts(model, log)
        return model

    @staticmethod
    def _smoke_model() -> nn.Module:
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

    # ── dataloaders ────────────────────────────────────────────────────────────

    def _build_dataloaders(self, cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
        if self.smoke_test and not self.smoke_test_real:
            return self._smoke_loaders()

        from orinode.data.whisper_collator import WhisperDataCollator
        from orinode.data.whisper_dataset import WhisperDataset, WhisperDatasetConfig

        data_cfg = cfg.data
        processor_name = cfg.encoder.model_id
        sample_rate = int(data_cfg.get("sample_rate", 16_000))
        max_duration = float(data_cfg.get("max_duration", 30.0))

        train_ds_cfg = WhisperDatasetConfig(
            manifest_path=Path(str(data_cfg.train_manifest)),
            processor_name=processor_name,
            sample_rate=sample_rate,
            max_audio_seconds=max_duration,
        )
        dev_ds_cfg = WhisperDatasetConfig(
            manifest_path=Path(str(data_cfg.dev_manifest)),
            processor_name=processor_name,
            sample_rate=sample_rate,
            max_audio_seconds=max_duration,
        )

        train_ds = WhisperDataset(train_ds_cfg)
        dev_ds = WhisperDataset(dev_ds_cfg)

        if self.smoke_test_real:
            train_ds = Subset(train_ds, list(range(min(8, len(train_ds)))))
            dev_ds = Subset(dev_ds, list(range(min(4, len(dev_ds)))))
            log.info(
                f"smoke-test-real: {len(train_ds)} train clips, {len(dev_ds)} dev clips"
            )

        collator = WhisperDataCollator()
        dl_cfg = cfg.get("dataloader", {})
        num_workers = int(dl_cfg.get("num_workers", 4))
        common = dict(
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=bool(dl_cfg.get("pin_memory", True)),
            persistent_workers=bool(dl_cfg.get("persistent_workers", True)) and num_workers > 0,
        )
        batch_size = int(cfg.training.get("batch_size_per_gpu", 8))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common)
        dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, **common)
        log.info(f"Train batches: {len(train_loader)}  Dev batches: {len(dev_loader)}")
        return train_loader, dev_loader

    @staticmethod
    def _smoke_loaders() -> tuple[DataLoader, DataLoader]:
        """Tiny synthetic loaders for smoke testing (no audio files or HF models needed)."""
        B, n_mels, T, S = 2, 128, 64, 8
        features = torch.randn(B, n_mels, T)
        labels = torch.randint(0, 100, (B, S))
        labels[:, -2:] = -100
        ds = TensorDataset(features, labels)
        loader = DataLoader(ds, batch_size=B)
        return loader, loader

    # ── training / validation steps ────────────────────────────────────────────

    def _training_step(
        self,
        batch: dict | tuple,
        model: nn.Module,
        state: TrainingState,
    ) -> torch.Tensor:
        if isinstance(batch, (tuple, list)):
            input_features, labels = batch
        else:
            input_features = batch["input_features"]
            labels = batch["labels"]
        out = model(input_features=input_features, labels=labels)
        return whisper_seq2seq_loss(out)

    def _validation_step(
        self,
        batch: dict | tuple,
        model: nn.Module,
        state: TrainingState,
    ) -> dict[str, float]:
        if isinstance(batch, (tuple, list)):
            input_features, labels = batch
        else:
            input_features = batch["input_features"]
            labels = batch["labels"]
        out = model(input_features=input_features, labels=labels)
        return {"val_loss": whisper_seq2seq_loss(out).item()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: Whisper encoder ASR")
    parser.add_argument("--smoke-test", action="store_true", help="Tiny synthetic run (no GPU)")
    parser.add_argument(
        "--smoke-test-real",
        action="store_true",
        help="Real Whisper large-v3, 8 train clips, 2 steps — validates VRAM budget",
    )
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    if args.smoke_test or args.smoke_test_real:
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "encoder": {
                    "model_id": "openai/whisper-large-v3",
                    "dora": {"enabled": False},
                },
                "model": {
                    "encoder": {"attn_implementation": "sdpa"},
                },
                "data": {
                    "sample_rate": 16000,
                    "max_duration": 30.0,
                    "train_manifest": "workspace/data/manifests/afrispeech_200_train.jsonl",
                    "dev_manifest": "workspace/data/manifests/afrispeech_200_dev.jsonl",
                },
                "training": {
                    "lr": 1e-4,
                    "num_epochs": 1,
                    "grad_accum_steps": 1,
                    "max_grad_norm": 1.0,
                    "warmup_steps": 0,
                    "checkpoint_every_steps": 9999,
                    "batch_size_per_gpu": 2,
                    "gradient_checkpointing": False,
                },
                "dataloader": {
                    "num_workers": 0,
                    "pin_memory": False,
                    "persistent_workers": False,
                },
                "run_id": "smoke_stage1" if args.smoke_test else "smoke_real_stage1",
            }
        )
    else:
        from orinode.utils.config import load_config

        cfg = load_config(
            args.config or Path("configs/training/stage1_encoder_adapt.yaml")
        )

    from orinode.paths import ensure_workspace

    ensure_workspace()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    trainer = Stage1Trainer(cfg, smoke_test=args.smoke_test, smoke_test_real=args.smoke_test_real)
    state = trainer.train()

    log.info(
        f"Stage 1 done  step={state.global_step}  train_loss={state.train_loss:.4f}"
    )

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        log.info(f"Peak GPU memory: {peak_gb:.2f} GB")


if __name__ == "__main__":
    main()
