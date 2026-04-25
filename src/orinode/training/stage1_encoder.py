"""Stage 1: Whisper encoder adaptation via seq2seq ASR fine-tuning.

Only the Whisper encoder is trained (with DoRA adapters).
The decoder, LLM adapter, and LLM are not involved.

Entry points:
    python -m orinode.training.stage1_encoder --smoke-test            # tiny synthetic model
    python -m orinode.training.stage1_encoder --smoke-test-real       # real model, 8 clips, 2 steps
    python -m orinode.training.stage1_encoder --smoke-test-real-full  # batch=16, DoRA, 4 opt steps
    python -m orinode.training.stage1_encoder --config <path>         # full training run
"""

from __future__ import annotations

import argparse
import math
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

    Smoke modes:
        smoke_test=True           — tiny synthetic Whisper (no GPU needed)
        smoke_test_real=True      — real Whisper large-v3, 8 clips, 2 steps
        smoke_test_real_full=True — batch=16, DoRA, grad-ckpt, 4 optimizer steps
    """

    stage_name = "stage1_encoder"

    def __init__(
        self,
        cfg: DictConfig,
        smoke_test: bool = False,
        smoke_test_real: bool = False,
        smoke_test_real_full: bool = False,
    ) -> None:
        self.smoke_test_real = smoke_test_real
        self.smoke_test_real_full = smoke_test_real_full
        self._processor = None  # set in _build_dataloaders for WER computation
        # smoke_test_real uses _smoke_train (2 steps, no Accelerate)
        # smoke_test_real_full uses the real Accelerate path with max_steps=4
        super().__init__(cfg, smoke_test=smoke_test or smoke_test_real)

    # ── model ──────────────────────────────────────────────────────────────────

    def _build_model(self, cfg: DictConfig) -> nn.Module:
        if self.smoke_test and not self.smoke_test_real:
            return self._smoke_model()

        from transformers import WhisperForConditionalGeneration

        from orinode.models.lora_utils import LoRAConfig, apply_lora, log_parameter_counts

        model_enc_cfg = cfg.get("model", {}).get("encoder", {})
        attn_impl = model_enc_cfg.get("attn_implementation", "sdpa")

        try:
            model = WhisperForConditionalGeneration.from_pretrained(
                cfg.encoder.model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl,
            )
        except ImportError:
            log.warning(f"attn={attn_impl} unavailable (missing package), falling back to sdpa")
            attn_impl = "sdpa"
            model = WhisperForConditionalGeneration.from_pretrained(
                cfg.encoder.model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
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

        if cfg.training.get("gradient_checkpointing", False):
            # enable_input_require_grads() is required when using PEFT + gradient
            # checkpointing. Without it, checkpointed layers recompute without a
            # connected grad_fn and loss.backward() raises "does not require grad".
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.config.use_cache = False
            log.info("Gradient checkpointing enabled")

        log_parameter_counts(model, log)

        # For smoke-test-real: move to GPU so VRAM is measured accurately.
        # Normal runs and smoke-test-real-full use Accelerate.prepare() for placement.
        if self.smoke_test_real and torch.cuda.is_available():
            model = model.to("cuda")
            log.info("Model moved to CUDA for smoke-test-real")

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

        train_ds = WhisperDataset(
            WhisperDatasetConfig(
                manifest_path=Path(str(data_cfg.train_manifest)),
                processor_name=processor_name,
                sample_rate=sample_rate,
                max_audio_seconds=max_duration,
            )
        )
        dev_ds = WhisperDataset(
            WhisperDatasetConfig(
                manifest_path=Path(str(data_cfg.dev_manifest)),
                processor_name=processor_name,
                sample_rate=sample_rate,
                max_audio_seconds=max_duration,
            )
        )
        self._processor = train_ds.processor  # store for WER computation

        if self.smoke_test_real:
            train_ds = Subset(train_ds, list(range(min(8, len(train_ds)))))
            dev_ds = Subset(dev_ds, list(range(min(4, len(dev_ds)))))
            log.info(f"smoke-test-real: {len(train_ds)} train, {len(dev_ds)} dev clips")
        elif self.smoke_test_real_full:
            train_ds = Subset(train_ds, list(range(min(32, len(train_ds)))))
            dev_ds = Subset(dev_ds, list(range(min(16, len(dev_ds)))))
            log.info(f"smoke-test-real-full: {len(train_ds)} train, {len(dev_ds)} dev clips")

        dl_cfg = cfg.get("dataloader", {})
        num_workers = int(dl_cfg.get("num_workers", 4))
        common = dict(
            collate_fn=WhisperDataCollator(),
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
        """Tiny synthetic loaders for smoke testing (no audio or HF models needed)."""
        B, n_mels, T, S = 2, 128, 64, 8
        features = torch.randn(B, n_mels, T)
        labels = torch.randint(0, 100, (B, S))
        labels[:, -2:] = -100
        ds = TensorDataset(features, labels)
        loader = DataLoader(ds, batch_size=B)
        return loader, loader

    # ── WER validation ─────────────────────────────────────────────────────────

    def _run_validation(
        self, loader: DataLoader, model: nn.Module
    ) -> dict[str, float]:
        """Override: compute val_loss (base) + WER (via generate) when processor is available."""
        metrics = super()._run_validation(loader, model)

        if self._processor is None:
            return metrics

        # WER — greedy decode on up to max_wer_batches batches
        max_wer_batches = int(self.cfg.training.get("max_wer_batches_at_eval", 20))
        try:
            from jiwer import wer as compute_wer

            hypotheses: list[str] = []
            references: list[str] = []
            tok = self._processor.tokenizer

            for i, batch in enumerate(loader):
                if i >= max_wer_batches:
                    break
                input_features, labels_batch = self._unpack_batch(batch, model)

                with torch.no_grad():
                    pred_ids = model.generate(
                        input_features,
                        language="en",
                        task="transcribe",
                        forced_decoder_ids=None,  # avoid conflict with processor presets
                        num_beams=1,              # greedy for speed
                    )

                hyps = tok.batch_decode(pred_ids, skip_special_tokens=True)
                refs = []
                for row in labels_batch:
                    valid = row[row != -100]
                    refs.append(tok.decode(valid, skip_special_tokens=True))

                hypotheses.extend(hyps)
                references.extend(refs)

            if references:
                wer_value = compute_wer(references, hypotheses)
                metrics["wer"] = wer_value
                log.info(f"Dev WER: {wer_value:.4f} ({len(references)} utterances)")
        except Exception as exc:
            log.warning(f"WER computation failed: {exc}")

        return metrics

    # ── training / validation steps ────────────────────────────────────────────

    @staticmethod
    def _unpack_batch(
        batch: dict | tuple,
        model: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Unpack batch and move to model device/dtype. Safe no-op under Accelerate."""
        if isinstance(batch, (tuple, list)):
            input_features, labels = batch
        else:
            input_features = batch["input_features"]
            labels = batch["labels"]
        p = next(model.parameters())
        input_features = input_features.to(device=p.device, dtype=p.dtype)
        labels = labels.to(device=p.device)
        return input_features, labels

    def _training_step(
        self,
        batch: dict | tuple,
        model: nn.Module,
        state: TrainingState,
    ) -> torch.Tensor:
        input_features, labels = self._unpack_batch(batch, model)
        out = model(input_features=input_features, labels=labels)
        return whisper_seq2seq_loss(out)

    def _validation_step(
        self,
        batch: dict | tuple,
        model: nn.Module,
        state: TrainingState,
    ) -> dict[str, float]:
        input_features, labels = self._unpack_batch(batch, model)
        out = model(input_features=input_features, labels=labels)
        return {"val_loss": whisper_seq2seq_loss(out).item()}


# ── helpers shared by smoke configs ───────────────────────────────────────────

def _base_smoke_cfg(run_name: str) -> dict:
    return {
        "encoder": {
            "model_id": "openai/whisper-large-v3",
            "dora": {"enabled": False},
        },
        "model": {"encoder": {"attn_implementation": "sdpa"}},
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
        "run_name": run_name,  # BaseTrainer generates unique run_id from this
    }


_CONFIGS_DIR = Path(__file__).parents[3] / "configs"
_DEFAULT_CFG = _CONFIGS_DIR / "training" / "stage1_encoder_adapt.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: Whisper encoder ASR")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-test-real", action="store_true",
                        help="Real Whisper large-v3, 8 clips, 2 steps")
    parser.add_argument("--smoke-test-real-full", action="store_true",
                        help="batch=16, DoRA, grad-ckpt, 4 optimizer steps — VRAM validation")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    from omegaconf import OmegaConf

    from orinode.paths import ensure_workspace

    ensure_workspace()

    def _load(path: Path = _DEFAULT_CFG):
        from orinode.utils.config import load_config
        return load_config(path.resolve())

    if args.smoke_test:
        cfg = OmegaConf.create(_base_smoke_cfg("smoke_stage1_encoder"))
        trainer = Stage1Trainer(cfg, smoke_test=True)

    elif args.smoke_test_real:
        cfg = OmegaConf.create(_base_smoke_cfg("smoke_stage1_encoder_real"))
        trainer = Stage1Trainer(cfg, smoke_test_real=True)

    elif args.smoke_test_real_full:
        cfg = _load()
        cfg = OmegaConf.merge(cfg, OmegaConf.create({
            "model": {
                "encoder": {"attn_implementation": "sdpa"},
                "decoder": {"attn_implementation": "sdpa"},
            },
            "training": {
                "max_steps": 4,
                "eval_interval": 0,
                "checkpoint_interval": 9999,
                "batch_size_per_gpu": 16,
                "gradient_checkpointing": True,
            },
            "dataloader": {
                "num_workers": 4,
                "pin_memory": True,
                "persistent_workers": True,
            },
            "run_name": "smoke_stage1_encoder_real_full",
        }))
        trainer = Stage1Trainer(cfg, smoke_test_real_full=True)

    else:
        cfg = _load(args.config.resolve() if args.config else _DEFAULT_CFG)
        trainer = Stage1Trainer(cfg)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    state = trainer.train()
    log.info(f"Stage 1 done  step={state.global_step}  train_loss={state.train_loss:.4f}")

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        log.info(f"Peak GPU memory: {peak_gb:.2f} GB")


if __name__ == "__main__":
    main()
