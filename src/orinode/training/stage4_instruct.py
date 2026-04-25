"""Stage 4: Instruction fine-tuning of the full Speech-LLM.

Adapter + LLM LoRA are jointly trained on instruction-formatted data.
Encoder remains frozen (DoRA weights from stage 1/2 are fixed).

Entry point: ``python -m orinode.training.stage4_instruct [--smoke-test]``
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


class Stage4Trainer(BaseTrainer):
    """Instruction fine-tune: adapter + LLM LoRA, encoder frozen."""

    stage_name = "stage4_instruct"

    def _build_model(self, cfg: DictConfig) -> nn.Module:
        if self.smoke_test:
            model = SpeechLLM.for_smoke_test()
            # Encoder frozen; adapter + decoder train
            for p in model.encoder.parameters():
                p.requires_grad_(False)
            return model

        from transformers import AutoModelForCausalLM

        from orinode.models.adapter import AudioLLMAdapter
        from orinode.models.lora_utils import LoRAConfig, apply_lora
        from orinode.models.whisper_encoder import WhisperEncoder

        encoder = WhisperEncoder.from_config(cfg.model.whisper_encoder)
        for p in encoder.parameters():
            p.requires_grad_(False)

        decoder = AutoModelForCausalLM.from_pretrained(
            cfg.model.decoder.model_id, torch_dtype=torch.bfloat16
        )
        lora_cfg = cfg.model.decoder.get("lora", {})
        if lora_cfg.get("enabled", True):
            lora = LoRAConfig(
                r=lora_cfg.get("r", 64),
                alpha=lora_cfg.get("alpha", 128),
                target_modules=list(
                    lora_cfg.get(
                        "target_modules",
                        ["q_proj", "k_proj", "v_proj", "o_proj"],
                    )
                ),
                dropout=lora_cfg.get("dropout", 0.05),
            )
            from peft import TaskType

            decoder = apply_lora(decoder, lora, task_type=TaskType.CAUSAL_LM)
            log.info(f"LoRA applied to decoder r={lora.r}")

        adapter = AudioLLMAdapter.from_config(
            cfg.model.adapter,
            decoder_hidden_size=decoder.config.hidden_size,
        )
        return SpeechLLM(encoder, adapter, decoder)

    def _build_dataloaders(self, cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
        if self.smoke_test:
            return self._smoke_loaders()
        raise NotImplementedError("Stage 4 real dataloader not yet implemented — use --smoke-test")

    @staticmethod
    def _smoke_loaders() -> tuple[DataLoader, DataLoader]:
        B, n_mels, T, S = 2, 128, 64, 6
        features = torch.randn(B, n_mels, T)
        input_ids = torch.randint(0, 256, (B, S))
        labels = input_ids.clone()
        labels[:, :2] = -100
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
    parser = argparse.ArgumentParser(description="Stage 4: Instruction fine-tuning")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    if args.smoke_test:
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "training": {
                    "lr": 2e-5,
                    "num_epochs": 1,
                    "grad_accum_steps": 1,
                    "max_grad_norm": 1.0,
                    "warmup_steps": 0,
                    "checkpoint_every_steps": 9999,
                },
                "run_id": "smoke_stage4",
            }
        )
    else:
        from orinode.utils.config import load_config

        cfg = load_config(args.config or Path("configs/training/stage4.yaml"))

    from orinode.paths import ensure_workspace

    ensure_workspace()

    trainer = Stage4Trainer(cfg, smoke_test=args.smoke_test)
    state = trainer.train()
    log.info(f"Stage 4 done  step={state.global_step}  " f"train_loss={state.train_loss:.4f}")


if __name__ == "__main__":
    main()
