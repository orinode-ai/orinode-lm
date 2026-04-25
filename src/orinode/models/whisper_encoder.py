"""WhisperEncoder wrapper with optional DoRA adapters.

Wraps ``transformers.WhisperModel.encoder`` so the rest of the codebase
can treat it as a single ``nn.Module`` that maps mel spectrograms to
encoder hidden states.

Stages 1 and 2 use ``WhisperForConditionalGeneration`` directly (via
``stage1_encoder.py`` / ``stage2_joint_asr.py``) for end-to-end seq2seq ASR.
This class is used in stages 3 and 4 where only the encoder output is
needed as input to the ``AudioLLMAdapter``.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig

from orinode.models.lora_utils import LoRAConfig, apply_lora
from orinode.utils.logging import get_logger

log = get_logger(__name__)


class WhisperEncoder(nn.Module):
    """Whisper encoder with optional DoRA adapters.

    Args:
        encoder: The ``transformers.WhisperEncoder`` module.
        output_dim: Hidden dimension of the encoder output (1280 for large-v3).
    """

    def __init__(self, encoder: nn.Module, output_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.output_dim = output_dim

    # ── factory methods ───────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        local_dir: str | Path | None = None,
        dora_config: LoRAConfig | None = None,
        freeze: bool = False,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> WhisperEncoder:
        """Load a Whisper encoder from HuggingFace or a local checkpoint.

        Args:
            model_id: HuggingFace model ID (e.g. ``openai/whisper-large-v3``).
            local_dir: Optional local directory; used as cache if provided.
            dora_config: DoRA adapter config. ``None`` = no adapters.
            freeze: Freeze all encoder parameters (for stage 3 / inference).
            torch_dtype: Weight dtype (bf16 on GPU, float32 for CPU testing).

        Returns:
            Initialised ``WhisperEncoder``.
        """
        from transformers import WhisperModel

        load_path = str(local_dir) if local_dir and Path(local_dir).exists() else model_id
        log.info(f"Loading Whisper encoder from {load_path}")

        whisper = WhisperModel.from_pretrained(load_path, torch_dtype=torch_dtype)
        encoder = whisper.encoder
        output_dim: int = whisper.config.d_model
        del whisper  # free decoder weights

        if dora_config is not None and dora_config.enabled:
            log.info(
                f"Applying DoRA r={dora_config.r} to encoder "
                f"modules: {dora_config.target_modules}"
            )
            encoder = apply_lora(encoder, dora_config)

        if freeze:
            for p in encoder.parameters():
                p.requires_grad_(False)
            log.info("Encoder frozen")

        return cls(encoder, output_dim)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> WhisperEncoder:
        """Build from an OmegaConf config node (``configs/model/whisper_encoder.yaml``).

        Args:
            cfg: Config with keys: ``model_id``, ``local_dir``, ``freeze``,
                ``dora`` (sub-node with ``enabled``, ``r``, etc.).

        Returns:
            Initialised ``WhisperEncoder``.
        """
        dora_cfg: LoRAConfig | None = None
        if cfg.dora.get("enabled", False):
            dora_cfg = LoRAConfig(
                r=cfg.dora.r,
                alpha=cfg.dora.get("alpha", cfg.dora.r * 2),
                target_modules=list(cfg.dora.target_modules),
                dropout=cfg.dora.get("dropout", 0.05),
                use_dora=True,
            )
        return cls.from_pretrained(
            model_id=cfg.model_id,
            local_dir=cfg.get("local_dir"),
            dora_config=dora_cfg,
            freeze=cfg.get("freeze", False),
        )

    @classmethod
    def for_smoke_test(cls, n_mels: int = 128) -> WhisperEncoder:
        """Create a tiny random-weight encoder for CPU smoke testing.

        No weights are downloaded. The architecture has 2 transformer layers
        and ``d_model=64`` so the forward pass completes in ~100 ms on CPU.

        Args:
            n_mels: Number of mel bins (must match the smoke-test input).

        Returns:
            Tiny ``WhisperEncoder`` with random weights.
        """
        from transformers import WhisperConfig, WhisperModel

        cfg = WhisperConfig(
            d_model=64,
            encoder_layers=2,
            encoder_attention_heads=4,
            encoder_ffn_dim=256,
            num_mel_bins=n_mels,
            max_source_positions=32,
            decoder_layers=2,
            decoder_attention_heads=4,
            decoder_ffn_dim=256,
            max_target_positions=32,
            vocab_size=100,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        return cls(WhisperModel(cfg).encoder, output_dim=64)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode mel spectrograms to encoder hidden states.

        Args:
            input_features: Log-mel spectrogram ``(B, n_mels, T)``.
                For production (whisper-large-v3): T=3000 (30 s padded).
                For smoke tests: T=64.
            attention_mask: Optional padding mask ``(B, T)``; ``1``=real,
                ``0``=pad. Pass when batches have variable-length audio.

        Returns:
            Encoder hidden states ``(B, T//2, output_dim)``.
        """
        out = self.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
        )
        return out.last_hidden_state
