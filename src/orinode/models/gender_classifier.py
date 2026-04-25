"""Gender classifier: wav2vec2-base encoder + linear head.

Architecture:
    wav2vec2-base (frozen)  →  mean-pool time dim  →  Linear(768, 2)  →  softmax
    Output: [p_male, p_female]

Expected accuracy: >95% clean speech, >90% 8 kHz telephony.
Checkpoint path: workspace/models/checkpoints/aux_gender/
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


class GenderClassifier(nn.Module):
    """wav2vec2-base backbone with a 2-class linear head.

    Args:
        encoder: HuggingFace ``Wav2Vec2Model`` (frozen by default).
        hidden_size: Encoder hidden dimension (768 for wav2vec2-base).
        dropout: Dropout before the head.
    """

    NUM_CLASSES = 2
    LABELS = ["male", "female"]

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int = 768,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.NUM_CLASSES),
        )

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "facebook/wav2vec2-base",
        freeze_encoder: bool = True,
        dropout: float = 0.1,
        torch_dtype: torch.dtype = torch.float32,
    ) -> GenderClassifier:
        from transformers import Wav2Vec2Model

        encoder = Wav2Vec2Model.from_pretrained(model_id, torch_dtype=torch_dtype)
        hidden_size: int = encoder.config.hidden_size

        if freeze_encoder:
            for p in encoder.parameters():
                p.requires_grad_(False)

        return cls(encoder, hidden_size=hidden_size, dropout=dropout)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> GenderClassifier:
        return cls.from_pretrained(
            model_id=cfg.get("model_id", "facebook/wav2vec2-base"),
            freeze_encoder=cfg.get("freeze_encoder", True),
            dropout=cfg.get("dropout", 0.1),
        )

    @classmethod
    def for_smoke_test(cls) -> GenderClassifier:
        """Tiny random-weight model; no downloads needed."""
        from transformers import Wav2Vec2Config, Wav2Vec2Model

        cfg = Wav2Vec2Config(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            conv_dim=(64,) * 7,
        )
        encoder = Wav2Vec2Model(cfg)
        return cls(encoder, hidden_size=64, dropout=0.0)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Classify gender from raw waveform.

        Args:
            input_values: Raw waveform ``(B, T)`` normalised to [-1, 1].
            attention_mask: Optional padding mask ``(B, T)``.

        Returns:
            Log-probabilities ``(B, 2)`` — [log_p_male, log_p_female].
        """
        out = self.encoder(
            input_values=input_values,
            attention_mask=attention_mask,
        )
        hidden = out.last_hidden_state  # (B, T', H)

        # Mean-pool over time (masked if attention_mask provided)
        if attention_mask is not None:
            # Wav2Vec2 subsamples by ~320x; approximate pooling mask
            pool_mask = attention_mask[:, ::320].float()
            pool_mask = pool_mask[:, : hidden.size(1)]
            pool_mask = pool_mask.unsqueeze(-1)
            pooled = (hidden * pool_mask).sum(1) / pool_mask.sum(1).clamp(min=1)
        else:
            pooled = hidden.mean(dim=1)  # (B, H)

        logits = self.head(pooled)  # (B, 2)
        return torch.log_softmax(logits, dim=-1)
