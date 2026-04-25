"""Emotion classifier: wav2vec2-large-xlsr-53 encoder + linear head.

Architecture:
    wav2vec2-large-xlsr-53 (frozen)  →  mean-pool  →  Linear(1024, 4)  →  softmax
    Output: [p_happy, p_angry, p_sad, p_neutral]

Data honesty note:
    Nigerian emotion corpora are scarce. This model is initially trained on
    English transfer data (IEMOCAP + RAVDESS). The UI **always** shows a
    "Preview" badge and methodology disclaimer until real Nigerian emotional
    speech is incorporated. See docs/AUX_MODELS.md for the full roadmap.

Checkpoint path: workspace/models/checkpoints/aux_emotion/
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


class EmotionClassifier(nn.Module):
    """wav2vec2-large-xlsr-53 backbone with a 4-class linear head.

    Args:
        encoder: HuggingFace ``Wav2Vec2Model`` (multilingual, frozen by default).
        hidden_size: Encoder hidden dimension (1024 for xlsr-53).
        dropout: Dropout before the head.
    """

    NUM_CLASSES = 4
    LABELS = ["happy", "angry", "sad", "neutral"]

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int = 1024,
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
        model_id: str = "facebook/wav2vec2-large-xlsr-53",
        freeze_encoder: bool = True,
        dropout: float = 0.1,
        torch_dtype: torch.dtype = torch.float32,
    ) -> EmotionClassifier:
        from transformers import Wav2Vec2Model

        encoder = Wav2Vec2Model.from_pretrained(model_id, torch_dtype=torch_dtype)
        hidden_size: int = encoder.config.hidden_size

        if freeze_encoder:
            for p in encoder.parameters():
                p.requires_grad_(False)

        return cls(encoder, hidden_size=hidden_size, dropout=dropout)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> EmotionClassifier:
        return cls.from_pretrained(
            model_id=cfg.get("model_id", "facebook/wav2vec2-large-xlsr-53"),
            freeze_encoder=cfg.get("freeze_encoder", True),
            dropout=cfg.get("dropout", 0.1),
        )

    @classmethod
    def for_smoke_test(cls) -> EmotionClassifier:
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
        """Classify emotion from raw waveform.

        Args:
            input_values: Raw waveform ``(B, T)`` normalised to [-1, 1].
            attention_mask: Optional padding mask ``(B, T)``.

        Returns:
            Log-probabilities ``(B, 4)`` — [happy, angry, sad, neutral].
        """
        out = self.encoder(
            input_values=input_values,
            attention_mask=attention_mask,
        )
        hidden = out.last_hidden_state  # (B, T', H)

        if attention_mask is not None:
            pool_mask = attention_mask[:, ::320].float()
            pool_mask = pool_mask[:, : hidden.size(1)]
            pool_mask = pool_mask.unsqueeze(-1)
            pooled = (hidden * pool_mask).sum(1) / pool_mask.sum(1).clamp(min=1)
        else:
            pooled = hidden.mean(dim=1)

        logits = self.head(pooled)  # (B, 4)
        return torch.log_softmax(logits, dim=-1)
