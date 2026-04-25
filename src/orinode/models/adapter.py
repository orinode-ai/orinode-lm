"""Audio-to-LLM adapter: 2-layer MLP + Q-Former resampler.

Compresses Whisper encoder output (50 tokens/sec, D=1280) to a fixed set
of ``num_query_tokens`` audio tokens that are prepended to the LLM's
token sequence.

Architecture (SALMONN / BLIP-2 style):

    encoder_hidden (B, T', D_enc)
        ↓  2-layer MLP
    projected (B, T', D_llm)
        ↓  Q-Former cross-attention
           learnable queries (B, N_q, D_llm) attend to projected
    audio_tokens (B, N_q, D_llm)

The fixed ``N_q = num_query_tokens`` (default 64) is independent of audio
duration. All frames of a 1–30 s clip map to the same 64 output tokens.
This is the BLIP-2 design rather than the windowed SALMONN design; it is
simpler and sufficient for utterances up to ~30 s.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


class AudioLLMAdapter(nn.Module):
    """2-layer MLP projection followed by a Q-Former resampler.

    Args:
        input_dim: Encoder hidden dimension (1280 for Whisper-large-v3).
        output_dim: LLM hidden dimension (3584 for Gemma-2-9B).
        mlp_hidden_dim: Intermediate dimension of the MLP (default 2048).
        num_query_tokens: Number of learnable query tokens ``N_q`` (default 64).
        num_qformer_layers: Number of transformer decoder layers in the
            Q-Former (default 2, matching BLIP-2).
        num_heads: Number of attention heads (must divide ``output_dim``).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mlp_hidden_dim: int = 2048,
        num_query_tokens: int = 64,
        num_qformer_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_query_tokens = num_query_tokens

        # 2-layer MLP with LayerNorm between layers
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(mlp_hidden_dim),
            nn.Linear(mlp_hidden_dim, output_dim),
        )

        # Learnable Q-Former query tokens
        self.query_tokens = nn.Parameter(torch.zeros(1, num_query_tokens, output_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        # Q-Former: queries cross-attend to projected encoder states
        # nn.TransformerDecoder: tgt=queries, memory=projected_encoder_states
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=output_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm for training stability
        )
        self.qformer = nn.TransformerDecoder(decoder_layer, num_layers=num_qformer_layers)
        self.output_norm = nn.LayerNorm(output_dim)

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: DictConfig, decoder_hidden_size: int) -> AudioLLMAdapter:
        """Build from a config node (``configs/model/adapter.yaml``).

        Args:
            cfg: Config with keys: ``input_dim``, ``mlp_hidden_dim``,
                ``num_query_tokens``, ``dropout``.
            decoder_hidden_size: LLM hidden dimension (resolved at runtime).

        Returns:
            Initialised ``AudioLLMAdapter``.
        """
        return cls(
            input_dim=cfg.input_dim,
            output_dim=decoder_hidden_size,
            mlp_hidden_dim=cfg.mlp_hidden_dim,
            num_query_tokens=cfg.num_query_tokens,
            dropout=cfg.get("dropout", 0.1),
        )

    @classmethod
    def for_smoke_test(
        cls,
        input_dim: int = 64,
        output_dim: int = 128,
    ) -> AudioLLMAdapter:
        """Create a tiny adapter for CPU smoke testing.

        Args:
            input_dim: Must match ``WhisperEncoder.for_smoke_test().output_dim``.
            output_dim: Must match the smoke-test decoder hidden size.

        Returns:
            Tiny ``AudioLLMAdapter`` with random weights.
        """
        return cls(
            input_dim=input_dim,
            output_dim=output_dim,
            mlp_hidden_dim=128,
            num_query_tokens=4,
            num_qformer_layers=1,
            num_heads=4,
            dropout=0.0,
        )

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project and compress encoder output to fixed-size audio tokens.

        Args:
            encoder_hidden_states: Whisper encoder output ``(B, T', D_enc)``.
            encoder_attention_mask: Padding mask ``(B, T')``; ``1``=real token,
                ``0``=padding.  When provided, padding positions are masked out
                in cross-attention so queries do not attend to pad tokens.

        Returns:
            Audio tokens ``(B, N_q, D_llm)`` ready to prepend to text embeddings.
        """
        B = encoder_hidden_states.size(0)

        # Project encoder states to LLM embedding dim
        projected = self.mlp(encoder_hidden_states)  # (B, T', D_llm)

        # Expand learnable queries to batch
        queries = self.query_tokens.expand(B, -1, -1)  # (B, N_q, D_llm)

        # Build memory key-padding mask (True = ignore in nn.TransformerDecoder)
        memory_key_padding_mask: torch.Tensor | None = None
        if encoder_attention_mask is not None:
            memory_key_padding_mask = encoder_attention_mask == 0  # (B, T')

        # Q-Former cross-attention: queries attend to projected encoder states
        output = self.qformer(
            tgt=queries,
            memory=projected,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # (B, N_q, D_llm)

        return self.output_norm(output)
