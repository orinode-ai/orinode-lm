"""SpeechLLM: end-to-end Whisper-encoder + adapter + causal-LLM decoder.

Forward pass:
    mel → WhisperEncoder → AudioLLMAdapter → audio_tokens (B, N_q, D_llm)
    input_ids → decoder.get_input_embeddings() → text_embeds (B, S, D_llm)
    cat([audio_tokens, text_embeds], dim=1) → decoder → logits → CE loss

Labels for the audio prefix positions are set to -100 so they are ignored
by ``language_model_loss``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from orinode.models.adapter import AudioLLMAdapter
from orinode.models.losses import language_model_loss
from orinode.models.whisper_encoder import WhisperEncoder


@dataclass
class SpeechLLMOutput:
    """Outputs from a single SpeechLLM forward pass."""

    loss: torch.Tensor | None
    logits: torch.Tensor  # (B, N_q + S, V)
    audio_tokens: torch.Tensor  # (B, N_q, D_llm)


class SpeechLLM(nn.Module):
    """Whisper encoder + adapter + causal LLM decoder.

    Args:
        encoder: ``WhisperEncoder`` wrapping a HuggingFace Whisper encoder.
        adapter: ``AudioLLMAdapter`` compressing encoder states to N_q tokens.
        decoder: Any HuggingFace ``PreTrainedModel`` with a causal LM head
            (e.g. ``GemmaForCausalLM``, ``GPT2LMHeadModel``).
    """

    def __init__(
        self,
        encoder: WhisperEncoder,
        adapter: AudioLLMAdapter,
        decoder: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter
        self.decoder = decoder

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        encoder: WhisperEncoder,
        adapter: AudioLLMAdapter,
        decoder_id: str,
        decoder_kwargs: dict | None = None,
    ) -> SpeechLLM:
        """Load decoder from HuggingFace and assemble SpeechLLM.

        Args:
            encoder: Already-initialised WhisperEncoder.
            adapter: Already-initialised AudioLLMAdapter.
            decoder_id: HuggingFace model ID for the decoder
                (e.g. ``google/gemma-2-9b-it``).
            decoder_kwargs: Extra kwargs forwarded to
                ``AutoModelForCausalLM.from_pretrained``.

        Returns:
            Assembled ``SpeechLLM``.
        """
        from transformers import AutoModelForCausalLM

        kw = decoder_kwargs or {}
        decoder = AutoModelForCausalLM.from_pretrained(decoder_id, **kw)
        return cls(encoder, adapter, decoder)

    @classmethod
    def for_smoke_test(cls) -> SpeechLLM:
        """Assemble a tiny CPU-only model for fast smoke tests.

        Uses tiny random-weight WhisperEncoder, AudioLLMAdapter, and a
        2-layer GPT-2 decoder so no downloads are needed.

        Returns:
            ``SpeechLLM`` ready for forward/generate calls on CPU.
        """
        from transformers import GPT2Config, GPT2LMHeadModel

        encoder = WhisperEncoder.for_smoke_test()  # output_dim=64
        adapter = AudioLLMAdapter.for_smoke_test(input_dim=64, output_dim=128)  # N_q=4, D_llm=128

        gpt2_cfg = GPT2Config(
            n_embd=128,
            n_layer=2,
            n_head=4,
            vocab_size=256,
            n_positions=256,
        )
        decoder = GPT2LMHeadModel(gpt2_cfg)
        return cls(encoder, adapter, decoder)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_embed(self) -> nn.Embedding:
        """Return the decoder's token embedding table."""
        # Works for GPT2LMHeadModel, GemmaForCausalLM, and most HF causal LMs.
        if hasattr(self.decoder, "get_input_embeddings"):
            return self.decoder.get_input_embeddings()
        raise AttributeError(f"{type(self.decoder).__name__} has no get_input_embeddings()")

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_features: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> SpeechLLMOutput:
        """Run the full speech-LLM forward pass.

        Args:
            input_features: Log-mel spectrogram ``(B, n_mels, T)``.
            input_ids: Text token IDs ``(B, S)``.  For inference these are the
                prompt tokens; for training they include the full target
                sequence.
            labels: Target token IDs ``(B, S)`` with ``-100`` at positions to
                ignore.  When provided, the loss is computed.  Audio-prefix
                positions are automatically masked (set to -100) before the
                loss call.
            encoder_attention_mask: Padding mask ``(B, T)`` for the encoder.
            label_smoothing: Forwarded to ``language_model_loss``.

        Returns:
            ``SpeechLLMOutput`` with ``loss`` (if labels provided), ``logits``,
            and ``audio_tokens``.
        """
        # 1. Encode audio
        enc_hidden = self.encoder(
            input_features=input_features,
            attention_mask=encoder_attention_mask,
        )  # (B, T', D_enc)

        # 2. Compress to N_q tokens
        audio_tokens = self.adapter(enc_hidden, encoder_attention_mask)  # (B, N_q, D_llm)

        # 3. Embed text tokens
        embed = self._get_embed()
        text_embeds = embed(input_ids)  # (B, S, D_llm)

        # 4. Concatenate: [audio | text]
        inputs_embeds = torch.cat([audio_tokens, text_embeds], dim=1)
        # (B, N_q + S, D_llm)

        # 5. Decoder forward (inputs_embeds bypasses the embedding layer)
        decoder_out = self.decoder(inputs_embeds=inputs_embeds)
        logits: torch.Tensor = decoder_out.logits  # (B, N_q + S, V)

        # 6. Loss (text portion only)
        loss: torch.Tensor | None = None
        if labels is not None:
            N_q = audio_tokens.size(1)
            # Prepend -100 for audio prefix so those positions are ignored
            audio_labels = labels.new_full((labels.size(0), N_q), -100)
            full_labels = torch.cat([audio_labels, labels], dim=1)
            # (B, N_q + S)

            vocab_size: int = logits.size(-1)
            loss = language_model_loss(logits, full_labels, vocab_size, label_smoothing)

        return SpeechLLMOutput(loss=loss, logits=logits, audio_tokens=audio_tokens)

    # ── generation ────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def generate(
        self,
        input_features: torch.Tensor,
        prompt_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        **generate_kwargs,
    ) -> torch.Tensor:
        """Transcribe / respond to speech.

        Encodes audio to ``audio_tokens``, embeds the prompt, concatenates,
        and calls ``decoder.generate`` with ``inputs_embeds``.

        Args:
            input_features: ``(B, n_mels, T)`` log-mel spectrogram.
            prompt_ids: ``(B, S_p)`` prompt token IDs (e.g. instruction text).
            encoder_attention_mask: Optional padding mask for the encoder.
            **generate_kwargs: Forwarded to ``decoder.generate``
                (e.g. ``max_new_tokens``, ``temperature``).

        Returns:
            Generated token IDs ``(B, T_out)`` from ``decoder.generate``.
        """
        enc_hidden = self.encoder(
            input_features=input_features,
            attention_mask=encoder_attention_mask,
        )
        audio_tokens = self.adapter(enc_hidden, encoder_attention_mask)

        embed = self._get_embed()
        prompt_embeds = embed(prompt_ids)

        inputs_embeds = torch.cat([audio_tokens, prompt_embeds], dim=1)

        return self.decoder.generate(
            inputs_embeds=inputs_embeds,
            **generate_kwargs,
        )
