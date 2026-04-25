"""Loss functions for Orinode training stages.

Stage 1/2 (encoder ASR): ``whisper_seq2seq_loss`` delegates to the HuggingFace
model's own loss so training stays identical to vanilla Whisper fine-tuning.

Stage 3/4 (Speech-LLM): ``language_model_loss`` computes causal cross-entropy
on the text portion of the sequence only (audio token positions have
labels=-100 and are ignored).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def language_model_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Compute causal LM cross-entropy loss.

    Positions where ``labels == -100`` are ignored, so callers should set
    audio-token positions and any instruction prefix positions to -100.

    The standard next-token prediction shift (predict token i+1 from position i)
    is applied internally.

    Args:
        logits: Raw LLM output logits ``(B, S, V)``.
        labels: Target token IDs ``(B, S)`` with -100 for ignored positions.
        vocab_size: Vocabulary size ``V``.
        label_smoothing: Label smoothing in ``[0, 1)``.

    Returns:
        Scalar loss tensor.
    """
    shift_logits = logits[..., :-1, :].contiguous()  # (B, S-1, V)
    shift_labels = labels[..., 1:].contiguous()  # (B, S-1)

    return F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
        label_smoothing=label_smoothing,
        reduction="mean",
    )


def whisper_seq2seq_loss(
    model_output: object,
) -> torch.Tensor:
    """Extract the loss from a HuggingFace Seq2SeqLM model output.

    HuggingFace Whisper models compute their own cross-entropy loss when
    ``labels`` is passed to the forward call. This function extracts it
    so training loops have a uniform interface.

    Args:
        model_output: Output from ``WhisperForConditionalGeneration.forward``.

    Returns:
        Scalar loss tensor.

    Raises:
        ValueError: If the output has no loss (i.e. labels were not passed).
    """
    if model_output.loss is None:
        raise ValueError(
            "model_output.loss is None — pass labels= to WhisperForConditionalGeneration"
        )
    return model_output.loss


def compute_token_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Compute per-token accuracy for non-ignored positions.

    Args:
        logits: ``(B, S, V)`` model output.
        labels: ``(B, S)`` target IDs with -100 for ignored positions.

    Returns:
        Fraction of non-ignored tokens predicted correctly.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    preds = shift_logits.argmax(dim=-1)
    mask = shift_labels != -100
    if mask.sum() == 0:
        return 0.0
    correct = (preds == shift_labels) & mask
    return correct.sum().item() / mask.sum().item()
