"""LoRA and DoRA adapter utilities.

Thin wrappers around PEFT so the rest of the codebase never imports PEFT
directly — config keys map cleanly to PEFT objects and we can swap
implementations without touching training code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn
from peft import (
    LoraConfig,
    PeftModel,  # noqa: F401 (re-exported for callers)
    TaskType,
    get_peft_model,
)


@dataclass
class LoRAConfig:
    """Validated LoRA / DoRA adapter configuration."""

    r: int = 64
    alpha: int = 128
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    dropout: float = 0.05
    bias: str = "none"
    use_dora: bool = False  # True → DoRA (magnitude decomposition)
    enabled: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LoRAConfig:
        return cls(
            r=d.get("r", 64),
            alpha=d.get("alpha", 128),
            target_modules=list(d.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
            dropout=d.get("dropout", 0.05),
            bias=d.get("bias", "none"),
            use_dora=d.get("use_dora", False),
            enabled=d.get("enabled", True),
        )


def apply_lora(
    model: nn.Module,
    config: LoRAConfig,
    task_type: TaskType | None = None,
) -> nn.Module:
    """Wrap a model with LoRA (or DoRA) adapters.

    Args:
        model: Base model to adapt.
        config: Adapter configuration.
        task_type: PEFT task type (``CAUSAL_LM`` for decoders, ``None`` for encoder-only).

    Returns:
        PEFT-wrapped model. Only adapter parameters are trainable by default.

    Raises:
        ValueError: If ``config.enabled`` is False.
    """
    if not config.enabled:
        raise ValueError("apply_lora called with config.enabled=False — check call site")

    peft_config = LoraConfig(
        r=config.r,
        lora_alpha=config.alpha,
        target_modules=config.target_modules,
        lora_dropout=config.dropout,
        bias=config.bias,
        use_dora=config.use_dora,
        task_type=task_type,
    )
    return get_peft_model(model, peft_config)


def freeze_base(model: nn.Module) -> None:
    """Freeze all non-adapter parameters in-place.

    Call after ``apply_lora`` to ensure only the adapter weights train.
    Safe to call on non-PEFT models (freezes everything).

    Args:
        model: Model whose non-LoRA parameters should be frozen.
    """
    for name, param in model.named_parameters():
        if not any(k in name for k in ("lora_", "dora_magnitude", "lora_embedding")):
            param.requires_grad_(False)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return ``(trainable_params, total_params)`` for ``model``.

    Args:
        model: Any ``nn.Module``.

    Returns:
        Tuple of ``(trainable, total)`` parameter counts.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def log_parameter_counts(model: nn.Module, logger: Any) -> None:
    """Log trainable vs total parameter counts.

    Args:
        model: Model to inspect.
        logger: Logger instance (must have ``.info`` method).
    """
    trainable, total = count_parameters(model)
    pct = 100 * trainable / total if total > 0 else 0.0
    logger.info(f"Parameters: {trainable:,} trainable / {total:,} total ({pct:.2f}%)")
