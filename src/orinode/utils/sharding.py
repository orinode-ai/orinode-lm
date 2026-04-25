"""FSDP helpers for multi-GPU training.

Thin wrappers around PyTorch FSDP to keep training scripts free of
boilerplate. Import-safe on CPU (FSDP is only instantiated when actually
called with a distributed process group active).
"""

from __future__ import annotations

import functools
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


def get_fsdp_mixed_precision() -> MixedPrecision:
    """Return bf16 mixed precision policy (params + grads + buffers)."""
    return MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )


def wrap_fsdp(
    module: torch.nn.Module,
    auto_wrap_cls: set[type[torch.nn.Module]] | None = None,
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD,
    cpu_offload: bool = False,
) -> FullyShardedDataParallel:
    """Wrap a module with FSDP FULL_SHARD + bf16.

    Args:
        module: Module to wrap.
        auto_wrap_cls: Transformer layer classes to auto-wrap individually
            (e.g. ``{GemmaDecoderLayer, WhisperEncoderLayer}``).
        sharding_strategy: FSDP sharding strategy.
        cpu_offload: Offload parameters to CPU to save VRAM at the cost of
            throughput. Useful for debugging on small machines.

    Returns:
        FSDP-wrapped module ready for distributed training.
    """
    from torch.distributed.fsdp import CPUOffload

    wrap_policy: Any = None
    if auto_wrap_cls:
        wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=auto_wrap_cls,
        )

    return FullyShardedDataParallel(
        module,
        sharding_strategy=sharding_strategy,
        mixed_precision=get_fsdp_mixed_precision(),
        auto_wrap_policy=wrap_policy,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        use_orig_params=True,
    )


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_distributed():
        dist.barrier()
