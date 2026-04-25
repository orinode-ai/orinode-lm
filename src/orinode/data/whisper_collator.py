"""DataCollator for batching WhisperDataset items.

Stacks input_features (all same shape) and right-pads labels with -100
so the cross-entropy loss ignores padding positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class WhisperDataCollator:
    """Collate a list of WhisperDataset items into a padded batch.

    Args:
        pad_token_id: Value used to pad label sequences (default -100 so
            the loss ignores those positions).
    """

    pad_token_id: int = -100

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = torch.stack(
            [f["input_features"] for f in features]
        )  # [B, n_mels, T]

        labels = self._pad_labels([f["labels"] for f in features])  # [B, max_seq_len]

        return {"input_features": input_features, "labels": labels}

    def _pad_labels(self, label_list: list[torch.Tensor]) -> torch.Tensor:
        max_len = max(t.shape[0] for t in label_list)
        padded = torch.full(
            (len(label_list), max_len),
            fill_value=self.pad_token_id,
            dtype=label_list[0].dtype,
        )
        for i, t in enumerate(label_list):
            padded[i, : t.shape[0]] = t
        return padded
