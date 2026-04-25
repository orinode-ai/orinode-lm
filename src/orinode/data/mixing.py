"""Temperature-weighted multilingual batch sampler.

Samples from multiple language-specific datasets with temperature α so that
low-resource languages (IG, PCM) are over-represented relative to their raw
dataset sizes while high-resource languages (EN, HA) are slightly
down-weighted.

Sampling formula::

    p(lang) = n(lang)^α / Σ_i n(i)^α     α ∈ (0, 1]

    α = 1.0  →  pure proportional sampling (proportional to dataset size)
    α → 0.0  →  uniform sampling (all languages equally likely)
    α = 0.3  →  default; recommended by MERaLiON and SALMONN papers

Reference: Multilingual pre-training paper (Conneau et al., 2020, CCNet).
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

from torch.utils.data import Dataset, Sampler


def compute_sampling_weights(
    counts: dict[str, int],
    temperature: float = 0.3,
) -> dict[str, float]:
    """Compute temperature-weighted sampling probabilities.

    Args:
        counts: Mapping from language code to number of training samples.
        temperature: Temperature α in ``(0, 1]``.

    Returns:
        Mapping from language code to sampling probability (values sum to 1.0).

    Raises:
        ValueError: If ``counts`` is empty or ``temperature`` is out of range.
    """
    if not counts:
        raise ValueError("counts must be non-empty")
    if not (0.0 < temperature <= 1.0):
        raise ValueError(f"temperature must be in (0, 1], got {temperature}")

    weighted = {lang: count**temperature for lang, count in counts.items()}
    total = sum(weighted.values())
    return {lang: w / total for lang, w in weighted.items()}


class MultilingualBatchSampler(Sampler[list[int]]):
    """Yield batches with temperature-weighted language mixing.

    Each batch is assembled by independently drawing ``batch_size`` samples,
    where each draw first picks a language with temperature-weighted probability
    and then picks uniformly from that language's index set.

    Suitable for use with multi-GPU training where the batch is split across
    ranks after sampling.

    Args:
        language_indices: Mapping from language code to list of flat dataset
            indices (indices into the concatenated multi-language dataset).
        batch_size: Number of samples per batch.
        temperature: Sampling temperature α.
        num_batches: Total batches per epoch.
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        language_indices: dict[str, list[int]],
        batch_size: int,
        temperature: float = 0.3,
        num_batches: int = 1000,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.language_indices = language_indices
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.seed = seed

        counts = {lang: len(idxs) for lang, idxs in language_indices.items()}
        self.weights = compute_sampling_weights(counts, temperature)
        self._langs = list(self.weights.keys())
        self._probs = [self.weights[lang] for lang in self._langs]

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed)
        for _ in range(self.num_batches):
            batch: list[int] = []
            for _ in range(self.batch_size):
                lang = rng.choices(self._langs, weights=self._probs, k=1)[0]
                idx = rng.choice(self.language_indices[lang])
                batch.append(idx)
            yield batch

    def __len__(self) -> int:
        return self.num_batches


class TemperatureWeightedDataset(Dataset):  # type: ignore[type-arg]
    """Wraps per-language datasets into a single temperature-sampled dataset.

    Useful when ``MultilingualBatchSampler`` cannot be used (e.g. the
    ``DataLoader`` does not accept a custom batch sampler). The sampling plan
    is deterministically pre-computed so ``__getitem__`` is O(1).

    Args:
        datasets: Mapping from language code to ``Dataset``.
        temperature: Sampling temperature.
        total_samples: Virtual length of the combined dataset.
        seed: RNG seed.
    """

    def __init__(
        self,
        datasets: dict[str, Dataset],  # type: ignore[type-arg]
        temperature: float = 0.3,
        total_samples: int = 100_000,
        seed: int = 42,
    ) -> None:
        self.datasets = datasets
        self.total_samples = total_samples

        counts = {lang: len(ds) for lang, ds in datasets.items()}  # type: ignore[arg-type]
        self.weights = compute_sampling_weights(counts, temperature)
        langs = list(self.weights.keys())
        probs = [self.weights[lang] for lang in langs]

        rng = random.Random(seed)
        self._plan: list[tuple[str, int]] = []
        for _ in range(total_samples):
            lang = rng.choices(langs, weights=probs, k=1)[0]
            ds_len = len(datasets[lang])  # type: ignore[arg-type]
            self._plan.append((lang, rng.randrange(ds_len)))

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        lang, ds_idx = self._plan[idx]
        item: dict[str, Any] = dict(self.datasets[lang][ds_idx])
        item.setdefault("language", lang)
        return item
