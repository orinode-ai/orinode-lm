"""Tests for mixing.py — temperature-weighted multilingual sampling."""

from __future__ import annotations

import pytest

from orinode.data.mixing import (
    MultilingualBatchSampler,
    TemperatureWeightedDataset,
    compute_sampling_weights,
)

# ── compute_sampling_weights ──────────────────────────────────────────────────


def test_weights_sum_to_one() -> None:
    counts = {"en": 1000, "ha": 500, "yo": 300, "ig": 80, "pcm": 200}
    weights = compute_sampling_weights(counts, temperature=0.3)
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_weights_all_languages_present() -> None:
    counts = {"en": 1000, "ha": 500, "yo": 300, "ig": 80, "pcm": 200}
    weights = compute_sampling_weights(counts, temperature=0.3)
    assert set(weights.keys()) == {"en", "ha", "yo", "ig", "pcm"}


def test_low_temperature_upsamples_minority() -> None:
    counts = {"en": 10000, "ig": 100}
    w_low = compute_sampling_weights(counts, temperature=0.1)
    w_prop = compute_sampling_weights(counts, temperature=1.0)
    # At low temperature, IG gets relatively more weight than at α=1
    assert w_low["ig"] > w_prop["ig"]


def test_temperature_one_is_proportional() -> None:
    counts = {"en": 600, "ha": 400}
    weights = compute_sampling_weights(counts, temperature=1.0)
    assert abs(weights["en"] - 0.6) < 1e-9
    assert abs(weights["ha"] - 0.4) < 1e-9


def test_temperature_approaches_uniform_at_zero() -> None:
    counts = {"en": 100000, "ig": 1}
    # Very small alpha → near-uniform
    weights = compute_sampling_weights(counts, temperature=0.01)
    # Both weights should be close to 0.5
    assert weights["en"] < 0.55
    assert weights["ig"] > 0.45


def test_weights_positive_all() -> None:
    counts = {"en": 5000, "ha": 1000, "yo": 800, "ig": 100, "pcm": 300}
    weights = compute_sampling_weights(counts, temperature=0.3)
    assert all(w > 0 for w in weights.values())


def test_weights_invalid_temperature_zero() -> None:
    with pytest.raises(ValueError, match="temperature"):
        compute_sampling_weights({"en": 100}, temperature=0.0)


def test_weights_invalid_temperature_negative() -> None:
    with pytest.raises(ValueError):
        compute_sampling_weights({"en": 100}, temperature=-0.5)


def test_weights_empty_counts_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        compute_sampling_weights({}, temperature=0.3)


def test_weights_single_language() -> None:
    weights = compute_sampling_weights({"ha": 500}, temperature=0.3)
    assert abs(weights["ha"] - 1.0) < 1e-9


def test_weights_equal_counts_equal_weights() -> None:
    counts = {"en": 100, "ha": 100, "yo": 100}
    weights = compute_sampling_weights(counts, temperature=0.3)
    vals = list(weights.values())
    assert abs(vals[0] - vals[1]) < 1e-9
    assert abs(vals[1] - vals[2]) < 1e-9


# ── MultilingualBatchSampler ──────────────────────────────────────────────────


def _make_lang_indices() -> dict[str, list[int]]:
    return {
        "en": list(range(0, 100)),
        "ha": list(range(100, 150)),
        "yo": list(range(150, 180)),
        "ig": list(range(180, 190)),
        "pcm": list(range(190, 210)),
    }


def test_batch_sampler_correct_batch_size() -> None:
    sampler = MultilingualBatchSampler(_make_lang_indices(), batch_size=8, num_batches=10, seed=0)
    for batch in sampler:
        assert len(batch) == 8


def test_batch_sampler_correct_num_batches() -> None:
    sampler = MultilingualBatchSampler(_make_lang_indices(), batch_size=4, num_batches=50, seed=0)
    batches = list(sampler)
    assert len(batches) == 50


def test_batch_sampler_indices_in_range() -> None:
    lang_indices = _make_lang_indices()
    all_valid = set(range(210))
    sampler = MultilingualBatchSampler(lang_indices, batch_size=16, num_batches=20, seed=1)
    for batch in sampler:
        assert all(idx in all_valid for idx in batch)


def test_batch_sampler_reproducible_with_same_seed() -> None:
    lang_indices = _make_lang_indices()
    s1 = list(MultilingualBatchSampler(lang_indices, batch_size=4, num_batches=5, seed=42))
    s2 = list(MultilingualBatchSampler(lang_indices, batch_size=4, num_batches=5, seed=42))
    assert s1 == s2


def test_batch_sampler_different_seeds_different_results() -> None:
    lang_indices = _make_lang_indices()
    s1 = list(MultilingualBatchSampler(lang_indices, batch_size=4, num_batches=10, seed=0))
    s2 = list(MultilingualBatchSampler(lang_indices, batch_size=4, num_batches=10, seed=99))
    assert s1 != s2


def test_batch_sampler_len() -> None:
    sampler = MultilingualBatchSampler(_make_lang_indices(), batch_size=4, num_batches=77)
    assert len(sampler) == 77


# ── TemperatureWeightedDataset ────────────────────────────────────────────────


class _MockDataset:
    """Tiny in-memory dataset stub."""

    def __init__(self, items: list[dict]) -> None:
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        return self._items[idx]


def test_weighted_dataset_len() -> None:
    datasets = {
        "en": _MockDataset([{"text": f"en_{i}"} for i in range(100)]),
        "ha": _MockDataset([{"text": f"ha_{i}"} for i in range(50)]),
    }
    ds = TemperatureWeightedDataset(datasets, total_samples=200, temperature=0.3)
    assert len(ds) == 200


def test_weighted_dataset_getitem_has_language() -> None:
    datasets = {
        "en": _MockDataset([{"text": "hello"}] * 20),
        "ha": _MockDataset([{"text": "sannu"}] * 20),
    }
    ds = TemperatureWeightedDataset(datasets, total_samples=50, temperature=0.3, seed=7)
    item = ds[0]
    assert "language" in item
    assert item["language"] in {"en", "ha"}


def test_weighted_dataset_all_languages_sampled() -> None:
    datasets = {lang: _MockDataset([{"text": lang}] * 100) for lang in ["en", "ha", "yo"]}
    ds = TemperatureWeightedDataset(datasets, total_samples=300, temperature=0.3, seed=0)
    langs_seen = {ds[i]["language"] for i in range(300)}
    assert langs_seen == {"en", "ha", "yo"}


def test_weighted_dataset_reproducible() -> None:
    datasets = {"en": _MockDataset([{"v": i} for i in range(50)])}
    d1 = TemperatureWeightedDataset(datasets, total_samples=10, seed=42)
    d2 = TemperatureWeightedDataset(datasets, total_samples=10, seed=42)
    assert [d1[i] for i in range(10)] == [d2[i] for i in range(10)]
