"""Shared pytest fixtures for the Orinode-LM test suite.

Fixture scopes:
- ``tmp_workspace``       session — one workspace tree per test run
- ``dummy_audio_tensor``  session — deterministic 3-second sine sweep
- ``dummy_audio_file``    session — the tensor written to a .flac file
- ``minimal_config``      session — smallest valid OmegaConf DictConfig
- ``dummy_manifest_row``  function — one ManifestRow per language
- ``event_bus``           function — EventBus writing to a tmp file

Tests that load real model weights must be decorated with ``@pytest.mark.slow``
and are excluded from ``make test`` (``-m 'not slow'``).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import soundfile as sf
import torch
from omegaconf import OmegaConf

from orinode.data.manifests import CSSpan, ManifestRow
from orinode.utils.events import EventBus

# ── workspace ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def tmp_workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create the full workspace/ directory tree in a temp dir.

    Sets ``ORINODE_WORKSPACE``, ``HF_HOME``, and ``TRANSFORMERS_CACHE``
    environment variables to point at this tree for the duration of the
    session.
    """
    root = tmp_path_factory.mktemp("workspace")

    # Mirror the real workspace layout
    subdirs = [
        "data/raw/naijavoices",
        "data/raw/afrispeech_200",
        "data/raw/bibletts",
        "data/raw/common_voice",
        "data/raw/crowdsourced_cs",
        "data/processed",
        "data/manifests",
        "models/base",
        "models/checkpoints",
        "logs/wandb",
        "logs/training",
        "evals",
        "cache/huggingface",
        "cache/transformers",
    ]
    for sub in subdirs:
        (root / sub).mkdir(parents=True, exist_ok=True)

    os.environ["ORINODE_WORKSPACE"] = str(root)
    os.environ["HF_HOME"] = str(root / "cache" / "huggingface")
    os.environ["TRANSFORMERS_CACHE"] = str(root / "cache" / "transformers")

    return root


# ── audio ─────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def dummy_audio_tensor() -> torch.Tensor:
    """Return a deterministic 3-second 16 kHz mono sine sweep tensor (1, 48000).

    Uses a fixed seed so the tensor is identical across all test runs.
    """
    sample_rate = 16_000
    duration = 3.0
    t = torch.linspace(0, duration, int(sample_rate * duration))
    # Sine sweep from 220 Hz to 880 Hz
    freq = 220.0 * (4.0 ** (t / duration))
    phase = 2 * math.pi * torch.cumsum(freq / sample_rate, dim=0)
    waveform = 0.5 * torch.sin(phase).unsqueeze(0)  # (1, 48000)
    return waveform


@pytest.fixture(scope="session")
def dummy_audio_file(tmp_workspace: Path, dummy_audio_tensor: torch.Tensor) -> Path:
    """Write the dummy audio tensor to a 16 kHz FLAC file and return its path."""
    path = tmp_workspace / "data" / "processed" / "test_audio_3s.flac"
    path.parent.mkdir(parents=True, exist_ok=True)
    audio_np = dummy_audio_tensor.squeeze(0).numpy()
    sf.write(str(path), audio_np, 16_000, format="flac")
    return path


# ── manifests ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def dummy_manifest_rows(dummy_audio_file: Path) -> list[ManifestRow]:
    """Return one ``ManifestRow`` per language with realistic diacritised text."""
    audio_str = str(dummy_audio_file)
    return [
        ManifestRow(
            audio_path=audio_str,
            duration=3.0,
            text="The quick brown fox jumps over the lazy dog",
            language="en",
            speaker_id="spk_en_001",
            domain="broadcast",
            corpus="test",
        ),
        ManifestRow(
            audio_path=audio_str,
            duration=3.0,
            text="Ina son ƙasar Najeriya da dukan zuciyata",
            language="ha",
            speaker_id="spk_ha_001",
            domain="call_center",
            corpus="test",
        ),
        ManifestRow(
            audio_path=audio_str,
            duration=3.0,
            text="Èdè Yorùbá jẹ́ èdè tí ó ní ọlọ́rọ̀ itan",
            language="yo",
            speaker_id="spk_yo_001",
            domain="broadcast",
            corpus="test",
        ),
        ManifestRow(
            audio_path=audio_str,
            duration=3.0,
            text="Asụsụ Igbo bụ asụsụ dị mma yana ọtụtụ ụdị",
            language="ig",
            speaker_id="spk_ig_001",
            domain="bible",
            corpus="test",
        ),
        ManifestRow(
            audio_path=audio_str,
            duration=3.0,
            text="I dey happy to meet you today for this place",
            language="pcm",
            speaker_id="spk_pcm_001",
            domain="call_center",
            corpus="test",
            is_code_switched=False,
        ),
        ManifestRow(
            audio_path=audio_str,
            duration=3.0,
            text="I go the market buy ƙayan abinci na gida",
            language="en",
            speaker_id="spk_cs_001",
            domain="call_center",
            corpus="test",
            is_code_switched=True,
            cs_spans=[
                CSSpan(start=0, end=19, language="en"),
                CSSpan(start=20, end=42, language="ha"),
            ],
        ),
    ]


@pytest.fixture(scope="function")
def dummy_manifest_row(dummy_manifest_rows: list[ManifestRow]) -> ManifestRow:
    """Return a single Hausa row (convenience alias)."""
    return dummy_manifest_rows[1]


@pytest.fixture(scope="session")
def tmp_manifest(tmp_workspace: Path, dummy_manifest_rows: list[ManifestRow]) -> Path:
    """Write a 6-row test manifest to workspace/data/manifests/test.jsonl."""
    from orinode.data.manifests import write_manifest

    path = tmp_workspace / "data" / "manifests" / "test.jsonl"
    write_manifest(dummy_manifest_rows, path)
    return path


# ── config ────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def minimal_config() -> OmegaConf:  # type: ignore[name-defined]
    """Return the smallest valid DictConfig for smoke-test model forward passes."""
    return OmegaConf.create(
        {
            "encoder": {
                "model_id": "openai/whisper-large-v3",
                "output_dim": 1280,
                "dora": {"enabled": False},
            },
            "adapter": {
                "type": "mlp_qformer",
                "input_dim": 1280,
                "mlp_hidden_dim": 256,
                "num_query_tokens": 4,
                "source_tokens_per_sec": 50,
                "target_tokens_per_sec": 5,
                "dropout": 0.0,
            },
            "decoder": {
                "hidden_size": 256,
                "lora": {"enabled": False},
            },
            "training": {
                "batch_size_per_gpu": 1,
                "max_steps": 1,
                "bf16": False,
                "gradient_checkpointing": False,
            },
            "data": {
                "sample_rate": 16000,
                "max_duration": 30.0,
                "min_duration": 0.5,
            },
        }
    )


# ── event bus ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="function")
def event_bus(tmp_path: Path) -> EventBus:
    """Return an ``EventBus`` writing to a temporary JSONL file."""
    events_file = tmp_path / "events.jsonl"
    return EventBus(path=events_file, run_id="test-run-001")


# ── disable UI auth for all tests ─────────────────────────────────────────────


@pytest.fixture(autouse=True)
def disable_ui_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure auth middleware is off so API tests don't need credentials."""
    monkeypatch.setenv("ORINODE_UI_USER", "")
    monkeypatch.setenv("ORINODE_UI_PASS", "")
    import orinode.ui.server as srv

    monkeypatch.setattr(srv, "_AUTH_ENABLED", False)


# ── mock inference pipelines ──────────────────────────────────────────────────


@pytest.fixture()
def mock_gender_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch get_gender_pipeline to return a deterministic stub."""
    from unittest.mock import MagicMock

    stub = MagicMock()
    stub.predict.return_value = {
        "prediction": "female",
        "confidence": 0.87,
        "model_version": "aux_gender/v1",
        "per_speaker": None,
    }

    import orinode.inference.gender_pipeline as gp_mod

    monkeypatch.setattr(gp_mod, "get_gender_pipeline", lambda device="cpu": stub)
    import orinode.ui.api_v1 as api_mod

    monkeypatch.setattr(
        api_mod,
        "get_gender_pipeline" if hasattr(api_mod, "get_gender_pipeline") else "__builtins__",
        lambda device="cpu": stub,
        raising=False,
    )


@pytest.fixture()
def mock_emotion_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch get_emotion_pipeline to return a deterministic stub."""
    from unittest.mock import MagicMock

    stub = MagicMock()
    stub.predict.return_value = {
        "top_prediction": "happy",
        "confidences": {"happy": 0.70, "angry": 0.10, "sad": 0.10, "neutral": 0.10},
        "segment_timeline": None,
        "model_version": "aux_emotion/v1",
        "disclaimer": "Preview model.",
    }

    import orinode.inference.emotion_pipeline as ep_mod

    monkeypatch.setattr(ep_mod, "get_emotion_pipeline", lambda device="cpu": stub)
    import orinode.ui.api_v1 as api_mod

    monkeypatch.setattr(
        api_mod,
        "get_emotion_pipeline" if hasattr(api_mod, "get_emotion_pipeline") else "__builtins__",
        lambda device="cpu": stub,
        raising=False,
    )
