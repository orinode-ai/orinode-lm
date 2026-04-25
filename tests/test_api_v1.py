"""Tests for the versioned public API (/api/v1/*)."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """TestClient with isolated workspace and mocked inference pipelines."""
    monkeypatch.setenv("ORINODE_WORKSPACE", str(tmp_path))

    import orinode.paths as paths_mod

    ws = paths_mod.WorkspacePaths(tmp_path)
    ws.ensure_all()
    monkeypatch.setattr(paths_mod, "WS", ws)

    from fastapi.testclient import TestClient

    from orinode.ui.server import app

    return TestClient(app)


@pytest.fixture()
def silent_audio_b64() -> str:
    """Return base64 of a minimal valid WAV (1-second silence, 16 kHz)."""
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000)
    return base64.b64encode(buf.getvalue()).decode()


# ── /api/v1/stats ─────────────────────────────────────────────────────────────


def test_stats_empty_workspace(client) -> None:
    resp = client.get("/api/v1/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert "total_hours" in body
    assert "languages" in body
    assert isinstance(body["languages"], list)
    assert body["checkpoints_count"] == 0
    assert body["runs_count"] == 0


# ── /api/v1/checkpoints ───────────────────────────────────────────────────────


def test_checkpoints_empty(client) -> None:
    resp = client.get("/api/v1/checkpoints")
    assert resp.status_code == 200
    assert resp.json() == []


def test_checkpoint_not_found(client) -> None:
    resp = client.get("/api/v1/checkpoints/nonexistent")
    assert resp.status_code == 404


# ── /api/v1/transcribe ────────────────────────────────────────────────────────


def test_transcribe_fallback_when_no_checkpoint(client, silent_audio_b64) -> None:
    resp = client.post("/api/v1/transcribe", json={"audio_b64": silent_audio_b64})
    assert resp.status_code == 200
    body = resp.json()
    assert "text" in body
    assert "language" in body
    assert "latency_ms" in body
    assert isinstance(body["word_tags"], list)


def test_transcribe_invalid_b64(client) -> None:
    resp = client.post("/api/v1/transcribe", json={"audio_b64": "not-valid-base64!!!"})
    assert resp.status_code == 422


# ── /api/v1/emotion ───────────────────────────────────────────────────────────


def test_emotion_no_checkpoint_returns_503(client, silent_audio_b64) -> None:
    resp = client.post("/api/v1/emotion", json={"audio_b64": silent_audio_b64})
    # No checkpoint → FileNotFoundError → 503
    assert resp.status_code == 503


def test_emotion_with_mock_pipeline(client, silent_audio_b64, monkeypatch) -> None:
    stub = MagicMock()
    stub.predict.return_value = {
        "top_prediction": "happy",
        "confidences": {"happy": 0.7, "angry": 0.1, "sad": 0.1, "neutral": 0.1},
        "segment_timeline": None,
        "model_version": "aux_emotion/v1",
        "disclaimer": "Preview.",
    }
    with patch("orinode.inference.emotion_pipeline.get_emotion_pipeline", return_value=stub):
        resp = client.post("/api/v1/emotion", json={"audio_b64": silent_audio_b64})
    assert resp.status_code == 200
    body = resp.json()
    assert body["top_prediction"] == "happy"
    assert "confidences" in body
    assert "disclaimer" in body


# ── /api/v1/gender ────────────────────────────────────────────────────────────


def test_gender_no_checkpoint_returns_503(client, silent_audio_b64) -> None:
    resp = client.post("/api/v1/gender", json={"audio_b64": silent_audio_b64})
    assert resp.status_code == 503


def test_gender_with_mock_pipeline(client, silent_audio_b64, monkeypatch) -> None:
    stub = MagicMock()
    stub.predict.return_value = {
        "prediction": "male",
        "confidence": 0.92,
        "model_version": "aux_gender/v1",
        "per_speaker": None,
    }
    with patch("orinode.inference.gender_pipeline.get_gender_pipeline", return_value=stub):
        resp = client.post("/api/v1/gender", json={"audio_b64": silent_audio_b64})
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] == "male"
    assert body["confidence"] == pytest.approx(0.92)


# ── /api/v1/samples ───────────────────────────────────────────────────────────


def test_samples_returns_list(client) -> None:
    resp = client.get("/api/v1/samples")
    assert resp.status_code == 200
    samples = resp.json()
    assert isinstance(samples, list)


def test_sample_audio_not_found(client) -> None:
    resp = client.get("/api/v1/samples/nonexistent/audio")
    assert resp.status_code == 404


# ── /api/v1/feedback ──────────────────────────────────────────────────────────


def test_feedback_recorded(client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORINODE_WORKSPACE", str(tmp_path))
    import orinode.paths as paths_mod

    ws = paths_mod.WorkspacePaths(tmp_path)
    ws.ensure_all()
    monkeypatch.setattr(paths_mod, "WS", ws)

    payload = {"task": "transcribe", "rating": 4, "comment": "Looks good"}
    resp = client.post("/api/v1/feedback", json=payload)
    assert resp.status_code == 201
    assert resp.json() == {"status": "recorded"}

    feedback_file = tmp_path / "logs" / "feedback" / "transcribe.jsonl"
    assert feedback_file.exists()
    line = json.loads(feedback_file.read_text().strip())
    assert line["rating"] == 4
    assert line["task"] == "transcribe"


def test_feedback_unknown_task_sanitised(client) -> None:
    payload = {"task": "../../../etc/passwd", "rating": 1, "comment": ""}
    resp = client.post("/api/v1/feedback", json=payload)
    assert resp.status_code == 201
