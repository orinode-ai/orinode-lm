"""Tests for the FastAPI UI backend."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Return a TestClient with the workspace pointing at a temp dir."""
    monkeypatch.setenv("ORINODE_WORKSPACE", str(tmp_path))

    # Patch WS so the store scans the tmp logs dir
    import orinode.paths as _paths

    ws = _paths.WorkspacePaths(tmp_path)
    ws.ensure_all()
    monkeypatch.setattr(_paths, "WS", ws)

    from orinode.ui.server import app

    return TestClient(app)


def test_health(client: TestClient) -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_runs_empty(client: TestClient) -> None:
    res = client.get("/api/runs")
    assert res.status_code == 200
    assert res.json() == []


def test_run_not_found(client: TestClient) -> None:
    res = client.get("/api/runs/does_not_exist")
    assert res.status_code == 404


def test_runs_with_events(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write a synthetic JSONL event file and verify the store parses it."""
    import orinode.paths as _paths

    logs_dir = _paths.WS.logs
    logs_dir.mkdir(parents=True, exist_ok=True)

    events = [
        {
            "ts": 1000.0,
            "run_id": "run_a",
            "type": "train_start",
            "total_steps": 100,
            "stage": 1,
            "config_yaml": "",
        },
        {
            "ts": 1001.0,
            "run_id": "run_a",
            "type": "step",
            "step": 10,
            "loss": 2.5,
            "lr": 1e-4,
            "grad_norm": 0.5,
            "epoch": 0,
        },
        {
            "ts": 1002.0,
            "run_id": "run_a",
            "type": "eval",
            "step": 10,
            "wer": {"en": 0.15},
            "eval_loss": 2.1,
        },
        {"ts": 1003.0, "run_id": "run_a", "type": "train_end", "total_steps": 100},
    ]

    event_file = logs_dir / "stage1_encoder_events.jsonl"
    event_file.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    res = client.get("/api/runs")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 1
    assert data[0]["run_id"] == "run_a"
    assert data[0]["status"] == "completed"
    assert data[0]["step"] == 10
    assert abs(data[0]["train_loss"] - 2.5) < 1e-6

    res2 = client.get("/api/runs/run_a")
    assert res2.status_code == 200
    detail = res2.json()
    assert detail["run_id"] == "run_a"
    assert len(detail["events"]) == 4


def test_transcribe_stub(client: TestClient) -> None:
    """Transcribe endpoint returns a valid response even without a checkpoint."""
    import base64

    # Tiny valid WAV: 44-byte header + silence
    wav_header = bytes(
        [
            0x52,
            0x49,
            0x46,
            0x46,
            0x24,
            0x00,
            0x00,
            0x00,  # RIFF
            0x57,
            0x41,
            0x56,
            0x45,
            0x66,
            0x6D,
            0x74,
            0x20,  # WAVE fmt
            0x10,
            0x00,
            0x00,
            0x00,
            0x01,
            0x00,
            0x01,
            0x00,  # PCM mono
            0x80,
            0x3E,
            0x00,
            0x00,
            0x00,
            0x7D,
            0x00,
            0x00,  # 16000 Hz
            0x02,
            0x00,
            0x10,
            0x00,
            0x64,
            0x61,
            0x74,
            0x61,  # data
            0x00,
            0x00,
            0x00,
            0x00,
        ]
    )
    b64 = base64.b64encode(wav_header).decode()

    res = client.post("/api/transcribe", json={"audio_b64": b64, "language": "en"})
    assert res.status_code == 200
    body = res.json()
    assert "text" in body
    assert "latency_ms" in body
    assert body["language"] == "en"
