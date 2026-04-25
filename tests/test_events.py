"""Tests for utils/events.py — JSONL-backed training event bus."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from orinode.utils.events import (
    EventBus,
)

# ── EventBus.emit + read_events round-trips ───────────────────────────────────


def test_step_event_roundtrip(event_bus: EventBus) -> None:
    event_bus.step(step=100, loss=2.34, lr=5e-5, grad_norm=0.8, epoch=1)
    events = EventBus.read_events(event_bus.path)
    assert len(events) == 1
    e = events[0]
    assert e["type"] == "step"
    assert e["step"] == 100
    assert abs(e["loss"] - 2.34) < 1e-6
    assert abs(e["lr"] - 5e-5) < 1e-9
    assert abs(e["grad_norm"] - 0.8) < 1e-6
    assert e["epoch"] == 1
    assert e["run_id"] == "test-run-001"
    assert "ts" in e


def test_eval_event_roundtrip(event_bus: EventBus) -> None:
    wer = {"en": 0.09, "ha": 0.14, "yo": 0.12, "ig": 0.21, "pcm": 0.18}
    event_bus.eval(step=1000, wer=wer, cs_wer=0.24, eval_loss=1.87)
    events = EventBus.read_events(event_bus.path)
    assert len(events) == 1
    e = events[0]
    assert e["type"] == "eval"
    assert e["step"] == 1000
    assert abs(e["wer"]["en"] - 0.09) < 1e-9
    assert abs(e["cs_wer"] - 0.24) < 1e-9
    assert abs(e["eval_loss"] - 1.87) < 1e-6


def test_checkpoint_saved_event_roundtrip(event_bus: EventBus) -> None:
    event_bus.checkpoint_saved(
        step=500, path="/workspace/models/checkpoints/run/step_500", is_best=True
    )
    events = EventBus.read_events(event_bus.path)
    assert len(events) == 1
    e = events[0]
    assert e["type"] == "checkpoint_saved"
    assert e["step"] == 500
    assert e["is_best"] is True
    assert "step_500" in e["path"]


def test_error_event_roundtrip(event_bus: EventBus) -> None:
    event_bus.error(message="CUDA OOM at step 300", step=300)
    events = EventBus.read_events(event_bus.path)
    e = events[0]
    assert e["type"] == "error"
    assert "CUDA OOM" in e["message"]
    assert e["step"] == 300


def test_train_start_roundtrip(event_bus: EventBus) -> None:
    event_bus.train_start(stage=2, config_yaml="run_name: test", total_steps=50000)
    events = EventBus.read_events(event_bus.path)
    e = events[0]
    assert e["type"] == "train_start"
    assert e["stage"] == 2
    assert e["total_steps"] == 50000


def test_train_end_roundtrip(event_bus: EventBus) -> None:
    event_bus.train_end(total_steps=50000, best_step=42000, best_eval_loss=1.23)
    events = EventBus.read_events(event_bus.path)
    e = events[0]
    assert e["type"] == "train_end"
    assert e["best_step"] == 42000


def test_multiple_events_in_order(event_bus: EventBus) -> None:
    event_bus.step(step=1, loss=3.0, lr=1e-4, grad_norm=1.0)
    event_bus.step(step=2, loss=2.9, lr=1e-4, grad_norm=0.9)
    event_bus.eval(step=2, wer={"en": 0.15})
    events = EventBus.read_events(event_bus.path)
    assert len(events) == 3
    assert events[0]["step"] == 1
    assert events[1]["step"] == 2
    assert events[2]["type"] == "eval"


def test_run_id_embedded_in_all_events(event_bus: EventBus) -> None:
    event_bus.step(step=1, loss=1.0, lr=1e-4, grad_norm=0.5)
    event_bus.eval(step=1, wer={"ha": 0.2})
    for e in EventBus.read_events(event_bus.path):
        assert e["run_id"] == "test-run-001"


def test_ts_is_float(event_bus: EventBus) -> None:
    event_bus.step(step=1, loss=1.0, lr=1e-4, grad_norm=0.5)
    e = EventBus.read_events(event_bus.path)[0]
    assert isinstance(e["ts"], float)
    assert e["ts"] > 0


# ── missing / malformed file handling ────────────────────────────────────────


def test_read_events_missing_file(tmp_path: Path) -> None:
    result = EventBus.read_events(tmp_path / "nonexistent.jsonl")
    assert result == []


def test_read_events_malformed_line_skipped(tmp_path: Path) -> None:
    p = tmp_path / "events.jsonl"
    p.write_text(
        '{"type": "step", "step": 1, "run_id": "r", "ts": 1.0}\n'
        "this is not json\n"
        '{"type": "step", "step": 2, "run_id": "r", "ts": 2.0}\n',
        encoding="utf-8",
    )
    events = EventBus.read_events(p)
    assert len(events) == 2
    assert events[0]["step"] == 1
    assert events[1]["step"] == 2


def test_read_events_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "events.jsonl"
    p.write_text("", encoding="utf-8")
    assert EventBus.read_events(p) == []


def test_read_events_blank_lines_skipped(tmp_path: Path) -> None:
    p = tmp_path / "events.jsonl"
    p.write_text('\n\n{"type":"step","step":5,"run_id":"r","ts":1.0}\n\n', encoding="utf-8")
    events = EventBus.read_events(p)
    assert len(events) == 1


# ── thread safety ─────────────────────────────────────────────────────────────


def test_concurrent_emitters_no_corruption(tmp_path: Path) -> None:
    """Multiple threads writing to the same EventBus must not corrupt the JSONL."""
    bus = EventBus(path=tmp_path / "events.jsonl", run_id="concurrent")
    n_threads = 8
    n_events_per_thread = 50

    def emit_steps(thread_id: int) -> None:
        for i in range(n_events_per_thread):
            bus.step(step=thread_id * 1000 + i, loss=float(i), lr=1e-4, grad_norm=0.5)

    threads = [threading.Thread(target=emit_steps, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    events = EventBus.read_events(bus.path)
    assert len(events) == n_threads * n_events_per_thread
    # Every line must parse cleanly (read_events silently skips bad lines)
    with bus.path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                obj = json.loads(line)  # raises on corruption
                assert obj["type"] == "step"
