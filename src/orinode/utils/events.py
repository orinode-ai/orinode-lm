"""Typed JSONL-backed training event bus.

Training scripts write events; the UI server tails the file with watchdog
and forwards updates over WebSocket. The two sides share only this schema
and the file path — training never imports from the UI, and the UI works
whether or not training is currently running.

Event types:
    step              — emitted every ``log_interval`` training steps
    eval              — emitted after each eval run (WER per language)
    checkpoint_saved  — emitted when a checkpoint is written to disk
    epoch_complete    — emitted at the end of each epoch
    error             — emitted on unhandled exceptions in training
    train_start       — emitted when a training run begins
    train_end         — emitted when a training run finishes

Wire format (one JSON object per line)::

    {"ts": 1700000000.1, "run_id": "stage2-20260115", "type": "step",
     "step": 1000, "loss": 2.34, "lr": 5e-05, "grad_norm": 0.8, "epoch": 1}
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ── event dataclasses ─────────────────────────────────────────────────────────


@dataclass
class BaseEvent:
    ts: float = field(default_factory=time.time)
    run_id: str = ""
    type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class StepEvent(BaseEvent):
    type: str = "step"
    step: int = 0
    loss: float = 0.0
    lr: float = 0.0
    grad_norm: float = 0.0
    epoch: int = 0


@dataclass
class EvalEvent(BaseEvent):
    type: str = "eval"
    step: int = 0
    wer: dict[str, float] = field(default_factory=dict)
    cs_wer: float | None = None
    eval_loss: float | None = None


@dataclass
class CheckpointSavedEvent(BaseEvent):
    type: str = "checkpoint_saved"
    step: int = 0
    path: str = ""
    is_best: bool = False


@dataclass
class EpochCompleteEvent(BaseEvent):
    type: str = "epoch_complete"
    epoch: int = 0
    step: int = 0
    avg_loss: float = 0.0


@dataclass
class ErrorEvent(BaseEvent):
    type: str = "error"
    message: str = ""
    step: int | None = None


@dataclass
class TrainStartEvent(BaseEvent):
    type: str = "train_start"
    stage: int = 0
    run_name: str = ""
    config_yaml: str = ""
    total_steps: int = 0
    augmentation: str | None = None


@dataclass
class TrainEndEvent(BaseEvent):
    type: str = "train_end"
    total_steps: int = 0
    best_step: int | None = None
    best_eval_loss: float | None = None


# ── event bus ─────────────────────────────────────────────────────────────────


class EventBus:
    """Thread-safe JSONL event writer.

    Each ``emit`` appends one JSON line to the file and flushes immediately
    so the UI's file-tail can pick it up without buffering delay. The write
    lock is per-instance, so multiple concurrent EventBus objects on the same
    file are safe.

    Args:
        path: Path to the JSONL output file.
        run_id: Training run identifier embedded in every event.

    Example::

        bus = EventBus(WS.events_file("stage2-20260115"), "stage2-20260115")
        bus.step(step=1000, loss=2.34, lr=5e-5, grad_norm=0.8)
        bus.eval(step=1000, wer={"en": 0.09, "ha": 0.14}, cs_wer=0.24)
    """

    def __init__(self, path: Path, run_id: str) -> None:
        self.path = path
        self.run_id = run_id
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: BaseEvent) -> None:
        """Write one event as a JSON line (thread-safe, non-blocking)."""
        event.run_id = self.run_id
        event.ts = time.time()
        line = json.dumps(event.to_dict(), ensure_ascii=False) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()

    # ── convenience emitters ──────────────────────────────────────────────────

    def step(
        self,
        step: int,
        loss: float,
        lr: float,
        grad_norm: float,
        epoch: int = 0,
    ) -> None:
        self.emit(StepEvent(step=step, loss=loss, lr=lr, grad_norm=grad_norm, epoch=epoch))

    def eval(
        self,
        step: int,
        wer: dict[str, float],
        cs_wer: float | None = None,
        eval_loss: float | None = None,
    ) -> None:
        self.emit(EvalEvent(step=step, wer=wer, cs_wer=cs_wer, eval_loss=eval_loss))

    def checkpoint_saved(self, step: int, path: str, is_best: bool = False) -> None:
        self.emit(CheckpointSavedEvent(step=step, path=path, is_best=is_best))

    def epoch_complete(self, epoch: int, step: int, avg_loss: float) -> None:
        self.emit(EpochCompleteEvent(epoch=epoch, step=step, avg_loss=avg_loss))

    def error(self, message: str, step: int | None = None) -> None:
        self.emit(ErrorEvent(message=message, step=step))

    def train_start(self, stage: int, config_yaml: str, total_steps: int) -> None:
        self.emit(TrainStartEvent(stage=stage, config_yaml=config_yaml, total_steps=total_steps))

    def train_end(
        self,
        total_steps: int,
        best_step: int | None = None,
        best_eval_loss: float | None = None,
    ) -> None:
        self.emit(
            TrainEndEvent(
                total_steps=total_steps,
                best_step=best_step,
                best_eval_loss=best_eval_loss,
            )
        )

    # ── reader (static) ───────────────────────────────────────────────────────

    @staticmethod
    def read_events(path: Path) -> list[dict[str, Any]]:
        """Read all events from a JSONL file; returns ``[]`` if file is missing.

        Malformed lines are silently skipped (a partially-written line from a
        crash should not abort a UI reload).
        """
        if not path.exists():
            return []
        events: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return events
