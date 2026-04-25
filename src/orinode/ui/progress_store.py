"""Read training event JSONL files and reconstruct run state.

Scanning is done on every call (no caching) so the store always reflects
the latest state on disk.  For a research setup with O(10) runs this is
fast enough; add caching if needed.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from orinode.ui.schemas import EventSchema, RunDetail, RunSummary
from orinode.utils.events import EventBus


def _logs_dir() -> Path:
    from orinode.paths import WS

    return WS.logs


def _derive_stage(path: Path) -> str:
    """Extract stage name from `stage1_encoder_events.jsonl` → `stage1_encoder`."""
    return path.stem.replace("_events", "")


def _summarise(
    run_id: str,
    stage: str,
    events: list[dict[str, Any]],
) -> RunSummary:
    status = "running"
    step = 0
    total_steps = 0
    train_loss: float | None = None
    val_loss: float | None = None
    wer: float | None = None
    created_at = events[0]["ts"] if events else 0.0
    updated_at = events[-1]["ts"] if events else 0.0

    for ev in events:
        t = ev.get("type", "")
        if t == "train_start":
            total_steps = ev.get("total_steps", 0)
        elif t == "step":
            step = max(step, ev.get("step", 0))
            train_loss = ev.get("loss")
        elif t == "eval":
            val_loss = ev.get("eval_loss")
            raw_wer = ev.get("wer", {})
            if raw_wer:
                vals = [v for v in raw_wer.values() if isinstance(v, int | float)]
                if vals:
                    wer = sum(vals) / len(vals)
        elif t == "train_end":
            status = "completed"
        elif t == "error":
            status = "error"

    return RunSummary(
        run_id=run_id,
        stage=stage,
        status=status,
        step=step,
        total_steps=total_steps,
        train_loss=None if train_loss is None or math.isnan(train_loss) else train_loss,
        val_loss=None if val_loss is None or math.isnan(val_loss) else val_loss,
        wer=None if wer is None or math.isnan(wer) else wer,
        created_at=created_at,
        updated_at=updated_at,
    )


class ProgressStore:
    """Read-only view over all `*_events.jsonl` files in the logs directory."""

    def _all_event_paths(self) -> list[Path]:
        d = _logs_dir()
        if not d.exists():
            return []
        return sorted(d.glob("*_events.jsonl"))

    def _events_for_path(self, path: Path) -> list[dict[str, Any]]:
        return EventBus.read_events(path)

    def _runs_in_file(self, path: Path) -> dict[str, list[dict[str, Any]]]:
        """Group events by run_id within a single file."""
        by_run: dict[str, list[dict[str, Any]]] = {}
        for ev in self._events_for_path(path):
            rid = ev.get("run_id", path.stem)
            by_run.setdefault(rid, []).append(ev)
        return by_run

    def get_runs(self) -> list[RunSummary]:
        summaries: list[RunSummary] = []
        for path in self._all_event_paths():
            stage = _derive_stage(path)
            for run_id, events in self._runs_in_file(path).items():
                summaries.append(_summarise(run_id, stage, events))
        summaries.sort(key=lambda r: r.updated_at, reverse=True)
        return summaries

    def get_run(self, run_id: str) -> RunDetail | None:
        for path in self._all_event_paths():
            stage = _derive_stage(path)
            by_run = self._runs_in_file(path)
            if run_id not in by_run:
                continue
            events = by_run[run_id]
            summary = _summarise(run_id, stage, events)
            ev_schemas = [EventSchema.from_dict(e) for e in events]
            return RunDetail(**summary.model_dump(), events=ev_schemas)
        return None

    def get_events(self, run_id: str) -> list[EventSchema]:
        detail = self.get_run(run_id)
        return detail.events if detail else []

    def tail_events(self, run_id: str, after_ts: float) -> list[EventSchema]:
        """Return events newer than ``after_ts`` for ``run_id``."""
        return [e for e in self.get_events(run_id) if e.ts > after_ts]
