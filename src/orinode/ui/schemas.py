"""Pydantic schemas shared by the REST API and WebSocket."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class EventSchema(BaseModel):
    ts: float
    run_id: str
    type: str
    data: dict[str, Any] = {}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EventSchema:
        top = {k: d[k] for k in ("ts", "run_id", "type") if k in d}
        top["data"] = {k: v for k, v in d.items() if k not in ("ts", "run_id", "type")}
        return cls(**top)


class RunSummary(BaseModel):
    run_id: str
    stage: str
    status: str  # "running" | "completed" | "error"
    step: int = 0
    total_steps: int = 0
    train_loss: float | None = None
    val_loss: float | None = None
    wer: float | None = None
    created_at: float = 0.0
    updated_at: float = 0.0


class RunDetail(RunSummary):
    events: list[EventSchema] = []


class TranscribeRequest(BaseModel):
    audio_b64: str  # base64-encoded WAV or FLAC
    language: str = "auto"
    max_new_tokens: int = 256


class TranscribeResponse(BaseModel):
    text: str
    language: str
    latency_ms: float
