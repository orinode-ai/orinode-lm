"""FastAPI REST router for the Orinode-LM dashboard."""

from __future__ import annotations

import base64
import time

from fastapi import APIRouter, HTTPException

from orinode.ui.progress_store import ProgressStore
from orinode.ui.schemas import (
    RunDetail,
    RunSummary,
    TranscribeRequest,
    TranscribeResponse,
)

router = APIRouter()
_store = ProgressStore()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/runs", response_model=list[RunSummary])
def list_runs() -> list[RunSummary]:
    return _store.get_runs()


@router.get("/runs/{run_id}", response_model=RunDetail)
def get_run(run_id: str) -> RunDetail:
    run = _store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return run


@router.post("/transcribe", response_model=TranscribeResponse)
def transcribe(req: TranscribeRequest) -> TranscribeResponse:
    """Transcribe uploaded audio using the best available checkpoint.

    Falls back to a stub response when no checkpoint is loaded.
    """
    t0 = time.perf_counter()
    try:
        audio_bytes = base64.b64decode(req.audio_b64)
        from orinode.inference.transcribe import get_transcriber

        transcriber = get_transcriber()
        result = transcriber.transcribe(audio_bytes, language=req.language)
        text = result["text"]
        lang = result["language"]
    except Exception:  # noqa: BLE001
        text = "[No checkpoint loaded — run a training stage first]"
        lang = req.language

    latency_ms = (time.perf_counter() - t0) * 1000
    return TranscribeResponse(text=text, language=lang, latency_ms=latency_ms)
