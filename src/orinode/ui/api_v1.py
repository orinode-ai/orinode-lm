"""Versioned public REST API — /api/v1/*.

Separate from the internal training-dashboard routes in api.py.
All inference endpoints fall back gracefully when no checkpoint is loaded.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter()

_no_model_ns = ConfigDict(protected_namespaces=())


# ── Request / Response schemas ────────────────────────────────────────────────


class V1TranscribeRequest(BaseModel):
    audio_b64: str
    language: str = "auto"
    max_new_tokens: int = 256


class WordTag(BaseModel):
    word: str
    start: float
    end: float
    confidence: float


class V1TranscribeResponse(BaseModel):
    model_config = _no_model_ns

    text: str
    language: str
    languages_detected: list[str]
    word_tags: list[WordTag] = []
    latency_ms: float
    model_version: str


class V1CompareRequest(BaseModel):
    audio_b64: str
    language: str = "auto"
    checkpoint_ids: list[str] = Field(default_factory=list)


class CompareResult(BaseModel):
    checkpoint_id: str
    text: str
    latency_ms: float


class V1CompareResponse(BaseModel):
    results: list[CompareResult]


class V1EmotionRequest(BaseModel):
    audio_b64: str


class V1EmotionResponse(BaseModel):
    model_config = _no_model_ns

    top_prediction: str
    confidences: dict[str, float]
    segment_timeline: list[dict[str, Any]] | None = None
    model_version: str
    disclaimer: str
    latency_ms: float


class V1GenderRequest(BaseModel):
    audio_b64: str
    per_speaker: bool = False


class V1GenderResponse(BaseModel):
    model_config = _no_model_ns

    prediction: str
    confidence: float
    model_version: str
    per_speaker: list[dict[str, Any]] | None = None
    latency_ms: float


class V1AnalyzeRequest(BaseModel):
    audio_b64: str
    language: str = "auto"
    include_emotion: bool = True
    include_gender: bool = True


class V1AnalyzeResponse(BaseModel):
    transcription: dict[str, Any]
    emotion: dict[str, Any] | None = None
    gender: dict[str, Any] | None = None
    total_latency_ms: float


class CheckpointInfo(BaseModel):
    id: str
    stage: str
    created_at: float
    size_bytes: int
    path: str


class V1StatsResponse(BaseModel):
    total_hours: float
    languages: list[str]
    checkpoints_count: int
    runs_count: int


class SampleMeta(BaseModel):
    id: str
    title: str
    language: str
    duration_sec: float
    emotion: str | None = None
    gender: str | None = None
    transcript: str | None = None


class V1FeedbackRequest(BaseModel):
    model_config = _no_model_ns

    task: str = Field(..., description="'transcribe' | 'emotion' | 'gender'")
    rating: int = Field(..., ge=1, le=5)
    comment: str = ""
    audio_b64: str | None = None
    model_version: str = ""


# ── Helpers ───────────────────────────────────────────────────────────────────


def _decode_audio(audio_b64: str) -> bytes:
    try:
        return base64.b64decode(audio_b64)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Invalid base64 audio") from exc


def _list_checkpoints() -> list[CheckpointInfo]:
    from orinode.paths import WS

    ckpt_dir = WS.models_checkpoints
    if not ckpt_dir.exists():
        return []

    results = []
    for pt in sorted(ckpt_dir.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
        stage = pt.parent.name if pt.parent != ckpt_dir else "unknown"
        stat = pt.stat()
        results.append(
            CheckpointInfo(
                id=pt.stem,
                stage=stage,
                created_at=stat.st_mtime,
                size_bytes=stat.st_size,
                path=str(pt.relative_to(ckpt_dir)),
            )
        )
    return results


def _samples_dir() -> Path:
    return Path(__file__).parent.parent.parent.parent / "ui" / "frontend" / "public" / "samples"


def _load_samples_json() -> list[dict[str, Any]]:
    p = _samples_dir() / "samples.json"
    if not p.exists():
        return []
    with p.open() as fh:
        return json.load(fh)


# ── Transcription ──────────────────────────────────────────────────────────────


@router.post("/transcribe", response_model=V1TranscribeResponse)
def v1_transcribe(req: V1TranscribeRequest) -> V1TranscribeResponse:
    t0 = time.perf_counter()
    audio = _decode_audio(req.audio_b64)
    text = "[No checkpoint loaded — run: make train-stage1]"
    lang = req.language if req.language != "auto" else "en"
    model_version = "none"

    try:
        from orinode.inference.transcribe import get_transcriber

        tr = get_transcriber()
        result = tr.transcribe(audio, language=req.language)
        text = result["text"]
        lang = result.get("language", lang)
        model_version = result.get("model_version", "speech_llm/v1")
    except Exception:  # noqa: BLE001
        pass

    latency_ms = (time.perf_counter() - t0) * 1000
    return V1TranscribeResponse(
        text=text,
        language=lang,
        languages_detected=[lang],
        word_tags=[],
        latency_ms=round(latency_ms, 1),
        model_version=model_version,
    )


@router.post("/transcribe/compare", response_model=V1CompareResponse)
def v1_transcribe_compare(req: V1CompareRequest) -> V1CompareResponse:
    audio = _decode_audio(req.audio_b64)
    ckpts = _list_checkpoints()
    ids = req.checkpoint_ids or [c.id for c in ckpts[:3]]

    results: list[CompareResult] = []
    for ckpt_id in ids:
        t0 = time.perf_counter()
        text = f"[Checkpoint {ckpt_id} not loaded]"
        try:
            from orinode.inference.transcribe import get_transcriber

            tr = get_transcriber()
            res = tr.transcribe(audio, language=req.language)
            text = res["text"]
        except Exception:  # noqa: BLE001
            pass
        latency_ms = (time.perf_counter() - t0) * 1000
        results.append(
            CompareResult(checkpoint_id=ckpt_id, text=text, latency_ms=round(latency_ms, 1))
        )

    return V1CompareResponse(results=results)


# ── Emotion ────────────────────────────────────────────────────────────────────


@router.post("/emotion", response_model=V1EmotionResponse)
def v1_emotion(req: V1EmotionRequest) -> V1EmotionResponse:
    t0 = time.perf_counter()
    audio = _decode_audio(req.audio_b64)

    try:
        from orinode.inference.emotion_pipeline import get_emotion_pipeline

        pipeline = get_emotion_pipeline()
        result = pipeline.predict(audio)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Emotion inference failed: {exc}") from exc

    latency_ms = (time.perf_counter() - t0) * 1000
    return V1EmotionResponse(
        top_prediction=result["top_prediction"],
        confidences=result["confidences"],
        segment_timeline=result.get("segment_timeline"),
        model_version=result["model_version"],
        disclaimer=result["disclaimer"],
        latency_ms=round(latency_ms, 1),
    )


# ── Gender ─────────────────────────────────────────────────────────────────────


@router.post("/gender", response_model=V1GenderResponse)
def v1_gender(req: V1GenderRequest) -> V1GenderResponse:
    t0 = time.perf_counter()
    audio = _decode_audio(req.audio_b64)

    try:
        from orinode.inference.gender_pipeline import get_gender_pipeline

        pipeline = get_gender_pipeline()
        result = pipeline.predict(audio, per_speaker=req.per_speaker)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Gender inference failed: {exc}") from exc

    latency_ms = (time.perf_counter() - t0) * 1000
    return V1GenderResponse(
        prediction=result["prediction"],
        confidence=result["confidence"],
        model_version=result["model_version"],
        per_speaker=result.get("per_speaker"),
        latency_ms=round(latency_ms, 1),
    )


# ── Combined analysis ──────────────────────────────────────────────────────────


@router.post("/analyze", response_model=V1AnalyzeResponse)
def v1_analyze(req: V1AnalyzeRequest) -> V1AnalyzeResponse:
    t0 = time.perf_counter()
    audio = _decode_audio(req.audio_b64)

    transcription: dict[str, Any] = {"text": "[No checkpoint]", "language": req.language}
    emotion: dict[str, Any] | None = None
    gender: dict[str, Any] | None = None

    try:
        from orinode.inference.transcribe import get_transcriber

        tr = get_transcriber()
        result = tr.transcribe(audio, language=req.language)
        transcription = {"text": result["text"], "language": result.get("language", req.language)}
    except Exception:  # noqa: BLE001
        pass

    if req.include_emotion:
        try:
            from orinode.inference.emotion_pipeline import get_emotion_pipeline

            emotion = get_emotion_pipeline().predict(audio)
        except Exception:  # noqa: BLE001
            emotion = None

    if req.include_gender:
        try:
            from orinode.inference.gender_pipeline import get_gender_pipeline

            gender = get_gender_pipeline().predict(audio)
        except Exception:  # noqa: BLE001
            gender = None

    total_latency_ms = (time.perf_counter() - t0) * 1000
    return V1AnalyzeResponse(
        transcription=transcription,
        emotion=emotion,
        gender=gender,
        total_latency_ms=round(total_latency_ms, 1),
    )


# ── Checkpoints ────────────────────────────────────────────────────────────────


@router.get("/checkpoints", response_model=list[CheckpointInfo])
def v1_list_checkpoints() -> list[CheckpointInfo]:
    return _list_checkpoints()


@router.get("/checkpoints/{checkpoint_id}", response_model=CheckpointInfo)
def v1_get_checkpoint(checkpoint_id: str) -> CheckpointInfo:
    for ckpt in _list_checkpoints():
        if ckpt.id == checkpoint_id:
            return ckpt
    raise HTTPException(status_code=404, detail=f"Checkpoint '{checkpoint_id}' not found")


# ── Stats ──────────────────────────────────────────────────────────────────────


@router.get("/stats", response_model=V1StatsResponse)
def v1_stats() -> V1StatsResponse:
    from orinode.paths import WS
    from orinode.ui.progress_store import ProgressStore

    store = ProgressStore()
    runs = store.get_runs()
    ckpts = _list_checkpoints()

    # Estimate total processed hours from manifest line counts (lightweight)
    total_hours = 0.0
    if WS.data_manifests.exists():
        for mf in WS.data_manifests.glob("*.jsonl"):
            try:
                lines = mf.read_text().count("\n")
                total_hours += lines * 5.0 / 3600  # rough: 5 s/utterance
            except Exception:  # noqa: BLE001
                pass

    return V1StatsResponse(
        total_hours=round(total_hours, 1),
        languages=["en", "ha", "yo", "ig", "pcm"],
        checkpoints_count=len(ckpts),
        runs_count=len(runs),
    )


# ── Samples ────────────────────────────────────────────────────────────────────


@router.get("/samples", response_model=list[SampleMeta])
def v1_list_samples() -> list[SampleMeta]:
    raw = _load_samples_json()
    return [SampleMeta(**s) for s in raw]


@router.get("/samples/{sample_id}/audio")
def v1_sample_audio(sample_id: str) -> FileResponse:
    raw = _load_samples_json()
    for s in raw:
        if s.get("id") == sample_id:
            audio_path = _samples_dir() / s.get("file", f"{sample_id}.flac")
            if audio_path.exists():
                return FileResponse(audio_path, media_type="audio/flac")
            raise HTTPException(status_code=404, detail="Audio file not found on disk")
    raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found")


# ── Feedback ───────────────────────────────────────────────────────────────────


@router.post("/feedback", status_code=201)
def v1_feedback(req: V1FeedbackRequest) -> dict[str, str]:
    from orinode.paths import WS

    feedback_dir = WS.logs / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)

    allowed_tasks = {"transcribe", "emotion", "gender", "analyze"}
    task = req.task if req.task in allowed_tasks else "other"
    dest = feedback_dir / f"{task}.jsonl"

    entry: dict[str, Any] = {
        "ts": time.time(),
        "task": task,
        "rating": req.rating,
        "comment": req.comment,
        "model_version": req.model_version,
    }

    with dest.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")

    return {"status": "recorded"}
