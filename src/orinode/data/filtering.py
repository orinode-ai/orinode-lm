"""Three-pass data filtering for Nigerian speech corpora.

Pass 1 — metadata_filter: duration, transcript length, speech rate.
Pass 2 — audio_quality_check: RMS, clipping, SNR, silence fraction (post-preprocessing).
Pass 3 — transcript_quality_check: Whisper CER against ground-truth transcript.
"""

from __future__ import annotations

import math
import multiprocessing
import random
from dataclasses import dataclass, field

NIGERIAN_ACCENTS: frozenset[str] = frozenset(
    {
        # Core ethnic groups
        "yoruba", "igbo", "hausa", "nigerian english", "nigerian",
        "edo", "ijaw", "tiv", "efik", "ibibio", "kanuri", "fulani",
        "fulfulde", "idoma", "igala", "urhobo", "nupe", "kalabari",
        "annang", "itsekiri", "ogoni", "isoko", "esan",
        # Niger Delta (alternate spellings + additional groups)
        "izon",      # alternate spelling of Ijaw
        "nembe",     # Ijoid, Bayelsa/Rivers
        "ikwere",    # Igboid, Rivers state
        "epie",      # Edoid, Bayelsa
        "bekwarra",  # Cross River state
        # Middle Belt / North Central
        "ebira",     # Kogi state
        "alago",     # Nasarawa state
        # Compound / hybrid labels
        "hausa/fulani",
        "hausa fulani",
        "pidgin",
        "nigerian pidgin",
    }
)

_MEDICAL_KEYWORDS: frozenset[str] = frozenset(
    {
        "patient",
        "hospital",
        "doctor",
        "nurse",
        "clinic",
        "medical",
        "diagnosis",
        "prescription",
        "treatment",
        "surgery",
        "symptom",
        "disease",
        "infection",
        "medicine",
        "blood",
        "laboratory",
        "health",
        "therapy",
    }
)


# ── shared result ─────────────────────────────────────────────────────────────


@dataclass
class FilterResult:
    keep: bool
    reason: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"keep": self.keep, "reason": self.reason, "details": self.details}

    @classmethod
    def from_dict(cls, d: dict) -> FilterResult:
        return cls(keep=d["keep"], reason=d["reason"], details=d.get("details", {}))


# ── Pass 1 — metadata ─────────────────────────────────────────────────────────


@dataclass
class MetadataFilterConfig:
    min_duration_sec: float = 1.0
    max_duration_sec: float = 30.0
    min_chars: int = 3
    min_words_per_sec: float = 0.5
    max_words_per_sec: float = 6.0
    max_chars_per_sec: float = 30.0
    max_clips_per_speaker: int = 200
    random_seed: int = 42


def _is_domain_clinical(text: str) -> bool:
    words = text.lower().split()
    return sum(1 for w in words if w in _MEDICAL_KEYWORDS) >= 2


def metadata_filter(row: dict, config: MetadataFilterConfig) -> FilterResult:
    """Apply Pass-1 metadata filters to one AfriSpeech row.

    Accent is not used for rejection — rows arrive pre-filtered by country at
    download time. NIGERIAN_ACCENTS is checked only to populate the
    _accent_in_allowlist flag for downstream Stage 2 analysis.

    Rejection reasons:
        too_short, too_long, empty_transcript, impossible_speech_rate,
        transcript_too_long, speaker_cap_exceeded
    """
    duration: float = float(row.get("duration", 0.0))
    transcript: str = str(row.get("transcript", "") or "")
    accent_raw: str = str(row.get("accent", "") or "")
    accent: str = accent_raw.strip().lower()

    if duration < config.min_duration_sec:
        return FilterResult(keep=False, reason="too_short", details={"duration": duration})

    if duration > config.max_duration_sec:
        return FilterResult(keep=False, reason="too_long", details={"duration": duration})

    text = transcript.strip()
    if len(text) < config.min_chars:
        return FilterResult(keep=False, reason="empty_transcript", details={"chars": len(text)})

    word_count = len(text.split())
    words_per_sec = word_count / max(duration, 1e-6)
    chars_per_sec = len(text) / max(duration, 1e-6)

    if words_per_sec < config.min_words_per_sec or words_per_sec > config.max_words_per_sec:
        return FilterResult(
            keep=False,
            reason="impossible_speech_rate",
            details={"words_per_sec": round(words_per_sec, 3), "duration": duration},
        )

    if chars_per_sec > config.max_chars_per_sec:
        return FilterResult(
            keep=False,
            reason="transcript_too_long",
            details={"chars_per_sec": round(chars_per_sec, 3)},
        )

    accent_in_allowlist = accent in NIGERIAN_ACCENTS or any(
        part in NIGERIAN_ACCENTS
        for part in accent.replace("/", " ").split()
    )

    domain = "clinical" if _is_domain_clinical(text) else "general"
    return FilterResult(
        keep=True,
        reason="pass",
        details={
            "duration": duration,
            "words": word_count,
            "words_per_sec": round(words_per_sec, 3),
            "domain": domain,
            "accent": accent,
            "_accent_in_allowlist": accent_in_allowlist,
        },
    )


def apply_speaker_cap(
    rows: list[dict],
    results: list[FilterResult],
    config: MetadataFilterConfig,
) -> list[FilterResult]:
    """Cap utterances per speaker by randomly sampling kept rows.

    Args:
        rows: Raw metadata rows (same order as ``results``).
        results: Pass-1 filter results — modified in-place for capped rows.
        config: ``MetadataFilterConfig`` with ``max_clips_per_speaker`` and ``random_seed``.

    Returns:
        Updated ``results`` list (same object, modified in-place).
    """
    rng = random.Random(config.random_seed)
    speaker_kept: dict[str, list[int]] = {}

    for idx, (row, result) in enumerate(zip(rows, results, strict=False)):
        if not result.keep:
            continue
        spk = str(row.get("speaker_id", "") or "")
        speaker_kept.setdefault(spk, []).append(idx)

    for spk, indices in speaker_kept.items():
        if len(indices) <= config.max_clips_per_speaker:
            continue
        to_drop = rng.sample(indices, len(indices) - config.max_clips_per_speaker)
        for idx in to_drop:
            results[idx] = FilterResult(
                keep=False,
                reason="speaker_cap",
                details={"speaker_id": spk, "kept": config.max_clips_per_speaker},
            )

    return results


# ── Pass 2 — audio quality ────────────────────────────────────────────────────


@dataclass
class AudioFilterConfig:
    min_samples: int = 1000
    min_rms_dbfs: float = -45.0
    max_clip_frac: float = 0.02
    min_snr_db: float = 10.0
    max_silence_frac: float = 0.70
    frame_size_ms: float = 25.0
    num_workers: int = 14


def audio_quality_check(audio_path: str, config: AudioFilterConfig) -> FilterResult:
    """Load pre-processed 16 kHz mono FLAC and run quality checks.

    Rejection reasons:
        corrupt_audio, too_few_samples, silent, clipped, low_snr, mostly_silence

    Always stores: sample_rate, duration_sec, rms_db, snr_db, peak_db.
    """
    import numpy as np

    try:
        import soundfile as sf

        data, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    except Exception as exc:
        return FilterResult(
            keep=False, reason="corrupt_audio", details={"error": str(exc), "path": audio_path}
        )

    if data.ndim > 1:
        data = data.mean(axis=1)

    n_samples = len(data)
    details: dict = {
        "sample_rate": sr,
        "duration_sec": round(n_samples / max(sr, 1), 4),
    }

    if n_samples < config.min_samples:
        details["n_samples"] = n_samples
        return FilterResult(keep=False, reason="too_few_samples", details=details)

    rms = float(np.sqrt(np.mean(data**2)))
    rms_db = 20 * math.log10(max(rms, 1e-9))
    peak_db = 20 * math.log10(max(float(np.abs(data).max()), 1e-9))
    details["rms_db"] = round(rms_db, 3)
    details["peak_db"] = round(peak_db, 3)

    if rms_db < config.min_rms_dbfs:
        return FilterResult(keep=False, reason="silent", details=details)

    clip_frac = float(np.mean(np.abs(data) >= 0.999))
    details["clip_frac"] = round(clip_frac, 6)
    if clip_frac > config.max_clip_frac:
        return FilterResult(keep=False, reason="clipped", details=details)

    # Simple SNR via RMS of frame-wise energy variance
    frame_size = max(1, int(sr * config.frame_size_ms / 1000.0))
    n_frames = n_samples // frame_size
    if n_frames > 1:
        frames = data[: n_frames * frame_size].reshape(n_frames, frame_size)
        frame_rms = np.sqrt(np.mean(frames**2, axis=1))
        signal_rms = float(frame_rms.max())
        noise_rms = float(frame_rms.min() + 1e-9)
        snr_db = 20 * math.log10(max(signal_rms, 1e-9) / noise_rms)
    else:
        snr_db = 0.0
    details["snr_db"] = round(snr_db, 3)

    if snr_db < config.min_snr_db:
        return FilterResult(keep=False, reason="low_snr", details=details)

    # Silence fraction: frames below -50 dBFS
    silence_threshold_rms = 10 ** (-50.0 / 20.0)
    silence_frac = (
        float(np.mean(frame_rms < silence_threshold_rms)) if n_frames > 0 else 1.0
    )
    details["silence_frac"] = round(silence_frac, 4)

    if silence_frac > config.max_silence_frac:
        return FilterResult(keep=False, reason="mostly_silence", details=details)

    return FilterResult(keep=True, reason="pass", details=details)


def _audio_worker(args: tuple[str, AudioFilterConfig]) -> tuple[str, FilterResult]:
    path, cfg = args
    return path, audio_quality_check(path, cfg)


def run_audio_filter_parallel(
    audio_paths: list[str],
    config: AudioFilterConfig,
) -> list[FilterResult]:
    """Run audio_quality_check in parallel using multiprocessing.Pool."""
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    jobs = [(p, config) for p in audio_paths]
    results_map: dict[str, FilterResult] = {}

    with multiprocessing.Pool(processes=config.num_workers) as pool:
        it = pool.imap_unordered(_audio_worker, jobs, chunksize=32)
        if tqdm is not None:
            it = tqdm(it, total=len(jobs), desc="Pass 2 audio quality")
        for path, result in it:
            results_map[path] = result

    return [results_map[p] for p in audio_paths]


# ── Pass 3 — transcript quality ───────────────────────────────────────────────


@dataclass
class TranscriptFilterConfig:
    # 0.40 is deliberately generous: Whisper baseline on Nigerian English is
    # ~10-15% WER. This catches obviously wrong transcripts, not borderline ones.
    max_cer: float = 0.40
    asr_model_path: str = "openai/whisper-large-v3"
    batch_size: int = 16
    device: str = "cuda"
    use_bf16: bool = True


def transcript_quality_check(
    audio_path: str,
    transcript: str,
    asr_pipeline: object,
    config: TranscriptFilterConfig,
) -> FilterResult:
    """Run Whisper on audio, compare to ground-truth via CER.

    Stores CER in details regardless of outcome.
    Flags non-English ASR output as is_code_switched without rejecting.
    """
    from jiwer import cer as compute_cer

    from orinode.data.preprocessing import load_audio
    from orinode.data.text_normalization import normalize_transcript

    details: dict = {"audio_path": audio_path}

    try:
        waveform = load_audio(audio_path)
        audio_np = waveform.squeeze(0).numpy()
        result = asr_pipeline(  # type: ignore[operator]
            {"raw": audio_np, "sampling_rate": 16000},
            return_language=True,
        )
        hypothesis: str = result["text"] if isinstance(result, dict) else str(result)
        detected_lang: str = (
            result.get("chunks", [{}])[0].get("language", "en")
            if isinstance(result, dict)
            else "en"
        )
    except Exception as exc:
        return FilterResult(
            keep=False,
            reason="asr_error",
            details={"error": str(exc), **details},
        )

    ref = normalize_transcript(transcript, language="en")
    hyp = normalize_transcript(hypothesis, language="en")

    try:
        cer_score = float(compute_cer(ref, hyp))
    except Exception:
        cer_score = 1.0

    details["cer"] = round(cer_score, 4)
    details["hypothesis"] = hypothesis[:200]
    details["detected_language"] = detected_lang

    if cer_score > config.max_cer:
        return FilterResult(
            keep=False,
            reason=f"transcript_mismatch_cer_{cer_score:.2f}",
            details=details,
        )

    if detected_lang != "en":
        details["is_code_switched"] = True

    return FilterResult(keep=True, reason="pass", details=details)
