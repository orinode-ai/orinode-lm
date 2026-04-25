"""Audio preprocessing pipeline.

Runtime path (model inference):
    ``load_audio`` / ``load_audio_bytes`` — 16 kHz mono float32 via torchaudio.

Offline data-prep path:
    ``preprocess_clip`` — resample + loudness-normalise + save 16-bit FLAC.
    Uses soundfile + librosa + pyloudnorm (not a training dependency).

This is the only place sample-rate logic lives.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T

TARGET_SAMPLE_RATE: int = 16_000
TARGET_CHANNELS: int = 1
MAX_DURATION_SECONDS: float = 30.0
MIN_DURATION_SECONDS: float = 0.5

# Whisper-large-v3 feature extraction constants
WHISPER_N_MELS: int = 128
WHISPER_N_FFT: int = 400
WHISPER_HOP_LENGTH: int = 160


def load_audio(
    path: str | Path,
    target_sr: int = TARGET_SAMPLE_RATE,
) -> torch.Tensor:
    """Load an audio file as a 16 kHz mono float32 tensor.

    Args:
        path: Path to the audio file (.wav, .flac, .mp3, .ogg, .opus).
        target_sr: Target sample rate; defaults to 16 000 Hz.

    Returns:
        Float32 tensor of shape ``(1, num_samples)``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        RuntimeError: If torchaudio cannot decode the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    waveform, sr = torchaudio.load(str(path))
    return _normalise(waveform, sr, target_sr)


def load_audio_bytes(
    audio_bytes: bytes,
    target_sr: int = TARGET_SAMPLE_RATE,
) -> torch.Tensor:
    """Load audio from raw bytes (e.g. an HTTP multipart upload).

    Args:
        audio_bytes: Raw bytes in any torchaudio-supported format.
        target_sr: Target sample rate.

    Returns:
        Float32 tensor of shape ``(1, num_samples)``.
    """
    waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
    return _normalise(waveform, sr, target_sr)


def _normalise(waveform: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = T.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
    return waveform.float()


def get_duration(waveform: torch.Tensor, sample_rate: int = TARGET_SAMPLE_RATE) -> float:
    """Return duration in seconds."""
    return waveform.shape[-1] / sample_rate


def is_valid_duration(
    waveform: torch.Tensor,
    sample_rate: int = TARGET_SAMPLE_RATE,
    min_s: float = MIN_DURATION_SECONDS,
    max_s: float = MAX_DURATION_SECONDS,
) -> bool:
    """Return ``True`` if the waveform is between ``min_s`` and ``max_s`` seconds."""
    dur = get_duration(waveform, sample_rate)
    return min_s <= dur <= max_s


def save_flac(
    waveform: torch.Tensor,
    path: Path,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> None:
    """Save a waveform as a 16 kHz mono FLAC file.

    Args:
        waveform: Float32 tensor ``(1, num_samples)``.
        path: Output path. Parent directory is created if absent.
        sample_rate: Sample rate of the tensor.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), waveform, sample_rate, format="flac")


# ── offline data-prep ─────────────────────────────────────────────────────────


@dataclass
class PreprocessConfig:
    target_sample_rate: int = 16_000
    target_lufs: float = -27.0
    peak_ceiling: float = 0.99
    output_subtype: str = "PCM_16"
    skip_if_exists: bool = True


@dataclass
class PreprocessResult:
    success: bool
    raw_path: str
    output_path: str
    original_sample_rate: int
    duration_sec: float
    applied_loudnorm: bool
    error: str | None = None


def preprocess_clip(
    input_path: str | Path,
    output_path: str | Path,
    config: PreprocessConfig | None = None,
) -> PreprocessResult:
    """Resample to 16 kHz mono, loudness-normalise, save as 16-bit FLAC.

    Rules (immutable):
    - 16 kHz — Whisper's required sample rate.
    - Mono by averaging channels, NEVER by taking only left.
    - -27 LUFS — headroom for µ-law codec simulation during augmentation.
    - 16-bit PCM FLAC output.
    - NEVER denoise, dereverb, or EQ.
    - NEVER precompute mel-spectrograms here.
    """
    import librosa
    import numpy as np
    import pyloudnorm as pyln
    import soundfile as sf

    cfg = config or PreprocessConfig()
    raw = str(input_path)
    out = Path(output_path)

    if cfg.skip_if_exists and out.exists():
        try:
            info = sf.info(str(out))
            return PreprocessResult(
                success=True,
                raw_path=raw,
                output_path=str(out),
                original_sample_rate=info.samplerate,
                duration_sec=info.duration,
                applied_loudnorm=False,
            )
        except Exception:
            pass

    try:
        data, orig_sr = sf.read(raw, dtype="float32", always_2d=True)
        # Stereo → mono by averaging
        data = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]

        # Resample to 16 kHz
        if orig_sr != cfg.target_sample_rate:
            data = librosa.resample(
                data, orig_sr=orig_sr, target_sr=cfg.target_sample_rate, res_type="kaiser_best"
            )

        duration_sec = len(data) / cfg.target_sample_rate

        # Loudness normalise via pyloudnorm
        meter = pyln.Meter(cfg.target_sample_rate)
        loudness = meter.integrated_loudness(data)
        applied_loudnorm = False
        if not (loudness == float("-inf") or np.isnan(loudness)):
            data = pyln.normalize.loudness(data, loudness, cfg.target_lufs)
            applied_loudnorm = True

        # Peak ceiling
        peak = float(np.abs(data).max())
        if peak > cfg.peak_ceiling:
            data = data * (cfg.peak_ceiling / peak)

        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), data, cfg.target_sample_rate, subtype=cfg.output_subtype)

        return PreprocessResult(
            success=True,
            raw_path=raw,
            output_path=str(out),
            original_sample_rate=orig_sr,
            duration_sec=round(duration_sec, 4),
            applied_loudnorm=applied_loudnorm,
        )

    except Exception as exc:
        return PreprocessResult(
            success=False,
            raw_path=raw,
            output_path=str(output_path),
            original_sample_rate=0,
            duration_sec=0.0,
            applied_loudnorm=False,
            error=str(exc),
        )


def extract_log_mel(
    waveform: torch.Tensor,
    sample_rate: int = TARGET_SAMPLE_RATE,
    n_mels: int = WHISPER_N_MELS,
    n_fft: int = WHISPER_N_FFT,
    hop_length: int = WHISPER_HOP_LENGTH,
) -> torch.Tensor:
    """Extract a log-mel spectrogram matching Whisper's feature extraction.

    Args:
        waveform: Float32 tensor ``(1, T)`` at ``sample_rate``.
        sample_rate: Audio sample rate (must be 16 000 for Whisper compat.).
        n_mels: Number of mel filter banks. Whisper-large-v3 uses 128.
        n_fft: FFT window size.
        hop_length: Hop length in samples.

    Returns:
        Log-mel spectrogram of shape ``(1, n_mels, T')``.
    """
    mel = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )(waveform)
    return torch.log(mel.clamp(min=1e-9))
