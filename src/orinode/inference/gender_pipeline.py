"""Gender prediction pipeline.

Lazy-loads the aux_gender checkpoint on first call.
Per-speaker mode uses pyannote diarization if available; falls back to
single-speaker when it is not installed or no checkpoint is found.

Usage::

    from orinode.inference.gender_pipeline import get_gender_pipeline
    result = get_gender_pipeline().predict(audio_bytes)
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import torch

from orinode.models.gender_classifier import GenderClassifier
from orinode.utils.logging import get_logger

log = get_logger(__name__)

_SAMPLE_RATE = 16_000
_PIPELINE: GenderPipeline | None = None


class GenderPipeline:
    """Load a GenderClassifier checkpoint and run prediction.

    Args:
        checkpoint_path: Path to a saved ``GenderClassifier`` state dict.
        device: ``"cuda"`` or ``"cpu"``.
    """

    MODEL_VERSION = "aux_gender/v1"

    def __init__(self, checkpoint_path: Path, device: str = "cpu") -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._model: GenderClassifier | None = None

    def _load(self) -> GenderClassifier:
        if self._model is not None:
            return self._model
        log.info(f"Loading gender checkpoint: {self.checkpoint_path}")
        model = GenderClassifier.for_smoke_test()
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        model.eval()
        self._model = model.to(self.device)
        return self._model

    def predict(
        self,
        audio_bytes: bytes,
        per_speaker: bool = False,
    ) -> dict[str, Any]:
        """Predict gender from raw audio bytes.

        Args:
            audio_bytes: Raw WAV or FLAC audio.
            per_speaker: If True, attempt diarization and return per-speaker
                predictions. Falls back to single-speaker silently.

        Returns:
            Dict with ``prediction``, ``confidence``, ``model_version``,
            optionally ``per_speaker`` list.
        """
        import torchaudio

        model = self._load()
        waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
        if sr != _SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, _SAMPLE_RATE)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.squeeze(0).unsqueeze(0).to(self.device)  # (1, T)

        with torch.inference_mode():
            log_probs = model(input_values=waveform)
            probs = log_probs.exp().squeeze(0)

        pred_idx = probs.argmax().item()
        prediction = GenderClassifier.LABELS[pred_idx]
        confidence = probs[pred_idx].item()

        result: dict[str, Any] = {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "model_version": self.MODEL_VERSION,
        }

        if per_speaker:
            result["per_speaker"] = _diarize_and_predict(audio_bytes, model, self.device)

        return result


def _diarize_and_predict(
    audio_bytes: bytes,
    model: GenderClassifier,
    device: str,
) -> list[dict[str, Any]] | None:
    """Attempt pyannote diarization; return None if unavailable."""
    try:
        from pyannote.audio import Pipeline as PyannotePipeline  # type: ignore[import]
    except ImportError:
        return None

    try:
        import torchaudio

        waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
        if sr != _SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, _SAMPLE_RATE)

        diarization = PyannotePipeline.from_pretrained("pyannote/speaker-diarization-3.1")(
            {"waveform": waveform, "sample_rate": _SAMPLE_RATE}
        )

        results = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            seg = waveform[:, int(turn.start * _SAMPLE_RATE) : int(turn.end * _SAMPLE_RATE)]
            if seg.shape[1] < 1600:
                continue
            seg = seg.mean(0, keepdim=True).to(device)
            with torch.inference_mode():
                probs = model(input_values=seg).exp().squeeze(0)
            pred_idx = probs.argmax().item()
            results.append(
                {
                    "speaker": speaker,
                    "start": round(turn.start, 2),
                    "end": round(turn.end, 2),
                    "prediction": GenderClassifier.LABELS[pred_idx],
                    "confidence": round(probs[pred_idx].item(), 4),
                }
            )
        return results
    except Exception as e:  # noqa: BLE001
        log.warning(f"Diarization failed: {e}")
        return None


def get_gender_pipeline(device: str = "cpu") -> GenderPipeline:
    """Return the module-level singleton.

    Raises:
        FileNotFoundError: If no aux_gender checkpoint exists.
    """
    global _PIPELINE  # noqa: PLW0603
    if _PIPELINE is not None:
        return _PIPELINE

    from orinode.paths import WS

    ckpt_dir = WS.models_checkpoints / "aux_gender"
    ckpts = sorted(
        ckpt_dir.glob("*.pt") if ckpt_dir.exists() else [],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not ckpts:
        raise FileNotFoundError(
            f"No aux_gender checkpoint found in {ckpt_dir}. " "Run: make train-gender"
        )

    _PIPELINE = GenderPipeline(ckpts[0], device=device)
    return _PIPELINE
