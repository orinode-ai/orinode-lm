"""Emotion prediction pipeline.

Lazy-loads the aux_emotion checkpoint on first call.
Segment timeline (emotion per 3-second window) is only computed when
audio duration exceeds 30 seconds, to avoid unnecessary computation.

**Disclaimer**: Initial training uses English transfer data (IEMOCAP + RAVDESS).
Accuracy on Nigerian speech varies and may be lower than on English.
The UI shows a "Preview" badge until Nigerian emotional speech data is added.
See docs/AUX_MODELS.md for the roadmap.

Usage::

    from orinode.inference.emotion_pipeline import get_emotion_pipeline
    result = get_emotion_pipeline().predict(audio_bytes)
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import torch

from orinode.models.emotion_classifier import EmotionClassifier
from orinode.utils.logging import get_logger

log = get_logger(__name__)

_SAMPLE_RATE = 16_000
_SEGMENT_SECONDS = 3
_SEGMENT_THRESHOLD_SECONDS = 30
_PIPELINE: EmotionPipeline | None = None

DISCLAIMER = (
    "Emotion classification is in preview. This model was trained on English "
    "transfer data (IEMOCAP + RAVDESS) and evaluated on limited Nigerian samples. "
    "Accuracy varies by language and context. Not suitable for high-stakes decisions."
)


class EmotionPipeline:
    """Load an EmotionClassifier checkpoint and run prediction.

    Args:
        checkpoint_path: Path to a saved ``EmotionClassifier`` state dict.
        device: ``"cuda"`` or ``"cpu"``.
    """

    MODEL_VERSION = "aux_emotion/v1"

    def __init__(self, checkpoint_path: Path, device: str = "cpu") -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._model: EmotionClassifier | None = None

    def _load(self) -> EmotionClassifier:
        if self._model is not None:
            return self._model
        log.info(f"Loading emotion checkpoint: {self.checkpoint_path}")
        model = EmotionClassifier.for_smoke_test()
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        model.eval()
        self._model = model.to(self.device)
        return self._model

    def _predict_segment(
        self, waveform: torch.Tensor, model: EmotionClassifier
    ) -> dict[str, float]:
        """Return probability dict for a single waveform segment."""
        x = waveform.unsqueeze(0).to(self.device)  # (1, T)
        with torch.inference_mode():
            log_probs = model(input_values=x)
            probs = log_probs.exp().squeeze(0)
        return {
            label: round(probs[i].item(), 4) for i, label in enumerate(EmotionClassifier.LABELS)
        }

    def predict(
        self,
        audio_bytes: bytes,
    ) -> dict[str, Any]:
        """Predict emotion from raw audio bytes.

        For audio longer than 30 seconds, also returns a per-segment timeline
        (3-second windows) as ``segment_timeline``.

        Returns:
            Dict with ``top_prediction``, ``confidences``,
            ``segment_timeline`` (or None), ``model_version``, ``disclaimer``.
        """
        import torchaudio

        model = self._load()
        waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
        if sr != _SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, _SAMPLE_RATE)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)  # (T,)

        duration = waveform.shape[0] / _SAMPLE_RATE
        confidences = self._predict_segment(waveform, model)
        top_prediction = max(confidences, key=lambda k: confidences[k])

        segment_timeline = None
        if duration > _SEGMENT_THRESHOLD_SECONDS:
            seg_len = _SEGMENT_SECONDS * _SAMPLE_RATE
            timeline = []
            for start_sample in range(0, waveform.shape[0], seg_len):
                seg = waveform[start_sample : start_sample + seg_len]
                if seg.shape[0] < 1600:
                    break
                seg_confs = self._predict_segment(seg, model)
                timeline.append(
                    {
                        "start": round(start_sample / _SAMPLE_RATE, 2),
                        "end": round(
                            min(
                                (start_sample + seg_len) / _SAMPLE_RATE,
                                duration,
                            ),
                            2,
                        ),
                        "top": max(seg_confs, key=lambda k: seg_confs[k]),
                        "confidences": seg_confs,
                    }
                )
            segment_timeline = timeline

        return {
            "top_prediction": top_prediction,
            "confidences": confidences,
            "segment_timeline": segment_timeline,
            "model_version": self.MODEL_VERSION,
            "disclaimer": DISCLAIMER,
        }


def get_emotion_pipeline(device: str = "cpu") -> EmotionPipeline:
    """Return the module-level singleton.

    Raises:
        FileNotFoundError: If no aux_emotion checkpoint exists.
    """
    global _PIPELINE  # noqa: PLW0603
    if _PIPELINE is not None:
        return _PIPELINE

    from orinode.paths import WS

    ckpt_dir = WS.models_checkpoints / "aux_emotion"
    ckpts = sorted(
        ckpt_dir.glob("*.pt") if ckpt_dir.exists() else [],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not ckpts:
        raise FileNotFoundError(
            f"No aux_emotion checkpoint found in {ckpt_dir}. " "Run: make train-emotion"
        )

    _PIPELINE = EmotionPipeline(ckpts[0], device=device)
    return _PIPELINE
