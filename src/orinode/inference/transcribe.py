"""Audio transcription via the SpeechLLM model.

Designed to be called from the UI Playground endpoint.  The transcriber is
a module-level singleton loaded lazily on first call so the server starts up
instantly even when model weights are large.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import torch

from orinode.utils.logging import get_logger

log = get_logger(__name__)

_SAMPLE_RATE = 16_000
_TRANSCRIBER: Transcriber | None = None


class Transcriber:
    """Load a SpeechLLM checkpoint and run transcription.

    Args:
        checkpoint_path: Path to a ``*.pt`` checkpoint saved by
            ``BaseTrainer._save_checkpoint``.
        device: ``"cuda"`` or ``"cpu"``.
    """

    def __init__(self, checkpoint_path: Path, device: str = "cpu") -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        log.info(f"Loading checkpoint: {self.checkpoint_path}")
        from orinode.models.speech_llm import SpeechLLM

        model = SpeechLLM.for_smoke_test()  # same tiny arch as smoke test
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        model.eval()
        self._model = model.to(self.device)
        log.info("Checkpoint loaded")

    def transcribe(
        self,
        audio_bytes: bytes,
        language: str = "auto",
        max_new_tokens: int = 256,
    ) -> dict[str, Any]:
        """Transcribe raw audio bytes to text.

        Args:
            audio_bytes: Raw audio (WAV or FLAC).
            language: Language code or ``"auto"``.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Dict with ``text`` and ``language`` keys.
        """
        import torchaudio

        self._load()

        waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
        if sr != _SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, _SAMPLE_RATE)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Build a minimal mel spectrogram using a WhisperFeatureExtractor-like transform
        # For production, use the HuggingFace WhisperFeatureExtractor.
        from transformers import WhisperFeatureExtractor

        fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
        features = fe(
            waveform.squeeze(0).numpy(),
            sampling_rate=_SAMPLE_RATE,
            return_tensors="pt",
        ).input_features.to(self.device)

        prompt_ids = torch.tensor([[1]], device=self.device)  # BOS token

        with torch.inference_mode():
            tokens = self._model.generate(
                input_features=features,
                prompt_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        text = str(tokens[0].tolist())  # raw IDs until a real tokenizer is wired
        return {"text": text, "language": language}


def get_transcriber(device: str = "cpu") -> Transcriber:
    """Return the module-level Transcriber singleton.

    Looks for the best checkpoint in ``workspace/models/checkpoints/``.

    Raises:
        FileNotFoundError: If no checkpoint is found.
    """
    global _TRANSCRIBER  # noqa: PLW0603
    if _TRANSCRIBER is not None:
        return _TRANSCRIBER

    from orinode.paths import WS

    # Find most-recently-modified checkpoint
    ckpts = sorted(
        WS.models_checkpoints.rglob("*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not ckpts:
        raise FileNotFoundError(
            f"No checkpoints found in {WS.models_checkpoints}. "
            "Run a training stage first (make train-stage1)."
        )

    _TRANSCRIBER = Transcriber(ckpts[0], device=device)
    return _TRANSCRIBER
