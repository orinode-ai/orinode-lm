"""Dataset class for Whisper fine-tuning from manifest JSONL.

Each manifest row describes one audio clip. Dataset loads the FLAC,
extracts mel-spectrogram, tokenizes the transcript, and returns a
dict ready for Whisper's forward pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import WhisperProcessor


@dataclass
class WhisperDatasetConfig:
    manifest_path: Path
    processor_name: str = "openai/whisper-large-v3"
    sample_rate: int = 16000
    max_audio_seconds: float = 30.0
    language: str = "en"
    task: str = "transcribe"


class WhisperDataset(Dataset):
    """Whisper training dataset backed by a JSONL manifest.

    Each manifest row must have:
        audio_path  — absolute path to a preprocessed 16 kHz mono FLAC
        text        — reference transcription (already stripped of whitespace)

    Returns per item:
        input_features  torch.Tensor [n_mels, 3000]  mel-spectrogram
        labels          torch.Tensor [seq_len]        tokenized transcript
    """

    def __init__(self, config: WhisperDatasetConfig) -> None:
        self.config = config
        self.rows: list[dict[str, Any]] = []
        with open(config.manifest_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

        if not self.rows:
            raise ValueError(f"Manifest {config.manifest_path} is empty.")

        self.processor = WhisperProcessor.from_pretrained(
            config.processor_name,
            language=config.language,
            task=config.task,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]

        # Load pre-processed 16 kHz mono FLAC
        audio, sr = sf.read(row["audio_path"], dtype="float32")

        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        if sr != self.config.sample_rate:
            raise ValueError(
                f"{row['audio_path']} has sr={sr}, "
                f"expected {self.config.sample_rate}. "
                "Preprocessing should have resampled to 16 kHz."
            )

        max_samples = int(self.config.max_audio_seconds * self.config.sample_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # Whisper feature extractor → [1, n_mels, 3000]
        inputs = self.processor.feature_extractor(
            audio,
            sampling_rate=self.config.sample_rate,
            return_tensors="pt",
        )
        input_features = inputs["input_features"].squeeze(0)  # [n_mels, 3000]

        # Tokenize transcript
        transcript = row.get("text", row.get("transcript", "")).strip()
        labels = self.processor.tokenizer(
            transcript,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        return {
            "input_features": input_features,
            "labels": labels,
        }
