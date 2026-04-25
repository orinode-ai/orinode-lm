"""PyTorch Dataset implementations backed by JSONL manifests.

``ManifestDataset`` is the main entry point for training. It reads a manifest,
filters by duration, and returns preprocessed (waveform, text, metadata) dicts.
``collate_fn`` pads waveforms for batch assembly.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from orinode.data.manifests import ManifestRow, read_manifest
from orinode.data.preprocessing import load_audio


class ManifestDataset(Dataset):  # type: ignore[type-arg]
    """Dataset backed by a JSONL manifest file.

    Each ``__getitem__`` returns a dict with:

    - ``waveform``:        float32 tensor ``(1, T)``
    - ``text``:            NFC-normalised transcript
    - ``language``:        primary language tag (e.g. ``"ha"``)
    - ``is_code_switched``: bool
    - ``cs_spans``:        list of ``{start, end, language}`` dicts
    - ``duration``:        float (seconds)
    - ``audio_path``:      str
    - ``speaker_id``:      str
    - ``domain``:          str
    - ``corpus``:          str

    Args:
        manifest_path: Path to the JSONL manifest file.
        audio_root: Prepended to relative ``audio_path`` values.
        max_duration: Rows longer than this are excluded at init time.
        min_duration: Rows shorter than this are excluded at init time.
        transform: Optional callable applied to ``waveform`` tensors.
        text_transform: Optional callable applied to ``text`` strings.
    """

    def __init__(
        self,
        manifest_path: Path,
        audio_root: Path | None = None,
        max_duration: float = 30.0,
        min_duration: float = 0.5,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        text_transform: Callable[[str], str] | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.audio_root = audio_root
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.transform = transform
        self.text_transform = text_transform

        all_rows = read_manifest(manifest_path)
        self.rows: list[ManifestRow] = [
            r for r in all_rows if min_duration <= r.duration <= max_duration
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        audio_path = Path(row.audio_path)
        if self.audio_root is not None and not audio_path.is_absolute():
            audio_path = self.audio_root / audio_path

        waveform = load_audio(audio_path)
        if self.transform is not None:
            waveform = self.transform(waveform)

        text = row.text
        if self.text_transform is not None:
            text = self.text_transform(text)

        return {
            "waveform": waveform,
            "text": text,
            "language": row.language,
            "is_code_switched": row.is_code_switched,
            "cs_spans": [
                {"start": s.start, "end": s.end, "language": s.language} for s in row.cs_spans
            ],
            "duration": row.duration,
            "audio_path": str(row.audio_path),
            "speaker_id": row.speaker_id,
            "domain": row.domain,
            "corpus": row.corpus,
        }

    def language_counts(self) -> dict[str, int]:
        """Return a ``{language: count}`` mapping over all kept rows."""
        return dict(Counter(r.language for r in self.rows))

    def split_by_language(self) -> dict[str, ManifestDataset]:
        """Return one sub-dataset per language (shares the loaded row list)."""
        from collections import defaultdict

        lang_rows: dict[str, list[ManifestRow]] = defaultdict(list)
        for row in self.rows:
            lang_rows[row.language].append(row)

        result: dict[str, ManifestDataset] = {}
        for lang, rows in lang_rows.items():
            ds = object.__new__(ManifestDataset)
            ds.manifest_path = self.manifest_path
            ds.audio_root = self.audio_root
            ds.max_duration = self.max_duration
            ds.min_duration = self.min_duration
            ds.transform = self.transform
            ds.text_transform = self.text_transform
            ds.rows = rows
            result[lang] = ds
        return result


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate a list of dataset items into a padded batch.

    Waveforms are right-padded with zeros to the longest in the batch.

    Args:
        batch: List of dicts from ``ManifestDataset.__getitem__``.

    Returns:
        Dict with ``waveforms`` (B, 1, T_max), ``waveform_lengths`` (B,),
        ``texts`` (list), ``languages`` (list), and all other scalar fields
        as lists.
    """
    waveforms = [item["waveform"] for item in batch]
    lengths = torch.tensor([w.shape[-1] for w in waveforms], dtype=torch.long)
    max_len = int(lengths.max().item())

    padded = torch.zeros(len(waveforms), 1, max_len)
    for i, wf in enumerate(waveforms):
        padded[i, :, : wf.shape[-1]] = wf

    return {
        "waveforms": padded,
        "waveform_lengths": lengths,
        "texts": [item["text"] for item in batch],
        "languages": [item["language"] for item in batch],
        "is_code_switched": [item["is_code_switched"] for item in batch],
        "cs_spans": [item["cs_spans"] for item in batch],
        "durations": [item["duration"] for item in batch],
        "audio_paths": [item["audio_path"] for item in batch],
        "speaker_ids": [item["speaker_id"] for item in batch],
        "domains": [item["domain"] for item in batch],
    }
