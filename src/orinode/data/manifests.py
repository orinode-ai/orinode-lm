"""JSONL manifest schema and I/O utilities.

Manifest files live under ``workspace/data/manifests/``.  Each line is a
JSON-serialised ``ManifestRow``.  The schema is the contract between the
data pipeline and the training dataset classes.

Wire format example::

    {
      "audio_path": "workspace/data/processed/afrispeech/spk001_utt003.flac",
      "duration": 4.52,
      "text": "Ọ dị mma ka ị bịa ebe a",
      "language": "ig",
      "dialect": "igbo_owerri",
      "speaker_id": "spk001",
      "domain": "call_center",
      "is_code_switched": false,
      "cs_spans": [],
      "sample_rate": 16000,
      "corpus": "afrispeech_200"
    }
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SUPPORTED_LANGUAGES: frozenset[str] = frozenset({"en", "ha", "yo", "ig", "pcm"})


@dataclass
class CSSpan:
    """A language-coherent character span within a code-switched utterance."""

    start: int
    end: int
    language: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CSSpan:
        return cls(start=d["start"], end=d["end"], language=d["language"])


@dataclass
class ManifestRow:
    """Single row in a training / eval manifest JSONL file."""

    audio_path: str
    duration: float
    text: str
    language: str
    dialect: str = ""
    speaker_id: str = ""
    domain: str = ""
    is_code_switched: bool = False
    cs_spans: list[CSSpan] = field(default_factory=list)
    sample_rate: int = 16000
    corpus: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ManifestRow:
        raw_spans = d.pop("cs_spans", [])
        row = cls(**d)
        row.cs_spans = [CSSpan.from_dict(s) for s in raw_spans]
        return row

    def validate(self) -> list[str]:
        """Return a list of validation error strings (empty list = valid).

        Does not raise — callers decide whether to skip or abort.
        """
        errors: list[str] = []
        if not self.audio_path:
            errors.append("audio_path is empty")
        if self.duration <= 0:
            errors.append(f"duration={self.duration} must be > 0")
        if not self.text.strip():
            errors.append("text is empty or whitespace-only")
        if self.language not in SUPPORTED_LANGUAGES:
            errors.append(f"language={self.language!r} not in {SUPPORTED_LANGUAGES}")
        for span in self.cs_spans:
            if span.language not in SUPPORTED_LANGUAGES:
                errors.append(f"cs_span language={span.language!r} not supported")
            if span.start >= span.end:
                errors.append(f"cs_span start={span.start} >= end={span.end}")
        return errors


# ── I/O ───────────────────────────────────────────────────────────────────────


class ManifestWriter:
    """Append-mode JSONL writer for manifest rows.

    Use as a context manager to ensure the file is flushed and closed::

        with ManifestWriter(path) as w:
            for row in rows:
                w.write(row)
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf-8")

    def write(self, row: ManifestRow) -> None:
        self._fh.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> ManifestWriter:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def read_manifest(path: Path) -> list[ManifestRow]:
    """Read all rows from a manifest JSONL file.

    Args:
        path: Path to a ``.jsonl`` manifest file.

    Returns:
        List of ``ManifestRow`` objects.

    Raises:
        FileNotFoundError: If the manifest file does not exist.
        ValueError: If any line fails to parse.
    """
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    rows: list[ManifestRow] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(ManifestRow.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"Parse error at {path}:{lineno}: {exc}") from exc
    return rows


def iter_manifest(path: Path) -> Iterator[ManifestRow]:
    """Yield rows from a manifest without loading all into memory."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield ManifestRow.from_dict(json.loads(line))


def write_manifest(rows: list[ManifestRow], path: Path) -> None:
    """Write rows to a manifest file (overwrites existing file).

    Args:
        rows: List of ``ManifestRow`` objects.
        path: Output path. Parent directory is created if it does not exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def manifest_stats(path: Path) -> dict[str, Any]:
    """Compute summary statistics for a manifest file.

    Returns:
        Dict with keys: ``total_rows``, ``total_hours``, ``languages``
        (mapping of lang→hours), ``cs_count``, ``mean_duration``,
        ``max_duration``.
    """
    rows = read_manifest(path)
    if not rows:
        return {
            "total_rows": 0,
            "total_hours": 0.0,
            "languages": {},
            "cs_count": 0,
            "mean_duration": 0.0,
            "max_duration": 0.0,
        }
    lang_hours: dict[str, float] = defaultdict(float)
    durations = []
    cs_count = 0
    for row in rows:
        lang_hours[row.language] += row.duration / 3600.0
        durations.append(row.duration)
        if row.is_code_switched:
            cs_count += 1
    return {
        "total_rows": len(rows),
        "total_hours": sum(durations) / 3600.0,
        "languages": dict(lang_hours),
        "cs_count": cs_count,
        "mean_duration": sum(durations) / len(durations),
        "max_duration": max(durations),
    }
