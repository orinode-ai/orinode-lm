"""Tests for data/manifests.py and data/datasets.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from orinode.data.manifests import (
    CSSpan,
    ManifestRow,
    ManifestWriter,
    manifest_stats,
    read_manifest,
    write_manifest,
)

# ── ManifestRow ────────────────────────────────────────────────────────────────


def test_manifest_row_roundtrip() -> None:
    row = ManifestRow(
        audio_path="workspace/data/processed/test.flac",
        duration=4.5,
        text="Ina son ƙasar Najeriya",
        language="ha",
        speaker_id="spk001",
        domain="broadcast",
        corpus="naijavoices",
        is_code_switched=False,
        cs_spans=[],
    )
    d = row.to_dict()
    restored = ManifestRow.from_dict(d)
    assert restored.text == row.text
    assert restored.language == row.language
    assert restored.duration == row.duration
    assert restored.cs_spans == []


def test_manifest_row_with_cs_spans_roundtrip() -> None:
    row = ManifestRow(
        audio_path="test.flac",
        duration=3.0,
        text="I go buy ƙayan abinci",
        language="en",
        is_code_switched=True,
        cs_spans=[
            CSSpan(start=0, end=9, language="en"),
            CSSpan(start=10, end=21, language="ha"),
        ],
    )
    restored = ManifestRow.from_dict(row.to_dict())
    assert len(restored.cs_spans) == 2
    assert restored.cs_spans[0].language == "en"
    assert restored.cs_spans[1].language == "ha"
    assert restored.cs_spans[1].start == 10


def test_manifest_row_validate_valid() -> None:
    row = ManifestRow(audio_path="test.flac", duration=2.0, text="hello", language="en")
    assert row.validate() == []


def test_manifest_row_validate_empty_path() -> None:
    row = ManifestRow(audio_path="", duration=2.0, text="hello", language="en")
    errors = row.validate()
    assert any("audio_path" in e for e in errors)


def test_manifest_row_validate_bad_duration() -> None:
    row = ManifestRow(audio_path="test.flac", duration=-1.0, text="hello", language="en")
    errors = row.validate()
    assert any("duration" in e for e in errors)


def test_manifest_row_validate_unknown_language() -> None:
    row = ManifestRow(audio_path="test.flac", duration=2.0, text="hello", language="sw")
    errors = row.validate()
    assert any("language" in e for e in errors)


def test_manifest_row_validate_empty_text() -> None:
    row = ManifestRow(audio_path="test.flac", duration=2.0, text="   ", language="en")
    errors = row.validate()
    assert any("text" in e for e in errors)


# ── write_manifest + read_manifest ────────────────────────────────────────────


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    rows = [
        ManifestRow(
            audio_path=f"audio_{i}.flac",
            duration=float(i + 1),
            text=f"utterance {i}",
            language="en",
        )
        for i in range(5)
    ]
    path = tmp_path / "test.jsonl"
    write_manifest(rows, path)
    loaded = read_manifest(path)
    assert len(loaded) == 5
    for orig, restored in zip(rows, loaded, strict=False):
        assert orig.audio_path == restored.audio_path
        assert orig.text == restored.text
        assert orig.duration == restored.duration


def test_read_manifest_preserves_diacritics(tmp_path: Path) -> None:
    row = ManifestRow(
        audio_path="ha_test.flac",
        duration=2.0,
        text="Ina son ƙasar Najeriya da dukan zuciyata",
        language="ha",
    )
    path = tmp_path / "ha.jsonl"
    write_manifest([row], path)
    loaded = read_manifest(path)
    assert loaded[0].text == row.text


def test_read_manifest_preserves_yoruba_tones(tmp_path: Path) -> None:
    row = ManifestRow(
        audio_path="yo.flac",
        duration=3.0,
        text="Èdè Yorùbá jẹ́ ẹkọ tí ó nifẹ",
        language="yo",
    )
    path = tmp_path / "yo.jsonl"
    write_manifest([row], path)
    loaded = read_manifest(path)
    assert loaded[0].text == row.text


def test_read_manifest_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        read_manifest(Path("/nonexistent/path/manifest.jsonl"))


def test_read_manifest_bad_line_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text("this is not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Parse error"):
        read_manifest(p)


# ── ManifestWriter ────────────────────────────────────────────────────────────


def test_manifest_writer_context_manager(tmp_path: Path) -> None:
    path = tmp_path / "written.jsonl"
    rows = [
        ManifestRow(audio_path=f"a_{i}.flac", duration=1.0, text=f"text {i}", language="en")
        for i in range(3)
    ]
    with ManifestWriter(path) as w:
        for row in rows:
            w.write(row)
    loaded = read_manifest(path)
    assert len(loaded) == 3


def test_manifest_writer_appends(tmp_path: Path) -> None:
    path = tmp_path / "appended.jsonl"
    row = ManifestRow(audio_path="a.flac", duration=1.0, text="hello", language="en")
    with ManifestWriter(path) as w:
        w.write(row)
    with ManifestWriter(path) as w:
        w.write(row)
    loaded = read_manifest(path)
    assert len(loaded) == 2


# ── manifest_stats ────────────────────────────────────────────────────────────


def test_manifest_stats_counts(tmp_manifest: Path) -> None:
    stats = manifest_stats(tmp_manifest)
    assert stats["total_rows"] == 6
    assert stats["total_hours"] > 0
    assert "en" in stats["languages"]
    assert "ha" in stats["languages"]


def test_manifest_stats_cs_count(tmp_manifest: Path) -> None:
    stats = manifest_stats(tmp_manifest)
    assert stats["cs_count"] >= 1


def test_manifest_stats_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    # Empty manifest should not raise, but read_manifest returns [] for empty file
    # write_manifest with empty list then stats
    write_manifest([], path)
    stats = manifest_stats(path)
    assert stats["total_rows"] == 0
    assert stats["total_hours"] == 0.0
