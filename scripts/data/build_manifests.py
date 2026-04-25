"""AfriSpeech-200 Nigerian data pipeline — three-pass filtering + manifest build.

Usage::

    # Metadata filter only (fast, no audio needed)
    python scripts/data/build_manifests.py \\
        --metadata-path $ORINODE_DATASETS_DIR/afrispeech-200/nigeria/metadata.json \\
        --filter-level metadata

    # Audio quality filter (requires pre-processed audio)
    python scripts/data/build_manifests.py \\
        --metadata-path ... --filter-level audio

    # Full pipeline including Whisper transcript verification
    python scripts/data/build_manifests.py \\
        --metadata-path ... --filter-level full

    # Dry run — report counts only
    python scripts/data/build_manifests.py --metadata-path ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf

# ── AfriSpeech metadata loader ────────────────────────────────────────────────

MEDICAL_KEYWORDS = frozenset(
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


def _infer_domain(text: str) -> str:
    words = text.lower().split()
    return "clinical" if sum(1 for w in words if w in MEDICAL_KEYWORDS) >= 2 else "general"


def _extract_speaker_id(file_field: str) -> str:
    """Extract speaker ID from AfriSpeech filename convention: NNN_SPEAKERID.flac"""
    stem = Path(file_field).stem
    parts = stem.split("_", 1)
    return parts[1] if len(parts) == 2 else stem


def load_afrispeech_metadata(metadata_path: Path) -> list[dict]:
    """Load AfriSpeech JSONL metadata file.

    Wire format (one JSON object per line)::

        {"file":"nigeria/train/000001_abc123.flac","transcript":"...",
         "accent":"hausa","country":"Nigeria","split":"train",
         "duration":4.52}

    Adds fields: ``raw_audio_path``, ``speaker_id``, ``domain``, ``language``,
    ``source_corpus``, ``accent`` (lowercased).
    """
    corpus_root = metadata_path.parent.parent
    rows: list[dict] = []

    with open(metadata_path, encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {metadata_path}:{line_num}: {exc}") from exc

            row["raw_audio_path"] = str(corpus_root / row["file"])
            row["speaker_id"] = _extract_speaker_id(row["file"])
            row["domain"] = _infer_domain(row.get("transcript", ""))
            row["language"] = "en"
            row["source_corpus"] = "afrispeech-200"
            row["accent"] = str(row.get("accent", "")).lower().strip()
            if not row.get("duration"):
                try:
                    info = sf.info(row["raw_audio_path"])
                    row["duration"] = info.frames / info.samplerate
                except Exception:
                    row["duration"] = 0.0
            rows.append(row)

    return rows


# ── cache helpers ─────────────────────────────────────────────────────────────


def _cache_path(cache_dir: Path, name: str) -> Path:
    return cache_dir / f"{name}.jsonl"


def _save_cache(results: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def _load_cache(path: Path, cls: type) -> list | None:
    if not path.exists():
        return None
    results = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                results.append(cls.from_dict(json.loads(line)))
    return results


def _preprocess_result_to_dict(r: object) -> dict:
    import dataclasses

    return dataclasses.asdict(r)  # type: ignore[arg-type]


def _preprocess_result_from_dict(d: dict) -> object:
    from orinode.data.preprocessing import PreprocessResult

    return PreprocessResult(**d)


# ── pipeline passes ────────────────────────────────────────────────────────────


def run_or_load_pass1(
    rows: list[dict],
    cache_dir: Path,
    cfg_meta,
    dry_run: bool,
) -> list:
    from orinode.data.filtering import FilterResult, apply_speaker_cap, metadata_filter

    cache = _cache_path(cache_dir, "metadata_results")
    cached = _load_cache(cache, FilterResult)
    if cached is not None and len(cached) == len(rows):
        print(f"  [Pass 1] Loaded {len(cached)} cached results from {cache}")
        return cached

    print(f"  [Pass 1] Running metadata filter on {len(rows)} rows…")
    results = [metadata_filter(r, cfg_meta) for r in rows]
    results = apply_speaker_cap(rows, results, cfg_meta)

    kept = sum(1 for r in results if r.keep)
    total = len(results)
    reject_pct = (total - kept) / max(total, 1) * 100
    print(f"  [Pass 1] {kept}/{total} kept ({reject_pct:.1f}% rejected)")

    if not dry_run:
        _save_cache(results, cache)

    return results


def run_or_load_preprocessing(
    pass1_kept: list[dict],
    processed_dir: Path,
    cache_dir: Path,
    cfg_prep,
    dry_run: bool,
) -> list:
    import dataclasses

    from orinode.data.preprocessing import PreprocessResult, preprocess_clip

    cache = _cache_path(cache_dir, "preprocess_results")
    if cache.exists():
        cached = []
        with cache.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    cached.append(PreprocessResult(**json.loads(line)))
        if len(cached) == len(pass1_kept):
            print(f"  [Preprocess] Loaded {len(cached)} cached results")
            return cached

    print(f"  [Preprocess] Resampling + loudnorm {len(pass1_kept)} clips…")
    results = []
    for row in pass1_kept:
        raw_path = row["raw_audio_path"]
        rel = Path(row["file"]).name
        out_path = processed_dir / "afrispeech-200" / row.get("split", "train") / rel
        out_path = out_path.with_suffix(".flac")
        result = preprocess_clip(raw_path, out_path, cfg_prep)
        results.append(result)

    ok = sum(1 for r in results if r.success)
    print(f"  [Preprocess] {ok}/{len(results)} succeeded")

    if not dry_run:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("w") as fh:
            for r in results:
                fh.write(json.dumps(dataclasses.asdict(r), ensure_ascii=False) + "\n")

    return results


def run_or_load_pass2(
    pass1_kept: list[dict],
    preprocess_results: list,
    cache_dir: Path,
    cfg_audio,
    dry_run: bool,
) -> list:
    from orinode.data.filtering import FilterResult, run_audio_filter_parallel

    cache = _cache_path(cache_dir, "audio_results")
    cached = _load_cache(cache, FilterResult)
    # match rows that were preprocessed successfully
    ok_rows = [
        (row, pr)
        for row, pr in zip(pass1_kept, preprocess_results, strict=False)
        if pr.success  # type: ignore[union-attr]
    ]
    if cached is not None and len(cached) == len(ok_rows):
        print(f"  [Pass 2] Loaded {len(cached)} cached results")
        return cached

    print(f"  [Pass 2] Audio quality filter on {len(ok_rows)} clips…")
    audio_paths = [pr.output_path for _, pr in ok_rows]  # type: ignore[union-attr]
    results = run_audio_filter_parallel(audio_paths, cfg_audio)

    kept = sum(1 for r in results if r.keep)
    print(f"  [Pass 2] {kept}/{len(results)} kept")

    if not dry_run:
        _save_cache(results, cache)

    return results


def run_or_load_pass3(
    pass2_kept_rows: list[dict],
    pass2_kept_paths: list[str],
    cache_dir: Path,
    cfg_tx,
    dry_run: bool,
) -> list:
    import torch
    from jiwer import cer as compute_cer
    from transformers import pipeline as hf_pipeline

    from orinode.data.filtering import FilterResult
    from orinode.data.preprocessing import load_audio
    from orinode.data.text_normalization import normalize_transcript

    cache = _cache_path(cache_dir, "transcript_results")
    cached = _load_cache(cache, FilterResult)
    if cached is not None and len(cached) == len(pass2_kept_rows):
        print(f"  [Pass 3] Loaded {len(cached)} cached results")
        return cached

    print(f"  [Pass 3] Whisper transcript check on {len(pass2_kept_rows)} clips…")
    dtype = torch.bfloat16 if cfg_tx.use_bf16 else torch.float32
    asr = hf_pipeline(
        "automatic-speech-recognition",
        model=cfg_tx.asr_model_path,
        device=0 if cfg_tx.device == "cuda" else -1,
        torch_dtype=dtype,
        batch_size=cfg_tx.batch_size,
        chunk_length_s=30,
    )

    def _audio_gen():
        for path in pass2_kept_paths:
            waveform = load_audio(path)
            yield {"raw": waveform.squeeze(0).numpy(), "sampling_rate": 16000}

    pairs = list(zip(pass2_kept_rows, pass2_kept_paths, strict=False))
    results = []

    try:
        from tqdm import tqdm
        pair_iter = tqdm(
            zip(pairs, asr(_audio_gen())),
            total=len(pairs),
            desc="Pass 3 CER",
        )
    except ImportError:
        pair_iter = zip(pairs, asr(_audio_gen()))  # type: ignore[assignment]

    for (row, path), asr_out in pair_iter:
        details: dict = {"audio_path": path}
        try:
            hypothesis: str = asr_out["text"] if isinstance(asr_out, dict) else str(asr_out)
        except Exception as exc:
            results.append(FilterResult(
                keep=False, reason="asr_error",
                details={"error": str(exc), **details},
            ))
            continue

        ref = normalize_transcript(row["transcript"], language="en")
        hyp = normalize_transcript(hypothesis, language="en")
        try:
            cer_score = float(compute_cer(ref, hyp))
        except Exception:
            cer_score = 1.0

        details["cer"] = round(cer_score, 4)
        details["hypothesis"] = hypothesis[:200]

        if cer_score > cfg_tx.max_cer:
            results.append(FilterResult(
                keep=False,
                reason=f"transcript_mismatch_cer_{cer_score:.2f}",
                details=details,
            ))
        else:
            results.append(FilterResult(keep=True, reason="pass", details=details))

    kept = sum(1 for r in results if r.keep)
    print(f"  [Pass 3] {kept}/{len(results)} kept")

    if not dry_run:
        _save_cache(results, cache)

    return results


# ── manifest + report writers ─────────────────────────────────────────────────


def write_manifest_from_rows(rows: list[dict], out_path: Path) -> None:

    from orinode.data.manifests import ManifestRow, ManifestWriter

    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen_paths: set[str] = set()
    with ManifestWriter(out_path) as w:
        for row in rows:
            audio_path = row.get("processed_audio_path", row["raw_audio_path"])
            if audio_path in seen_paths:
                continue
            seen_paths.add(audio_path)
            mr = ManifestRow(
                audio_path=audio_path,
                duration=float(row.get("duration", 0.0)),
                text=(row.get("transcript") or "").strip(),
                language=row.get("language", "en"),
                dialect=row.get("accent", ""),
                speaker_id=row.get("speaker_id", ""),
                domain=row.get("domain", ""),
                corpus=row.get("source_corpus", "afrispeech-200"),
            )
            errs = mr.validate()
            if not errs:
                w.write(mr)


def write_reject_log(results_by_pass: list[tuple[str, list]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for pass_name, results in results_by_pass:
            for r in results:
                if not r.keep:
                    fh.write(
                        json.dumps({"pass": pass_name, **r.to_dict()}, ensure_ascii=False) + "\n"
                    )


def write_filter_report(
    pass_counts: dict[str, dict],
    filter_level: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"filter_level": filter_level, "passes": pass_counts}
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"  Filter report → {out_path}")


# ── guardrails ────────────────────────────────────────────────────────────────


def _check_pass1_guardrails(total: int, kept: int) -> None:
    reject_pct = (total - kept) / max(total, 1) * 100
    if reject_pct > 30:
        print(f"FAIL: Pass 1 rejected {reject_pct:.1f}% (> 30%). Inspect metadata.", file=sys.stderr)
        sys.exit(1)
    # Floor is 2% (not 5%) — afrispeech-200 arrives pre-filtered by country at
    # download time, so genuine rejection rate is naturally lower than raw corpora.
    if reject_pct < 2:
        print(f"WARN: Pass 1 rejected only {reject_pct:.1f}% (< 2%). Filter may be too loose.")


def _check_preprocess_guardrails(total: int, ok: int) -> None:
    fail_pct = (total - ok) / max(total, 1) * 100
    if fail_pct > 2:
        print(f"FAIL: Preprocessing failed {fail_pct:.1f}% (> 2%).", file=sys.stderr)
        sys.exit(1)


def _check_pass2_guardrails(total: int, kept: int) -> None:
    reject_pct = (total - kept) / max(total, 1) * 100
    if reject_pct > 20:
        print(f"FAIL: Pass 2 rejected {reject_pct:.1f}% (> 20%).", file=sys.stderr)
        sys.exit(1)
    if reject_pct < 1:
        print(f"WARN: Pass 2 rejected only {reject_pct:.1f}% (< 1%). Filter may be too loose.")


def _check_pass3_guardrails(total: int, kept: int) -> None:
    reject_pct = (total - kept) / max(total, 1) * 100
    if reject_pct > 25:
        print(f"FAIL: Pass 3 rejected {reject_pct:.1f}% (> 25%).", file=sys.stderr)
        sys.exit(1)
    if reject_pct < 1:
        print(f"WARN: Pass 3 rejected only {reject_pct:.1f}% (< 1%). Filter may be too loose.")


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="AfriSpeech-200 Nigerian data pipeline")
    parser.add_argument(
        "--metadata-path",
        type=Path,
        required=True,
        help="Path to AfriSpeech JSONL metadata file",
    )
    parser.add_argument(
        "--split-name",
        default="train",
        choices=["train", "validation", "test"],
        help="Logical split name used in output filenames and cache dirs",
    )
    parser.add_argument(
        "--filter-level",
        choices=["metadata", "audio", "full"],
        default="metadata",
        help="How deep to filter: metadata only, +audio quality, or +transcript check",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report counts only, no writes")
    args = parser.parse_args()

    # Normalise: HF calls it 'validation', manifests call it 'dev'
    output_split = "dev" if args.split_name == "validation" else args.split_name

    from orinode.data.filtering import (
        AudioFilterConfig,
        MetadataFilterConfig,
        TranscriptFilterConfig,
    )
    from orinode.data.preprocessing import PreprocessConfig
    from orinode.paths import WS, ensure_workspace

    ensure_workspace()

    processed_dir = WS.data / "processed"
    manifests_dir = WS.data / "manifests"
    cache_dir = WS.data / "filter_cache" / f"afrispeech_200_{output_split}"
    logs_dir = WS.logs / "filter_reports"

    cfg_meta = MetadataFilterConfig()
    cfg_prep = PreprocessConfig()
    cfg_audio = AudioFilterConfig()
    cfg_tx = TranscriptFilterConfig()

    print("=== AfriSpeech-200 Nigerian pipeline ===")
    print(f"  metadata: {args.metadata_path}")
    print(f"  split: {output_split}")
    print(f"  filter level: {args.filter_level}")
    print(f"  dry run: {args.dry_run}")
    print()

    # Load metadata
    print("[1] Loading AfriSpeech metadata…")
    raw_rows = load_afrispeech_metadata(args.metadata_path)
    print(f"    {len(raw_rows)} rows loaded")

    # Pass 1
    print("[2] Pass 1 — metadata filter")
    pass1_results = run_or_load_pass1(raw_rows, cache_dir, cfg_meta, args.dry_run)
    pass1_kept_rows = [r for r, res in zip(raw_rows, pass1_results, strict=False) if res.keep]
    _check_pass1_guardrails(len(raw_rows), len(pass1_kept_rows))

    if args.filter_level == "metadata":
        if not args.dry_run:
            out = manifests_dir / f"afrispeech_200_{output_split}.jsonl"
            write_manifest_from_rows(pass1_kept_rows, out)
            write_reject_log(
                [("pass1", pass1_results)],
                logs_dir / f"rejects_{output_split}_metadata.jsonl",
            )
            write_filter_report(
                {"pass1": {"total": len(raw_rows), "kept": len(pass1_kept_rows)}},
                "metadata",
                logs_dir / f"filter_report_{output_split}.json",
            )
            print(f"\nManifest → {out}")
        print(f"\nDone. {len(pass1_kept_rows)} clips passed metadata filter.")
        return

    # Preprocess
    print("[3] Preprocessing (resample + loudnorm)")
    preprocess_results = run_or_load_preprocessing(
        pass1_kept_rows, processed_dir, cache_dir, cfg_prep, args.dry_run
    )
    ok_prep = [r for r in preprocess_results if r.success]
    _check_preprocess_guardrails(len(preprocess_results), len(ok_prep))
    ok_rows = [row for row, pr in zip(pass1_kept_rows, preprocess_results, strict=False) if pr.success]

    # Pass 2
    print("[4] Pass 2 — audio quality filter")
    pass2_results = run_or_load_pass2(ok_rows, preprocess_results, cache_dir, cfg_audio, args.dry_run)
    pass2_kept_rows = [r for r, res in zip(ok_rows, pass2_results, strict=False) if res.keep]
    pass2_kept_paths = [pr.output_path for pr, res in zip(ok_prep, pass2_results, strict=False) if res.keep]
    _check_pass2_guardrails(len(pass2_results), len(pass2_kept_rows))

    if args.filter_level == "audio":
        if not args.dry_run:
            # Enrich rows with processed audio paths
            for row, pr in zip(ok_rows, ok_prep, strict=False):
                row["processed_audio_path"] = pr.output_path
            out = manifests_dir / f"afrispeech_200_{output_split}.jsonl"
            write_manifest_from_rows(pass2_kept_rows, out)
            write_reject_log(
                [("pass1", pass1_results), ("pass2", pass2_results)],
                logs_dir / f"rejects_{output_split}_audio.jsonl",
            )
            write_filter_report(
                {
                    "pass1": {"total": len(raw_rows), "kept": len(pass1_kept_rows)},
                    "pass2": {"total": len(pass2_results), "kept": len(pass2_kept_rows)},
                },
                "audio",
                logs_dir / f"filter_report_{output_split}.json",
            )
            print(f"\nManifest → {out}")
        print(f"\nDone. {len(pass2_kept_rows)} clips passed audio quality filter.")
        return

    # Pass 3
    print("[5] Pass 3 — transcript quality (Whisper CER)")
    pass3_results = run_or_load_pass3(
        pass2_kept_rows, pass2_kept_paths, cache_dir, cfg_tx, args.dry_run
    )
    pass3_kept = [
        (r, p)
        for r, p, res in zip(pass2_kept_rows, pass2_kept_paths, pass3_results, strict=False)
        if res.keep
    ]
    pass3_kept_rows = [r for r, _ in pass3_kept]
    _check_pass3_guardrails(len(pass3_results), len(pass3_kept_rows))

    if not args.dry_run:
        for row, path in pass3_kept:
            row["processed_audio_path"] = path
        out = manifests_dir / f"afrispeech_200_{output_split}.jsonl"
        write_manifest_from_rows(pass3_kept_rows, out)
        write_reject_log(
            [("pass1", pass1_results), ("pass2", pass2_results), ("pass3", pass3_results)],
            logs_dir / f"rejects_{output_split}_full.jsonl",
        )
        write_filter_report(
            {
                "pass1": {"total": len(raw_rows), "kept": len(pass1_kept_rows)},
                "pass2": {"total": len(pass2_results), "kept": len(pass2_kept_rows)},
                "pass3": {"total": len(pass3_results), "kept": len(pass3_kept_rows)},
            },
            "full",
            logs_dir / f"filter_report_{output_split}.json",
        )
        print(f"\nManifest → {out}")

    print(f"\nDone. {len(pass3_kept_rows)} clips passed all three passes.")


if __name__ == "__main__":
    main()
