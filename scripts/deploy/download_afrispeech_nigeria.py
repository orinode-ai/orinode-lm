import os
"""Download AfriSpeech-200 Nigerian subset from HuggingFace.

Streams intronhealth/afrispeech-200, filters country == "Nigeria",
saves audio files to disk, and appends each row to metadata_<split>.json (JSONL).

Idempotent: if output audio file already exists, writes are skipped.
Resume-safe: can be re-run if interrupted — continues where it left off.

Usage:
  python scripts/deploy/download_afrispeech_nigeria.py              # train split
  python scripts/deploy/download_afrispeech_nigeria.py --split test # test split
  python scripts/deploy/download_afrispeech_nigeria.py --split test --dry-run
  python scripts/deploy/download_afrispeech_nigeria.py --max-rows 100  # limit for testing
"""
import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm


HF_DATASET = "intronhealth/afrispeech-200"
DEFAULT_OUTPUT_DIR = "os.environ.get("ORINODE_DATASETS_DIR", os.path.expanduser("~/datasets"))/afrispeech-200"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                    help="Root directory for downloaded data")
    ap.add_argument("--split", default="train", choices=["train", "validation", "test"],
                    help="HuggingFace dataset split to download (HF uses 'validation' for dev)")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Stop after this many Nigerian rows (for testing)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview first 5 Nigerian rows, don't save anything")
    args = ap.parse_args()

    out_root = Path(args.output_dir) / "nigeria"
    out_root.mkdir(parents=True, exist_ok=True)
    # Per-split metadata file so train/dev/test rows never mix
    metadata_path = out_root / f"metadata_{args.split}.json"

    # Build set of already-downloaded files for idempotent resume
    already_downloaded: set[str] = set()
    if metadata_path.exists() and not args.dry_run:
        with open(metadata_path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    already_downloaded.add(row["file"])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"Found {len(already_downloaded):,} already-downloaded rows in {metadata_path.name}")

    # Stream from HuggingFace
    print(f"Streaming {HF_DATASET} split={args.split!r} from HuggingFace...")
    try:
        ds = load_dataset(HF_DATASET, "all", split=args.split, streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"ERROR: failed to load dataset: {e}", file=sys.stderr)
        print("If this is an auth issue, run: huggingface-cli login", file=sys.stderr)
        sys.exit(1)

    # Peek at first row to debug schema mismatches
    first = next(iter(ds))
    print(f"First row keys: {list(first.keys())}")
    ds = load_dataset(HF_DATASET, "all", split=args.split, streaming=True, trust_remote_code=True)

    count_total = 0
    count_nigerian = 0
    count_written = 0
    count_preview = 0

    split_dir = out_root / args.split
    if not args.dry_run:
        split_dir.mkdir(exist_ok=True)

    meta_out = None if args.dry_run else open(metadata_path, "a", encoding="utf-8")

    try:
        for row in tqdm(ds, desc="scanning"):
            count_total += 1

            country = str(row.get("country", "")).strip().upper()
            if country not in ("NG", "NIGERIA"):
                continue
            count_nigerian += 1

            audio = row.get("audio")
            transcript = row.get("transcript", "")
            accent = row.get("accent", "")

            # Resolve a filename
            raw_path = row.get("path") or (audio.get("path") if isinstance(audio, dict) else None)
            if raw_path:
                basename = Path(raw_path).stem + ".flac"
            else:
                basename = f"clip_{count_nigerian:07d}.flac"

            relative_path = f"nigeria/{args.split}/{basename}"

            # Dry-run: preview and continue
            if args.dry_run:
                if count_preview < 5:
                    print(f"  keep: {relative_path} | "
                          f"speaker={row.get('speaker_id', '?')} | "
                          f"{transcript[:50]}...")
                    count_preview += 1
                if count_preview >= 5:
                    break
                continue

            # Skip already-downloaded
            if relative_path in already_downloaded:
                continue

            # Write audio
            if not isinstance(audio, dict) or "array" not in audio:
                print(f"WARNING: row {count_nigerian} has no audio array, skipping",
                      file=sys.stderr)
                continue

            out_audio = split_dir / basename

            array = audio["array"]
            sample_rate = audio["sampling_rate"]

            if array.ndim > 1:
                array = array.mean(axis=1)  # mono
            if sample_rate not in (8000, 16000, 22050, 24000, 44100, 48000):
                print(f"WARNING: unusual sample rate {sample_rate} for {out_audio}, skipping",
                      file=sys.stderr)
                continue

            try:
                if not out_audio.exists():
                    sf.write(str(out_audio), array, sample_rate, subtype="PCM_16", format="FLAC")
            except Exception as e:
                print(f"WARNING: failed to write {out_audio}: {e}", file=sys.stderr)
                continue

            # Append metadata row
            meta_row = {
                "file": relative_path,
                "transcript": transcript,
                "accent": accent,
                "country": row.get("country", ""),
                "split": args.split,
                "duration": round(len(array) / sample_rate, 4),
                "speaker_id": row.get("speaker_id", ""),
                "age_group": row.get("age_group", ""),
                "gender": row.get("gender", ""),
            }
            meta_out.write(json.dumps(meta_row, ensure_ascii=False) + "\n")
            meta_out.flush()
            count_written += 1

            if args.max_rows and count_written >= args.max_rows:
                print(f"\nReached --max-rows={args.max_rows}, stopping")
                break

    finally:
        if meta_out:
            meta_out.close()

    print()
    print(f"=== Summary ===")
    print(f"Split:                  {args.split}")
    print(f"Total rows scanned:     {count_total:,}")
    print(f"Nigerian rows found:    {count_nigerian:,}")
    if args.dry_run:
        print(f"Dry-run previewed:      {count_preview} rows")
    else:
        print(f"Newly downloaded:       {count_written:,}")
        print(f"Metadata appended to:   {metadata_path}")


if __name__ == "__main__":
    main()
