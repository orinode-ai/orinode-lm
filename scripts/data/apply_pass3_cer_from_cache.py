"""Apply Pass 3 CER threshold to manifests using cached results."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cer-threshold", type=float, default=0.40)
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    args = ap.parse_args()

    # Load cache from all split-specific cache dirs
    cer_lookup = {}
    cache_root = Path("workspace/data/filter_cache")
    for cache_dir in cache_root.iterdir():
        cache_file = cache_dir / "transcript_results.jsonl"
        if not cache_file.exists():
            continue
        with open(cache_file) as f:
            for line in f:
                r = json.loads(line)
                d = r.get("details", r)
                path = r.get("audio_path") or d.get("audio_path")
                if path:
                    cer_lookup[path] = {
                        "cer": d.get("cer"),
                        "is_code_switched": d.get("is_code_switched", False),
                        "detected_language": d.get("detected_language"),
                    }

    print(f"Cache loaded: {len(cer_lookup)} entries")

    for split in args.splits:
        manifest_path = Path(f"workspace/data/manifests/afrispeech_200_{split}.jsonl")
        backup_path = manifest_path.with_suffix(".pre_cer.jsonl")

        manifest_path.rename(backup_path)

        kept = 0
        dropped = 0
        missing = 0

        with open(backup_path) as fin, open(manifest_path, "w") as fout:
            for line in fin:
                row = json.loads(line)
                audio_path = row["audio_path"]
                cache_entry = cer_lookup.get(audio_path)

                if cache_entry is None:
                    missing += 1
                    continue

                cer = cache_entry["cer"]
                if cer is not None and cer > args.cer_threshold:
                    dropped += 1
                    continue

                row["cer"] = cer
                row["is_code_switched"] = cache_entry["is_code_switched"]
                if cache_entry["detected_language"]:
                    row["detected_language"] = cache_entry["detected_language"]

                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1

        print(f"{split}: kept {kept}, dropped {dropped} (CER>{args.cer_threshold}), missing {missing}")


if __name__ == "__main__":
    main()
