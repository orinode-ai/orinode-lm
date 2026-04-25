import os
"""Merge real speaker_id from speaker_id_map.json into existing split manifests."""
import json
from collections import Counter
from pathlib import Path


def main():
    speaker_map_path = Path(os.environ.get("ORINODE_DATASETS_DIR", os.path.expanduser("~/datasets"))) / "afrispeech-200/nigeria/speaker_id_map.json"
    with open(speaker_map_path, encoding="utf-8") as f:
        raw_map = json.load(f)
    # Re-key by stem (strip extension) so .wav keys match .flac basenames
    speaker_map = {Path(k).stem: v for k, v in raw_map.items()}

    manifests_dir = Path("workspace/data/manifests")
    splits = ["train", "dev", "test"]

    for split in splits:
        input_path = manifests_dir / f"afrispeech_200_{split}.jsonl"
        if not input_path.exists():
            print(f"SKIP: {input_path} not found")
            continue

        backup_path = input_path.with_suffix(".jsonl.bak")
        input_path.rename(backup_path)

        out_rows = []
        unmatched = 0
        speakers_in_split: Counter = Counter()

        with open(backup_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                audio_path = row.get("audio_path", "")
                stem = Path(audio_path).stem  # match by stem; HF has .wav, we have .flac

                if stem in speaker_map:
                    info = speaker_map[stem]
                    row["speaker_id"] = info["speaker_id"] or "unknown"
                    row["age_group"] = info["age_group"]
                    row["gender"] = info["gender"]
                    row["hf_split_origin"] = info["split_origin"]
                    speakers_in_split[info["speaker_id"] or "unknown"] += 1
                else:
                    unmatched += 1
                    row["speaker_id"] = "unknown"

                out_rows.append(row)

        with open(input_path, "w", encoding="utf-8") as f:
            for row in out_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"\n{split}: {len(out_rows)} clips, {unmatched} unmatched")
        print(f"  Unique speakers: {len(speakers_in_split)}")
        print(f"  Top 5 speakers by clip count:")
        for spk, n in speakers_in_split.most_common(5):
            print(f"    {spk}: {n} clips")


if __name__ == "__main__":
    main()
