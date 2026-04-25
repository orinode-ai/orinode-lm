"""Enforce strict speaker disjointness by removing from dev/test any
speaker whose speaker_id appears in train.

Creates 'Orinode test' (ODT) and 'Orinode dev' (ODD) — speaker-disjoint
versions alongside the HF official splits preserved separately.
"""

import json
from pathlib import Path


def load_speakers(path: Path) -> set[str]:
    speakers: set[str] = set()
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            spk = row.get("speaker_id", "unknown")
            if spk and spk != "unknown":
                speakers.add(spk)
    return speakers


def filter_out_speakers(
    input_path: Path, output_path: Path, exclude_speakers: set[str]
) -> tuple[int, int]:
    kept = 0
    removed = 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            row = json.loads(line)
            spk = row.get("speaker_id", "unknown")
            if spk in exclude_speakers:
                removed += 1
            else:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
    return kept, removed


def main() -> None:
    train_path = Path("workspace/data/manifests/afrispeech_200_train.jsonl")
    train_speakers = load_speakers(train_path)
    print(f"Train has {len(train_speakers)} unique speakers")

    for split in ["dev", "test"]:
        hf_path = Path(f"workspace/data/manifests/afrispeech_200_{split}_hf_official.jsonl")
        clean_path = Path(f"workspace/data/manifests/afrispeech_200_{split}.jsonl")

        hf_speakers = load_speakers(hf_path)
        leaked = train_speakers & hf_speakers

        kept, removed = filter_out_speakers(hf_path, clean_path, train_speakers)

        remaining_speakers = load_speakers(clean_path)

        print(f"\n{split}:")
        print(f"  Removed: {removed} clips from {len(leaked)} leaking speakers")
        print(f"  Kept:    {kept} clips from {len(remaining_speakers)} disjoint speakers")


if __name__ == "__main__":
    main()
