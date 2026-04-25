import os
"""Stream AfriSpeech-200 Nigerian metadata to capture speaker_id for existing clips.

Does NOT re-download audio. Just streams HF metadata and builds a
mapping: filename -> speaker_id that we merge into existing manifests.
"""
import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

HF_DATASET = "intronhealth/afrispeech-200"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    ap.add_argument(
        "--output",
        default=os.path.join(os.environ.get("ORINODE_DATASETS_DIR", os.path.expanduser("~/datasets")), "afrispeech-200/nigeria/speaker_id_map.json"),
    )
    args = ap.parse_args()

    speaker_map: dict[str, dict] = {}  # basename -> {speaker_id, age_group, gender, split_origin}

    for split in args.splits:
        print(f"\nStreaming split={split!r}...")
        ds = load_dataset(HF_DATASET, "all", split=split, streaming=True, trust_remote_code=True)

        for row in tqdm(ds, desc=split):
            country = str(row.get("country") or "").strip().upper()
            if country not in ("NG", "NIGERIA"):
                continue

            # HF audio dict carries the original path used as filename key
            audio = row.get("audio")
            raw_path = row.get("path") or (audio.get("path") if isinstance(audio, dict) else None)
            if not raw_path:
                continue
            basename = Path(raw_path).name  # e.g. "c4f7...flac" or "speaker_id/clip.wav" -> "clip.wav"

            speaker_map[basename] = {
                "speaker_id": str(row.get("speaker_id") or ""),
                "age_group": str(row.get("age_group") or ""),
                "gender": str(row.get("gender") or ""),
                "split_origin": split,
            }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(speaker_map, f, ensure_ascii=False, indent=2)

    unique_speakers = len({v["speaker_id"] for v in speaker_map.values() if v["speaker_id"]})
    unknown = sum(1 for v in speaker_map.values() if not v["speaker_id"])

    print(f"\nMapped {len(speaker_map):,} filenames to speaker IDs")
    print(f"Unique speakers:    {unique_speakers:,}")
    print(f"Unknown speaker_id: {unknown}")
    print(f"Output:             {output_path}")


if __name__ == "__main__":
    main()
