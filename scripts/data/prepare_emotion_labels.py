"""Prepare emotion classification training manifests.

Three bootstrap strategies for Nigerian emotion data — honest about scarcity.

=====================================================================
STRATEGY 1 (default, auto-executed):
  Transfer-learn from IEMOCAP + RAVDESS (English speech).
  These are well-labeled English emotion corpora. We train on them first
  to get a working baseline, then fine-tune on Nigerian data as it becomes
  available.

  IEMOCAP: ~12h, 4 label classes (happy, angry, sad, neutral mapped from
  original 9 categories). Requires institutional request form at
  https://sail.usc.edu/iemocap/iemocap_release.htm

  RAVDESS: ~24 actors, 8 emotions mapped to our 4 classes.
  Available on Zenodo: https://zenodo.org/record/1188976

  This script ONLY implements Strategy 1 by default.
  Run: python scripts/data/prepare_emotion_labels.py

=====================================================================
STRATEGY 2 (documented, requires manual work, NOT auto-executed):
  Nollywood scene-labeled audio.

  Target: 20–50 hours of scene-labeled audio from Nigerian movies/TV.
  Process:
    1. Obtain licensed audio from Nollywood distributors
    2. Use scene-level metadata (fight scene → angry, reunion → happy, etc.)
       as weak labels
    3. Validate with human annotators (target: 3 annotations per clip)
    4. Build manifest with orinode.data.manifests.ManifestWriter

  Status: Pending rights negotiation. Not auto-executed.

=====================================================================
STRATEGY 3 (documented, future work, NOT auto-executed):
  Crowdsource emotion labels via Orinode's platform.

  Target: 5,000+ labelled clips across 5 languages from Nigerian annotators.
  Process:
    1. Randomly sample NaijaVoices clips (1–5 sec each)
    2. Present to Nigerian annotators on the Orinode platform with
       4 emotion options (happy / angry / sad / neutral)
    3. Require 3+ annotator agreement for inclusion
    4. Stratify final dataset by language: EN, HA, YO, IG, PCM

  Status: Platform UI for annotation not yet built.

=====================================================================
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LABEL_MAP_IEMOCAP = {
    "hap": "happy",
    "exc": "happy",  # excited → happy
    "ang": "angry",
    "sad": "sad",
    "neu": "neutral",
    "fru": "neutral",  # frustrated → neutral (imperfect but workable)
}

LABEL_MAP_RAVDESS = {
    "01": "neutral",
    "02": "neutral",  # calm → neutral
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "neutral",  # fearful → neutral (out of scope)
    "07": "neutral",  # disgust → neutral
    "08": "neutral",  # surprised → neutral
}


def strategy1_iemocap(iemocap_root: Path, out_manifest: Path) -> int:
    """Build a manifest from IEMOCAP label files."""
    rows = []
    for label_file in iemocap_root.rglob("*.lab"):
        for line in label_file.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            utt_id, raw_label = parts[0], parts[2]
            label = LABEL_MAP_IEMOCAP.get(raw_label)
            if label is None:
                continue
            audio = label_file.parent.parent / "sentences" / "wav" / (utt_id + ".wav")
            if not audio.exists():
                continue
            rows.append({"audio_path": str(audio), "label": label, "source": "iemocap"})

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return len(rows)


def strategy1_ravdess(ravdess_root: Path, out_manifest: Path) -> int:
    """Build a manifest from RAVDESS filename conventions."""
    rows = []
    for wav in ravdess_root.rglob("*.wav"):
        parts = wav.stem.split("-")
        if len(parts) < 3:
            continue
        emotion_code = parts[2]
        label = LABEL_MAP_RAVDESS.get(emotion_code)
        if label is None:
            continue
        rows.append({"audio_path": str(wav), "label": label, "source": "ravdess"})

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare emotion labels (Strategy 1)")
    parser.add_argument("--iemocap-root", type=Path, default=None)
    parser.add_argument("--ravdess-root", type=Path, default=None)
    parser.add_argument("--out-train", type=Path, default=None)
    parser.add_argument("--out-val", type=Path, default=None)
    args = parser.parse_args()

    from orinode.paths import WS, ensure_workspace

    ensure_workspace()
    out_train = args.out_train or WS.data / "manifests" / "emotion_train.jsonl"
    out_val = args.out_val or WS.data / "manifests" / "emotion_val.jsonl"

    total = 0

    iemocap = args.iemocap_root or WS.data / "raw" / "iemocap"
    if iemocap.exists():
        n = strategy1_iemocap(iemocap, out_train)
        print(f"IEMOCAP: {n} utterances → {out_train}")
        total += n
    else:
        print(f"IEMOCAP not found at {iemocap}. Request access from USC SAIL.")

    ravdess = args.ravdess_root or WS.data / "raw" / "ravdess"
    if ravdess.exists():
        n = strategy1_ravdess(ravdess, out_train)
        print(f"RAVDESS: {n} clips → {out_train}")
        total += n
    else:
        print(f"RAVDESS not found at {ravdess}. Download from Zenodo #1188976.")

    if total == 0:
        print("\nNo data found. Scaffold only — see docstring for download instructions.")
        return

    # Simple 90/10 train/val split
    import random

    lines = out_train.read_text().splitlines()
    random.shuffle(lines)
    split = int(len(lines) * 0.9)
    out_train.write_text("\n".join(lines[:split]) + "\n")
    out_val.write_text("\n".join(lines[split:]) + "\n")
    print(f"\n{len(lines[:split])} train / {len(lines[split:])} val")
    print("Strategy 2 (Nollywood) and Strategy 3 (crowdsource) not auto-executed.")
    print("See script docstring and docs/AUX_MODELS.md for details.")


if __name__ == "__main__":
    main()
