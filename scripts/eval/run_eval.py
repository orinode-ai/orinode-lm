"""ASR and code-switching evaluation script.

Usage::

    python scripts/eval/run_eval.py --run-id <run_id> --mode asr
    python scripts/eval/run_eval.py --run-id <run_id> --mode cs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=["asr", "cs"], default="asr")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    from orinode.paths import WS, ensure_workspace

    ensure_workspace()

    # Resolve checkpoint
    ckpt = args.checkpoint
    if ckpt is None:
        stage_dir = WS.models_checkpoints / "stage4_instruct"
        if not stage_dir.exists():
            stage_dir = WS.models_checkpoints / "stage3_speech_llm"
        ckpts = sorted(stage_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not ckpts:
            print(f"No checkpoints found under {WS.models_checkpoints}. Run training first.")
            return
        ckpt = ckpts[0]

    # Resolve manifest
    manifest = args.manifest
    if manifest is None:
        suffix = "cs_test.jsonl" if args.mode == "cs" else "test.jsonl"
        manifest = WS.data / "manifests" / suffix
    if not manifest.exists():
        print(f"Manifest not found: {manifest}. Run make build-manifests first.")
        return

    print(f"Evaluating run={args.run_id}  mode={args.mode}")
    print(f"  checkpoint: {ckpt}")
    print(f"  manifest:   {manifest}")
    print("  (full evaluation not yet implemented — wire in Transcriber + WER)")

    # Placeholder result
    result = {
        "run_id": args.run_id,
        "mode": args.mode,
        "checkpoint": str(ckpt),
        "manifest": str(manifest),
        "wer": {"en": None, "ha": None, "ig": None, "yo": None, "pcm": None},
        "cs_wer": None,
    }

    out = WS.evals / f"{args.run_id}_{args.mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"Result written → {out}")


if __name__ == "__main__":
    main()
