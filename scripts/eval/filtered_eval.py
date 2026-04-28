"""Filtered WER evaluation on a single checkpoint.

Evaluates on all dev clips, then reports WER broken down by duration filter.
Saves per-clip results for later analysis.

Usage (VPS, CPU only):
    python scripts/eval/filtered_eval.py \
        --ckpt workspace/models/checkpoints/.../best.pt \
        --label step_500 \
        --manifest workspace/data/manifests/afrispeech_200_dev.jsonl \
        --out workspace/diagnostics/filtered_eval_step_500.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import unicodedata
from pathlib import Path


# ── WER helpers ───────────────────────────────────────────────────────────────

def clip_wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    if not r:
        return float("nan")
    d = list(range(len(h) + 1))
    for rw in r:
        nd = [d[0] + 1] + [0] * len(h)
        for j, hw in enumerate(h):
            nd[j + 1] = min(d[j] + (0 if rw == hw else 1), d[j + 1] + 1, nd[j] + 1)
        d = nd
    return d[len(h)] / len(r)


def is_garbage(hyp: str, ref: str) -> bool:
    """True if hyp looks like a loop or wrong-language output."""
    if not hyp.strip():
        return False
    # Non-latin characters
    nl = sum(1 for c in hyp if unicodedata.category(c).startswith("L") and ord(c) > 0x024F)
    if nl / max(len(hyp), 1) > 0.1:
        return True
    # Repetition: any 4-char run of same character
    if any(hyp.count(c * 4) > 0 for c in "abcdefghijklmnopqrstuvwxyz"):
        return True
    # Output length >> ref length
    if len(hyp.split()) > max(len(ref.split()) * 3, len(ref.split()) + 10):
        return True
    return False


def tier_stats(records: list[dict], label: str, wer_key: str = "wer") -> dict:
    wers = [r[wer_key] for r in records if r[wer_key] is not None and not math.isnan(r[wer_key])]
    if not wers:
        return {"label": label, "n": 0}
    wers_s = sorted(wers)
    n = len(wers_s)
    empty = sum(1 for r in records if not r["hyp"].strip())
    loops = sum(1 for r in records if r.get("is_garbage", False))
    over_100 = sum(1 for w in wers_s if w > 1.0)
    return {
        "label": label,
        "n": n,
        "mean_wer": round(statistics.mean(wers_s), 4),
        "median_wer": round(statistics.median(wers_s), 4),
        "p25_wer": round(wers_s[int(n * 0.25)], 4),
        "p75_wer": round(wers_s[int(n * 0.75)], 4),
        "p95_wer": round(wers_s[int(n * 0.95)], 4),
        "empty_hyps": empty,
        "garbage_hyps": loops,
        "wer_over_100pct": over_100,
    }


def print_tier(t: dict) -> None:
    if t["n"] == 0:
        print(f"  {t['label']}: no clips")
        return
    print(f"  {t['label']}  n={t['n']}")
    print(f"    Mean WER:   {t['mean_wer']*100:5.2f}%")
    print(f"    Median WER: {t['median_wer']*100:5.2f}%")
    print(f"    p25/p75/p95: {t['p25_wer']*100:.1f}% / {t['p75_wer']*100:.1f}% / {t['p95_wer']*100:.1f}%")
    print(f"    Empty hyps: {t['empty_hyps']}  Garbage/loops: {t['garbage_hyps']}  WER>100%: {t['wer_over_100pct']}")


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path: Path, cache_dir: Path):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from orinode.models.lora_utils import LoRAConfig, apply_lora

    model_id = "openai/whisper-large-v3"
    print(f"Loading base model from cache ...", flush=True)
    t0 = time.time()
    processor = WhisperProcessor.from_pretrained(model_id, cache_dir=str(cache_dir))
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float32, attn_implementation="sdpa",
        cache_dir=str(cache_dir),
    )
    print(f"Applying DoRA (r=32, alpha=64) ...", flush=True)
    model.model.encoder = apply_lora(model.model.encoder, LoRAConfig(
        r=32, alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        dropout=0.05, use_dora=True,
    ))
    print(f"Loading weights from {ckpt_path.name} ...", flush=True)
    ckpt = __import__("torch").load(ckpt_path, map_location="cpu", weights_only=True)
    missing, _ = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"WARNING missing keys: {missing[:3]}", flush=True)
    print(f"  step={ckpt.get('global_step','?')}  "
          f"saved_wer={ckpt.get('val_wer', float('nan')):.4f}  "
          f"loaded in {time.time()-t0:.1f}s", flush=True)
    model.eval()
    return model, processor


# ── inference ─────────────────────────────────────────────────────────────────

def transcribe_all(model, processor, clips: list[dict]) -> list[dict]:
    import torch
    import torchaudio
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer

    normalizer = BasicTextNormalizer()
    tok = processor.tokenizer
    results: list[dict] = []
    n = len(clips)
    t_start = time.time()

    for i, clip in enumerate(clips):
        t0 = time.time()
        ref_raw = clip.get("text", "")
        ref = normalizer(ref_raw)
        path = clip["audio_path"]
        hyp = ""

        try:
            wav, sr = torchaudio.load(path)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            inputs = processor(wav.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt")
            feats = inputs.input_features.to(torch.float32)
            with torch.no_grad():
                ids = model.generate(
                    feats,
                    language="en",
                    task="transcribe",
                    num_beams=1,
                    max_new_tokens=444,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2,
                )
            hyp = normalizer(tok.decode(ids[0], skip_special_tokens=True))
        except Exception as exc:
            print(f"  [clip {i+1}] ERROR: {exc}", flush=True)

        wer = clip_wer(ref, hyp) if ref.strip() else float("nan")
        garbage = is_garbage(hyp, ref)
        elapsed = time.time() - t0
        elapsed_total = time.time() - t_start
        eta = (elapsed_total / (i + 1)) * (n - i - 1)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1:3d}/{n}] dur={clip.get('duration',0):.1f}s  "
                  f"cpu={elapsed:.1f}s  wer={wer*100:.0f}%  "
                  f"ETA={eta/60:.1f}min  hyp={hyp[:50]!r}", flush=True)

        results.append({
            "audio": Path(path).name,
            "duration": clip.get("duration", 0.0),
            "dialect": clip.get("dialect", ""),
            "ref": ref,
            "hyp": hyp,
            "wer": round(wer, 4) if not math.isnan(wer) else None,
            "is_garbage": garbage,
        })

    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import os, sys
    sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
    os.environ.setdefault(
        "TRANSFORMERS_CACHE",
        str(Path(__file__).parents[2] / "workspace/cache/transformers"),
    )
    cache_dir = Path(os.environ["TRANSFORMERS_CACHE"])

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--label", default="checkpoint")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--min-duration", type=float, default=0.0,
                    help="Only evaluate clips >= this duration (seconds)")
    ap.add_argument("--max-clips", type=int, default=0,
                    help="Sample at most N clips (seed=42). 0 = use all.")
    ap.add_argument("--out", type=Path,
                    default=Path("workspace/diagnostics/filtered_eval.json"))
    args = ap.parse_args()

    import random
    all_clips = [json.loads(l) for l in args.manifest.read_text().splitlines() if l.strip()]
    clips = [c for c in all_clips if c.get("duration", 0) >= args.min_duration]
    if args.max_clips > 0 and len(clips) > args.max_clips:
        rng = random.Random(42)
        clips = rng.sample(clips, args.max_clips)
        clips.sort(key=lambda c: c.get("duration", 0))  # stable order for ETA
    print(f"Manifest: {len(all_clips)} total → {len(clips)} selected "
          f"(min_dur={args.min_duration}s, max_clips={args.max_clips or 'all'})", flush=True)
    print(f"Checkpoint: {args.ckpt}", flush=True)

    model, processor = load_model(args.ckpt, cache_dir)

    print(f"\nRunning inference on all {len(clips)} clips ...", flush=True)
    results = transcribe_all(model, processor, clips)

    # Duration tiers (always report sub-tiers of the evaluated set)
    all_r   = [r for r in results if r["wer"] is not None]
    ge3_r   = [r for r in all_r if r["duration"] >= 3.0]
    ge5_r   = [r for r in all_r if r["duration"] >= 5.0]
    short_r = [r for r in all_r if r["duration"] < 3.0]

    tiers = [tier_stats(all_r, f"Evaluated ({len(all_r)} clips, min_dur={args.min_duration}s)")]
    if short_r:
        tiers.append(tier_stats(short_r, "  sub: Clips <3s"))
    if ge3_r and args.min_duration < 3.0:
        tiers.append(tier_stats(ge3_r, "  sub: Clips >=3s"))
    if ge5_r:
        tiers.append(tier_stats(ge5_r, "  sub: Clips >=5s"))

    print(f"\n{'='*60}", flush=True)
    print(f"FILTERED EVAL — {args.label}", flush=True)
    print(f"{'='*60}", flush=True)
    for t in tiers:
        print_tier(t)
        print()

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "label": args.label,
        "checkpoint": str(args.ckpt),
        "manifest": str(args.manifest),
        "tiers": tiers,
        "per_clip": results,
    }
    args.out.write_text(json.dumps(output, indent=2))
    print(f"Saved → {args.out}", flush=True)


if __name__ == "__main__":
    main()
