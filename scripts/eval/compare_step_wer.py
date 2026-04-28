"""Compare per-clip WER between two checkpoints on a dev sample.

Usage (on VPS):
    python scripts/eval/compare_step_wer.py \
        --ckpt-a workspace/models/checkpoints/.../step_500.pt \
        --ckpt-b workspace/models/checkpoints/.../step_1000.pt \
        --label-a step_500 --label-b step_1000 \
        --manifest workspace/data/manifests/afrispeech_200_dev.jsonl \
        --n-clips 30 --out /tmp/wer_compare.json

Runs entirely on CPU (GPU is occupied by training).
Selects n_clips shortest clips for tractable CPU runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path


def load_manifest_sample(manifest: Path, n: int) -> list[dict]:
    """Return n shortest clips from manifest (deterministic, no random)."""
    rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: float(r.get("duration", 0)))
    return rows[:n]


def load_model(ckpt_path: Path, cache_dir: Path) -> tuple:
    """Load Whisper-large-v3 + DoRA on CPU, return (model, processor)."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    from orinode.models.lora_utils import LoRAConfig, apply_lora

    model_id = "openai/whisper-large-v3"
    print(f"  Loading base model from cache ...", flush=True)
    processor = WhisperProcessor.from_pretrained(model_id, cache_dir=str(cache_dir))
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
        cache_dir=str(cache_dir),
    )

    print(f"  Applying DoRA (r=32, alpha=64) ...", flush=True)
    lora_config = LoRAConfig(
        r=32,
        alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        dropout=0.05,
        use_dora=True,
    )
    model.model.encoder = apply_lora(model.model.encoder, lora_config)

    print(f"  Loading weights from {ckpt_path.name} ...", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"  WARNING missing keys: {missing[:5]}", flush=True)
    if unexpected:
        print(f"  WARNING unexpected keys: {unexpected[:5]}", flush=True)
    saved_step = ckpt.get("global_step", "?")
    saved_wer = ckpt.get("val_wer", float("nan"))
    print(f"  Loaded: step={saved_step}  saved_wer={saved_wer:.4f}", flush=True)

    model.eval()
    return model, processor


def transcribe_clips(model, processor, clips: list[dict]) -> list[str]:
    """Return list of hypotheses, one per clip, in clip order."""
    import torch
    import torchaudio
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer

    normalizer = BasicTextNormalizer()
    hyps: list[str] = []
    tok = processor.tokenizer

    for i, clip in enumerate(clips):
        t0 = time.time()
        path = clip["audio_path"]
        try:
            waveform, sr = torchaudio.load(path)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(0, keepdim=True)
            audio_np = waveform.squeeze(0).numpy()

            inputs = processor(
                audio_np,
                sampling_rate=16000,
                return_tensors="pt",
            )
            input_features = inputs.input_features.to(torch.float32)

            with torch.no_grad():
                pred_ids = model.generate(
                    input_features,
                    language="en",
                    task="transcribe",
                    num_beams=1,
                    max_new_tokens=444,
                    no_repeat_ngram_size=3,
                )
            hyp = tok.decode(pred_ids[0], skip_special_tokens=True)
            hyp_norm = normalizer(hyp)
        except Exception as exc:
            print(f"  [clip {i}] ERROR: {exc}", flush=True)
            hyp_norm = ""

        elapsed = time.time() - t0
        print(f"  [{i+1:2d}/{len(clips)}] {Path(path).name[:30]} "
              f"dur={clip.get('duration', 0):.1f}s  "
              f"cpu={elapsed:.1f}s  hyp={hyp_norm[:60]!r}", flush=True)
        hyps.append(hyp_norm)

    return hyps


def clip_wer(ref: str, hyp: str) -> float:
    """Token-level WER for a single utterance."""
    r = ref.split()
    h = hyp.split()
    if not r:
        return float("nan")
    # DP edit distance
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r):
        nd = [i + 1] + [0] * len(h)
        for j, hw in enumerate(h):
            nd[j + 1] = min(d[j] + (0 if rw == hw else 1),
                            d[j + 1] + 1,
                            nd[j] + 1)
        d = nd
    return d[len(h)] / len(r)


def main() -> None:
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer
    normalizer = BasicTextNormalizer()

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-a", type=Path, required=True)
    ap.add_argument("--ckpt-b", type=Path, required=True)
    ap.add_argument("--label-a", default="ckpt_a")
    ap.add_argument("--label-b", default="ckpt_b")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--n-clips", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("/tmp/wer_compare.json"))
    args = ap.parse_args()

    import sys, os
    sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
    os.environ.setdefault(
        "TRANSFORMERS_CACHE",
        str(Path(__file__).parents[2] / "workspace/cache/transformers"),
    )
    cache_dir = Path(os.environ["TRANSFORMERS_CACHE"])

    print(f"\n=== Clip sample ===", flush=True)
    clips = load_manifest_sample(args.manifest, args.n_clips)
    print(f"Selected {len(clips)} shortest clips "
          f"(dur range {clips[0]['duration']:.1f}s – {clips[-1]['duration']:.1f}s)", flush=True)

    refs_norm = [normalizer(c["text"]) for c in clips]

    print(f"\n=== {args.label_a} inference ===", flush=True)
    model_a, proc_a = load_model(args.ckpt_a, cache_dir)
    hyps_a = transcribe_clips(model_a, proc_a, clips)
    del model_a, proc_a

    print(f"\n=== {args.label_b} inference ===", flush=True)
    model_b, proc_b = load_model(args.ckpt_b, cache_dir)
    hyps_b = transcribe_clips(model_b, proc_b, clips)
    del model_b, proc_b

    # Per-clip WER
    results = []
    for i, (ref, ha, hb) in enumerate(zip(refs_norm, hyps_a, hyps_b)):
        wa = clip_wer(ref, ha)
        wb = clip_wer(ref, hb)
        delta = (wb - wa) if not (math.isnan(wa) or math.isnan(wb)) else float("nan")
        results.append({
            "idx": i,
            "audio": Path(clips[i]["audio_path"]).name,
            "duration": clips[i].get("duration"),
            "dialect": clips[i].get("dialect", ""),
            "ref": ref,
            f"hyp_{args.label_a}": ha,
            f"hyp_{args.label_b}": hb,
            f"wer_{args.label_a}": round(wa, 4) if not math.isnan(wa) else None,
            f"wer_{args.label_b}": round(wb, 4) if not math.isnan(wb) else None,
            "delta_wer": round(delta, 4) if not math.isnan(delta) else None,
        })

    results.sort(key=lambda r: (r["delta_wer"] or -999), reverse=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nFull results → {args.out}", flush=True)

    # Print top 10 regressions
    print(f"\n{'='*70}")
    print(f"TOP 10 regressions ({args.label_a} → {args.label_b}, WER delta worst first)")
    print(f"{'='*70}")
    for r in results[:10]:
        wa = r[f"wer_{args.label_a}"]
        wb = r[f"wer_{args.label_b}"]
        d = r["delta_wer"]
        print(f"\n[{r['idx']:3d}] {r['audio']}  dur={r['duration']:.1f}s  "
              f"dialect={r['dialect']}")
        print(f"  WER: {args.label_a}={wa}  {args.label_b}={wb}  delta={d:+.3f}")
        print(f"  REF : {r['ref']}")
        print(f"  {args.label_a}: {r[f'hyp_{args.label_a}']}")
        print(f"  {args.label_b}: {r[f'hyp_{args.label_b}']}")

    # Aggregate stats
    valid = [r for r in results if r["delta_wer"] is not None]
    regressions = [r for r in valid if r["delta_wer"] > 0]
    improvements = [r for r in valid if r["delta_wer"] < 0]
    avg_a = sum(r[f"wer_{args.label_a}"] for r in valid if r[f"wer_{args.label_a}"] is not None) / max(len(valid), 1)
    avg_b = sum(r[f"wer_{args.label_b}"] for r in valid if r[f"wer_{args.label_b}"] is not None) / max(len(valid), 1)

    print(f"\n=== Aggregate ===")
    print(f"  Clips analysed: {len(valid)}")
    print(f"  Avg WER  {args.label_a}: {avg_a:.3f}   {args.label_b}: {avg_b:.3f}")
    print(f"  Regressions: {len(regressions)}  Improvements: {len(improvements)}")

    # Failure mode counts
    longer, wrong_lang, truncated, other = 0, 0, 0, 0
    import re, unicodedata

    def _non_latin_ratio(s: str) -> float:
        if not s:
            return 0.0
        n = sum(1 for c in s if unicodedata.category(c).startswith("L") and ord(c) > 0x024F)
        return n / max(len(s), 1)

    for r in regressions:
        ha = r[f"hyp_{args.label_a}"]
        hb = r[f"hyp_{args.label_b}"]
        ref = r["ref"]
        len_ratio = len(hb.split()) / max(len(ha.split()), 1) if ha else 1
        if _non_latin_ratio(hb) > 0.1:
            wrong_lang += 1
        elif len(hb.split()) < 0.5 * len(ref.split()):
            truncated += 1
        elif len_ratio > 1.4:
            longer += 1
        else:
            other += 1

    print(f"\n  Failure modes (among {len(regressions)} regressions):")
    print(f"    Longer / insertions: {longer}")
    print(f"    Wrong language/garbage: {wrong_lang}")
    print(f"    Truncated / deletions: {truncated}")
    print(f"    Other (new specific errors): {other}")


if __name__ == "__main__":
    main()
