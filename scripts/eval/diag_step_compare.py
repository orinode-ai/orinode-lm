"""Focused diagnostic: compare two checkpoints on a fixed clip list.

Reads clip list from a previous wer_compare.json (or samples fresh with seed=42).
Runs both checkpoints on CPU with current production eval params.

Usage:
    python scripts/eval/diag_step_compare.py \
        --ckpt-a .../step_500.pt  --label-a step_500 \
        --ckpt-b .../step_2500.pt --label-b step_2500 \
        --prev-result /tmp/wer_compare.json \
        --manifest workspace/data/manifests/afrispeech_200_dev.jsonl \
        --out /tmp/diag_compare.json
"""

from __future__ import annotations

import argparse
import json
import math
import unicodedata
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────

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


def non_latin_ratio(s: str) -> float:
    if not s:
        return 0.0
    n = sum(1 for c in s if unicodedata.category(c).startswith("L") and ord(c) > 0x024F)
    return n / max(len(s), 1)


def classify(wa: float, wb: float, hyp_b: str, ref: str) -> str:
    if math.isnan(wa) or math.isnan(wb):
        return "UNKNOWN"
    if non_latin_ratio(hyp_b) > 0.1:
        return "CATASTROPHIC"
    ref_words = len(ref.split())
    hyp_b_words = len(hyp_b.split())
    # Repetition loop heuristic: output much longer than ref or contains repeat chars
    if hyp_b_words > max(ref_words * 2, ref_words + 5):
        return "CATASTROPHIC"
    if any(hyp_b.count(c * 4) > 0 for c in "abcdefghijklmnopqrstuvwxyz "):
        return "CATASTROPHIC"
    delta = wb - wa
    if delta > 0.05:
        return "REGRESSED"
    if delta < -0.05:
        return "IMPROVED"
    return "SAME"


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path: Path, cache_dir: Path):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from orinode.models.lora_utils import LoRAConfig, apply_lora

    model_id = "openai/whisper-large-v3"
    print(f"  Loading base model ...", flush=True)
    processor = WhisperProcessor.from_pretrained(model_id, cache_dir=str(cache_dir))
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float32, attn_implementation="sdpa",
        cache_dir=str(cache_dir),
    )
    print(f"  Applying DoRA (r=32, alpha=64) ...", flush=True)
    model.model.encoder = apply_lora(model.model.encoder, LoRAConfig(
        r=32, alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        dropout=0.05, use_dora=True,
    ))
    print(f"  Loading {ckpt_path.name} ...", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"  WARNING missing: {missing[:3]}", flush=True)
    print(f"  step={ckpt.get('global_step','?')}  saved_wer={ckpt.get('val_wer',float('nan')):.4f}",
          flush=True)
    model.eval()
    return model, processor


# ── inference ─────────────────────────────────────────────────────────────────

def transcribe(model, processor, audio_paths: list[str]) -> list[str]:
    import time
    import torch
    import torchaudio
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer

    normalizer = BasicTextNormalizer()
    tok = processor.tokenizer
    hyps: list[str] = []

    for i, path in enumerate(audio_paths):
        t0 = time.time()
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
            hyp = ""
        elapsed = time.time() - t0
        print(f"  [{i+1:2d}/{len(audio_paths)}] {Path(path).name[:32]}  cpu={elapsed:.1f}s  "
              f"hyp={hyp[:60]!r}", flush=True)
        hyps.append(hyp)
    return hyps


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
    ap.add_argument("--ckpt-a", type=Path, required=True)
    ap.add_argument("--ckpt-b", type=Path, required=True)
    ap.add_argument("--label-a", default="ckpt_a")
    ap.add_argument("--label-b", default="ckpt_b")
    ap.add_argument("--prev-result", type=Path, default=None,
                    help="Previous wer_compare.json — reuse its clip list")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("/tmp/diag_compare.json"))
    args = ap.parse_args()

    from transformers.models.whisper.english_normalizer import BasicTextNormalizer
    normalizer = BasicTextNormalizer()

    # ── clip list ──
    if args.prev_result and args.prev_result.exists():
        prev = json.loads(args.prev_result.read_text())
        manifest_rows = [json.loads(l) for l in args.manifest.read_text().splitlines() if l.strip()]
        by_name = {Path(r["audio_path"]).name: r for r in manifest_rows}
        clips = []
        for entry in prev:
            name = entry["audio"]
            if name in by_name:
                clips.append(by_name[name])
        print(f"Reusing {len(clips)}/{len(prev)} clips from previous diagnostic", flush=True)
    else:
        import random
        manifest_rows = [json.loads(l) for l in args.manifest.read_text().splitlines() if l.strip()]
        rng = random.Random(42)
        clips = rng.sample(manifest_rows, min(30, len(manifest_rows)))
        print(f"Fresh sample of {len(clips)} clips (seed=42)", flush=True)

    audio_paths = [c["audio_path"] for c in clips]
    refs = [normalizer(c["text"]) for c in clips]

    print(f"\n=== {args.label_a} ===", flush=True)
    model_a, proc_a = load_model(args.ckpt_a, cache_dir)
    hyps_a = transcribe(model_a, proc_a, audio_paths)
    del model_a, proc_a

    print(f"\n=== {args.label_b} ===", flush=True)
    model_b, proc_b = load_model(args.ckpt_b, cache_dir)
    hyps_b = transcribe(model_b, proc_b, audio_paths)
    del model_b, proc_b

    # ── per-clip results ──
    results = []
    for i, (ref, ha, hb) in enumerate(zip(refs, hyps_a, hyps_b)):
        wa = clip_wer(ref, ha)
        wb = clip_wer(ref, hb)
        label = classify(wa, wb, hb, ref)
        results.append({
            "idx": i,
            "audio": Path(audio_paths[i]).name,
            "duration": clips[i].get("duration"),
            "dialect": clips[i].get("dialect", ""),
            "ref": ref,
            f"hyp_{args.label_a}": ha,
            f"hyp_{args.label_b}": hb,
            f"wer_{args.label_a}": round(wa, 4) if not math.isnan(wa) else None,
            f"wer_{args.label_b}": round(wb, 4) if not math.isnan(wb) else None,
            "delta": round(wb - wa, 4) if not (math.isnan(wa) or math.isnan(wb)) else None,
            "verdict": label,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nFull results → {args.out}", flush=True)

    # ── summary ──
    counts = {"IMPROVED": 0, "SAME": 0, "REGRESSED": 0, "CATASTROPHIC": 0, "UNKNOWN": 0}
    for r in results:
        counts[r["verdict"]] += 1

    valid = [r for r in results if r["delta"] is not None]
    avg_a = sum(r[f"wer_{args.label_a}"] for r in valid) / max(len(valid), 1)
    avg_b = sum(r[f"wer_{args.label_b}"] for r in valid) / max(len(valid), 1)

    print(f"\n{'='*65}")
    print(f"DIAGNOSTIC: {args.label_a} vs {args.label_b}  ({len(results)} clips)")
    print(f"{'='*65}")
    print(f"  Avg WER  {args.label_a}: {avg_a:.3f}   {args.label_b}: {avg_b:.3f}  "
          f"delta={avg_b-avg_a:+.3f}")
    print(f"  IMPROVED:     {counts['IMPROVED']:3d}  ({counts['IMPROVED']/len(results)*100:.0f}%)")
    print(f"  SAME:         {counts['SAME']:3d}  ({counts['SAME']/len(results)*100:.0f}%)")
    print(f"  REGRESSED:    {counts['REGRESSED']:3d}  ({counts['REGRESSED']/len(results)*100:.0f}%)")
    print(f"  CATASTROPHIC: {counts['CATASTROPHIC']:3d}  ({counts['CATASTROPHIC']/len(results)*100:.0f}%)")

    reg_pct = (counts["REGRESSED"] + counts["CATASTROPHIC"]) / len(results) * 100
    if reg_pct >= 60:
        verdict = "BROAD REGRESSION — recommend kill + restart with adjusted params"
    elif reg_pct >= 30:
        verdict = "MIXED SIGNAL — consider adjustment"
    else:
        verdict = "NARROW FAILURE — hypothesis supported, let training continue"
    print(f"\n  VERDICT: {verdict}")
    print(f"  (regressed+catastrophic: {reg_pct:.0f}% of clips)")

    # ── top regressions ──
    regressed = sorted([r for r in results if r["verdict"] in ("REGRESSED", "CATASTROPHIC")],
                       key=lambda r: (r["delta"] or 0), reverse=True)
    if regressed:
        print(f"\n--- Top regressions ---")
        for r in regressed[:5]:
            print(f"\n  [{r['idx']:2d}] {r['audio']}  dur={r['duration']:.1f}s  "
                  f"dialect={r['dialect']}  [{r['verdict']}]")
            print(f"    REF: {r['ref']}")
            print(f"    {args.label_a}: {r[f'hyp_{args.label_a}']}")
            print(f"    {args.label_b}: {r[f'hyp_{args.label_b}']}")
            print(f"    WER: {r[f'wer_{args.label_a}']} → {r[f'wer_{args.label_b}']}  "
                  f"delta={r['delta']:+.3f}")

    # ── top improvements ──
    improved = sorted([r for r in results if r["verdict"] == "IMPROVED"],
                      key=lambda r: (r["delta"] or 0))
    if improved:
        print(f"\n--- Top improvements ---")
        for r in improved[:3]:
            print(f"\n  [{r['idx']:2d}] {r['audio']}  dur={r['duration']:.1f}s  "
                  f"dialect={r['dialect']}  [IMPROVED]")
            print(f"    REF: {r['ref']}")
            print(f"    {args.label_a}: {r[f'hyp_{args.label_a}']}")
            print(f"    {args.label_b}: {r[f'hyp_{args.label_b}']}")
            print(f"    WER: {r[f'wer_{args.label_a}']} → {r[f'wer_{args.label_b}']}  "
                  f"delta={r['delta']:+.3f}")


if __name__ == "__main__":
    main()
