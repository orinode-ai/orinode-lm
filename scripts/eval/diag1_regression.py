"""Diagnostic 1: step_500 vs step_3000 regression analysis.

Reads cached step_500 per-clip results, runs step_3000 inference on
the same clips, computes deltas, identifies regression patterns.
"""
from __future__ import annotations
import argparse, json, math, time, unicodedata
from pathlib import Path


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
    if not hyp.strip():
        return False
    nl = sum(1 for c in hyp if unicodedata.category(c).startswith("L") and ord(c) > 0x024F)
    if nl / max(len(hyp), 1) > 0.1:
        return True
    if any(hyp.count(c * 4) > 0 for c in "abcdefghijklmnopqrstuvwxyz"):
        return True
    if len(hyp.split()) > max(len(ref.split()) * 3, len(ref.split()) + 10):
        return True
    return False


def load_model(ckpt_path: Path, cache_dir: Path):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from orinode.models.lora_utils import LoRAConfig, apply_lora
    model_id = "openai/whisper-large-v3"
    print(f"Loading base model ...", flush=True)
    proc = WhisperProcessor.from_pretrained(model_id, cache_dir=str(cache_dir))
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float32, attn_implementation="sdpa",
        cache_dir=str(cache_dir))
    print(f"Applying DoRA ...", flush=True)
    model.model.encoder = apply_lora(model.model.encoder, LoRAConfig(
        r=32, alpha=64, target_modules=["q_proj","k_proj","v_proj","out_proj"],
        dropout=0.05, use_dora=True))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"], strict=False)
    print(f"  step={ckpt.get('global_step','?')}  saved_wer={ckpt.get('val_wer',float('nan')):.4f}", flush=True)
    model.eval()
    return model, proc


def transcribe(model, proc, audio_paths: list[str]) -> list[str]:
    import torch, torchaudio
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer
    normalizer = BasicTextNormalizer()
    tok = proc.tokenizer
    hyps = []
    n = len(audio_paths)
    t0_all = time.time()
    for i, path in enumerate(audio_paths):
        t0 = time.time()
        try:
            wav, sr = torchaudio.load(path)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            inp = proc(wav.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt")
            feats = inp.input_features.to(torch.float32)
            with torch.no_grad():
                ids = model.generate(feats, language="en", task="transcribe",
                                     num_beams=1, max_new_tokens=444,
                                     no_repeat_ngram_size=3, repetition_penalty=1.2)
            hyp = normalizer(tok.decode(ids[0], skip_special_tokens=True))
        except Exception as e:
            print(f"  [clip {i+1}] ERROR: {e}", flush=True)
            hyp = ""
        elapsed = time.time() - t0
        eta = (time.time() - t0_all) / (i + 1) * (n - i - 1)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1:3d}/{n}] cpu={elapsed:.1f}s  ETA={eta/60:.1f}min  hyp={hyp[:55]!r}", flush=True)
        hyps.append(hyp)
    return hyps


def main():
    import os, sys
    sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
    os.environ.setdefault("TRANSFORMERS_CACHE",
        str(Path(__file__).parents[2] / "workspace/cache/transformers"))
    cache_dir = Path(os.environ["TRANSFORMERS_CACHE"])

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-b", type=Path, required=True, help="step_3000.pt")
    ap.add_argument("--label-b", default="step_3000")
    ap.add_argument("--cached-a", type=Path, required=True, help="filtered_eval_step_500.json")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("/tmp/diag1_regression.json"))
    args = ap.parse_args()

    # Load cached step_500 results
    cached = json.loads(args.cached_a.read_text())
    clips_a = cached["per_clip"]  # 150 clips with ref, hyp, wer already computed

    # Rebuild audio path mapping from manifest
    manifest = {Path(json.loads(l)["audio_path"]).name: json.loads(l)
                for l in args.manifest.read_text().splitlines() if l.strip()}
    audio_paths = []
    for c in clips_a:
        row = manifest.get(c["audio"])
        audio_paths.append(row["audio_path"] if row else "")

    print(f"Running step_3000 inference on {len(audio_paths)} clips ...", flush=True)
    model_b, proc_b = load_model(args.ckpt_b, cache_dir)
    hyps_b = transcribe(model_b, proc_b, audio_paths)
    del model_b, proc_b

    from transformers.models.whisper.english_normalizer import BasicTextNormalizer
    normalizer = BasicTextNormalizer()

    results = []
    for i, (ca, hb) in enumerate(zip(clips_a, hyps_b)):
        ref = ca["ref"]
        ha  = ca["hyp"]
        wa  = ca["wer"]
        wb  = clip_wer(ref, hb) if ref.strip() else float("nan")
        wb_r = round(wb, 4) if not math.isnan(wb) else None
        delta = round(wb - wa, 4) if (wa is not None and wb_r is not None) else None

        if delta is None:
            verdict = "UNKNOWN"
        elif delta < -0.10:
            verdict = "IMPROVED"
        elif delta > 0.10:
            verdict = "CATASTROPHIC" if is_garbage(hb, ref) else "REGRESSED"
        else:
            verdict = "SAME"

        row = manifest.get(ca["audio"], {})
        results.append({
            "idx": i,
            "audio": ca["audio"],
            "duration": ca["duration"],
            "dialect": ca.get("dialect", row.get("dialect", "")),
            "speaker_id": row.get("speaker_id", ""),
            "domain": row.get("domain", ""),
            "ref": ref,
            "hyp_step_500": ha,
            "hyp_step_3000": hb,
            "wer_step_500": wa,
            "wer_step_3000": wb_r,
            "delta": delta,
            "verdict": verdict,
            "is_garbage_3000": is_garbage(hb, ref),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))

    # Summary
    counts = {k: sum(1 for r in results if r["verdict"] == k)
              for k in ["IMPROVED","SAME","REGRESSED","CATASTROPHIC","UNKNOWN"]}
    valid = [r for r in results if r["delta"] is not None]
    avg_a = sum(r["wer_step_500"] for r in valid) / max(len(valid), 1)
    avg_b = sum(r["wer_step_3000"] for r in valid) / max(len(valid), 1)

    print(f"\n{'='*65}", flush=True)
    print(f"REGRESSION ANALYSIS: step_500 vs {args.label_b}", flush=True)
    print(f"{'='*65}", flush=True)
    print(f"  Avg WER  500: {avg_a*100:.2f}%   3000: {avg_b*100:.2f}%   delta={( avg_b-avg_a)*100:+.2f}pp")
    for k,v in counts.items():
        print(f"  {k:12s}: {v:3d}  ({v/len(results)*100:.0f}%)")

    regressed = [r for r in results if r["verdict"] in ("REGRESSED","CATASTROPHIC")]
    regressed.sort(key=lambda r: r["delta"] or 0, reverse=True)

    print(f"\n--- Top 10 regressions ---", flush=True)
    for r in regressed[:10]:
        print(f"\n  [{r['idx']:3d}] {r['audio']}  dur={r['duration']:.1f}s  "
              f"dialect={r['dialect']}  [{r['verdict']}]  delta={r['delta']:+.3f}")
        print(f"    REF:  {r['ref']}")
        print(f"    500:  {r['hyp_step_500']}")
        print(f"    3000: {r['hyp_step_3000']}")

    # Pattern analysis
    print(f"\n--- Regression pattern ---", flush=True)
    if regressed:
        durs = [r["duration"] for r in regressed]
        all_durs = [r["duration"] for r in results]
        print(f"  Regressed clip dur: mean={sum(durs)/len(durs):.1f}s  "
              f"vs overall mean={sum(all_durs)/len(all_durs):.1f}s")
        dial = {}
        for r in regressed:
            d = r["dialect"] or "unknown"
            dial[d] = dial.get(d, 0) + 1
        dial_all = {}
        for r in results:
            d = r["dialect"] or "unknown"
            dial_all[d] = dial_all.get(d, 0) + 1
        print(f"  Regressed by dialect: {sorted(dial.items(), key=lambda x: -x[1])}")
        print(f"  All clips by dialect: {sorted(dial_all.items(), key=lambda x: -x[1])}")
        ref_lens = [len(r["ref"].split()) for r in regressed]
        print(f"  Regressed ref length: mean={sum(ref_lens)/len(ref_lens):.1f} words")
        all_lens = [len(r["ref"].split()) for r in results]
        print(f"  All clips ref length: mean={sum(all_lens)/len(all_lens):.1f} words")

    print(f"\nSaved → {args.out}", flush=True)


if __name__ == "__main__":
    main()
