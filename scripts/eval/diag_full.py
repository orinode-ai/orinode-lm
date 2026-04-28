"""Combined D1 + D3: regression analysis + audio quality check.

D1: step_500 (cached) vs step_3000 (GPU inference) on 150-clip sample.
D3: audio quality check on 20 worst-regressed clips.

Usage:
    python scripts/eval/diag_full.py \
        --ckpt-b workspace/models/checkpoints/.../step_3000.pt \
        --cached-a workspace/diagnostics/filtered_eval_step_500.json \
        --manifest workspace/data/manifests/afrispeech_200_dev.jsonl \
        --out workspace/diagnostics/regression_analysis.json
"""
from __future__ import annotations
import argparse, json, math, os, subprocess, sys, time, unicodedata
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


def load_model(ckpt_path: Path, cache_dir: Path, device: str):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from orinode.models.lora_utils import LoRAConfig, apply_lora

    model_id = "openai/whisper-large-v3"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Loading model → {device} ({dtype}) ...", flush=True)
    proc = WhisperProcessor.from_pretrained(model_id, cache_dir=str(cache_dir))
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation="sdpa",
        cache_dir=str(cache_dir))
    print("Applying DoRA (r=32, alpha=64) ...", flush=True)
    model.model.encoder = apply_lora(model.model.encoder, LoRAConfig(
        r=32, alpha=64, target_modules=["q_proj","k_proj","v_proj","out_proj"],
        dropout=0.05, use_dora=True))
    ckpt = __import__("torch").load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(device)
    print(f"  step={ckpt.get('global_step','?')}  saved_wer={ckpt.get('val_wer',float('nan')):.4f}", flush=True)
    model.eval()
    return model, proc


def transcribe_gpu(model, proc, audio_paths: list[str], device: str) -> list[str]:
    import torch, torchaudio
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer
    normalizer = BasicTextNormalizer()
    tok = proc.tokenizer
    hyps, n = [], len(audio_paths)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
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
            feats = inp.input_features.to(device=device, dtype=dtype)
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
        if (i + 1) % 15 == 0 or i == 0 or i == n - 1:
            print(f"  [{i+1:3d}/{n}]  {elapsed:.1f}s/clip  ETA={eta/60:.1f}min  hyp={hyp[:55]!r}", flush=True)
        hyps.append(hyp)
    return hyps


def audio_quality(path: str) -> dict:
    """Extract audio metadata using soxi/ffprobe."""
    p = Path(path)
    q: dict = {"path": path, "exists": p.exists(), "size_kb": round(p.stat().st_size / 1024, 1) if p.exists() else None}
    try:
        r = subprocess.run(["soxi", "-D", path], capture_output=True, text=True, timeout=5)
        q["duration_soxi"] = round(float(r.stdout.strip()), 3) if r.returncode == 0 else None
    except Exception:
        pass
    try:
        r = subprocess.run(
            ["ffprobe","-v","error","-show_entries","format=duration,bit_rate",
             "-show_entries","stream=sample_rate,channels","-of","json", path],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            d = json.loads(r.stdout)
            fmt = d.get("format", {})
            streams = d.get("streams", [{}])
            q["bit_rate"] = fmt.get("bit_rate")
            q["sample_rate"] = streams[0].get("sample_rate") if streams else None
            q["channels"] = streams[0].get("channels") if streams else None
    except Exception:
        pass
    return q


def main():
    sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
    os.environ.setdefault("TRANSFORMERS_CACHE",
        str(Path(__file__).parents[2] / "workspace/cache/transformers"))
    cache_dir = Path(os.environ["TRANSFORMERS_CACHE"])

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-b", type=Path, required=True)
    ap.add_argument("--label-b", default="step_3000")
    ap.add_argument("--cached-a", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("workspace/diagnostics/regression_analysis.json"))
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    # Load cached step_500
    cached = json.loads(args.cached_a.read_text())
    clips_a = cached["per_clip"]
    manifest = {Path(json.loads(l)["audio_path"]).name: json.loads(l)
                for l in args.manifest.read_text().splitlines() if l.strip()}
    audio_paths = [manifest[c["audio"]]["audio_path"] if c["audio"] in manifest else "" for c in clips_a]

    # ── D1: inference ──
    print(f"\n=== D1: {args.label_b} inference ({len(audio_paths)} clips) ===", flush=True)
    model_b, proc_b = load_model(args.ckpt_b, cache_dir, device)
    hyps_b = transcribe_gpu(model_b, proc_b, audio_paths, device)
    del model_b, proc_b
    if device == "cuda":
        torch.cuda.empty_cache()

    from transformers.models.whisper.english_normalizer import BasicTextNormalizer
    normalizer = BasicTextNormalizer()

    results = []
    for i, (ca, hb) in enumerate(zip(clips_a, hyps_b)):
        ref, ha, wa = ca["ref"], ca["hyp"], ca["wer"]
        wb = clip_wer(ref, hb) if ref.strip() else float("nan")
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
        results.append(dict(
            idx=i, audio=ca["audio"], duration=ca["duration"],
            dialect=ca.get("dialect", row.get("dialect","")),
            speaker_id=row.get("speaker_id",""), domain=row.get("domain",""),
            ref=ref, hyp_step_500=ha, hyp_step_3000=hb,
            wer_step_500=wa, wer_step_3000=wb_r, delta=delta, verdict=verdict,
            is_garbage_3000=is_garbage(hb, ref),
        ))

    # ── D1 summary ──
    counts = {k: sum(1 for r in results if r["verdict"]==k)
              for k in ["IMPROVED","SAME","REGRESSED","CATASTROPHIC","UNKNOWN"]}
    valid = [r for r in results if r["delta"] is not None]
    avg_a = sum(r["wer_step_500"] for r in valid) / max(len(valid), 1)
    avg_b = sum(r["wer_step_3000"] for r in valid) / max(len(valid), 1)

    print(f"\n{'='*65}", flush=True)
    print(f"D1 REGRESSION ANALYSIS: step_500 vs {args.label_b}  ({len(results)} clips)", flush=True)
    print(f"{'='*65}", flush=True)
    print(f"  Avg WER  500: {avg_a*100:.2f}%   3000: {avg_b*100:.2f}%   delta={( avg_b-avg_a)*100:+.2f}pp")
    for k, v in counts.items():
        print(f"  {k:12s}: {v:3d}  ({v/len(results)*100:.0f}%)")

    regressed = sorted([r for r in results if r["verdict"] in ("REGRESSED","CATASTROPHIC")],
                       key=lambda r: r["delta"] or 0, reverse=True)
    improved  = sorted([r for r in results if r["verdict"] == "IMPROVED"],
                       key=lambda r: r["delta"] or 0)

    print(f"\n--- Top 10 regressions ---")
    for r in regressed[:10]:
        print(f"\n  [{r['idx']:3d}] {r['audio']}  dur={r['duration']:.1f}s  "
              f"dialect={r['dialect']}  [{r['verdict']}]  delta={r['delta']:+.3f}")
        print(f"    REF:  {r['ref']}")
        print(f"    500:  {r['hyp_step_500']}")
        print(f"    3000: {r['hyp_step_3000']}")

    print(f"\n--- Top 5 improvements ---")
    for r in improved[:5]:
        print(f"\n  [{r['idx']:3d}] {r['audio']}  dur={r['duration']:.1f}s  dialect={r['dialect']}  delta={r['delta']:+.3f}")
        print(f"    REF:  {r['ref']}")
        print(f"    500:  {r['hyp_step_500']}")
        print(f"    3000: {r['hyp_step_3000']}")

    # Pattern analysis
    print(f"\n--- Regression patterns ---")
    if regressed:
        durs_r = [r["duration"] for r in regressed]
        durs_all = [r["duration"] for r in results]
        print(f"  Regressed mean dur: {sum(durs_r)/len(durs_r):.1f}s  vs all: {sum(durs_all)/len(durs_all):.1f}s")
        import collections
        dial_r = collections.Counter(r["dialect"] for r in regressed)
        dial_all = collections.Counter(r["dialect"] for r in results)
        print(f"  Regressed dialects: {dict(dial_r.most_common(8))}")
        print(f"  All dialects:       {dict(dial_all.most_common(8))}")
        ref_r = [len(r["ref"].split()) for r in regressed]
        ref_all = [len(r["ref"].split()) for r in results]
        print(f"  Regressed ref len:  mean={sum(ref_r)/len(ref_r):.1f}w  vs all: {sum(ref_all)/len(ref_all):.1f}w")
        garbage_r = sum(1 for r in regressed if r["is_garbage_3000"])
        print(f"  Garbage outputs in regressions: {garbage_r}/{len(regressed)}")
        speakers_r = collections.Counter(r["speaker_id"] for r in regressed if r["speaker_id"])
        if speakers_r:
            print(f"  Repeated speakers in regressions: {dict(speakers_r.most_common(5))}")

    # ── D3: audio quality ──
    worst_20 = regressed[:20]
    print(f"\n{'='*65}", flush=True)
    print(f"D3 AUDIO QUALITY — {len(worst_20)} worst-regressed clips", flush=True)
    print(f"{'='*65}", flush=True)
    for r in worst_20:
        path = manifest.get(r["audio"], {}).get("audio_path", "")
        if not path:
            print(f"  {r['audio']}: path not found in manifest")
            continue
        q = audio_quality(path)
        print(f"  {r['audio']}  dur={r['duration']:.1f}s  delta={r['delta']:+.3f}")
        print(f"    size={q.get('size_kb','?')}KB  rate={q.get('sample_rate','?')}Hz  "
              f"ch={q.get('channels','?')}  bitrate={q.get('bit_rate','?')}")

    # Size/quality stats for regressed vs all
    if worst_20:
        all_paths = [manifest.get(r["audio"], {}).get("audio_path","") for r in results]
        sizes_all = [Path(p).stat().st_size/1024 for p in all_paths if p and Path(p).exists()]
        sizes_r   = [Path(manifest.get(r["audio"],{}).get("audio_path","")).stat().st_size/1024
                     for r in worst_20 if manifest.get(r["audio"],{}).get("audio_path","")]
        if sizes_all and sizes_r:
            print(f"\n  File size (KB): regressed mean={sum(sizes_r)/len(sizes_r):.0f}  "
                  f"all mean={sum(sizes_all)/len(sizes_all):.0f}")

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "label_a": "step_500", "label_b": args.label_b,
        "counts": counts, "avg_wer_500": round(avg_a, 4), "avg_wer_3000": round(avg_b, 4),
        "per_clip": results,
    }
    args.out.write_text(json.dumps(output, indent=2))
    print(f"\nSaved → {args.out}", flush=True)


if __name__ == "__main__":
    main()
