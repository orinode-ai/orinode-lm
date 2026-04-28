"""Diagnostic 2: Train vs Dev distribution analysis."""
import json, random, re, collections

STOPWORDS = {"the","a","an","and","or","of","to","in","is","was","were","be","been","being",
             "have","has","had","do","does","did","will","would","could","should","may","might",
             "for","with","on","at","by","from","as","it","its","he","she","they","we","i",
             "this","that","these","those","his","her","their","our","but","not","are","also"}

def stats(durs):
    s = sorted(durs)
    n = len(s)
    return dict(n=n, mean=round(sum(s)/n,1), median=round(s[n//2],1),
                p25=round(s[n//4],1), p75=round(s[3*n//4],1),
                p95=round(s[int(n*.95)],1), min=round(s[0],1), max=round(s[-1],1))

def text_stats(texts):
    lens = [len(t.split()) for t in texts]
    s = sorted(lens)
    n = len(s)
    words = collections.Counter()
    for t in texts:
        for w in re.findall(r"[a-z]+", t.lower()):
            if w not in STOPWORDS and len(w) > 2:
                words[w] += 1
    return dict(
        word_len=dict(mean=round(sum(lens)/n,1), median=s[n//2], p75=s[3*n//4], p95=s[int(n*.95)]),
        top_30=[w for w,_ in words.most_common(30)],
    )

train_all = [json.loads(l) for l in open("/home/user/orinode-lm/workspace/data/manifests/afrispeech_200_train.jsonl") if l.strip()]
dev_all   = [json.loads(l) for l in open("/home/user/orinode-lm/workspace/data/manifests/afrispeech_200_dev.jsonl") if l.strip()]

rng = random.Random(42)
train_s = rng.sample(train_all, 100)
dev_ge3 = [c for c in dev_all if c.get("duration",0) >= 3.0]

def report(label, clips):
    d = stats([c["duration"] for c in clips])
    t = text_stats([c["text"] for c in clips])
    dialects = collections.Counter(c.get("dialect","?") for c in clips)
    print(f"\n=== {label} (n={len(clips)}) ===")
    print(f"  Duration:  mean={d['mean']}s  median={d['median']}s  p25={d['p25']}s  p75={d['p75']}s  p95={d['p95']}s  max={d['max']}s")
    print(f"  Ref words: mean={t['word_len']['mean']}  median={t['word_len']['median']}  p75={t['word_len']['p75']}  p95={t['word_len']['p95']}")
    print(f"  Dialects:  {dict(dialects.most_common(8))}")
    print(f"  Top words: {t['top_30']}")

report("TRAIN sample (100 clips, seed=42)", train_s)
report("TRAIN full", train_all)
report("DEV >=3s", dev_ge3)
report("DEV full", dev_all)

# Mismatch summary
print("\n=== MISMATCH SUMMARY ===")
td = stats([c["duration"] for c in train_all])
dd = stats([c["duration"] for c in dev_all])
print(f"  Train mean dur: {td['mean']}s   Dev mean dur: {dd['mean']}s")
td_ge3 = stats([c["duration"] for c in train_all if c.get("duration",0) >= 3.0])
print(f"  Train clips <3s: {sum(1 for c in train_all if c.get('duration',0)<3.0)} / {len(train_all)} ({100*sum(1 for c in train_all if c.get('duration',0)<3.0)/len(train_all):.1f}%)")
print(f"  Dev   clips <3s: {sum(1 for c in dev_all if c.get('duration',0)<3.0)} / {len(dev_all)} ({100*sum(1 for c in dev_all if c.get('duration',0)<3.0)/len(dev_all):.1f}%)")
train_d = collections.Counter(c.get("dialect","?") for c in train_all)
dev_d   = collections.Counter(c.get("dialect","?") for c in dev_all)
print(f"  Train dialects: {dict(train_d.most_common(8))}")
print(f"  Dev   dialects: {dict(dev_d.most_common(8))}")
