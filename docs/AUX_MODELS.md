# Auxiliary Models

Orinode-LM ships two auxiliary classifiers that run alongside the main
speech-LLM: a **gender classifier** and an **emotion classifier**.
Both are lightweight wav2vec2-based heads and load lazily on first API call.

---

## Gender Classifier

**Architecture:** `facebook/wav2vec2-base` (frozen) → mean-pool → `Linear(768, 2)` → log-softmax  
**Labels:** `male`, `female`  
**Config:** `configs/training/aux_gender_classifier.yaml`  
**Training script:** `python scripts/training/train_gender.py`  
**Makefile target:** `make train-gender`

### Data

Trained on VCTK and Common Voice speaker-gender annotations.
Manifest: `workspace/data/manifests/gender_train.jsonl`

### Per-speaker mode

When `per_speaker=true` is passed to `POST /api/v1/gender`, the pipeline
attempts to use **pyannote** (`pyannote/speaker-diarization-3.1`) for
speaker diarization before classifying each turn. Falls back to single-speaker
if pyannote is not installed.

---

## Emotion Classifier

**Architecture:** `facebook/wav2vec2-large-xlsr-53` (frozen) → mean-pool → `Linear(1024, 4)` → log-softmax  
**Labels:** `happy`, `angry`, `sad`, `neutral`  
**Config:** `configs/training/aux_emotion_classifier.yaml`  
**Training script:** `python scripts/training/train_emotion.py`  
**Makefile target:** `make train-emotion`

> **Preview badge**: The UI always shows a "Preview" badge until Nigerian emotional
> speech data replaces the English transfer data. See the disclaimer field in API
> responses.

### Data Strategies

#### Strategy 1 — English transfer (default, auto-executed)

Transfer-learn from IEMOCAP + RAVDESS. Run:

```bash
python scripts/data/prepare_emotion_labels.py \
  --iemocap-root /path/to/iemocap \
  --ravdess-root /path/to/ravdess
```

- **IEMOCAP** (~12 h): request access at https://sail.usc.edu/iemocap/
- **RAVDESS** (~24 actors): download from https://zenodo.org/record/1188976

#### Strategy 2 — Nollywood scene labels (pending)

Obtain licensed audio from Nollywood distributors.
Use scene-level metadata (fight → angry, reunion → happy) as weak labels.
Validate with ≥3 human annotations per clip.
Status: **pending rights negotiation**.

#### Strategy 3 — Crowdsourced Nigerian labels (future)

5,000+ clips annotated by Nigerian speakers via the Orinode platform,
stratified by language (EN / HA / YO / IG / PCM).
Status: **annotation UI not yet built**.

---

## Inference Pipelines

```python
from orinode.inference.emotion_pipeline import get_emotion_pipeline
result = get_emotion_pipeline().predict(audio_bytes)
# result: {top_prediction, confidences, segment_timeline, model_version, disclaimer}

from orinode.inference.gender_pipeline import get_gender_pipeline
result = get_gender_pipeline().predict(audio_bytes, per_speaker=False)
# result: {prediction, confidence, model_version, per_speaker}
```

Both pipelines are module-level singletons, loaded lazily on first call.
Checkpoints are read from `workspace/models/checkpoints/aux_emotion/` and
`workspace/models/checkpoints/aux_gender/` respectively (newest `.pt` file).
