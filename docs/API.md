# Public REST API — /api/v1

Base URL (local): `http://127.0.0.1:7860`

All endpoints accept and return JSON. Audio must be base64-encoded WAV or FLAC.

---

## POST /api/v1/transcribe

Transcribe audio.

**Request**
```json
{ "audio_b64": "<base64>", "language": "auto", "max_new_tokens": 256 }
```

`language` options: `auto`, `en`, `ha`, `yo`, `ig`, `pcm`

**Response**
```json
{
  "text": "I dey happy to meet you today",
  "language": "pcm",
  "languages_detected": ["pcm"],
  "word_tags": [],
  "latency_ms": 312.4,
  "model_version": "speech_llm/v1"
}
```

---

## POST /api/v1/transcribe/compare

Transcribe with multiple checkpoints and compare outputs.

**Request**
```json
{ "audio_b64": "<base64>", "language": "auto", "checkpoint_ids": [] }
```
Pass `checkpoint_ids: []` to auto-select the 3 most recent checkpoints.

---

## POST /api/v1/emotion

Classify emotion: `happy` | `angry` | `sad` | `neutral`.

**Request**
```json
{ "audio_b64": "<base64>" }
```

**Response**
```json
{
  "top_prediction": "happy",
  "confidences": { "happy": 0.72, "angry": 0.10, "sad": 0.08, "neutral": 0.10 },
  "segment_timeline": null,
  "model_version": "aux_emotion/v1",
  "disclaimer": "Preview model trained on English transfer data...",
  "latency_ms": 120.1
}
```

`segment_timeline` is populated for audio > 30 seconds (3-second windows).

Returns **503** if no emotion checkpoint is loaded. Run `make train-emotion`.

---

## POST /api/v1/gender

Classify speaker gender.

**Request**
```json
{ "audio_b64": "<base64>", "per_speaker": false }
```

**Response**
```json
{
  "prediction": "female",
  "confidence": 0.91,
  "model_version": "aux_gender/v1",
  "per_speaker": null,
  "latency_ms": 85.2
}
```

Returns **503** if no gender checkpoint is loaded. Run `make train-gender`.

---

## POST /api/v1/analyze

Run transcription + emotion + gender in a single request.

**Request**
```json
{
  "audio_b64": "<base64>",
  "language": "auto",
  "include_emotion": true,
  "include_gender": true
}
```

---

## GET /api/v1/checkpoints

List all checkpoint files.

---

## GET /api/v1/checkpoints/{id}

Get metadata for a single checkpoint.

---

## GET /api/v1/stats

Return aggregate workspace statistics.

```json
{
  "total_hours": 42.5,
  "languages": ["en", "ha", "yo", "ig", "pcm"],
  "checkpoints_count": 3,
  "runs_count": 5
}
```

---

## GET /api/v1/samples

List public sample audio clips.

---

## GET /api/v1/samples/{id}/audio

Download a sample clip as `audio/flac`.

---

## POST /api/v1/feedback

Submit a quality rating for a model output.

**Request**
```json
{ "task": "transcribe", "rating": 4, "comment": "Mostly correct", "model_version": "speech_llm/v1" }
```

`task` options: `transcribe`, `emotion`, `gender`, `analyze`  
`rating`: integer 1–5

**Response**: `{ "status": "recorded" }`

Feedback is appended to `workspace/logs/feedback/{task}.jsonl`.
