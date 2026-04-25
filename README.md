# Orinode-LM

Nigerian Speech-LLM — English / Hausa / Igbo / Yoruba / Nigerian Pidgin + intra-sentential code-switching.

Architecture: Whisper-large-v3 encoder (DoRA) → 2-layer MLP + Q-Former adapter (50→5 tokens/sec) → Gemma-2-9B-it decoder (LoRA r=64). Modelled on MERaLiON/SALMONN/Qwen-Audio.

Everything runs **locally**. No S3, no HuggingFace Hub push, no W&B cloud sync by default. The local web UI at `http://127.0.0.1:7860` lets you test every checkpoint as training progresses.

## Quickstart

```bash
# 1. Clone and enter the repo
git clone <this-repo> orinode-lm && cd orinode-lm

# 2. Copy env config (edit ORINODE_WORKSPACE if you want a different disk)
cp .env.example .env

# 3. Install Python deps + build frontend
make install          # uv sync + npm ci

# 4. Verify the foundation (no GPU needed)
make smoke-test       # synthetic forward+backward, exits clean

# 5. Download corpora (2–3 TB total; skip individually with env flags)
make download-data

# 6. Preprocess to 16 kHz FLAC + JSONL manifests
make build-manifests

# 7. Start the local UI (any terminal; keep it running)
make ui               # → http://127.0.0.1:7860

# 8. Train (separate terminal)
make train-stage1
# UI updates live as training writes events to workspace/logs/training/
```

## Four-stage curriculum

| Stage | What trains | Data | Approx runtime (8×H100) |
|-------|------------|------|--------------------------|
| 1 | Whisper encoder DoRA | AfriSpeech (EN only) | 12 h |
| 2 | Encoder + adapter (joint ASR) | All 5 langs + CS | 4 d |
| 3 | Adapter + decoder LoRA | All 5 langs + CS | 5 d |
| 4 | Instruction fine-tune | Call-center, translation, Q&A | 2 d |

## Disk budget

| Item | Size |
|------|------|
| NaijaVoices (raw) | ~2 TB |
| AfriSpeech + BibleTTS + Common Voice | ~1 TB |
| Gemma-2-9B-it weights | ~18 GB |
| Whisper-large-v3 weights | ~3 GB |
| Checkpoints (keep-best-3 per run) | ~100 GB/run |
| **Minimum recommended** | **5 TB NVMe** |

## Local UI

Six pages served at `http://127.0.0.1:7860`:

- **Dashboard** — active runs, latest eval table, storage usage
- **Runs** — sortable table of all training runs
- **RunDetail** — live loss/WER charts, checkpoint list, log tail
- **Playground** — upload/record audio → pick checkpoint → transcribe with per-word language tags
  *(Sample FLAC files are not bundled; upload your own audio or record in-browser)*
- **Compare** — same audio across N checkpoints with word-level diff
- **Data** — corpus stats, manifest explorer, random sample viewer

> **Security note:** The UI binds `127.0.0.1` with no authentication. Do not expose port 7860 to the public internet. If running on a remote machine, SSH-forward the port locally.

## Key make targets

```
make install          # first-run setup
make smoke-test       # CPU forward+backward test
make ui               # build frontend + serve on :7860
make ui-dev           # hot-reload dev mode (vite :5173 + uvicorn :7860)
make train-stage1     # kick off stage 1
make eval RUN=<id>    # run eval for a checkpoint run
make test             # fast tests (< 30 s, no GPU)
make lint             # ruff + format check
make typecheck        # mypy strict
```

## Project layout

```
configs/          YAML configs (_base/ inherited by stage configs)
src/orinode/      Python package
  data/           manifests, preprocessing, diacritics, augmentation, mixing
  models/         WhisperEncoder, AudioLLMAdapter, SpeechLLM, LoRA utils
  training/       4-stage trainers, callbacks, EventBus integration
  eval/           WER, CS-WER, LID accuracy, reports
  inference/      pipeline.py, checkpoint registry
  ui/             FastAPI backend, WebSocket, progress_store
  utils/          logging, config loader, FSDP helpers, EventBus
ui/frontend/      React + Vite + Tailwind source
scripts/          data download, training launchers, eval runners, UI build
tests/            pytest suite (fast + slow tiers)
workspace/        runtime data — gitignored; created by make install
```

See `docs/` for detailed documentation on each component (including `docs/training_stages.md` for the full four-stage training curriculum).
