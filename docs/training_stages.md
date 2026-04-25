# The 4-stage training curriculum, explained

This document walks through what happens at each stage of training Orinode-LM, why the stages exist in that order, what you're actually changing in the model at each step, and what "done" looks like before you move on. Written for someone who understands the product but isn't a Speech-LLM researcher.

---

## Why stages at all?

You could in theory throw all the data at one model and train it end-to-end. In practice this fails in predictable ways: the model forgets English while learning Hausa, the decoder never learns to listen to the encoder, loss spikes to NaN, training runs cost 5× more than necessary. The 4-stage curriculum is how MERaLiON, SALMONN, Qwen-Audio, and every other working Speech-LLM solves this. Each stage does one specific thing, you verify it worked, then you move on. Failures are localized.

Think of it like teaching a kid a new language. You don't simultaneously teach them to hear the language, understand it, follow instructions in it, and have conversations. You do it in that order, and each skill builds on the last.

---

## Stage 1 — Teach the ears to hear Nigerian accents

**What changes:** Only the Whisper encoder. Specifically, DoRA adapters injected into the encoder's attention layers. The rest of the model (decoder, adapter, LLM) doesn't exist yet.

**What training data:** AfriSpeech-200. English only. 200 hours of Nigerian-accented English speech with transcripts.

**What it's learning:** Whisper was trained mostly on American and British English. When a Nigerian speaker says "wahala" (trouble) or pronounces "water" with a particular accent, vanilla Whisper sometimes mishears. Stage 1 pushes the encoder to correctly represent Nigerian phonetics while not breaking its understanding of other English accents.

**Why English-only at this stage:** Two reasons. First, AfriSpeech has Nigerian-accented English from all three major language groups (Hausa, Yoruba, Igbo speakers speaking English), which gives the encoder exposure to the phonetic influences from each without needing labeled Hausa/Yoruba/Igbo yet. Second, English is what Whisper knows best — if we mess up this stage, it's obvious because English WER goes up, which is easy to measure. Low-resource languages would hide the damage.

**Why DoRA and not full fine-tuning:** The AfricanNLP 2026 paper tested this exact setup. Full fine-tuning is slightly better on the target task but **catastrophically forgets the other 97 languages Whisper knew** — which matters because Stage 2 needs those representations to transfer to Hausa/Yoruba/Igbo. DoRA updates only ~1% of the parameters, preserves Whisper's multilingual priors, and trains in a third the time with half the VRAM.

**What "done" looks like:**
- English WER on AfriSpeech-200 held-out test set drops by 15–25% relative (e.g., from 14% to 10–12%)
- English WER on standard LibriSpeech doesn't go up (catastrophic forgetting check)
- Loss curve is smooth and plateaued — no spikes, no NaN
- Checkpoint size ~200 MB (just the DoRA weights, not the full encoder)

**Hardware:** 1× RTX A6000, 3–5 days, ~$30–50 compute.

**What can go wrong:**
- Learning rate too high → loss spikes, model diverges. Drop to 5e-5, restart.
- Over-training → English WER gets worse on LibriSpeech. Stop earlier, use the best-eval checkpoint, not the last.
- Batch size too small → noisy gradients, slow convergence. Use gradient accumulation to hit effective batch size 64+.

**What comes out:** A Whisper-large-v3 encoder that's slightly better at Nigerian English and unchanged on everything else. It's still just an encoder — it can't transcribe by itself at this stage (the decoder is still vanilla Whisper's decoder, untouched).

---

## Stage 2 — Teach it the Nigerian languages

**What changes:** The Whisper encoder (with Stage 1 DoRA adapters as starting point) continues training, now on all five languages. The tokenizer gets extended to cover Hausa/Yoruba/Igbo characters that Whisper's original BPE over-segments. A CTC auxiliary loss is added on the encoder for alignment stability.

**What training data:** Everything mixed together with temperature-weighted sampling:
- NaijaVoices (Hausa, Yoruba, Igbo — 1,800h total)
- AfriSpeech-200 (Nigerian English — 200h)
- BibleTTS (Hausa + Yoruba single-speaker — 166h)
- Nigerian Common Voice (read speech, all four languages)
- Crowdsourced Pidgin data (target 100h+)
- Crowdsourced code-switched conversational data (target 100h+)

Total corpus size: 2,000–2,500 hours across five languages + code-switching.

**What temperature-weighted sampling means in practice:** If you just mixed data uniformly, English would dominate training because there's more of it. The model would end up great at English and mediocre at Igbo. Temperature sampling with α=0.3 re-weights the sampling probability so low-resource languages appear in training batches more often than their raw data share.

Concretely: English is 10% of batches even though it's 30% of the corpus, Igbo is 15% of batches even though it's 8% of the corpus. The model sees each language often enough to learn it well.

**Why joint training, not per-language:** This is the single biggest architectural decision. If you train one model per language and stitch them together, code-switching never works — the model literally has no concept that Hausa and English can appear in the same utterance. Joint training means one model sees "za a isa gobe confirmed" during training and learns to handle it. MERaLiON works because it's joint-trained on Singlish. Same principle here.

**Why add a CTC auxiliary loss:** Whisper's default objective is cross-entropy on next-token prediction, which is great for generation but doesn't give the encoder per-frame alignment signal. Adding CTC (connectionist temporal classification) on the encoder's top layer as a secondary loss forces the encoder to produce timestamp-aligned representations. This matters later for streaming inference (you can tell when a word ended) and improves alignment stability on 8kHz telephony. The 2024/2025 literature (Zhao & Shi, arXiv:2412.16507) shows consistent 4–8% WER improvements from this.

**What "done" looks like:**
- Monolingual WER per language meets target: HA <10%, YO <10%, IG <12%, PCM <14%, EN <8%
- **Code-switched WER** on held-out CS data <25% (this is the hard metric — any model that handles CS at all is already ahead)
- Language-ID accuracy at segment level >90%
- Per-language WER curves all trending down together — no language getting sacrificed for others

**Hardware:** 2× RTX A6000 with FSDP, 7–10 days, ~$150–250 compute. Single A6000 works but takes 2–3 weeks.

**What can go wrong:**
- One language's WER drops while another's rises → mixing weights wrong. Adjust α higher (0.2 → more aggressive low-resource weighting) or add more data for the struggling language.
- Overall WER improves but CS-WER stagnates → not enough CS data. Crowdsource more intra-sentential examples.
- Igbo WER refuses to go below 20% → Whisper never pretrained on Igbo. Consider swapping to **Meta Omnilingual ASR** as the encoder base for Igbo specifically, or use MMS-1B-all adapter merging.

**What comes out:** A multilingual Whisper encoder that produces strong acoustic representations for Nigerian speech across five languages and code-switching. It can transcribe via Whisper's decoder with reasonable quality, but it doesn't yet "understand" — it's purely an acoustic model. Asking it "what did the customer want?" gets you nothing.

---

## Stage 3 — Teach it to understand

**What changes:** This is where the architecture grows. The Stage 2 encoder is **frozen** (no more training). On top of it, two new components train:

1. **The adapter** — a 2-layer MLP plus a Q-Former-style resampler that converts encoder outputs (50 tokens/sec of 1280-dim vectors) into the LLM's embedding space (5 tokens/sec of ~3000-dim vectors). The 10× downsampling is critical — without it, audio would dominate the LLM's context window.
2. **The LLM decoder** — Gemma-2-9B-it (or Gemma-2-2B for bootstrap) with LoRA adapters. The LLM itself stays mostly frozen; LoRA adds ~50M trainable parameters that let it learn to attend to the audio tokens prepended to its text embeddings.

**What training data:** Paired `<audio, text>` examples. The audio can be a Nigerian speech clip, the text is the transcript. Crucially, we also mix in:
- Audio + multilingual translations (EN audio → HA text, etc.)
- Audio + summaries (long audio → short text summary)
- Audio + Q&A pairs (audio + "what is the speaker asking?" → answer text)
- Monolingual text-only examples (to prevent the LLM from forgetting how to read text)

The ratio matters. Too much audio-only training and the LLM forgets how to do text tasks. Roughly 70% audio-text pairs, 30% text-only.

**What it's learning:** The model is learning to connect audio features to language. Specifically, the LLM is learning that the 5-token-per-second audio representation coming out of the adapter is meaningful and attendable. Before Stage 3, the LLM sees audio tokens as noise. After Stage 3, it treats them as a modality it can reason about.

**Why freeze the encoder:** Two reasons. First, training the encoder and the adapter and the LLM decoder simultaneously is a recipe for instability — gradients flow badly across 10B parameters of mismatched components. Second, the encoder is already good from Stage 2. It doesn't need more training; it needs to be connected to something that can use its output.

**Why LoRA on the LLM, not full fine-tuning:** Gemma-2-9B has 9B parameters. Full fine-tuning would require 200+ GB of VRAM for gradients and optimizer states alone, needing 4-8× H100s. LoRA r=64 adds 50M trainable parameters (0.5% of the model), trains on 2× A6000 for the 2B variant or 4× H100 for the 9B variant, and preserves all of Gemma's pretrained knowledge (reasoning, instruction-following, multilingualism).

**The Gemma-2B vs Gemma-9B decision point:** This is where you decide the scope of your v0.1. Gemma-2B gives you something demo-able for ~$500 in compute; Gemma-9B needs H100s and ~$10k-20k. Start with 2B to validate the recipe, scale to 9B when funded. Every major Speech-LLM paper did the same — SALMONN was published at 7B, MERaLiON initially shipped at 3B and then 10B. Small-first is normal.

**What "done" looks like:**
- WER on Nigerian ASR roughly matches Stage 2 (slightly better or slightly worse; LoRA LLM decoder has different strengths than Whisper decoder)
- The model can **follow basic audio-based instructions**: "Transcribe this audio in Yoruba" works. "Summarize this call" produces a coherent summary.
- Eval loss on held-out audio-text pairs plateaus
- Text-only performance of the LLM doesn't degrade (measure on a simple text eval set as a sanity check)

**Hardware:**
- With Gemma-2-2B: 2× RTX A6000, 7–14 days, ~$500–1,500 compute
- With Gemma-2-9B: 4× H100 80GB (different provider), 10–20 days, ~$5k–15k compute

**What can go wrong:**
- LLM ignores the audio tokens → adapter outputs are too small in norm. Add a projection layer normalization, or use a higher LR on the adapter than the LLM LoRA.
- LLM hallucinates transcriptions that don't match audio → not enough grounded audio-text data relative to text-only. Shift the mix toward audio-text.
- Loss plateaus high → the encoder is bottlenecking. Revisit Stage 2; consider unfreezing the last 2 encoder layers with very low LR.
- Out-of-memory during training → enable gradient checkpointing, drop effective batch size, use 8-bit Adam optimizer, or switch to the 2B decoder.

**What comes out:** A model that can transcribe Nigerian speech *and* answer questions about what was said. You can upload a customer service call in Yoruba-English code-switching and ask "what product is the customer asking about?" and get a reasonable answer. This is the first version that's recognizably a Speech-LLM, not just an ASR system.

---

## Stage 4 — Teach it to follow instructions well

**What changes:** The Stage 3 model keeps its architecture (frozen encoder, trained adapter, LoRA'd LLM). LoRA on the LLM continues training, now on instruction-formatted data.

**What training data:** Curated instruction-following examples in all five languages, covering the tasks Orinode actually cares about:

- **Call-center scenarios** — "The customer called to report a lost card. Summarize their issue and next steps." Multi-turn dialogue audio + expected structured response.
- **Translation** — "Translate this Hausa audio to English."
- **Question answering** — "The speaker mentioned three product names. List them."
- **Intent classification** — "Is the caller angry, calm, or frustrated?" (note: this overlaps with emotion classifier; at the LLM level it's a zero-shot capability)
- **Entity extraction** — "Extract all phone numbers, Naira amounts, and dates mentioned in this audio."
- **Routing** — "Based on this call opening, which department should handle it: billing, technical support, or sales?"
- **Tone/register transformation** — "Rewrite this customer complaint as a formal incident report."

Target: **50k–100k instruction examples** across the five languages. These don't all need real audio — you can synthesize a lot of them by pairing existing audio with Claude-generated instructions + responses. Then human-validate a sample of 5k for quality.

**What it's learning:** How to be useful. Stage 3 made the model capable of connecting audio to language; Stage 4 makes it follow instructions about audio. The difference is whether the model, given audio + "summarize this call", produces a coherent summary or just transcribes verbatim.

**Why this stage is separate from Stage 3:** You could in theory mix instruction data into Stage 3. In practice, Stage 3 is about establishing the basic audio-language connection, and mixing instruction data early makes it harder to tell if the model is understanding audio or just pattern-matching on the instruction format. Clean separation: Stage 3 teaches listening, Stage 4 teaches helpfulness.

**Why the instruction data matters enormously:** The quality of your Stage 4 data directly determines product quality. If your instruction data is 50k high-quality call-center scenarios in Nigerian languages, your product handles Nigerian call-center scenarios well. If your instruction data is generic English NLP tasks translated poorly, you ship a generic model. Spend real effort on instruction data curation — this is where you encode your product differentiation.

**What "done" looks like:**
- Instruction-following accuracy (judged by Claude or human raters on 5 axes: factuality, language fidelity, instruction-following, fluency, cultural appropriateness) >75% acceptable
- Model responds in the requested language (e.g., when asked to respond in Hausa, it does, not in English)
- ASR WER from Stage 3 doesn't regress (measure as sanity check)
- Response format matches the task (summaries are short, transcriptions are verbatim, classifications are single labels)

**Hardware:** Same as Stage 3. Stage 4 is usually shorter (3–7 days) because it's fine-tuning fine-tuning.

**What can go wrong:**
- Model over-fits to instruction templates → always responds "The customer is asking about X" regardless of actual audio. Diversify instruction phrasing in training data.
- Model responds in English when asked to respond in Hausa → insufficient Hausa instruction data. Oversample Hausa instructions.
- Quality degrades vs Stage 3 → instruction data is low quality. Stop, audit data, fix before continuing.
- Model hallucinates plausible-sounding details that aren't in the audio → this is an inherent LLM failure mode. Add contrastive examples where the correct answer is "the speaker didn't say" or "not mentioned in audio".

**What comes out:** Aria. Or at least, Aria v0.1. A model you can plug into Orinode's production pipeline and have it handle Nigerian customer calls end-to-end — transcribe, understand intent, generate appropriate responses, hand off to TTS.

---

## The auxiliary models, for completeness

These train **in parallel** with the main Speech-LLM stages. They don't depend on Stages 1–4 and can start any time you have data.

### Gender classifier

**Architecture:** wav2vec2-base (frozen) → mean-pool over time dimension → Linear(768, 2) → softmax.

**Data:** NaijaVoices speakers filtered to those with gender metadata. Target 10 hours balanced across male and female, pooled across languages. Hold out ~200 speakers for eval (never see them in training).

**Time:** 1–2 days on 1× RTX A6000. About $15–25.

**What done looks like:** >95% accuracy on clean speech, >90% on 8kHz telephony simulation. These are reliable numbers because the underlying task (acoustic correlates of gender) is well-understood.

### Emotion classifier

**Architecture:** wav2vec2-large-xlsr-53 (frozen) → mean-pool → Linear(1024, 4) → softmax for happy/angry/sad/neutral.

**Data:** This is the honest problem. Three strategies documented in `prepare_emotion_labels.py`:

1. **Transfer-learn from IEMOCAP + RAVDESS** (English emotion corpora) as a warm start. This works but the resulting model's Nigerian language performance is unknown until evaluated — the acoustic correlates of "angry" may differ across cultures.
2. **Nollywood scene-labeled audio** — manually curate 20–50 hours of Nollywood movie audio with scene-level emotion labels. Labor-intensive.
3. **Crowdsource emotion labels on NaijaVoices subset** via Orinode's platform. Recommended path for v1.0.

For v0.1, ship strategy 1 only, label it **Preview**, show a methodology disclaimer in the UI, add a feedback widget. Strategies 2 and 3 are roadmap items, not launch items.

**Time:** 2–3 days on 1× RTX A6000. About $25–40.

**What done looks like:** 60–70% accuracy on English (IEMOCAP held-out). Nigerian language accuracy is genuinely unknown until you have labeled eval data. Ship with humility.

---

## The honest timeline and budget

| Stage | Gemma-2B path | Gemma-9B path |
|---|---|---|
| Stage 1 (encoder English adapt) | 3–5 days, $30–50, 1× A6000 | Same |
| Stage 2 (joint multilingual) | 7–10 days, $150–250, 2× A6000 | Same |
| Stage 3 (Speech-LLM) | 7–14 days, $500–1,500, 2× A6000 | 10–20 days, $5k–15k, 4× H100 |
| Stage 4 (instruction tune) | 3–7 days, $200–500, 2× A6000 | 5–10 days, $2k–5k, 4× H100 |
| Aux gender | 1–2 days, $15–25, 1× A6000 | Same |
| Aux emotion | 2–3 days, $25–40, 1× A6000 | Same |
| **Total wall clock** | **~5–7 weeks** | **~8–12 weeks** |
| **Total compute** | **~$900–2,400** | **~$10k–25k** |

This is wall clock for sequential stages on rented GPUs. Add 30–50% buffer for failed runs, hyperparameter tuning, data pipeline debugging — every production Speech-LLM project has a few restarts.

The Gemma-2B path gets you a working demo. The Gemma-9B path gets you MERaLiON-class quality. Go 2B first, validate the recipe, then raise money to go 9B.

---

## A note on when things are actually "done"

The WER and accuracy numbers above are realistic targets, not floors. In practice, the question at the end of each stage is: **would I let this model touch a real customer call?**

- Stage 1 done: the encoder is better than vanilla Whisper on Nigerian English. Yes, ship it as an STT upgrade.
- Stage 2 done: you have working multilingual Nigerian ASR including code-switching. Yes, ship it as the new Aria STT — it's already better than your current English-fallback.
- Stage 3 done: you have a Speech-LLM that can transcribe and understand. Maybe — it depends on instruction-following quality, which Stage 4 fixes.
- Stage 4 done: you have a model that can replace the current Whisper + Claude pipeline for Nigerian calls. This is when you flip the switch.

At each stage, ship what's ready. Don't wait for Stage 4 to replace the current pipeline — Stage 2 alone is a significant upgrade and unlocks product differentiation (Hausa/Yoruba/Igbo STT that actually works). The staged model ships in stages. That's the point.
