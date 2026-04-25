"""WER, CER, and code-switching WER for Nigerian speech evaluation.

Three evaluation modes:

1. ``tone_sensitive`` (default) — diacritics preserved, exact NFC match.
   Use this as the primary metric; it measures whether the model produces
   linguistically correct output for speakers who read/write with diacritics.

2. ``tone_stripped`` — combining tone marks removed before comparison.
   Useful for robustness analysis when the evaluation reference transcripts
   are inconsistently diacritised (common in crowdsourced data).

3. CS-WER — a hypothesis word is a hit only when BOTH the word token AND the
   language tag match the reference.  Penalises models that transcribe the
   correct word but misidentify the language.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import jiwer
import jiwer.transforms as tr

from orinode.data.diacritics import normalize_text, strip_tone_marks

# jiwer 3.x uses ReduceToListOfListOfWords (renamed from SentencesToListOfWords)
_WORD_TRANSFORM = tr.Compose(
    [
        tr.ToLowerCase(),
        tr.RemoveMultipleSpaces(),
        tr.Strip(),
        tr.ReduceToListOfListOfWords(),
    ]
)

_CHAR_TRANSFORM = tr.Compose(
    [
        tr.ToLowerCase(),
        tr.RemoveMultipleSpaces(),
        tr.Strip(),
        tr.ReduceToListOfListOfChars(),
    ]
)


# ── standard WER ──────────────────────────────────────────────────────────────


@dataclass
class WERResult:
    wer: float
    cer: float
    insertions: int
    deletions: int
    substitutions: int
    hits: int
    total_ref_words: int
    total_hyp_words: int


def compute_wer(
    references: list[str],
    hypotheses: list[str],
    mode: str = "tone_sensitive",
) -> WERResult:
    """Compute WER and CER between reference and hypothesis transcripts.

    Args:
        references: Ground-truth transcripts.
        hypotheses: Model-output transcripts.
        mode: ``"tone_sensitive"`` (default) or ``"tone_stripped"``.

    Returns:
        ``WERResult`` with WER, CER and alignment counts.

    Raises:
        ValueError: If ``references`` and ``hypotheses`` have different lengths.
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"Length mismatch: {len(references)} references vs {len(hypotheses)} hypotheses"
        )
    if not references:
        return WERResult(
            wer=0.0,
            cer=0.0,
            insertions=0,
            deletions=0,
            substitutions=0,
            hits=0,
            total_ref_words=0,
            total_hyp_words=0,
        )

    if mode == "tone_stripped":
        refs = [strip_tone_marks(normalize_text(r)) for r in references]
        hyps = [strip_tone_marks(normalize_text(h)) for h in hypotheses]
    else:
        refs = [normalize_text(r) for r in references]
        hyps = [normalize_text(h) for h in hypotheses]

    wm = jiwer.process_words(
        refs,
        hyps,
        reference_transform=_WORD_TRANSFORM,
        hypothesis_transform=_WORD_TRANSFORM,
    )
    cm = jiwer.process_characters(
        refs,
        hyps,
        reference_transform=_CHAR_TRANSFORM,
        hypothesis_transform=_CHAR_TRANSFORM,
    )

    total_ref = wm.hits + wm.deletions + wm.substitutions
    total_hyp = wm.hits + wm.insertions + wm.substitutions

    return WERResult(
        wer=wm.wer,
        cer=cm.cer,
        insertions=wm.insertions,
        deletions=wm.deletions,
        substitutions=wm.substitutions,
        hits=wm.hits,
        total_ref_words=total_ref,
        total_hyp_words=total_hyp,
    )


def compute_wer_per_language(
    references: list[str],
    hypotheses: list[str],
    languages: list[str],
    mode: str = "tone_sensitive",
) -> dict[str, WERResult]:
    """Compute per-language WER from parallel reference/hypothesis/language lists.

    Args:
        references: Ground-truth transcripts.
        hypotheses: Model outputs.
        languages: Language tag per utterance.
        mode: Evaluation mode (see ``compute_wer``).

    Returns:
        Mapping from language code to ``WERResult``.
    """
    lang_refs: dict[str, list[str]] = defaultdict(list)
    lang_hyps: dict[str, list[str]] = defaultdict(list)
    for ref, hyp, lang in zip(references, hypotheses, languages, strict=False):
        lang_refs[lang].append(ref)
        lang_hyps[lang].append(hyp)

    return {lang: compute_wer(lang_refs[lang], lang_hyps[lang], mode=mode) for lang in lang_refs}


# ── code-switching WER ────────────────────────────────────────────────────────


@dataclass
class TaggedWord:
    """A word annotated with its language tag."""

    word: str
    language: str


@dataclass
class CSWERResult:
    """Code-switching WER result.

    ``cs_wer`` penalises both word-token errors and language-tag errors.
    ``token_wer`` is the ordinary WER ignoring tags (for comparison).
    """

    cs_wer: float
    token_wer: float
    tag_accuracy: float
    per_language_wer: dict[str, float] = field(default_factory=dict)
    total_ref_words: int = 0


def compute_cs_wer(
    ref_tagged: list[list[TaggedWord]],
    hyp_tagged: list[list[TaggedWord]],
) -> CSWERResult:
    """Compute code-switching WER.

    A hypothesis word is a hit only when both the word string (normalised,
    case-folded) and the language tag match the aligned reference word.
    Language-tag errors on otherwise-correct words are counted as additional
    errors in the CS-WER numerator.

    Args:
        ref_tagged: List of utterances; each utterance is a list of
            ``TaggedWord`` objects.
        hyp_tagged: Parallel hypothesis list.

    Returns:
        ``CSWERResult`` with overall CS-WER and per-language breakdown.

    Raises:
        ValueError: If ``ref_tagged`` and ``hyp_tagged`` have different lengths.
    """
    if len(ref_tagged) != len(hyp_tagged):
        raise ValueError("Reference and hypothesis lists must have the same length")

    if not ref_tagged:
        return CSWERResult(cs_wer=0.0, token_wer=0.0, tag_accuracy=1.0)

    ref_texts = [" ".join(w.word for w in utt) for utt in ref_tagged]
    hyp_texts = [" ".join(w.word for w in utt) for utt in hyp_tagged]
    token_result = compute_wer(ref_texts, hyp_texts, mode="tone_sensitive")

    lang_ref: dict[str, int] = defaultdict(int)
    lang_tag_err: dict[str, int] = defaultdict(int)

    for ref_utt, hyp_utt in zip(ref_tagged, hyp_tagged, strict=False):
        ref_str = " ".join(w.word.lower() for w in ref_utt)
        hyp_str = " ".join(w.word.lower() for w in hyp_utt)

        output = jiwer.process_words(ref_str, hyp_str)
        ref_idx = 0
        for chunk in output.alignments[0]:
            if chunk.type == "equal":
                n = chunk.ref_end_idx - chunk.ref_start_idx
                for i in range(n):
                    if ref_idx < len(ref_utt):
                        lang = ref_utt[ref_idx].language
                        lang_ref[lang] += 1
                        hyp_pos = chunk.hyp_start_idx + i
                        if hyp_pos < len(hyp_utt) and hyp_utt[hyp_pos].language != lang:
                            lang_tag_err[lang] += 1
                        ref_idx += 1
            elif chunk.type in ("delete", "substitute"):
                for _ in range(chunk.ref_end_idx - chunk.ref_start_idx):
                    if ref_idx < len(ref_utt):
                        lang = ref_utt[ref_idx].language
                        lang_ref[lang] += 1
                        lang_tag_err[lang] += 1
                        ref_idx += 1

    total_ref = token_result.total_ref_words
    token_errors = int(round(total_ref * token_result.wer))
    tag_errors = sum(lang_tag_err.values())
    cs_errors = token_errors + tag_errors
    cs_wer = min(cs_errors / total_ref, 1.0) if total_ref > 0 else 0.0

    total_tagged = sum(lang_ref.values())
    tag_accuracy = 1.0 - (tag_errors / total_tagged) if total_tagged > 0 else 1.0

    per_lang_wer = {
        lang: lang_tag_err[lang] / count for lang, count in lang_ref.items() if count > 0
    }

    return CSWERResult(
        cs_wer=cs_wer,
        token_wer=token_result.wer,
        tag_accuracy=tag_accuracy,
        per_language_wer=per_lang_wer,
        total_ref_words=total_ref,
    )
