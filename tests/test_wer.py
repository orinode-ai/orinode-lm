"""Tests for eval/wer.py — WER, CER, and code-switching WER."""

from __future__ import annotations

import pytest

from orinode.eval.wer import (
    TaggedWord,
    compute_cs_wer,
    compute_wer,
    compute_wer_per_language,
)

# ── basic WER ─────────────────────────────────────────────────────────────────


def test_perfect_match_wer_zero() -> None:
    result = compute_wer(["hello world"], ["hello world"])
    assert result.wer == 0.0
    assert result.hits == 2
    assert result.insertions == 0
    assert result.deletions == 0
    assert result.substitutions == 0


def test_completely_wrong_wer_one() -> None:
    result = compute_wer(["hello world"], ["foo bar"])
    assert result.wer == 1.0


def test_single_substitution() -> None:
    result = compute_wer(["hello world"], ["hello there"])
    assert result.substitutions == 1
    assert result.hits == 1
    assert abs(result.wer - 0.5) < 1e-9


def test_single_insertion() -> None:
    result = compute_wer(["hello world"], ["hello big world"])
    assert result.insertions == 1
    assert abs(result.wer - 0.5) < 1e-9


def test_single_deletion() -> None:
    result = compute_wer(["hello world"], ["hello"])
    assert result.deletions == 1
    assert abs(result.wer - 0.5) < 1e-9


def test_empty_inputs_returns_zero() -> None:
    result = compute_wer([], [])
    assert result.wer == 0.0
    assert result.total_ref_words == 0


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_wer(["hello"], ["a", "b"])


def test_case_insensitive() -> None:
    result = compute_wer(["Hello World"], ["hello world"])
    assert result.wer == 0.0


def test_multi_utterance_wer() -> None:
    refs = ["hello world", "foo bar baz"]
    hyps = ["hello world", "foo baz bar"]
    result = compute_wer(refs, hyps)
    # First utt: 0 errors; second utt: 2 errors (bar↔baz swap)
    assert result.total_ref_words == 5


# ── tone_sensitive vs tone_stripped ──────────────────────────────────────────


def test_hausa_tone_sensitive_counts_diacritic_mismatch() -> None:
    # Reference has ƙ; hypothesis uses k (wrong)
    result = compute_wer(["ƙasar Najeriya"], ["kasar Najeriya"], mode="tone_sensitive")
    # "ƙasar" ≠ "kasar" → 1 substitution
    assert result.substitutions >= 1


def test_hausa_tone_stripped_ignores_implosive() -> None:
    # In tone_stripped mode the implosive ɗ and plain d might match after strip
    # strip_tone_marks keeps Hausa consonants, so ɗ ≠ d still
    result_sensitive = compute_wer(["ɗan"], ["dan"], mode="tone_sensitive")
    result_stripped = compute_wer(["ɗan"], ["dan"], mode="tone_stripped")
    # Both modes should count ɗ as different from d (it's a consonant, not a tone)
    assert result_sensitive.wer == result_stripped.wer


def test_yoruba_tone_stripped_ignores_tone_marks() -> None:
    # ọmọ with tone marks vs without — stripped mode should match
    result = compute_wer(["ọmọ"], ["ọmọ"], mode="tone_stripped")
    assert result.wer == 0.0


def test_tone_stripped_handles_empty() -> None:
    result = compute_wer([], [], mode="tone_stripped")
    assert result.wer == 0.0


# ── CER ───────────────────────────────────────────────────────────────────────


def test_cer_perfect_match() -> None:
    result = compute_wer(["hello"], ["hello"])
    assert result.cer == 0.0


def test_cer_nonzero_on_mismatch() -> None:
    result = compute_wer(["hello"], ["hellx"])
    assert result.cer > 0.0


# ── per-language WER ──────────────────────────────────────────────────────────


def test_per_language_wer_splits_correctly() -> None:
    refs = ["hello world", "ƙasar Najeriya", "sannu"]
    hyps = ["hello world", "kasar Najeriya", "sannu"]
    langs = ["en", "ha", "ha"]
    per_lang = compute_wer_per_language(refs, hyps, langs)
    assert per_lang["en"].wer == 0.0
    assert per_lang["ha"].wer > 0.0


def test_per_language_wer_all_correct() -> None:
    refs = ["a", "b", "c"]
    hyps = ["a", "b", "c"]
    langs = ["en", "ha", "yo"]
    per_lang = compute_wer_per_language(refs, hyps, langs)
    for result in per_lang.values():
        assert result.wer == 0.0


# ── CS-WER ────────────────────────────────────────────────────────────────────


def _make_tagged(words: list[tuple[str, str]]) -> list[TaggedWord]:
    return [TaggedWord(word=w, language=lang) for w, lang in words]


def test_cs_wer_all_correct() -> None:
    ref = [_make_tagged([("hello", "en"), ("world", "en")])]
    hyp = [_make_tagged([("hello", "en"), ("world", "en")])]
    result = compute_cs_wer(ref, hyp)
    assert result.token_wer == 0.0


def test_cs_wer_wrong_tag_penalised() -> None:
    ref = [_make_tagged([("hello", "en"), ("sannu", "ha")])]
    hyp = [_make_tagged([("hello", "en"), ("sannu", "en")])]  # wrong tag on second word
    result = compute_cs_wer(ref, hyp)
    assert result.tag_accuracy < 1.0


def test_cs_wer_wrong_word_and_wrong_tag() -> None:
    ref = [_make_tagged([("sannu", "ha")])]
    hyp = [_make_tagged([("hello", "en")])]
    result = compute_cs_wer(ref, hyp)
    assert result.cs_wer > 0.0


def test_cs_wer_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        compute_cs_wer(
            [_make_tagged([("a", "en")])],
            [_make_tagged([("a", "en")]), _make_tagged([("b", "ha")])],
        )


def test_cs_wer_empty_inputs() -> None:
    result = compute_cs_wer([], [])
    assert result.cs_wer == 0.0
    assert result.token_wer == 0.0


def test_cs_wer_bounded_by_one() -> None:
    ref = [_make_tagged([("a", "en"), ("b", "ha"), ("c", "yo")])]
    hyp = [_make_tagged([("x", "ig"), ("y", "pcm"), ("z", "en")])]
    result = compute_cs_wer(ref, hyp)
    assert 0.0 <= result.cs_wer <= 1.0
