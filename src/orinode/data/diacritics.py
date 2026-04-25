"""NFC normalisation and diacritic preservation for Nigerian languages.

Rules enforced everywhere in the pipeline:
- All text is Unicode NFC before storage and before tokenisation.
- Diacritics are NEVER stripped from training data.
- ``strip_tone_marks`` exists solely for the tone-stripped WER evaluation mode.

Language-specific characters:
- Hausa:        implosives ɗ Ɗ ɓ Ɓ, ejective ƙ Ƙ, labio-palatal ƴ Ƴ
- Yoruba:       tonal vowels (à á ā + combining marks on e o a), sub-dot ẹ Ẹ ọ Ọ, nasal ṅ Ṅ
- Igbo:         sub-dot ị Ị ụ Ụ ọ Ọ, nasal ṅ Ṅ, dotted ẹ Ẹ
- Nigerian Pidgin: base Latin; no special diacritics required
- English:      no diacritics; preserve any present in loanwords
"""

from __future__ import annotations

import unicodedata

# Characters that are language-specific AND should survive tone stripping
# (because they mark consonant class, not tone)
_HAUSA_CONSONANTS: frozenset[str] = frozenset("ɗɓƙƴŊŋÐð") | frozenset("ƊƁƘƳ")
_PRESERVED_THROUGH_STRIP: frozenset[str] = _HAUSA_CONSONANTS

# Full character inventories (used for tokenizer extension analysis)
HAUSA_ALPHABET: list[str] = list("aAbBcCdDɗɓeEfFgGhHiIjJkKƙlLmMnNoOpPrRsStTuUwWyYzZƴ")
YORUBA_ALPHABET: list[str] = list("aAbBdDeẹEẸfFgGhHiIjJkKlLmMnNoọOỌpPrRsṣSṢtTuUwWyY")
IGBO_ALPHABET: list[str] = list("aAbBcCdDeEfFgGhHiịIỊjJkKlLmMnNoọOỌpPrRsStTuụUỤvVwWyYzZ")
PCM_ALPHABET: list[str] = list("aAbBcCdDeEfFgGhHiIjJkKlLmMnNoOpPrRsStTuUvVwWyYzZ")


def nfc(text: str) -> str:
    """Return the Unicode NFC form of ``text``.

    All text entering the pipeline must go through this function. It is a
    no-op when text is already NFC (the common case), so it is safe to call
    eagerly on every string.
    """
    return unicodedata.normalize("NFC", text)


def normalize_text(text: str) -> str:
    """NFC-normalise and collapse interior whitespace.

    Does NOT strip diacritics, lowercase, or remove punctuation. Those
    decisions belong to the tokenizer, not the preprocessing pipeline.

    Args:
        text: Raw transcript string (any Unicode normalisation form).

    Returns:
        NFC-normalised string with collapsed whitespace.
    """
    return " ".join(nfc(text).split())


def strip_tone_marks(text: str) -> str:
    """Remove combining tone-mark characters while preserving base letters.

    ONLY used in the tone-stripped WER evaluation mode. Never called during
    training data preprocessing.

    Hausa consonant diacritics (ɗ ɓ ƙ ƴ) are NOT stripped — they mark
    consonant class (phonemic), not tone (prosodic).

    Args:
        text: NFC-normalised text.

    Returns:
        Text with combining non-spacing marks (Unicode category Mn) removed,
        except for those belonging to ``_PRESERVED_THROUGH_STRIP``.
    """
    nfd = unicodedata.normalize("NFD", text)
    stripped = "".join(
        ch for ch in nfd if unicodedata.category(ch) != "Mn" or ch in _PRESERVED_THROUGH_STRIP
    )
    return unicodedata.normalize("NFC", stripped)


def is_nfc(text: str) -> bool:
    """Return ``True`` if ``text`` is already in NFC form."""
    return unicodedata.is_normalized("NFC", text)


def validate_nfc_roundtrip(text: str) -> bool:
    """Return ``True`` if NFD→NFC round-trips without loss of information."""
    nfd = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", nfd) == nfc(text)


def count_diacritics(text: str, language: str) -> int:
    """Count language-specific diacritic/special characters in ``text``.

    Args:
        text: NFC-normalised text.
        language: One of ``en``, ``ha``, ``yo``, ``ig``, ``pcm``.

    Returns:
        Number of language-specific special characters found.  Returns 0 for
        ``en`` and ``pcm`` (neither requires special characters).
    """
    nfc_text = nfc(text)
    if language == "ha":
        return sum(1 for ch in nfc_text if ch in _HAUSA_CONSONANTS)
    if language == "yo":
        subdot = frozenset("ẹọṣẸỌṢṅṄ")
        return sum(1 for ch in nfc_text if ch in subdot)
    if language == "ig":
        subdot = frozenset("ịụọṅẹỊỤỌṄẸ")
        return sum(1 for ch in nfc_text if ch in subdot)
    return 0
