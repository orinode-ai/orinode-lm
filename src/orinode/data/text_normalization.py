"""Text normalisation for Nigerian speech transcripts.

Rules:
- NFC Unicode normalisation always applied
- Tone marks and dotted vowels (HA/YO/IG) NEVER stripped
- English: lowercase, collapse whitespace, keep only apostrophes + hyphens
- Nigerian languages: preserve diacritics, no case folding
- All languages: strip leading/trailing whitespace, collapse internal runs
"""

from __future__ import annotations

import re
import unicodedata

_KEEP_PUNCT = re.compile(r"[^\w\s'\-]", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+")

# Languages where we lowercase (English only; preserve case for others)
_LOWERCASE_LANGS = frozenset({"en"})


def normalize_transcript(text: str, language: str = "en") -> str:
    """Normalise a transcript for training and CER comparison.

    Args:
        text: Raw transcript string.
        language: ISO code — ``en``, ``ha``, ``yo``, ``ig``, ``pcm``, etc.

    Returns:
        Normalised string. Never raises; empty input returns empty string.
    """
    if not text:
        return ""

    # NFC: compose diacritics into precomposed forms (ọ not o + combining)
    text = unicodedata.normalize("NFC", text)

    # Case folding only for English
    if language in _LOWERCASE_LANGS:
        text = text.lower()

    # Strip punctuation except apostrophe and hyphen
    text = _KEEP_PUNCT.sub(" ", text)

    # Collapse whitespace
    text = _WHITESPACE.sub(" ", text).strip()

    return text
