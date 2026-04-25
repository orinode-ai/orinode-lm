"""Tokenizer extension for Hausa / Yoruba / Igbo / Nigerian Pidgin.

Adds language-specific diacritic characters and control tokens to a pretrained
tokenizer (Whisper or Gemma) so they are represented as single tokens rather
than falling back to byte-level representations.
"""

from __future__ import annotations

from pathlib import Path

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from orinode.data.diacritics import (
    HAUSA_ALPHABET,
    IGBO_ALPHABET,
    YORUBA_ALPHABET,
    nfc,
)

LANGUAGE_TOKENS: dict[str, str] = {
    "en": "<|en|>",
    "ha": "<|ha|>",
    "yo": "<|yo|>",
    "ig": "<|ig|>",
    "pcm": "<|pcm|>",
    "mixed": "<|mixed|>",
}

TASK_TOKENS: dict[str, str] = {
    "transcribe": "<|transcribe|>",
    "translate": "<|translate|>",
    "understand": "<|understand|>",
}

_ALL_SPECIAL_TOKENS: list[str] = list(LANGUAGE_TOKENS.values()) + list(TASK_TOKENS.values())


def build_nigerian_vocab() -> list[str]:
    """Return new tokens needed for full Nigerian language coverage.

    Collects all non-ASCII characters from the language alphabets plus a small
    set of high-frequency Hausa diacritic bigrams.

    Returns:
        Sorted list of unique new token strings.
    """
    candidates: set[str] = set(HAUSA_ALPHABET + YORUBA_ALPHABET + IGBO_ALPHABET)
    new_chars = {nfc(ch) for ch in candidates if ord(ch) > 127}
    hausa_bigrams = {"ɗa", "ɓa", "ƙa", "ƴa", "ɗi", "ɓi", "ƙi", "ƙu", "ɗu"}
    return sorted(new_chars | hausa_bigrams)


def extend_tokenizer(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    save_dir: Path | None = None,
) -> tuple[PreTrainedTokenizer | PreTrainedTokenizerFast, int]:
    """Add Nigerian character tokens and control tokens to a tokenizer.

    Only tokens not already in the vocabulary are added, so this function is
    safe to call more than once.

    Args:
        tokenizer: Base tokenizer (Whisper or Gemma).
        save_dir: If given, save the extended tokenizer to this directory.

    Returns:
        Tuple of ``(extended_tokenizer, num_tokens_added)``.
    """
    existing = set(tokenizer.get_vocab().keys())

    new_tokens = [t for t in build_nigerian_vocab() if t not in existing]
    new_specials = [t for t in _ALL_SPECIAL_TOKENS if t not in existing]

    added = tokenizer.add_tokens(new_tokens)
    added += tokenizer.add_special_tokens({"additional_special_tokens": new_specials})

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(save_dir))

    return tokenizer, added


def get_language_token_id(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    language: str,
) -> int:
    """Return the token ID for a language control token.

    Args:
        tokenizer: Extended tokenizer (must have had ``extend_tokenizer`` called).
        language: One of ``en``, ``ha``, ``yo``, ``ig``, ``pcm``, ``mixed``.

    Returns:
        Integer token ID.

    Raises:
        KeyError: If ``language`` is unknown or the token is not in the vocab.
    """
    token = LANGUAGE_TOKENS.get(language)
    if token is None:
        raise KeyError(f"Unknown language: {language!r}. Expected one of {list(LANGUAGE_TOKENS)}")
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id == tokenizer.unk_token_id:
        raise KeyError(f"Language token {token!r} maps to UNK — call extend_tokenizer() first")
    return int(token_id)
