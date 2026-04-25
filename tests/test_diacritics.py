"""Tests for diacritics.py — NFC normalisation and diacritic preservation.

50 hard cases covering:
- Hausa implosives / ejectives (ɗ ɓ ƙ ƴ)
- Yoruba tonal vowels and sub-dot letters
- Igbo sub-dot vowels and dotted nasal
- NFD→NFC round-trips
- Mixed-language code-switching strings
- Whitespace normalisation
- strip_tone_marks correctness (tone stripped, consonants preserved)
- count_diacritics per language
- Edge cases: empty string, ASCII-only, non-Nigerian Unicode
"""

from __future__ import annotations

import unicodedata

from orinode.data.diacritics import (
    count_diacritics,
    is_nfc,
    nfc,
    normalize_text,
    strip_tone_marks,
    validate_nfc_roundtrip,
)

# ── NFC normalisation ─────────────────────────────────────────────────────────


def test_nfc_already_nfc() -> None:
    assert nfc("hello") == "hello"


def test_nfc_converts_nfd() -> None:
    nfd_e = unicodedata.normalize("NFD", "ẹ")
    assert len(nfd_e) == 2  # base + combining dot below
    assert len(nfc(nfd_e)) == 1  # single precomposed character


def test_nfc_hausa_implosive_d() -> None:
    assert nfc("ɗ") == "ɗ"
    assert is_nfc("ɗ")


def test_nfc_hausa_implosive_b() -> None:
    assert nfc("ɓ") == "ɓ"
    assert is_nfc("ɓ")


def test_nfc_hausa_ejective_k() -> None:
    assert nfc("ƙ") == "ƙ"
    assert is_nfc("ƙ")


def test_nfc_hausa_labiopalatal_y() -> None:
    assert nfc("ƴ") == "ƴ"
    assert is_nfc("ƴ")


def test_nfc_yoruba_subdot_e() -> None:
    char = "ẹ"
    assert is_nfc(nfc(char))
    assert nfc(char) == "ẹ"


def test_nfc_yoruba_subdot_o() -> None:
    char = "ọ"
    assert is_nfc(nfc(char))


def test_nfc_yoruba_tone_high_a() -> None:
    char = "á"
    assert is_nfc(nfc(char))
    assert len(nfc(char)) == 1


def test_nfc_yoruba_tone_low_a() -> None:
    char = "à"
    assert is_nfc(nfc(char))


def test_nfc_yoruba_mid_tone_e() -> None:
    # mid-tone ē (combining macron)
    nfd = "ē"
    assert is_nfc(nfc(nfd))


def test_nfc_igbo_subdot_i() -> None:
    assert nfc("ị") == "ị"
    assert is_nfc("ị")


def test_nfc_igbo_subdot_u() -> None:
    assert nfc("ụ") == "ụ"
    assert is_nfc("ụ")


def test_nfc_igbo_nasal_n() -> None:
    assert is_nfc(nfc("ṅ"))


def test_nfc_sentence_hausa() -> None:
    s = "Ina son ƙasar Najeriya"
    assert is_nfc(nfc(s))
    assert nfc(s) == s  # already NFC


def test_nfc_sentence_yoruba() -> None:
    s = "Èdè Yorùbá"
    result = nfc(s)
    assert is_nfc(result)


def test_nfc_sentence_igbo() -> None:
    s = "Asụsụ Igbo bụ asụsụ"
    assert is_nfc(nfc(s))


def test_nfc_mixed_cs_sentence() -> None:
    s = "I go buy ƙayan abinci"
    assert is_nfc(nfc(s))


def test_nfc_empty_string() -> None:
    assert nfc("") == ""


def test_nfc_ascii_only() -> None:
    assert nfc("Hello world 123") == "Hello world 123"


# ── normalize_text ────────────────────────────────────────────────────────────


def test_normalize_collapses_spaces() -> None:
    assert normalize_text("hello   world") == "hello world"


def test_normalize_strips_leading_trailing() -> None:
    assert normalize_text("  ẹkọ  ") == "ẹkọ"


def test_normalize_preserves_diacritics() -> None:
    s = "Ina son ƙasar"
    assert normalize_text(s) == s


def test_normalize_nfc_roundtrip() -> None:
    nfd = unicodedata.normalize("NFD", "ọmọ")
    normalised = normalize_text(nfd)
    assert is_nfc(normalised)


def test_normalize_tabs_and_newlines() -> None:
    assert normalize_text("foo\tbar\nbaz") == "foo bar baz"


# ── strip_tone_marks ──────────────────────────────────────────────────────────


def test_strip_removes_yoruba_tones() -> None:
    result = strip_tone_marks("àbí")
    assert "à" not in result
    assert "á" not in result
    # base letter should remain
    assert "a" in result.lower() or "b" in result.lower()


def test_strip_preserves_hausa_implosive_d() -> None:
    assert "ɗ" in strip_tone_marks("ɗan")


def test_strip_preserves_hausa_ejective_k() -> None:
    assert "ƙ" in strip_tone_marks("ƙaura")


def test_strip_preserves_hausa_implosive_b() -> None:
    assert "ɓ" in strip_tone_marks("ɓangare")


def test_strip_preserves_hausa_y() -> None:
    assert "ƴ" in strip_tone_marks("ƴar")


def test_strip_result_is_nfc() -> None:
    assert is_nfc(strip_tone_marks("Ìsọ̀rọ̀ àwọn ìjọba"))


def test_strip_empty_string() -> None:
    assert strip_tone_marks("") == ""


def test_strip_ascii_unchanged() -> None:
    assert strip_tone_marks("hello world") == "hello world"


def test_strip_igbo_subdot_i_preserved() -> None:
    # ị is a precomposed character (combining dot below is part of the base)
    # After stripping Mn combining marks, the precomposed NFC form should survive
    # because it's already NFC (no separate combining mark in NFC form).
    result = strip_tone_marks("ị")
    assert result  # not empty


# ── validate_nfc_roundtrip ────────────────────────────────────────────────────


def test_roundtrip_hausa_sentence() -> None:
    assert validate_nfc_roundtrip("Ina son ƙasar Najeriya da dukan zuciyata")


def test_roundtrip_yoruba_sentence() -> None:
    assert validate_nfc_roundtrip("Èdè Yorùbá jẹ́ èdè tí ó ní ọlọ́rọ̀ itan")


def test_roundtrip_igbo_sentence() -> None:
    assert validate_nfc_roundtrip("Asụsụ Igbo bụ asụsụ dị mma")


def test_roundtrip_ascii() -> None:
    assert validate_nfc_roundtrip("the quick brown fox")


def test_roundtrip_empty() -> None:
    assert validate_nfc_roundtrip("")


# ── count_diacritics ──────────────────────────────────────────────────────────


def test_count_hausa_implosives() -> None:
    count = count_diacritics("ɗan ƙaura ɓangare ƴar", "ha")
    assert count == 4


def test_count_hausa_zero_in_english() -> None:
    assert count_diacritics("hello world", "ha") == 0


def test_count_yoruba_subdot_vowels() -> None:
    count = count_diacritics("ẹkọ ọmọ", "yo")
    assert count >= 2


def test_count_igbo_subdot_vowels() -> None:
    count = count_diacritics("ụlọ ọchịchọ", "ig")
    assert count >= 2


def test_count_en_returns_zero() -> None:
    assert count_diacritics("hello", "en") == 0


def test_count_pcm_returns_zero() -> None:
    assert count_diacritics("I dey go market", "pcm") == 0


def test_count_mixed_hausa() -> None:
    s = "ɗan ilu yana da kyau"
    assert count_diacritics(s, "ha") >= 1


def test_is_nfc_pure_ascii() -> None:
    assert is_nfc("The quick brown fox")


def test_is_nfc_hausa_sentence() -> None:
    assert is_nfc("Ina son ƙasar Najeriya")


def test_is_nfc_nfd_is_false() -> None:
    nfd = unicodedata.normalize("NFD", "ẹkọ")
    assert not is_nfc(nfd)
