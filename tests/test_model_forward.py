"""Fast model forward-pass tests (no GPU, no downloads)."""

from __future__ import annotations

import pytest
import torch

# ── WhisperEncoder ────────────────────────────────────────────────────────────


def test_whisper_encoder_forward() -> None:
    from orinode.models.whisper_encoder import WhisperEncoder

    enc = WhisperEncoder.for_smoke_test()
    enc.eval()

    B, n_mels, T = 2, 128, 64
    features = torch.randn(B, n_mels, T)
    out = enc(features)

    assert out.ndim == 3
    assert out.shape[0] == B
    assert out.shape[2] == 64  # output_dim


def test_whisper_encoder_attention_mask() -> None:
    from orinode.models.whisper_encoder import WhisperEncoder

    enc = WhisperEncoder.for_smoke_test()
    enc.eval()

    B, n_mels, T = 2, 128, 64
    features = torch.randn(B, n_mels, T)
    mask = torch.ones(B, T, dtype=torch.long)
    mask[1, 48:] = 0  # second item shorter

    out_masked = enc(features, attention_mask=mask)
    out_plain = enc(features)

    assert out_masked.shape == out_plain.shape


# ── AudioLLMAdapter ───────────────────────────────────────────────────────────


def test_adapter_output_shape() -> None:
    from orinode.models.adapter import AudioLLMAdapter

    adapter = AudioLLMAdapter.for_smoke_test(input_dim=64, output_dim=128)
    adapter.eval()

    B, T_prime, D_enc = 2, 32, 64
    hidden = torch.randn(B, T_prime, D_enc)
    out = adapter(hidden)

    assert out.shape == (B, 4, 128)  # N_q=4 for smoke test


def test_adapter_with_padding_mask() -> None:
    from orinode.models.adapter import AudioLLMAdapter

    adapter = AudioLLMAdapter.for_smoke_test(input_dim=64, output_dim=128)
    adapter.eval()

    B, T_prime, D_enc = 2, 32, 64
    hidden = torch.randn(B, T_prime, D_enc)
    mask = torch.ones(B, T_prime, dtype=torch.long)
    mask[1, 20:] = 0

    out = adapter(hidden, encoder_attention_mask=mask)
    assert out.shape == (B, 4, 128)


def test_adapter_fixed_output_regardless_of_seq_len() -> None:
    """N_q is fixed even when encoder sequence length changes."""
    from orinode.models.adapter import AudioLLMAdapter

    adapter = AudioLLMAdapter.for_smoke_test(input_dim=64, output_dim=128)
    adapter.eval()

    for T_prime in (8, 16, 32):
        hidden = torch.randn(2, T_prime, 64)
        out = adapter(hidden)
        assert out.shape == (2, 4, 128), f"Failed for T_prime={T_prime}"


# ── language_model_loss ───────────────────────────────────────────────────────


def test_language_model_loss_basic() -> None:
    from orinode.models.losses import language_model_loss

    B, S, V = 2, 10, 50
    logits = torch.randn(B, S, V)
    labels = torch.randint(0, V, (B, S))

    loss = language_model_loss(logits, labels, vocab_size=V)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_language_model_loss_ignores_minus100() -> None:
    from orinode.models.losses import language_model_loss

    B, S, V = 2, 10, 50
    logits = torch.randn(B, S, V)
    labels = torch.full((B, S), -100, dtype=torch.long)
    labels[:, 5:] = torch.randint(0, V, (B, 5))

    loss_partial = language_model_loss(logits, labels, vocab_size=V)
    assert loss_partial.item() > 0


def test_compute_token_accuracy() -> None:
    from orinode.models.losses import compute_token_accuracy

    B, S, V = 2, 8, 32
    logits = torch.zeros(B, S, V)
    labels = torch.ones(B, S, dtype=torch.long) * 5
    labels[:, 0] = -100  # ignore first token

    # Make position 5 the argmax everywhere
    logits[:, :, 5] = 10.0

    acc = compute_token_accuracy(logits, labels)
    assert 0.0 <= acc <= 1.0


# ── SpeechLLM ─────────────────────────────────────────────────────────────────


def test_speech_llm_forward_no_labels() -> None:
    from orinode.models.speech_llm import SpeechLLM

    model = SpeechLLM.for_smoke_test()
    model.eval()

    B, n_mels, T, S = 2, 128, 64, 6
    features = torch.randn(B, n_mels, T)
    input_ids = torch.randint(0, 256, (B, S))

    out = model(input_features=features, input_ids=input_ids)
    assert out.loss is None
    assert out.logits.shape[0] == B
    assert out.audio_tokens.shape == (B, 4, 128)  # N_q=4


def test_speech_llm_forward_with_labels() -> None:
    from orinode.models.speech_llm import SpeechLLM

    model = SpeechLLM.for_smoke_test()
    model.eval()

    B, n_mels, T, S = 2, 128, 64, 6
    features = torch.randn(B, n_mels, T)
    input_ids = torch.randint(0, 256, (B, S))
    labels = input_ids.clone()
    labels[:, :2] = -100

    out = model(input_features=features, input_ids=input_ids, labels=labels)
    assert out.loss is not None
    assert out.loss.ndim == 0
    assert out.loss.item() > 0


def test_speech_llm_logits_sequence_length() -> None:
    """Logit sequence = N_q + S."""
    from orinode.models.speech_llm import SpeechLLM

    model = SpeechLLM.for_smoke_test()
    model.eval()

    B, n_mels, T, S = 1, 128, 64, 8
    features = torch.randn(B, n_mels, T)
    input_ids = torch.randint(0, 256, (B, S))

    out = model(input_features=features, input_ids=input_ids)
    N_q = model.adapter.num_query_tokens  # 4
    assert out.logits.shape == (B, N_q + S, 256)  # vocab_size=256


@pytest.mark.slow
def test_speech_llm_generate() -> None:
    from orinode.models.speech_llm import SpeechLLM

    model = SpeechLLM.for_smoke_test()
    model.eval()

    B, n_mels, T = 1, 128, 64
    features = torch.randn(B, n_mels, T)
    prompt_ids = torch.randint(0, 256, (B, 3))

    tokens = model.generate(
        input_features=features,
        prompt_ids=prompt_ids,
        max_new_tokens=5,
        do_sample=False,
    )
    assert tokens.ndim == 2
    assert tokens.shape[0] == B
