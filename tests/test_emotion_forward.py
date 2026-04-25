"""Smoke tests for EmotionClassifier forward pass."""

from __future__ import annotations

import torch


def test_emotion_classifier_forward_smoke() -> None:
    from orinode.models.emotion_classifier import EmotionClassifier

    model = EmotionClassifier.for_smoke_test()
    model.eval()
    x = torch.randn(2, 3200)
    with torch.inference_mode():
        log_probs = model(input_values=x)
    assert log_probs.shape == (2, EmotionClassifier.NUM_CLASSES)
    assert (log_probs <= 0).all()


def test_emotion_classifier_labels() -> None:
    from orinode.models.emotion_classifier import EmotionClassifier

    assert len(EmotionClassifier.LABELS) == EmotionClassifier.NUM_CLASSES
    for label in ("happy", "angry", "sad", "neutral"):
        assert label in EmotionClassifier.LABELS


def test_emotion_classifier_probabilities_sum_to_one() -> None:
    from orinode.models.emotion_classifier import EmotionClassifier

    model = EmotionClassifier.for_smoke_test()
    model.eval()
    x = torch.randn(1, 8000)
    with torch.inference_mode():
        log_probs = model(input_values=x)
    probs = log_probs.exp()
    assert abs(probs.sum().item() - 1.0) < 1e-5


def test_emotion_disclaimer_constant() -> None:
    from orinode.inference.emotion_pipeline import DISCLAIMER

    assert "IEMOCAP" in DISCLAIMER or "English" in DISCLAIMER
    assert len(DISCLAIMER) > 20
