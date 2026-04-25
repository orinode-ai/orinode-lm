"""Smoke tests for GenderClassifier forward pass."""

from __future__ import annotations

import torch


def test_gender_classifier_forward_smoke() -> None:
    from orinode.models.gender_classifier import GenderClassifier

    model = GenderClassifier.for_smoke_test()
    model.eval()
    x = torch.randn(2, 3200)
    with torch.inference_mode():
        log_probs = model(input_values=x)
    assert log_probs.shape == (2, GenderClassifier.NUM_CLASSES)
    # log-probs must be <= 0
    assert (log_probs <= 0).all()


def test_gender_classifier_labels() -> None:
    from orinode.models.gender_classifier import GenderClassifier

    assert len(GenderClassifier.LABELS) == GenderClassifier.NUM_CLASSES
    assert "male" in GenderClassifier.LABELS
    assert "female" in GenderClassifier.LABELS


def test_gender_classifier_probabilities_sum_to_one() -> None:
    from orinode.models.gender_classifier import GenderClassifier

    model = GenderClassifier.for_smoke_test()
    model.eval()
    x = torch.randn(1, 8000)
    with torch.inference_mode():
        log_probs = model(input_values=x)
    probs = log_probs.exp()
    assert abs(probs.sum().item() - 1.0) < 1e-5
