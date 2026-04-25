"""Data augmentation callables for speech training.

All augmentations are ``torch.nn.Module`` subclasses so they compose with
``torchaudio.transforms`` pipelines and serialise with ``torch.save``.

Augmentation schedule (applied to raw waveforms in training):
- Speed perturbation: random 0.9× / 1.0× / 1.1× (p=0.3)
- Telephony simulation: G.711 μ-law codec + 300–3400 Hz band-pass (p=0.5)
- Gaussian noise injection at 10–20 dB SNR to approximate MUSAN (p=0.3)

SpecAugment is applied to the log-mel spectrogram after feature extraction.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
import torchaudio.functional as F
import torchaudio.transforms as T

from orinode.data.preprocessing import TARGET_SAMPLE_RATE


class SpecAugment(nn.Module):
    """Frequency masking + time masking on log-mel spectrograms (SpecAugment).

    Args:
        freq_mask_param: Maximum frequency mask width in mel bins.
        time_mask_param: Maximum time mask width in frames.
        num_freq_masks: Number of independent frequency masks to apply.
        num_time_masks: Number of independent time masks to apply.
    """

    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
    ) -> None:
        super().__init__()
        self.freq_masks = nn.ModuleList(
            [T.FrequencyMasking(freq_mask_param) for _ in range(num_freq_masks)]
        )
        self.time_masks = nn.ModuleList(
            [T.TimeMasking(time_mask_param) for _ in range(num_time_masks)]
        )

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment.

        Args:
            spec: Log-mel spectrogram ``(batch, n_mels, T)`` or ``(n_mels, T)``.

        Returns:
            Augmented spectrogram with same shape.
        """
        for mask in self.freq_masks:
            spec = mask(spec)
        for mask in self.time_masks:
            spec = mask(spec)
        return spec


class SpeedPerturbation(nn.Module):
    """Speed perturbation via integer-ratio resampling.

    Randomly selects one of ``rates`` and resample to simulate the target
    speed while returning audio at the original sample rate.

    Args:
        rates: Speed factors to choose from.
        sample_rate: Native sample rate (Hz).
    """

    def __init__(
        self,
        rates: tuple[float, ...] = (0.9, 1.0, 1.1),
        sample_rate: int = TARGET_SAMPLE_RATE,
    ) -> None:
        super().__init__()
        self.rates = rates
        self.sample_rate = sample_rate

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random speed perturbation.

        Args:
            waveform: Float32 tensor ``(1, T)``.

        Returns:
            Speed-perturbed waveform at the original sample rate.
        """
        rate = random.choice(self.rates)
        if rate == 1.0:
            return waveform
        # Simulate rate by treating the waveform as if recorded at rate*sr,
        # then resampling back to sr.
        orig_freq = round(self.sample_rate * rate)
        return T.Resample(orig_freq=orig_freq, new_freq=self.sample_rate)(waveform)


class TelephonySimulator(nn.Module):
    """G.711 μ-law codec + 300–3400 Hz band-pass (telephone line simulation).

    Replicates the audio degradation of a typical Nigerian mobile or PSTN call.
    The combination of quantisation noise and bandwidth restriction is the
    dominant acoustic mismatch between lab speech and production call-center
    audio.

    Args:
        sample_rate: Audio sample rate (Hz).
        quantization_channels: μ-law quantisation depth (256 = 8-bit G.711).
        low_cutoff: Band-pass lower cutoff (Hz).
        high_cutoff: Band-pass upper cutoff (Hz).
    """

    def __init__(
        self,
        sample_rate: int = TARGET_SAMPLE_RATE,
        quantization_channels: int = 256,
        low_cutoff: float = 300.0,
        high_cutoff: float = 3400.0,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.quantization_channels = quantization_channels
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply telephone codec + band-pass simulation.

        Args:
            waveform: Float32 tensor ``(1, T)`` in ``[-1, 1]``.

        Returns:
            Degraded waveform with same shape.
        """
        encoded = F.mu_law_encoding(waveform, self.quantization_channels)
        decoded = F.mu_law_decoding(encoded, self.quantization_channels)
        decoded = F.highpass_biquad(decoded, self.sample_rate, self.low_cutoff)
        decoded = F.lowpass_biquad(decoded, self.sample_rate, self.high_cutoff)
        return decoded


class AddGaussianNoise(nn.Module):
    """Add white Gaussian noise at a random SNR to approximate MUSAN conditions.

    Args:
        snr_min_db: Minimum SNR in dB.
        snr_max_db: Maximum SNR in dB.
    """

    def __init__(self, snr_min_db: float = 10.0, snr_max_db: float = 20.0) -> None:
        super().__init__()
        self.snr_min_db = snr_min_db
        self.snr_max_db = snr_max_db

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add noise at a uniformly-random SNR.

        Args:
            waveform: Float32 tensor ``(1, T)``.

        Returns:
            Noisy waveform clamped to ``[-1, 1]``.
        """
        snr_db = random.uniform(self.snr_min_db, self.snr_max_db)
        signal_power = waveform.pow(2).mean().clamp(min=1e-9)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(waveform) * noise_power.sqrt()
        return (waveform + noise).clamp(-1.0, 1.0)


class RandomAugmentPipeline(nn.Module):
    """Stochastic waveform augmentation pipeline.

    Each augmentation is gated by an independent Bernoulli draw. The three
    augmentations are applied in order: speed → telephony → noise.

    Args:
        p_telephony: Probability of telephony simulation.
        p_speed: Probability of speed perturbation.
        p_noise: Probability of noise addition.
        sample_rate: Audio sample rate.
    """

    def __init__(
        self,
        p_telephony: float = 0.5,
        p_speed: float = 0.3,
        p_noise: float = 0.3,
        sample_rate: int = TARGET_SAMPLE_RATE,
    ) -> None:
        super().__init__()
        self.p_telephony = p_telephony
        self.p_speed = p_speed
        self.p_noise = p_noise
        self.telephony = TelephonySimulator(sample_rate=sample_rate)
        self.speed = SpeedPerturbation(sample_rate=sample_rate)
        self.noise = AddGaussianNoise()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply stochastic augmentation pipeline.

        Args:
            waveform: Float32 tensor ``(1, T)``.

        Returns:
            Augmented waveform.
        """
        if random.random() < self.p_speed:
            waveform = self.speed(waveform)
        if random.random() < self.p_telephony:
            waveform = self.telephony(waveform)
        if random.random() < self.p_noise:
            waveform = self.noise(waveform)
        return waveform
