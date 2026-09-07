import torch
import torch.nn as nn
import torch.nn.functional as F
from aug.base import *

class MagnitudePerturbation(TimeSeriesTransform):
    """Phase-Selective Augmentation for time series data.

    This transform applies frequency-domain modifications by scaling specific frequency bands 
    to simulate variations like sensor noise, environmental differences, etc.
    """

    def __init__(self, win_length: int = 10, hop_length: int = 5, freq_range: tuple = (0.2, 0.5), scale: float = 0.1):
        """
        Args:
            win_length (int): Window size for STFT.
            hop_length (int): Hop length for STFT.
            freq_range (tuple): Range of frequencies to modify (0-1, where 1 corresponds to Nyquist frequency).
            scale (float): Strength of the frequency perturbation.
        """
        if win_length <= 0:
            raise ValueError(f"win_length must be > 0, got {win_length}")
        if hop_length <= 0:
            raise ValueError(f"hop_length must be > 0, got {hop_length}")
        if not (0 < freq_range[0] < freq_range[1] < 1):
            raise ValueError(f"freq_range must be a tuple (low, high) where 0 < low < high < 1, got {freq_range}")
        if scale < 0:
            raise ValueError(f"scale must be non-negative, got {scale}")

        self.win_length = win_length
        self.hop_length = hop_length
        self.freq_range = freq_range  # Range of frequencies to modify
        self.scale = scale  # Perturbation intensity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Phase-Selective Augmentation to a batch of time series.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, time_steps, channels)

        Returns:
            torch.Tensor: Augmented tensor with the same shape.
        """
        batch_size, time_steps, channels = x.shape
        x_aug = torch.zeros_like(x)  # Tensor to store the augmented result

        for c in range(channels):
            # Convert to frequency domain (STFT)
            x_freq = torch.stft(
                x[:, :, c],  # Select channel c
                n_fft=self.win_length,
                hop_length=self.hop_length,
                return_complex=True
            )

            # Get frequency bins (frequencies corresponding to the STFT)
            freqs = torch.fft.fftfreq(self.win_length, d=self.hop_length / time_steps)
            freqs = freqs[:self.win_length // 2]  # Positive frequencies

            # Apply perturbation to the selected frequency range
            low_freq_idx = int(self.freq_range[0] * len(freqs))
            high_freq_idx = int(self.freq_range[1] * len(freqs))

            # Scale the magnitude of frequencies in the selected range
            x_freq[:, low_freq_idx:high_freq_idx] *= (1 + self.scale * torch.randn_like(x_freq[:, low_freq_idx:high_freq_idx]))

            # Inverse transform to get the time domain signal back
            x_aug[:, :, c] = torch.istft(
                x_freq,
                n_fft=self.win_length,
                hop_length=self.hop_length
            )

        return x_aug

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(win_length={self.win_length}, hop_length={self.hop_length}, freq_range={self.freq_range}, scale={self.scale})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(freq_range={self.freq_range}, scale={self.scale:.3f})"


