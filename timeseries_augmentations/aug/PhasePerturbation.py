from aug.base import *

class PhasePerturbation(TimeSeriesTransform):
    """Applies selective phase perturbation to specific frequency bands in the frequency domain."""

    def __init__(self, win_length: int = 10, hop_length: int = 5, freq_range: tuple = (0.2, 0.5), scale: float = 0.1):
        """
        Args:
            win_length (int): Window size for STFT.
            hop_length (int): Hop length for STFT.
            freq_range (tuple): Fractional range (0 to 1) of frequencies to perturb (e.g., (0.2, 0.5) for mid-range frequencies).
            scale (float): Intensity of phase perturbation applied to selected frequencies.
        """
        if win_length <= 0:
            raise ValueError(f"win_length must be > 0, got {win_length}")
        if hop_length <= 0:
            raise ValueError(f"hop_length must be > 0, got {hop_length}")
        if not (0 <= freq_range[0] < freq_range[1] <= 1):
            raise ValueError(f"freq_range must be within (0, 1), got {freq_range}")
        if scale < 0:
            raise ValueError(f"scale must be non-negative, got {scale}")
        
        self.win_length = win_length
        self.hop_length = hop_length
        self.freq_range = freq_range
        self.scale = scale  # Strength of phase perturbation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Phase-Selective Augmentation to a batch of time series.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, time_steps, channels)
        
        Returns:
            torch.Tensor: Augmented tensor with the same shape.
        """
        batch_size, time_steps, channels = x.shape
        x_aug = torch.zeros_like(x)  # Tensor to store the augmented output

        window = torch.hann_window(self.win_length, device=x.device)

        for c in range(channels):
            # Convert to frequency domain using STFT
            x_freq = torch.stft(
                x[:, :, c],
                n_fft=self.win_length,
                hop_length=self.hop_length,
                window=window,
                return_complex=True
            )

            # Determine frequency range indices to modify
            num_freqs = x_freq.shape[1]
            low_idx = int(self.freq_range[0] * num_freqs)
            high_idx = int(self.freq_range[1] * num_freqs)

            # Apply phase perturbation
            phase_shift = self.scale * (2 * torch.rand_like(x_freq[low_idx:high_idx]).real - 1)  # Small phase shift
            x_freq[low_idx:high_idx] *= torch.exp(1j * phase_shift)  # Apply phase shift

            # Convert back to time domain using ISTFT
            x_aug[:, :, c] = torch.istft(
                x_freq,
                n_fft=self.win_length,
                hop_length=self.hop_length,
                window=window
            )
        
        return x_aug

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(win_length={self.win_length}, hop_length={self.hop_length}, freq_range={self.freq_range}, scale={self.scale})"
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(freq_range={self.freq_range}, scale={self.scale})"

