from aug.base import *  

class TADA(TimeSeriesTransform):
    """Temporal Adversarial Data Augmentation (TADA) respecting constraints."""

    def __init__(self, win_length: int = 10, hop_length: int = 5, gamma: float = 0.1, phi_max: float = 0.5):
        """
        Args:
            win_length (int): Window size for STFT.
            hop_length (int): Hop length for STFT.
            gamma (float): Strength of adversarial perturbation.
            phi_max (float): Maximum allowed perturbation for stability.
        """
        if win_length <= 0:
            raise ValueError(f"win_length must be > 0, got {win_length}")
        if hop_length <= 0:
            raise ValueError(f"hop_length must be > 0, got {hop_length}")
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")

        self.win_length = win_length
        self.hop_length = hop_length
        self.gamma = gamma
        self.phi_max = phi_max  # Maximum warping constraint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply TADA with constraints.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, time_steps, channels)
        
        Returns:
            torch.Tensor: Augmented tensor with constraints applied.
        """
        batch_size, time_steps, channels = x.shape
        x_tada = torch.zeros_like(x)  # Store the result

        for c in range(channels):
            # Convert to frequency domain (STFT)
            x_freq = torch.stft(
                x[:, :, c], 
                n_fft=self.win_length,
                hop_length=self.hop_length,
                return_complex=True
            )

            # Generate a progressive phase shift to avoid non-monotonicity
            phase_shift = self.gamma * torch.cumsum(torch.randn_like(x_freq).real, dim=-1)

            # Apply a maximum distance constraint
            phase_shift = torch.clamp(phase_shift, -self.phi_max, self.phi_max)

            # Reduce warping effect at the edges to maintain alignment
            weights = torch.linspace(0.2 , 0.3, steps=x_freq.shape[-1])
            phase_shift = phase_shift * weights

            # Apply the phase shift
            x_freq_shifted = x_freq * torch.exp(1j * phase_shift)

            # Convert back to the time domain (ISTFT)
            x_tada[:, :, c] = torch.istft(
                x_freq_shifted,
                n_fft=self.win_length,
                hop_length=self.hop_length
            )

        return x_tada

# class TADA(TimeSeriesTransform):
#     """Temporal Adversarial Data Augmentation (TADA) for time series data.

#     This transform applies a differentiable phase perturbation in the frequency domain 
#     to simulate adversarial time shifts.
#     """

#     def __init__(self, win_length: int = 10, hop_length: int = 5, gamma: float = 0.1):
#         """
#         Args:
#             win_length (int): Window size for STFT.
#             hop_length (int): Hop length for STFT.
#             gamma (float): Strength of adversarial perturbation.
#         """
#         if win_length <= 0:
#             raise ValueError(f"win_length must be > 0, got {win_length}")
#         if hop_length <= 0:
#             raise ValueError(f"hop_length must be > 0, got {hop_length}")
#         if gamma < 0:
#             raise ValueError(f"gamma must be non-negative, got {gamma}")

#         self.win_length = win_length
#         self.hop_length = hop_length
#         self.gamma = gamma  # Perturbation intensity

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Apply TADA to a batch of time series.
        
#         Args:
#             x (torch.Tensor): Input tensor of shape (batch, time_steps, channels)
        
#         Returns:
#             torch.Tensor: Augmented tensor with the same shape.
#         """
#         batch_size, time_steps, channels = x.shape
#         x_tada = torch.zeros_like(x)  # Tensor pour stocker le résultat

#         for c in range(channels):
#             # Convertir en domaine fréquentiel (STFT)
#             x_freq = torch.stft(
#                 x[:, :, c],  # Sélection du canal c
#                 n_fft=self.win_length,
#                 hop_length=self.hop_length,
#                 return_complex=True
#             )

#             # Générer un décalage de phase aléatoire
#             phase_shift = self.gamma * torch.randn_like(x_freq).real  # Perturbation faible
#             x_freq_shifted = x_freq * torch.exp(1j * phase_shift)  # Appliquer le décalage de phase

#             # Revenir en domaine temporel (ISTFT)
#             x_tada[:, :, c] = torch.istft(
#                 x_freq_shifted,
#                 n_fft=self.win_length,
#                 hop_length=self.hop_length
#             )

#         return x_tada

#     def __repr__(self) -> str:
#         return f"{self.__class__.__name__}(win_length={self.win_length}, hop_length={self.hop_length}, gamma={self.gamma})"

#     def __str__(self) -> str:
#         return f"{self.__class__.__name__}(γ={self.gamma:.3f})"