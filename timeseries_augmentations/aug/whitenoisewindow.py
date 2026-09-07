from timeseries_augmentations.aug.base import TimeSeriesTransform
import torch
import numpy as np

class WhiteNoiseWindow(TimeSeriesTransform):
    def __init__(self, window_ratio: float = 0.2, noise_std: float = 0.05):
        """
        Args:
            window_ratio (float): Proportion de la fenêtre temporelle à bruiter (ex: 0.2 → 20% du temps)
            noise_std (float): Écart-type du bruit blanc ajouté (gaussien)
        """
        if not (0 < window_ratio <= 1):
            raise ValueError("window_ratio must be in (0, 1]")
        self.window_ratio = window_ratio
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tenseur d'entrée (G, T, C)
        Returns:
            Tenseur bruité (G, T, C)
        """
        G, T, C = x.shape

        # Taille de la fenêtre
        win_len = int(T * self.window_ratio)
        if win_len == 0:
            return x  # Pas de fenêtre

        # Choisir un point de départ aléatoire
        start = torch.randint(0, T - win_len + 1, (1,)).item()
        end = start + win_len

        # Générer du bruit
        noise = torch.randn((G, win_len, C), device=x.device) * self.noise_std

        # Appliquer le bruit dans la fenêtre
        x_noisy = x.clone()
        x_noisy[:, start:end, :] += noise

        return x_noisy

    def __repr__(self):
        return f"{self.__class__.__name__}(window_ratio={self.window_ratio}, noise_std={self.noise_std})"

    def __str__(self):
        return f"{self.__class__.__name__}(ratio={self.window_ratio:.2f}, σ={self.noise_std:.3f})"
