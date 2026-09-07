import torch
import torch.nn.functional as F
from typing import Optional
from timeseries_augmentations.aug.base import TimeSeriesTransform
import numpy as np

class TemporalDropout(TimeSeriesTransform):
    def __init__(self, drop_rate: float = 0.3):
        """
        Args:
            drop_rate (float): Proportion de pas de temps à supprimer aléatoirement (entre 0 et 1).
        """
        if not (0 <= drop_rate <= 1):
            raise ValueError("drop_rate must be in [0, 1]")
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (G, T, C)
        Returns:
            Tensor with some timesteps dropped and linearly interpolated
        """
        G, T, C = x.shape
        min_keep = 3  # garder au moins 3 points dans chaque série

        retain_length = max(int(T * (1 - self.drop_rate)), min_keep)
        x_out = torch.empty_like(x)

        for g in range(G):
            # 1. Sélection aléatoire des indices temporels à garder
            keep_idx = torch.randperm(T)[:retain_length].sort()[0]

            # 2. Extraction et interpolation
            for c in range(C):
                kept_time = keep_idx.cpu().numpy()
                kept_vals = x[g, keep_idx, c].cpu().numpy()

                # Interpolation linéaire sur tous les points de temps
                interp = torch.tensor(
                    np.interp(np.arange(T), kept_time, kept_vals),
                    dtype=x.dtype,
                    device=x.device
                )
                x_out[g, :, c] = interp

        return x_out

    def __repr__(self):
        return f"{self.__class__.__name__}(drop_rate={self.drop_rate})"

    def __str__(self):
        return f"{self.__class__.__name__}(drop={self.drop_rate:.2f})"
