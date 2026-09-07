
from timeseries_augmentations.aug.base import TimeSeriesTransform
import torch
import numpy as np

class InterExtrapolation(TimeSeriesTransform):
    def __init__(self, alpha: float = 0.3, mode: str = "both"):
        """
        Args:
            alpha (float): Contrôle l'amplitude de l'interpolation/extrapolation (via Beta distribution)
            mode (str): "interpolation", "extrapolation", ou "both"
        """
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        if mode not in ["interpolation", "extrapolation", "both"]:
            raise ValueError("mode must be 'interpolation', 'extrapolation' or 'both'")
        self.alpha = alpha
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor de forme (G, T, C)
        Returns:
            Tensor transformé (G, T, C)
        """
        G, T, C = x.shape

        # Mélange de batchs
        perm = torch.randperm(G, device=x.device)
        x_pair = x[perm]

        # Tirage du coefficient lambda
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((G, 1, 1)).to(x.device)

        # Choix du mode : interpolation, extrapolation ou mélange
        if self.mode == "interpolation":
            x_new = lam * x + (1 - lam) * x_pair
        elif self.mode == "extrapolation":
            x_new = (1 + lam) * x - lam * x_pair
        else:  # mode == "both"
            use_extrap = torch.rand(G, 1, 1, device=x.device) > 0.5
            x_new = torch.where(use_extrap, (1 + lam) * x - lam * x_pair, lam * x + (1 - lam) * x_pair)

        return x_new

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha}, mode='{self.mode}')"

    def __str__(self):
        return f"{self.__class__.__name__}(α={self.alpha:.2f}, mode={self.mode})"
