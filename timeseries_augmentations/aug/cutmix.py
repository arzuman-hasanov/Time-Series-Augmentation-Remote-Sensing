from timeseries_augmentations.aug.base import TimeSeriesTransform
import torch
import numpy as np



class CutMix(TimeSeriesTransform):
    def __init__(self, alpha: float = 0.4):
        """
        Args:
            alpha (float): Contrôle la taille de la fenêtre temporelle à couper (via Beta distribution).
        """
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (G, T, C)
        Returns:
            Augmented tensor with temporal CutMix applied (same shape).
        """
        G, T, C = x.shape

        # 1. Shuffle batch dimension
        perm = torch.randperm(G, device=x.device)
        x_shuffled = x[perm]

        # 2. Sample mixing coefficient (Beta) and compute cut length
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        cut_len = int(T * lam)

        if cut_len == 0:
            return x  # Aucun changement si la fenêtre est vide

        # 3. Select start index for temporal window
        start = torch.randint(0, T - cut_len + 1, (1,)).item()
        end = start + cut_len

        # 4. Apply CutMix: replace temporal window
        x_cutmix = x.clone()
        x_cutmix[:, start:end, :] = x_shuffled[:, start:end, :]

        return x_cutmix

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha})"

    def __str__(self):
        return f"{self.__class__.__name__}(α={self.alpha:.2f})"
