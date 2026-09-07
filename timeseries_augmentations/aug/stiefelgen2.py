from timeseries_augmentations.aug.base import TimeSeriesTransform

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from scipy.linalg import expm




class StiefelGen(TimeSeriesTransform):
    def __init__(self, beta: float = 0.3, smooth: Optional[int] = None):
        """
        Args:
            beta (float): Perturbation scale in [0, 1], relative to injectivity radius.
            smooth (Optional[int]): Temporal smoothing kernel size (e.g. 4).
        """
        if not (0 <= beta <= 1):
            raise ValueError("beta must be in [0, 1]")
        self.beta = beta
        self.smooth = smooth
        self.radius = 0.89 * np.pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        G, T, C = x.shape

        # 1. Padding miroir pour stabiliser les bords
        pad_size = self.smooth if self.smooth and self.smooth > 1 else 0
        x_padded = F.pad(x, (0, 0, pad_size, pad_size), mode='reflect')  # (G, T+2p, C)
        T_pad = x_padded.shape[1]

        # 2. Mise à plat pour SVD : (T, GC)
        X = x_padded.permute(1, 0, 2).reshape(T_pad, G * C)

        # 3. Décomposition SVD
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)

        # 4. Perturbation géométrique
        U_new = self._perturb_stiefel(U, self.beta)
        V_new = self._perturb_stiefel(Vh.T, self.beta)

        # 5. Reconstruction
        S_mat = torch.diag(S)
        X_new = U_new @ S_mat @ V_new.T  # (T_pad, GC)

        # 6. Reshape vers (G, T+2p, C)
        X_new = X_new.reshape(T_pad, G, C).permute(1, 0, 2)  # (G, T+2p, C)

        # 7. Retrait du padding miroir
        X_new = X_new[:, pad_size:-pad_size, :]  # (G, T, C)

        # 8. Lissage optionnel (temporal)
        if self.smooth and self.smooth > 1:
            X_new = F.avg_pool1d(
                X_new.permute(0, 2, 1),
                kernel_size=self.smooth,
                stride=1,
                padding=self.smooth // 2,
                count_include_pad=False
            ).permute(0, 2, 1)

        return X_new

    def _perturb_stiefel(self, M: torch.Tensor, beta: float) -> torch.Tensor:
        """
        Perturbe la matrice M (orthogonale) sur la variété de Stiefel.
        """
        m, n = M.shape
        device = M.device

        # Skew-symmetric random matrix
        A = torch.randn(n, n, device=device)
        A = A - A.T
        delta = M @ A

        # Normalisation par la métrique canonique
        w = torch.eye(m, device=device) - 0.5 * M @ M.T
        delta_norm = torch.norm((w @ delta).flatten())
        delta_scaled = delta / delta_norm * (self.radius * beta)

        # Exponential map (sur CPU via scipy)
        delta_np = delta_scaled.detach().cpu().numpy()
        M_np = M.detach().cpu().numpy()
        M_new = M_np @ expm(np.linalg.pinv(M_np.T @ M_np) @ (M_np.T @ delta_np))

        return torch.tensor(M_new, dtype=M.dtype, device=device)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(beta={self.beta}, smooth={self.smooth})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(β={self.beta:.2f}, smooth={self.smooth})"
