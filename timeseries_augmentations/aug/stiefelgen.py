from timeseries_augmentations.aug.base import TimeSeriesTransform
import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import expm
from typing import Optional




##################################

# class StiefelGen(TimeSeriesTransform):
#     def __init__(self, window_size: int = 10, beta: float = 0.05, smoothing: bool = True):
#         """
#         Args:
#             window_size: taille m pour la page matrix (doit diviser la longueur temporelle T)
#             beta: facteur de déplacement dans la variété de Stiefel (0 = aucun, 1 = max)
#             smoothing: True pour lisser la série générée
#         """
#         assert 0 <= beta <= 1, "β doit être entre 0 et 1"
#         assert window_size >= 2, "window_size doit être ≥ 2"
#         self.m = window_size
#         self.beta = beta
#         self.smoothing = smoothing

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: (G, T, C) - G groupes, T temps, C bandes spectrales
#         """
#         G, T, C = x.shape
#         output = []

#         for i in range(G):  # pour chaque groupe (champ)
#             transformed = []
#             for c in range(C):  # pour chaque bande spectrale
#                 signal = x[i, :, c]  # (T,)

#                 # 🔹 Normalisation (z-score)
#                 mean, std = signal.mean(), signal.std()
#                 signal_norm = (signal - mean) / (std + 1e-8)

#                 # 🔹 Transformation Stiefel
#                 signal_aug = self._augment_signal(signal_norm)

#                 # 🔹 Dénormalisation
#                 signal_aug = signal_aug * std + mean
#                 transformed.append(signal_aug.unsqueeze(1))

#             output.append(torch.cat(transformed, dim=1))  # (T, C)

#         return torch.stack(output, dim=0)  # (G, T, C)

#     def _augment_signal(self, signal: torch.Tensor) -> torch.Tensor:
#         m = self.m
#         N = signal.shape[0]
#         n = N // m
#         T_mat = signal[:n * m].reshape(n, m).T  # page matrix (m, n)

#         # 🔹 Décomposition SVD
#         U, S, Vh = torch.linalg.svd(T_mat, full_matrices=False)

#         # 🔹 Perturbation géométrique
#         U_new = self._stiefel_perturb(U)
#         V_new = self._stiefel_perturb(Vh.T)

#         # 🔹 Reconstruction
#         T_mat_new = U_new @ torch.diag(S) @ V_new.T
#         signal_new = T_mat_new.T.reshape(-1)

#         # 🔹 Smoothing optionnel (moyenne glissante)
#         if self.smoothing:
#             signal_new = F.avg_pool1d(signal_new.view(1, 1, -1), kernel_size=3, stride=1, padding=1).squeeze()

#         return signal_new

#     def _stiefel_perturb(self, U: torch.Tensor) -> torch.Tensor:
#         # 🔹 Étape 1 : vecteur tangent aléatoire
#         Δ = self._random_tangent_vector(U)

#         # 🔹 Étape 2 : normalisation selon le produit scalaire canonique
#         Δ_norm = self._canonical_norm(U, Δ)
#         Δ_normalized = Δ / Δ_norm * self.beta * (0.89 * torch.pi)

#         # 🔹 Étape 3 : Retraction QR (approximation de l’exponentielle)
#         return self._qr_retraction(U + Δ_normalized)

#     def _random_tangent_vector(self, U: torch.Tensor) -> torch.Tensor:
#         A = torch.randn_like(U)
#         proj = A - U @ (U.T @ A)  # projection tangentielle
#         return proj

#     def _canonical_norm(self, U: torch.Tensor, Δ: torch.Tensor) -> torch.Tensor:
#         I = torch.eye(U.shape[0], device=U.device)
#         M = I - 0.5 * U @ U.T
#         inner = torch.trace(Δ.T @ M @ Δ)
#         return torch.sqrt(inner + 1e-8)

#     def _qr_retraction(self, Y: torch.Tensor) -> torch.Tensor:
#         # QR orthonormalise la matrice Y pour rester dans Stiefel
#         Q, _ = torch.linalg.qr(Y, mode='reduced')
#         return Q








#################################################################################################


# class StiefelGen(TimeSeriesTransform):
#     def __init__(self, window_size: int = 10, beta: float = 0.05, smoothing: bool = True):
#         self.m = window_size
#         self.beta = beta
#         self.smoothing = smoothing

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         G, T, C = x.shape
#         output = []

#         for i in range(G):
#             transformed = []
#             for c in range(C):
#                 signal = x[i, :, c]

#                 # 🔹 Normalisation
#                 mean, std = signal.mean(), signal.std()
#                 signal_norm = (signal - mean) / (std + 1e-6)

#                 # 🔹 Augmentation
#                 signal_aug = self._augment_signal(signal_norm)

#                 # 🔹 Dénormalisation
#                 signal_aug = signal_aug * std + mean
#                 transformed.append(signal_aug.unsqueeze(1))
#             output.append(torch.cat(transformed, dim=1))
#         return torch.stack(output, dim=0)

#     def _augment_signal(self, signal: torch.Tensor) -> torch.Tensor:
#         m = self.m
#         N = signal.shape[0]
#         n = N // m
#         signal = signal[:n * m]  # Crop
#         T_mat = signal.reshape(n, m).T  # Shape (m, n)

#         # 🔹 Décomposition SVD
#         U, S, Vh = torch.linalg.svd(T_mat, full_matrices=False)

#         # 🔹 Rotation géométrique via exp(A) U
#         U_new = self._rotate_on_stiefel(U)
#         V_new = self._rotate_on_stiefel(Vh.T)

#         # 🔹 Reconstruction
#         T_new = U_new @ torch.diag(S) @ V_new.T
#         signal_new = T_new.T.reshape(-1)

#         # 🔹 Smoothing final
#         if self.smoothing:
#             signal_new = F.avg_pool1d(signal_new.view(1, 1, -1), kernel_size=3, stride=1, padding=1).squeeze()

#         return signal_new

#     def _rotate_on_stiefel(self, U: torch.Tensor) -> torch.Tensor:
#         """
#         Perturbe U en appliquant une rotation sur la variété de Stiefel : U2 = exp(A) * U
#         """
#         m, n = U.shape
#         # 🔹 Crée une matrice antisymétrique A (exp(A) ∈ SO(m) => groupe des rotations orthogonales)
#         A = torch.randn(m, m, device=U.device)
#         A = A - A.T  # antisymétrique

#         # 🔹 Échelle selon beta (plus petit = plus fluide)
#         A = A * self.beta

#         # 🔹 Calcul exponentielle de matrice (expm de scipy)
#         A_np = A.cpu().numpy()
#         R = torch.tensor(expm(A_np), dtype=U.dtype, device=U.device)

#         # 🔹 Nouvelle base orthogonale
#         return R @ U





#     def __repr__(self) -> str:
#         return f"{self.__class__.__name__}(beta={self.beta})"
#     def __str__(self) -> str:
#         return f"{self.__class__.__name__}(beta={self.beta})"





import torch
import torch.nn.functional as F
from typing import Optional

class StiefelGen(TimeSeriesTransform):
    def __init__(self, window_size: int = 10, beta: float = 0.05, smoothing: bool = True):
        """
        Args:
            window_size: taille m pour la page matrix (doit diviser la longueur temporelle T)
            beta: facteur de déplacement dans la variété de Stiefel (0 = aucun, 1 = max)
            smoothing: True pour lisser la série générée
        """
        assert 0 <= beta <= 1, "β doit être entre 0 et 1"
        assert window_size >= 2, "window_size doit être ≥ 2"
        self.m = window_size
        self.beta = beta
        self.smoothing = smoothing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (G, T, C) - G groupes, T temps, C bandes spectrales
        """
        G, T, C = x.shape
        output = []

        for i in range(G):  # pour chaque groupe (champ)
            transformed = []
            for c in range(C):  # pour chaque bande spectrale
                signal = x[i, :, c]  # (T,)

                # 🔹 Normalisation (z-score)
                mean, std = signal.mean(), signal.std()
                signal_norm = (signal - mean) / (std + 1e-8)

                # 🔹 Transformation Stiefel
                signal_aug = self._augment_signal(signal_norm)

                # 🔹 Dénormalisation
                signal_aug = signal_aug * std + mean
                transformed.append(signal_aug.unsqueeze(1))

            output.append(torch.cat(transformed, dim=1))  # (T, C)

        return torch.stack(output, dim=0)  # (G, T, C)

    def _augment_signal(self, signal: torch.Tensor) -> torch.Tensor:
        m = self.m
        N = signal.shape[0]
        n = N // m
        T_mat = signal[:n * m].reshape(n, m).T  # page matrix (m, n)

        # 🔹 Décomposition SVD
        U, S, Vh = torch.linalg.svd(T_mat, full_matrices=False)

        # 🔹 Perturbation géométrique
        U_new = self._stiefel_perturb(U)
        V_new = self._stiefel_perturb(Vh.T)

        # 🔹 Reconstruction
        T_mat_new = U_new @ torch.diag(S) @ V_new.T
        signal_new = T_mat_new.T.reshape(-1)

        # 🔹 Smoothing optionnel (moyenne glissante)
        if self.smoothing:
            signal_new = F.avg_pool1d(signal_new.view(1, 1, -1), kernel_size=3, stride=1, padding=1).squeeze()

        return signal_new

    def _stiefel_perturb(self, U: torch.Tensor) -> torch.Tensor:
        # 🔹 Étape 1 : vecteur tangent aléatoire
        Δ = self._random_tangent_vector(U)

        # 🔹 Étape 2 : normalisation selon le produit scalaire canonique
        Δ_norm = self._canonical_norm(U, Δ)
        Δ_normalized = Δ / Δ_norm * self.beta * (0.89 * torch.pi)

        # 🔹 Étape 3 : Retraction QR (approximation de l’exponentielle)
        return self._qr_retraction(U + Δ_normalized)

    def _random_tangent_vector(self, U: torch.Tensor) -> torch.Tensor:
        A = torch.randn_like(U)
        proj = A - U @ (U.T @ A)  # projection tangentielle
        return proj

    def _canonical_norm(self, U: torch.Tensor, Δ: torch.Tensor) -> torch.Tensor:
        I = torch.eye(U.shape[0], device=U.device)
        M = I - 0.5 * U @ U.T
        inner = torch.trace(Δ.T @ M @ Δ)
        return torch.sqrt(inner + 1e-8)

    def _qr_retraction(self, Y: torch.Tensor) -> torch.Tensor:
        # QR orthonormalise la matrice Y pour rester dans Stiefel
        Q, _ = torch.linalg.qr(Y, mode='reduced')
        return Q

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(beta={self.beta})"
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(beta={self.beta})"