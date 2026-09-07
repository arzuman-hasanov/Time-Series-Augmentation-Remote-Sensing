from timeseries_augmentations.aug.base import TimeSeriesTransform


from typing import List, Literal, Optional, Tuple
import torch
import torch.nn.functional as F
from geomstats.geometry.stiefel import Stiefel, StiefelCanonicalMetric

class StiefelGen(TimeSeriesTransform):
    """Stiefel manifold-based time series augmentation for FranceCrops data.
    
    Perturbs input time series along geodesics on the Stiefel manifold to preserve
    orthonormality constraints while introducing realistic variations.
    
    Args:
        beta (float): Perturbation scale (0=no perturbation, 1=maximum perturbation).
        bands_dim (int): Dimension index for spectral bands (default: -1).
        apply_to (str): Whether to perturb 'time' (temporal) or 'bands' (spectral) modes.
    """
    
    def __init__(self, beta: float = 0.3, bands_dim: int = -1, apply_to: str = 'time'):
        if not 0 <= beta <= 1:
            raise ValueError(f"beta must be in [0,1], got {beta}")
        if apply_to not in ['time', 'bands']:
            raise ValueError(f"apply_to must be 'time' or 'bands', got {apply_to}")
            
        self.beta = beta
        self.bands_dim = bands_dim
        self.apply_to = apply_to
        self.inj_radius = 0.89 * torch.pi  # Global injectivity radius for Stiefel
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Stiefel perturbation to input tensor of shape (G, T, C).
        
        Args:
            x: Input tensor with shape (group_size, timesteps, bands)
            
        Returns:
            Perturbed tensor with same shape as input
        """
        if self.beta == 0:
            return x
            
        # Reshape based on perturbation mode
        if self.apply_to == 'time':
            # Reshape to (G*C, T) for temporal perturbation
            G, T, C = x.shape
            x_reshaped = x.permute(0, 2, 1).reshape(-1, T)  # (G*C, T)
        else:
            # Reshape to (G*T, C) for spectral band perturbation
            G, T, C = x.shape
            x_reshaped = x.reshape(-1, C)  # (G*T, C)
        
        # Process each slice with Stiefel perturbation
        x_perturbed = torch.stack([
            self._perturb_slice(slice_2d) 
            for slice_2d in x_reshaped.unbind(0)
        ])
        
        # Reshape back to original dimensions
        if self.apply_to == 'time':
            x_perturbed = x_perturbed.reshape(G, C, T).permute(0, 2, 1)
        else:
            x_perturbed = x_perturbed.reshape(G, T, C)
            
        return x_perturbed
    
    def _perturb_slice(self, x: torch.Tensor) -> torch.Tensor:
        # Compute SVD
        U, S, Vh = torch.linalg.svd(x.unsqueeze(-1), full_matrices=False)
        U = U.squeeze(-1)  # Shape: (T,) or (C,)

        # Initialize Stiefel manifold and metric
        stiefel = Stiefel(U.shape[0], 1)
        metric = stiefel.metric

        # Generate random tangent vector orthogonal to U
        tangent_vec = torch.randn_like(U)
        tangent_vec = tangent_vec - tangent_vec.dot(U) * U

        # Scale tangent vector
        tangent_vec = self.beta * self.inj_radius * F.normalize(tangent_vec, p=2, dim=0)

        # Exponential map on the manifold
        U_perturbed = metric.exp(tangent_vec.unsqueeze(-1), U.unsqueeze(-1)).squeeze(-1)

        # Reconstruct signal
        return U_perturbed * S[0] * Vh[0, 0]

    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(beta={self.beta}, apply_to='{self.apply_to}')"
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(β={self.beta:.2f}, mode={self.apply_to})"