import torch.nn as nn
from aug.base import *
from timeseries_augmentations.aug.base import TimeSeriesTransform  # Ensure correct import path


class MaskedTimeSeries(TimeSeriesTransform):
    """Applies Masked Signal Modeling (MST) transformation to time series."""
    
    def __init__(self, mask_ratio: float = 0.2):
        """
        Args:
            mask_ratio: Fraction of the time series to mask. E.g., 0.2 will mask 20% of the data.
        """
        if not (0 < mask_ratio <= 1):
            raise ValueError("mask_ratio must be in the range (0,1]")
        self.mask_ratio = mask_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, time_steps, features) where batch_size is the number of instances,
               time_steps is the length of the time series, and features is the number of features at each timestep.
        
        Returns:
            A tensor with some parts masked, where the model is supposed to reconstruct the missing parts.
        """
        G, T, C = x.shape  # G: batch size, T: time steps, C: channels
        
        # Create a mask to hide portions of the time series
        mask = torch.ones_like(x)
        
        # Calculate how many time steps to mask
        mask_count = int(T * self.mask_ratio)
        
        # For each batch, choose random time steps to mask
        for i in range(G):
            mask_idx = torch.randperm(T)[:mask_count]  # Random indices to mask
            mask[i, mask_idx, :] = 0  # Set those positions to 0 in the mask
        
        # Apply the mask to the input data
        masked_data = x * mask  # This zeros out the selected portions of the input
        
        return masked_data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mask_ratio={self.mask_ratio})"

