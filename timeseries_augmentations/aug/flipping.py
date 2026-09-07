from aug.base import *  

import torch
class Flipping(TimeSeriesTransform):
    """Flip the time series along the time axis.

    This transform randomly flips the time series horizontally (along time axis)
    with a given probability. When applied, it reverses the temporal order of
    the sequence while maintaining the shape and scale of the data.

    Note:
        This transform should be used with caution as it may not be appropriate
        for all types of time series data. For example, in cases where the temporal
        ordering is crucial for the task (e.g., forecasting), flipping might
        destroy important patterns in the data.
    """

    def __init__(self, p: float = 0.5):
        """Initialize the flipping transform.

        Args:
            p: Probability of applying the flip. Must be between 0 and 1.
        """
        if not 0 <= p <= 1:
            raise ValueError(f"p must be between 0 and 1, got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sample a mask of shape (group_size, 1, 1) to decide which sequences to flip
        flip_mask = torch.rand(x.shape[0], 1, 1, device=x.device) < self.p
        # Flip along time dimension (dim=1) where mask is True
        x = torch.where(
            flip_mask,
            x.flip(dims=[1]),  # Flipped version
            x,  # Original version
        )
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p:.2f})"