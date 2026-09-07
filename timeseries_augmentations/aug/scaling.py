from aug.base import *


class Scaling(TimeSeriesTransform):
    """Randomly scale the time series by a factor.

    This transform multiplies the input by a random scaling factor sampled from
    a uniform distribution between [1-magnitude, 1+magnitude].
    """

    def __init__(self, magnitude: float = 0.2):
        """Initialize the scaling transform.

        Args:
            magnitude: Maximum scaling deviation from 1. The actual scaling factor
                      will be uniformly sampled from [1-magnitude, 1+magnitude].
                      Must be in (0, 1) to ensure positive scaling factors.
        """
        if not 0 < magnitude < 1:
            raise ValueError(f"magnitude must be between 0 and 1, got {magnitude}")
        self.magnitude = magnitude

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sample scaling factor for each sample in the group
        scale = (
            1.0
            + (2 * torch.rand(x.shape[0], 1, 1, device=x.device) - 1) * self.magnitude
        )
        return x * scale

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(magnitude={self.magnitude})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(m={self.magnitude:.3f})"