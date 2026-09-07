from timeseries_augmentations.aug.base import TimeSeriesTransform

import torch
class Jittering(TimeSeriesTransform):
    """Add Gaussian noise to the input for jittering effect.

    This transform adds random noise sampled from N(0, sigma²) to create
    a jittering effect that helps with robustness to noise in the data.

    Important:
        This transform assumes the input data is normalized with approximately unit
        standard deviation. If your data has a different scale, the sigma parameter
        should be adjusted accordingly, or the jittering effect might be too strong
        or too weak.

        In the context of this codebase, this transform is designed to be applied
        after the normalization step in SSLGroupedTimeSeriesDataset, which normalizes
        the data using pre-computed percentiles.
    """

    def __init__(self, sigma: float = 0.9):
        """Initialize the jittering transform.

        Args:
            sigma: Standard deviation of the Gaussian noise relative to unit-scaled data.
                  Must be non-negative. A value of 0.1 means the noise standard deviation
                  will be 10% of the data's expected standard deviation.
        """
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)  # Convertir numpy.ndarray en torch.Tensor
        noise = torch.randn_like(x) * self.sigma

        return x + noise

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sigma={self.sigma})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(σ={self.sigma:.3f})"
    