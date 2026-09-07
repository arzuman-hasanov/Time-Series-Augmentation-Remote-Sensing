from aug.base import *  

class Resizing(TimeSeriesTransform):
    """Resize the time series by cropping and scaling.

    This transform changes the temporal resolution of the time series by randomly
    cropping a portion of it and then scaling that portion back to the original length.
    The crop size is determined by a factor sampled uniformly from [1-magnitude, 1].
    A resize factor of 1 means no cropping, while smaller values mean taking a smaller crop.

    Note:
        This transform first crops a portion of the time series and then uses linear
        interpolation to scale it back to the original length. This creates a different
        effect than pure interpolation-based resizing, as it focuses on a specific
        temporal segment of the data.
    """

    def __init__(self, magnitude: float = 0.2):
        """Initialize the resizing transform.

        Args:
            magnitude: Maximum amount to crop. The resize factor will be sampled
                      uniformly from [1-magnitude, 1]. Must be between 0 and 1.
                      A larger magnitude means more aggressive cropping.
        """
        if not 0 < magnitude < 1:
            raise ValueError(f"magnitude must be between 0 and 1, got {magnitude}")
        self.magnitude = magnitude

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original shape
        group_size, time_steps, channels = x.shape

        # Sample resize factor between [1-magnitude, 1]
        # This is the portion of the sequence to keep
        resize_factor = 1.0 - torch.rand(group_size, device=x.device) * self.magnitude

        # Calculate crop sizes - ensure at least 1 timestep
        crop_sizes = (time_steps * resize_factor).round().long().clamp(min=1)

        # Prepare output tensor
        out = torch.zeros_like(x)

        # Process each sample in the group
        for i in range(group_size):
            crop_size = crop_sizes[i].item()

            # Randomly select start position for crop
            if crop_size < time_steps:
                start = torch.randint(0, time_steps - crop_size + 1, (1,)).item()
            else:
                start = 0
                crop_size = time_steps

            # Extract crop
            crop = x[
                i : i + 1, start : start + crop_size
            ]  # Keep batch dim for interpolate

            # Scale back to original size
            temp = F.interpolate(
                crop.transpose(1, 2),  # Shape: (1, C, crop_size)
                size=time_steps,
                mode="linear",
                align_corners=True,
            )

            out[i] = temp.transpose(1, 2).squeeze(0)

        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(magnitude={self.magnitude})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(m={self.magnitude:.3f})"
