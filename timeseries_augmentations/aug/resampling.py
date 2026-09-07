from aug.base import *  

import torch
from typing import Union


class Resampling(TimeSeriesTransform):
    """Resampling augmentation that processes a single view.

    This transform performs temporal resampling in three steps:
    1. Upsamples the original time series to T_up timesteps using linear interpolation
    2. Samples one subsequence that maintains temporal coverage
    3. Resamples the subsequence back to the original temporal resolution
    """

    def __init__(
        self,
        upsampling_factor: float = 2.0,
        subsequence_length_ratio: float = 0.5,
    ):
        """Initialize the resampling augmentation.

        Args:
            upsampling_factor: Factor to upsample the original time series (typically 2.0)
            subsequence_length_ratio: Length of the subsequence as a ratio of upsampled length (typically 0.5)
        """
        if upsampling_factor <= 1.0:
            raise ValueError(
                f"upsampling_factor must be > 1.0, got {upsampling_factor}"
            )
        if not 0.0 < subsequence_length_ratio < 1.0:
            raise ValueError(
                f"subsequence_length_ratio must be between 0 and 1, got {subsequence_length_ratio}"
            )

        self.upsampling_factor = upsampling_factor
        self.subsequence_length_ratio = subsequence_length_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the resampling augmentation to a single tensor.

        Args:
            x: A tensor of shape (G, T, C), where
                - G is the group size (batch)
                - T is the number of timesteps
                - C is the number of channels

        Returns:
            A tensor of the same shape (G, T, C) after augmentation.
        """
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"Input must be a torch.Tensor, got {type(x)}")

        group_size, time_steps, channels = x.shape

        # Step 1: Upsample to higher temporal resolution
        up_steps = int(time_steps * self.upsampling_factor)
        x_up = torch.nn.functional.interpolate(
            x.transpose(1, 2),  # (G, C, T)
            size=up_steps,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)  # Back to (G, T, C)

        # Step 2: Sample subsequence
        subseq_length = int(up_steps * self.subsequence_length_ratio)
        quarters = up_steps // 4
        points_per_quarter = subseq_length // 4

        all_indices = torch.randperm(up_steps)
        indices = []

        for q in range(4):
            quarter_start = q * quarters
            quarter_end = (q + 1) * quarters
            quarter_indices = all_indices[
                (all_indices >= quarter_start) & (all_indices < quarter_end)
            ]
            indices.extend(quarter_indices[:points_per_quarter].tolist())

        indices.sort()

        # Extract subsequence
        subseq = x_up[:, indices]  # (G, subseq_length, C)

        # Step 3: Resample back to original resolution
        x_aug = torch.nn.functional.interpolate(
            subseq.transpose(1, 2),
            size=time_steps,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)  # (G, T, C)

        return x_aug

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"upsampling_factor={self.upsampling_factor}, "
            f"subsequence_length_ratio={self.subsequence_length_ratio})"
        )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"up={self.upsampling_factor:.1f}, "
            f"len={self.subsequence_length_ratio:.2f})"
        )
