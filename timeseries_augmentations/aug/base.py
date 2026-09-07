from typing import List, Literal, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F



class TimeSeriesTransform:
    """Base class for all time series transforms.

    All transforms should inherit from this class and implement the forward method.
    The base class handles input validation and provides a standard interface.

    TODO(augmentations):
        Currently, each transform treats each sample in the group independently.
        We might want to add support for group-level operations where the same
        transform is applied consistently across all samples in a group.
        This would require:
        1. A way to specify if a transform should operate at group or sample level
        2. Different implementations for group-level operations
        3. Careful consideration of how this interacts with the sampling strategy
    """

    def validate_input(self, x: torch.Tensor) -> None:
        """Validate input tensor shape and values

        Args:
            x: Input tensor expected to be of shape (G, T, C) where:
               - G is the group size (number of samples in the group)
               - T is the number of timesteps
               - C is the number of channels
        """
        if not (
            isinstance(x, torch.Tensor)
            or (isinstance(x, list) and all(isinstance(e, torch.Tensor) for e in x))
        ):
            raise TypeError(
                f"Input must be a torch.Tensor or list of torch.Tensor, got {type(x)}"
            )

        if (isinstance(x, torch.Tensor) and x.dim() != 3) or (
            isinstance(x, list) and not all(e.dim() == 3 for e in x)
        ):
            raise ValueError(
                f"Input tensor must have 3 dimensions (G, T, C), got shape {x.shape}"
            )

        if (isinstance(x, torch.Tensor) and not torch.isfinite(x).all()) or (
            isinstance(x, list) and not all(torch.isfinite(e).all() for e in x)
        ):
            raise ValueError("Input tensor contains NaN or infinite values")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.validate_input(x)
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform implementation to be defined by subclasses"""
        raise NotImplementedError

    def __repr__(self) -> str:
        """Detailed string representation for debugging"""
        return self.__class__.__name__

    def __str__(self) -> str:
        """Simple string representation for logging"""
        return self.__class__.__name__


