from timeseries_augmentations.aug.base import *

class Compose:
    """Composes several transforms together to be applied sequentially.

    This class allows you to chain multiple transforms together in a fixed order.
    Each transform is applied to the output of the previous transform.

    Example:
        >>> transforms = Compose([
        ...     Jittering(sigma=0.1),
        ...     AnotherTransform(),
        ... ])
        >>> output = transforms(input)  # Applies transforms in order
    """

    def __init__(self, transforms: List[TimeSeriesTransform]):
        """Initialize the compose transform.

        Args:
            transforms: List of transforms to apply in order. Must not be empty.
        """
        if not transforms:
            raise ValueError("transforms list must not be empty")
        if not all(isinstance(t, TimeSeriesTransform) for t in transforms):
            raise TypeError("All transforms must inherit from TimeSeriesTransform")
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {repr(t)}"
        format_string += "\n)"
        return format_string

    def __str__(self) -> str:
        return " → ".join(str(t) for t in self.transforms)
