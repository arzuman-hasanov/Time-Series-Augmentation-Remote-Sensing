from timeseries_augmentations.aug.base import TimeSeriesTransform
import torch
import numpy as np

class WindowSlice(TimeSeriesTransform):
    """Applies Window Slicing (WS) transformation to time series."""
    
    def __init__(self, reduce_ratio: float = 0.9):
        """
        Args:
            reduce_ratio: Fraction of the original time series length to retain.
        """
        if not (0 < reduce_ratio <= 1):
            raise ValueError("reduce_ratio must be in the range (0,1]")
        self.reduce_ratio = reduce_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        G, T, C = x.shape
        target_len = int(np.ceil(self.reduce_ratio * T))
        
        if target_len >= T:
            return x  # No slicing needed
        
        starts = torch.randint(0, T - target_len, (G,))
        ends = starts + target_len

        ret = torch.zeros_like(x)
        for i in range(G):
            for dim in range(C):
                sliced = x[i, starts[i]:ends[i], dim]
                ret[i, :, dim] = torch.from_numpy(
                    np.interp(
                        np.linspace(0, target_len - 1, num=T),
                        np.arange(target_len),
                        sliced.numpy()
                    )
                )
        return ret

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(reduce_ratio={self.reduce_ratio})"
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(reduce_ratio={self.reduce_ratio})"