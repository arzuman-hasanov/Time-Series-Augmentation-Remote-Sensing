from aug.base import *  
import numpy as np

class WindowWarping(TimeSeriesTransform):
    """Applies Window Warping (WW) transformation to time series."""
    
    def __init__(self, window_ratio: float = 0.1, scales: List[float] = [0.5, 2.]):
        """
        Args:
            window_ratio: Fraction of the time series length used for warping.
            scales: List of scaling factors to randomly apply to the window.
        """
        self.window_ratio = window_ratio
        self.scales = scales

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        G, T, C = x.shape  # G: number of samples, T: time steps, C: number of channels
        
        # Select a random scaling factor for each sample
        warp_scales = torch.tensor(np.random.choice(self.scales, G), dtype=torch.float32)
        
        # Compute the size of the window to warp (ceil ensures rounding up)
        warp_size = int(np.ceil(self.window_ratio * T))
        window_steps = torch.arange(warp_size, dtype=torch.float32)  # Steps within the warp window

        # Select random start positions for the warp window in each sample
        window_starts = torch.randint(1, T - warp_size - 1, (G,))
        window_ends = window_starts + warp_size  # Compute end positions of the window

        # Initialize the output tensor with zeros
        ret = torch.zeros_like(x)
        
        # Iterate over each sample in the batch
        for i in range(G):
            # Iterate over each channel
            for dim in range(C):
                
                # Extract the segment before the warped window
                start_seg = x[i, :window_starts[i], dim]
                
                # Apply warping by interpolating within the selected window
                window_seg = torch.from_numpy(
                    np.interp(
                        np.linspace(0, warp_size - 1, int(warp_size * warp_scales[i].item())),
                        window_steps.numpy(),
                        x[i, window_starts[i]:window_ends[i], dim].numpy()
                    )
                )
                
                # Extract the segment after the warped window
                end_seg = x[i, window_ends[i]:, dim]
                
                # Concatenate the unmodified start segment, warped segment, and end segment
                warped = torch.cat((start_seg, window_seg, end_seg))
                
                # Resample the full sequence to match the original length using interpolation
                ret[i, :, dim] = torch.from_numpy(
                    np.interp(
                        np.arange(T),  # Target time steps
                        np.linspace(0, T - 1, warped.shape[0]),  # Current warped time steps
                        warped.numpy()
                    )
                )
        
        return ret

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(window_ratio={self.window_ratio}, scales={self.scales})"
