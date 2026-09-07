from aug.base import *
from scipy.interpolate import CubicSpline
import utils.dtw as dtw
from tqdm import tqdm


class Spawner(TimeSeriesTransform):
    """Apply SPAWNER augmentation to time series for contrastive learning."""

    def __init__(self, sigma: float = 0.05, window_fraction: float = 0.1, verbose: int = 0):
        """
        Args:
            sigma: Standard deviation for jittering after SPAWNER.
            window_fraction: Fraction of the time series length to use for DTW window.
            verbose: Level of verbosity (-1 for silent, 1 for debug visualization).
        """
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        if not (0 < window_fraction <= 1):
            raise ValueError(f"window_fraction must be in (0,1], got {window_fraction}")
        
        self.sigma = sigma
        self.window_fraction = window_fraction
        self.verbose = verbose

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.cpu().numpy()
        random_points = np.random.randint(low=1, high=x_np.shape[1]-1, size=x_np.shape[0])
        window = np.ceil(x_np.shape[1] * self.window_fraction).astype(int)
        orig_steps = np.arange(x_np.shape[1])
        
        ret = np.zeros_like(x_np)
        for i, pat in enumerate(tqdm(x_np)):
            choices = np.delete(np.arange(x_np.shape[0]), i)
            if choices.size > 0:
                random_sample = x_np[np.random.choice(choices)]
                path1 = dtw.dtw(pat[:random_points[i]], random_sample[:random_points[i]], dtw.RETURN_PATH, slope_constraint="symmetric", window=window)
                path2 = dtw.dtw(pat[random_points[i]:], random_sample[random_points[i]:], dtw.RETURN_PATH, slope_constraint="symmetric", window=window)
                combined = np.concatenate((np.vstack(path1), np.vstack(path2+random_points[i])), axis=1)
                mean = np.mean([pat[combined[0]], random_sample[combined[1]]], axis=0)
                for dim in range(x_np.shape[2]):
                    ret[i,:,dim] = np.interp(orig_steps, np.linspace(0, x_np.shape[1]-1., num=mean.shape[0]), mean[:,dim]).T
            else:
                if self.verbose > -1:
                    print(f"Skipping pattern {i}, no matching sample found.")
                ret[i,:] = pat
        
        return torch.tensor(ret, dtype=x.dtype, device=x.device)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sigma={self.sigma}, window_fraction={self.window_fraction}, verbose={self.verbose})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(σ={self.sigma:.3f}, window_fraction={self.window_fraction})"
