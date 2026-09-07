from aug.base import *
from scipy.interpolate import CubicSpline

class MagnitudeWarping(TimeSeriesTransform):
    """Apply magnitude warping transformation using cubic spline interpolation."""

    def __init__(self, sigma: float = 0.2, knot: int = 4):
        """
        Args:
            sigma: Standard deviation of the random warp.
            knot: Number of control points for the spline interpolation.
        """
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        if knot < 2:
            raise ValueError(f"knot must be at least 2, got {knot}")
        
        self.sigma = sigma
        self.knot = knot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.cpu().numpy()
        orig_steps = np.arange(x_np.shape[1])
        
        random_warps = np.random.normal(loc=1.0, scale=self.sigma, size=(x_np.shape[0], self.knot + 2, x_np.shape[2]))
        warp_steps = np.linspace(0, x_np.shape[1] - 1, num=self.knot + 2).reshape(-1, 1)
        
        ret = np.zeros_like(x_np)
        for i, pat in enumerate(x_np):
            warper = np.array([CubicSpline(warp_steps.flatten(), random_warps[i, :, dim])(orig_steps) for dim in range(x_np.shape[2])]).T
            ret[i] = pat * warper
        
        return torch.tensor(ret, dtype=x.dtype, device=x.device)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sigma={self.sigma}, knot={self.knot})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(σ={self.sigma:.3f}, knot={self.knot})"
