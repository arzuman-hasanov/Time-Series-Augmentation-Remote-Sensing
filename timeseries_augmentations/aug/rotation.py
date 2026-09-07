from aug.base import *

class Rotation(TimeSeriesTransform):
    """
    Applies a random rotation matrix to the feature space (D-dimensions).
    Each time step is rotated in the D-dimensional feature space.
    """

    def __init__(self, seed: int = None):
        """
        Args:
            seed: Optional random seed for reproducibility.
        """
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        B, T, D = x.shape

        # Generate a random rotation matrix (D x D)
        A = torch.randn(D, D)
        Q, R = torch.linalg.qr(A)  # QR decomposition to get orthonormal matrix
        rotation_matrix = Q

        # Apply rotation: each time step (dim D) gets rotated
        x_rotated = torch.matmul(x, rotation_matrix)  # (B, T, D) x (D, D) => (B, T, D)

        return x_rotated

    def __repr__(self):
        return f"{self.__class__.__name__}(seed={self.seed})"

    def __str__(self):
        return f"{self.__class__.__name__}(seed={self.seed})"
