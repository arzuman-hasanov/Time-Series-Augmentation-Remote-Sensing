from aug.base import *

class MixUp(TimeSeriesTransform):
    """Applies MixUp augmentation to time series data."""

    def __init__(self, alpha: float = 0.4):
        """
        Args:
            alpha (float): Parameter for Beta distribution that controls mixing strength.
        """
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply MixUp to a batch of time series.
        
        Args:
            x: Tensor of shape (batch, time_steps, channels)
        
        Returns:
            Mixed tensor with the same shape.
        """
        batch_size = x.shape[0]
        # Sample lambda from Beta distribution
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((batch_size,)).to(x.device)
        lam = lam.view(batch_size, 1, 1)  # Reshape for broadcasting
        
        # Generate shuffled indices to mix different samples
        indices = torch.randperm(batch_size, device=x.device)
        x_shuffled = x[indices]  # Shuffle the batch
        
        # Compute the mixed sample
        x_mixed = lam * x + (1 - lam) * x_shuffled
        
        return x_mixed

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alpha={self.alpha})"


