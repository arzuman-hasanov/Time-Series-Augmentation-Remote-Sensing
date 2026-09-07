from aug.base import *


class Permutation(TimeSeriesTransform):
    """Randomly permute segments of the time series.

    This transform divides the time series into n_segments of equal length and
    randomly permutes a subset of these segments. This creates a new sequence that
    preserves some of the original temporal structure while modifying others.

    Note:
        This transform preserves local temporal structure within segments while
        disrupting the global temporal structure. The segment_size parameter
        controls this trade-off: larger segments preserve more local structure
        but allow for fewer permutations.

        When n_permute < n_segments, only a random subset of segments will be
        permuted, while others remain in their original positions. This allows
        for more subtle temporal modifications.
    """

    def __init__(self, n_segments: int = 5, n_permute: Optional[int] = None):
        """Initialize the permutation transform.

        Args:
            n_segments: Number of segments to divide the sequence into.
                       Must be at least 2 (otherwise no permutation is possible).
            n_permute: Number of segments to permute. Must be at least 2 and not greater
                      than n_segments. If None, all segments will be permuted.
        """
        if n_segments < 2:
            raise ValueError(
                f"n_segments must be at least 2 for permutation, got {n_segments}"
            )

        # If n_permute is not specified, permute all segments
        n_permute = n_permute if n_permute is not None else n_segments

        if n_permute < 2:
            raise ValueError(
                f"n_permute must be at least 2 for permutation, got {n_permute}"
            )
        if n_permute > n_segments:
            raise ValueError(
                f"n_permute ({n_permute}) cannot be greater than n_segments ({n_segments})"
            )

        self.n_segments = n_segments
        self.n_permute = n_permute

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        group_size, time_steps, channels = x.shape

        # Calculate segment size (handle cases where time_steps isn't perfectly divisible)
        segment_size = time_steps // self.n_segments
        if segment_size == 0:
            raise ValueError(
                f"Time series length ({time_steps}) is too short to be divided "
                f"into {self.n_segments} segments"
            )

        # Adjust n_segments if necessary to handle remainder
        actual_n_segments = time_steps // segment_size
        if actual_n_segments < 2:
            raise ValueError(
                f"Time series length ({time_steps}) is too short for permutation "
                f"with segment size {segment_size}"
            )

        # Adjust n_permute if necessary
        actual_n_permute = min(self.n_permute, actual_n_segments)

        # Prepare output tensor
        out = x.clone()  # Start with a copy since we might keep some segments unchanged

        # Process each sample in the group
        for i in range(group_size):
            # Randomly select segments to permute
            segment_indices = torch.randperm(actual_n_segments)[:actual_n_permute]
            # Generate permutation for selected segments
            perm = torch.randperm(actual_n_permute)

            # Create temporary storage for segments to be permuted
            temp = torch.zeros(
                (actual_n_permute, segment_size, channels),
                device=x.device,
                dtype=x.dtype,
            )

            # Collect segments to be permuted
            for j, idx in enumerate(segment_indices):
                start = idx * segment_size
                end = start + segment_size
                temp[j] = x[i, start:end]

            # Place permuted segments back
            for j, idx in enumerate(segment_indices):
                start = idx * segment_size
                end = start + segment_size
                out[i, start:end] = temp[perm[j]]

        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_segments={self.n_segments}, n_permute={self.n_permute})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(n={self.n_segments}, p={self.n_permute})"
