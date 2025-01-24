import numpy as np
import gymnasium
from gymnasium.spaces import MultiDiscrete

class MaskedMultiDiscrete(MultiDiscrete):
    """
    A custom MultiDiscrete that supports masking during sampling.
    If self.mask is set, sample() will only return valid indices.
    Otherwise, it falls back to full unmasked sampling.
    """

    def __init__(self, nvec, dtype=np.int64):
        """
        Args:
            nvec (tuple or list): The shape or bounds for each dimension,
                e.g., for a 2D grid of shape (4,4), pass nvec=(4,4).
            dtype: The NumPy dtype for the space (default is np.int64).
        """
        super().__init__(nvec, dtype=dtype)

        # We'll store a boolean mask here. The environment is responsible
        # for calling set_mask(...) whenever it changes.
        self.mask = None

        # If you want to pre-allocate a buffer for large spaces,
        # you can do so here to avoid frequent allocations.
        # Example: self.valid_buffer = np.empty(np.prod(nvec), dtype=np.int64)

    def set_mask(self, mask: np.ndarray):
        """
        Updates the current action mask.

        Args:
            mask (np.ndarray): A boolean array of the same shape as self.nvec.
                               True indicates valid actions, False invalid.
        """
        if mask.shape != tuple(self.nvec):
            raise ValueError(f"Mask shape {mask.shape} must match {self.nvec}")
        if mask.dtype != bool:
            raise ValueError("Mask must be a boolean array.")
        self.mask = mask

    def sample(self):
        """
        Samples from the valid actions if a mask is set, otherwise samples
        from the full space.

        Returns:
            A NumPy array of shape (len(nvec),) representing the sampled action.
            e.g. for a 2D grid, returns [row, col].
        """
        # If no mask, sample from entire space
        if self.mask is None:
            return super().sample()

        # 1) Flatten the mask and find indices of valid cells
        valid_indices = np.flatnonzero(self.mask)
        if len(valid_indices) == 0:
            raise RuntimeError("No valid actions in the mask!")

        # 2) Randomly pick one valid index among them
        idx = self.np_random.integers(0, len(valid_indices))
        flat_idx = valid_indices[idx]

        # 3) If nvec is 2D (like a grid), decode row/col
        #    For general MultiDiscrete shapes, you'd do something like
        #    np.unravel_index(flat_idx, self.nvec).
        if len(self.nvec) == 2:
            h, w = self.nvec
            row = flat_idx // w
            col = flat_idx % w
            return np.array([row, col], dtype=self.dtype)

        # For higher dimensions, e.g. nvec=(d1, d2, d3,...)
        return np.array(np.unravel_index(flat_idx, self.nvec), dtype=self.dtype)
