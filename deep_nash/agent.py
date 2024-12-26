"""
Meant as a container for a trained policy. Simplifies the interface for interacting
with the environment
"""
from typing import Any, Union

import numpy as np
import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule

from deep_nash.network import DeepNashNet


class DeepNashAgent(TensorDictModule):
    def __init__(self, config=None, *args, **kwargs):
        # TODO: Add Loading from the config at some point. Device should also be a part of the config
        net = DeepNashNet(10, 20, 0, 0) # Default

        # Call TensorDictModule constructor
        super().__init__(
            module=net,
            in_keys=["obs", "action_mask"],  # Input key from TensorDict
            out_keys=["policy", "value", "log_probs", "logits"]  # Output key to TensorDict
        )

    def forward(
        self,
        tensordict: TensorDictBase,
        *args,
        tensordict_out: Union[TensorDictBase, None] = None,
        **kwargs: Any,
    ) -> TensorDictBase:
        # Ensure batch dimensions for obs and action_mask
        if len(tensordict["obs"].shape) == 3:
            tensordict["obs"] = tensordict["obs"].unsqueeze(0)

        if len(tensordict["action_mask"].shape) == 2:
            tensordict["action_mask"] = tensordict["action_mask"].unsqueeze(0)

        # Call the parent forward method to compute policy logits
        tensordict = super().forward(tensordict)
        shape = tensordict["obs"].shape  # Shape: (B, C, H, W)

        # Sample actions using policy logits
        sampled_indices = torch.multinomial(tensordict["policy"], num_samples=1, replacement=True)  # Shape: (B, 1)

        # Convert sampled indices into rows and columns
        rows = sampled_indices // shape[-1]  # Shape: (B, 1)
        cols = sampled_indices % shape[-1]  # Shape: (B, 1)

        # Generate one-hot encodings for rows and columns
        def one_hot_encode(indices, size):
            one_hot = torch.zeros(indices.size(0), size, device=indices.device)
            return one_hot.scatter_(1, indices, 1)

        # One-hot encode rows and columns
        rows_onehot = one_hot_encode(rows, shape[-2])  # One-hot encoding for rows
        cols_onehot = one_hot_encode(cols, shape[-1])  # One-hot encoding for columns

        # Pad one-hot encodings to max(H, W) for rectangular grids
        max_dim = max(shape[-2], shape[-1])
        rows_onehot_padded = torch.nn.functional.pad(rows_onehot, (0, max_dim - rows_onehot.shape[1]))
        cols_onehot_padded = torch.nn.functional.pad(cols_onehot, (0, max_dim - cols_onehot.shape[1]))

        # Stack padded one-hot encodings along a new dimension
        action_onehot = torch.stack((rows_onehot_padded, cols_onehot_padded), dim=1)  # Shape: (B, 2, max(H, W))

        # Add the one-hot encoded action to the TensorDict
        tensordict["action"] = action_onehot

        return tensordict