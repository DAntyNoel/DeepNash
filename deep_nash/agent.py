"""
Meant as a container for a trained policy. Simplifies the interface for interacting
with the environment
"""
from typing import Any, Union

import numpy as np
import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule
from torch import Tensor

from deep_nash.network import DeepNashNet


class DeepNashAgent(TensorDictModule):
    def __init__(self, config=None, *args, **kwargs):
        # TODO: Add Loading from the config at some point. Device should also be a part of the config
        net = DeepNashNet(15, 16, 0, 0) # Default

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

        leading_shape = tensordict.shape

        # Ensure batch dimensions for obs and action_mask
        if len(leading_shape) == 0:
            tensordict = tensordict.unsqueeze(0)

        if len(leading_shape) > 1:
            tensordict = tensordict.reshape(-1)

        # Call the parent forward method to compute policy logits
        tensordict = super().forward(tensordict)
        shape = tensordict["obs"].shape  # Shape: (B, C, H, W)

        policy = tensordict["policy"]
        if "collector" in tensordict and "mask" in tensordict["collector"]:
            valid = tensordict["collector"]["mask"]  # boolean mask
        else:
            valid = torch.ones(policy.shape[:-1]).bool()

        # 1. Flatten if there's a complex batch shape
        #    We'll flatten everything except the last dimension (which is N).
        #    For example, if policy.shape = (B1, B2, ..., Bk, N),
        #    flatten_size = B1*B2*...*Bk.
        original_batch_size = policy.shape[:-1]  # all but the last dim
        N = policy.shape[-1]

        policy_flat = policy.view(-1, N)  # (flatten_size, N)
        valid_flat = valid.view(-1)  # (flatten_size,)

        # 2. Identify which entries are valid
        valid_indices = valid_flat.nonzero(as_tuple=True)[0]  # shape = (#valid,)

        # 3. We'll create a container for the final samples in flattened form.
        #    shape = (flatten_size, 1) because we sample 1 action index per row.
        sampled_indices_flat = torch.zeros(
            (policy_flat.size(0), 1),
            dtype=torch.long,
            device=policy_flat.device
        )

        if len(valid_indices) > 0:
            # 4. Subset only the valid rows of policy
            valid_policy: Tensor = policy_flat[valid_indices]  # (#valid, N)
            assert not (valid_policy.isinf() | valid_policy.isnan()).any()

            # 5. Sample from the subset
            sampled_valid = torch.multinomial(valid_policy, num_samples=1, replacement=True)
            # shape = (#valid, 1)

            # 6. Copy those sampled indices back into our container
            sampled_indices_flat[valid_indices] = sampled_valid

        # 7. Unflatten to original batch shape
        #    so we get (B1, B2, ..., Bk, 1)
        sampled_indices = sampled_indices_flat.view(*original_batch_size, 1)

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
        tensordict["action_one_hot"] = one_hot_encode(sampled_indices, shape[-2] * shape[-1])

        if len(leading_shape) == 0:
            tensordict = tensordict.squeeze(0)

        if len(leading_shape) > 1:
            tensordict = tensordict.view(*leading_shape)

        return tensordict