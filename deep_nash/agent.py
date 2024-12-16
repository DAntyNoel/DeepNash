"""
Meant as a container for a trained policy. Simplifies the interface for interacting
with the environment
"""
import numpy as np
import torch

from deep_nash.network import DeepNashNet
from deep_nash.utils import batch_info_dicts


class DeepNashAgent:
    def __init__(self, device, config=None):
        if config:
            pass
        else:
            self.policy = DeepNashNet(10, 20, 0, 0) # Default

        self.policy.to(device)
        self.device = device


    def get_action(self, obs, mask, info):
        obs = obs[None, :]
        mask = mask[None, :]

        with torch.no_grad():
            # Convert observations to tensor
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            info_batch = batch_info_dicts([info])

            # Forward pass
            (deploy_policy, _), (select_policy, _), (move_policy, _) = self.policy.forward(obs_t, info_batch)

            # Extract phases and shape info
            game_phase = info_batch["game_phase"][0]
            H, W = info_batch["board_shape"][0]

            # Convert policies to numpy and select the appropriate policy per environment
            deploy_np = deploy_policy.squeeze(1).cpu().numpy()
            select_np = select_policy.squeeze(1).cpu().numpy()
            move_np = move_policy.squeeze(1).cpu().numpy()

            if info_batch["game_phase"][0] == 0:
                actions = deploy_np
            elif info_batch["game_phase"][0] == 1:
                actions = select_np
            else:
                actions = move_np

            # Apply masks and normalize
            actions *= mask
            sums = actions.sum(axis=(1, 2), keepdims=True)
            sums[sums < 1e-8] = 1e-8
            actions /= sums

            # Sample actions
            flat_actions = actions.reshape(1, -1)
            rand_vals = np.random.rand(1)
            cumulative_sums = np.cumsum(flat_actions, axis=1)
            sampled_indices = np.argmax(cumulative_sums >= rand_vals[:, None], axis=1)

            rows, cols = divmod(sampled_indices, W)
            action_coords = np.stack([rows, cols], axis=1)
            return tuple(action_coords[0])