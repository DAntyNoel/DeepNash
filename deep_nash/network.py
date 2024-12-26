import numpy as np
import torch
import torch.nn as nn

from stratego_gym.envs.stratego import GAME_CONFIG_4x4, FLAG, BOMB

class ConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvResBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    '''
    The forward method of the ConvResBlock takes an input tensor x
    and returns the output and skip-out connection
    '''
    def forward(self, x):
        residual = x
        out = self.layers(x) + self.res_conv(residual)
        return out, residual


class DeconvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DeconvResBlock, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels // 2, kernel_size=3, stride=stride, padding=1)
        self.relu1 = nn.ReLU()
        self.skip_in = nn.LazyConvTranspose2d(out_channels // 2, kernel_size=1, stride=stride)
        self.deconv2 = nn.ConvTranspose2d(out_channels // 2, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x, skip_in=None):
        x = self.deconv1(x)
        x = self.relu1(x)
        if skip_in is not None:
            x += self.skip_in(skip_in)
        x = self.deconv2(x)
        x = self.relu2(x)
        return x

class PyramidModule(nn.Module):
    def __init__(self, inner_channels, outer_channels, N, M, stride=1):
        super(PyramidModule, self).__init__()
        self.initial_conv = nn.LazyConv2d(outer_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.outer_conv_blocks = nn.ModuleList([ConvResBlock(outer_channels, outer_channels) for _ in range(N)])
        self.strided_conv = ConvResBlock(outer_channels, inner_channels, stride=stride)
        self.inner_conv_blocks = nn.ModuleList([ConvResBlock(inner_channels, inner_channels) for _ in range(M)])
        self.inner_deconv_blocks = nn.ModuleList([DeconvResBlock(inner_channels, inner_channels) for _ in range(M)])
        self.strided_deconv = DeconvResBlock(inner_channels, outer_channels, stride=stride)
        self.outer_deconv_blocks = nn.ModuleList([DeconvResBlock(outer_channels, outer_channels) for _ in range(N)])

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.relu(x)
        skip_connections = []
        for block in self.outer_conv_blocks:
            x, res = block(x)
            skip_connections.append(res)
        x, res = self.strided_conv(x)
        skip_connections.append(res)
        for block in self.inner_conv_blocks:
            x, res = block(x)
            skip_connections.append(res)
        for block in self.inner_deconv_blocks:
            x = block(x, skip_connections.pop())
        x = self.strided_deconv(x, skip_connections.pop())
        for block in self.outer_deconv_blocks:
            x = block(x, skip_connections.pop())
        return x

class DeepNashNet(nn.Module):
    def __init__(self, inner_channels, outer_channels, N, M, game_config=None):
        super(DeepNashNet, self).__init__()

        self.pyramid = PyramidModule(inner_channels, outer_channels, N, M)
        self.deployment_head = nn.Sequential(
            PyramidModule(inner_channels, outer_channels, 1, 0),
            nn.Conv2d(outer_channels,1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(start_dim=-3, end_dim=-1),
        )
        self.selection_head = nn.Sequential(
            PyramidModule(inner_channels, outer_channels, 1, 0),
            nn.Conv2d(outer_channels,1, kernel_size=3, padding=1),
            nn.Flatten(start_dim=-3, end_dim=-1),
        )
        self.movement_head = nn.Sequential(
            PyramidModule(inner_channels, outer_channels, 1, 0),
            nn.Conv2d(outer_channels,1, kernel_size=3, padding=1),
            nn.Flatten(start_dim=-3, end_dim=-1),
        )
        self.value_head = nn.Sequential(
            PyramidModule(inner_channels, outer_channels, 0, 0),
            nn.Conv2d(outer_channels,1, kernel_size=3, padding=1),
            nn.Flatten(start_dim=-3, end_dim=-1),
            nn.LazyLinear(1)
        )

        self.device = next(self.parameters()).device

        if game_config is None:
            game_config = GAME_CONFIG_4x4

        self.pieces = np.array(list(game_config['pieces']))
        invalid_piece_ids = [FLAG, BOMB]  # Define invalid piece IDs
        valid_piece_ids = np.setdiff1d(self.pieces, invalid_piece_ids)  # Exclude invalid pieces
        self.valid_mask = torch.tensor([id_ in valid_piece_ids for id_ in self.pieces], dtype=torch.bool, device=self.device,
                                       requires_grad=False)  # Create a mask for valid piece IDs

    def forward(self, state, action_mask):
        # state shape: (..., C, H, W), mask shape: (..., H, W)
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device)
        state = state.float()

        if isinstance(action_mask, np.ndarray):
            action_mask = torch.from_numpy(action_mask).to(self.device)

        num_pieces = self.pieces.size

        # Pass observation through the pyramid (feature extractor)
        board_embed = self.pyramid(state)  # shape: (..., C, H, W)

        tiled_no_attack = state[..., -4:-3, :, :] # shape: (... 1, H, W)

        one_hot_last_selected = state[..., 1:num_pieces + 1, :, :] # shape: (..., num_pieces, H, W)
        one_hot_last_selected = one_hot_last_selected[..., self.valid_mask, :, :] # shape: (..., num_movable_pieces, H, W)
        one_hot_last_selected = one_hot_last_selected * state[..., -1:, :, :] # shape: (..., num_movable_pieces, H, W)

        # Calculate game phase tensors
        finished_deploying = state[..., -3:-2, :, :] # shape: (..., 1, H, W)
        moving_piece = state[..., -2:-1, :, :] # shape: (..., 1, H, W)
        game_phase_tensor = (2 * moving_piece + finished_deploying).long()
        game_phase_tensor = game_phase_tensor.flatten(start_dim=-2) # shape (..., 1, H * W)

        # Compute policies -> shape: (..., 1 * H * W)
        deployment_logits = self.deployment_head(board_embed)
        selection_logits = self.selection_head(torch.cat((board_embed, tiled_no_attack), dim=-3))
        movement_logits = self.movement_head(torch.cat((board_embed, tiled_no_attack, one_hot_last_selected), dim=-3))

        all_logits = torch.stack([deployment_logits, selection_logits, movement_logits], dim=-2) # shape: (..., 3, 1 * H * W)
        logits = torch.gather(all_logits, dim=-2, index=game_phase_tensor) # shape: (..., 1, 1 * H * W)
        logits = torch.squeeze(logits, dim=-2) # shape: (..., H * W)

        # Action Masking
        action_mask = action_mask.flatten(start_dim=-2, end_dim=-1).bool() # shape: (..., H * W)
        masked_logits = torch.where(action_mask, logits, -torch.inf)
        masked_policy = nn.functional.softmax(masked_logits, dim=-1)
        masked_log_probs = nn.functional.log_softmax(masked_logits, dim=-1)

        # Compute value -> shape: (..., 1)
        value = self.value_head(torch.cat((board_embed,
                                          tiled_no_attack * finished_deploying * moving_piece,
                                          one_hot_last_selected * moving_piece), dim=-3))

        return masked_policy, value, masked_log_probs, logits




