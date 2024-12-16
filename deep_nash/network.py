import numpy as np
import torch
import torch.nn as nn

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
    def __init__(self, inner_channels, outer_channels, N, M):
        super(DeepNashNet, self).__init__()
        self.pyramid = PyramidModule(inner_channels, outer_channels, N, M)
        self.deployment_head = nn.Sequential(
            PyramidModule(inner_channels, outer_channels, 1, 0),
            nn.Conv2d(outer_channels,1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Softmax(dim=-1)
        )
        self.selection_head = nn.Sequential(
            PyramidModule(inner_channels, outer_channels, 1, 0),
            nn.Conv2d(outer_channels,1, kernel_size=3, padding=1),
            nn.Flatten(),
            nn.Softmax(dim=-1),
        )
        self.movement_head = nn.Sequential(
            PyramidModule(inner_channels, outer_channels, 1, 0),
            nn.Conv2d(outer_channels,1, kernel_size=3, padding=1),
            nn.Flatten(),
            nn.Softmax(dim=-1)
        )
        self.value_head = nn.Sequential(
            PyramidModule(inner_channels, outer_channels, 0, 0),
            nn.Conv2d(outer_channels,1, kernel_size=3, padding=1),
            nn.Flatten(),
            nn.LazyLinear(1)
        )

    def forward(self, obs, env_info):
        # Assumptions about env_info:
        # - env_info["board_shape"] = (H, W)
        # - env_info["num_pieces"] = number of possible piece types
        # - env_info["moves_since_attack"] = shape (B,) array of floats or ints
        # - env_info["cur_board"] = list/array of length B, each element is a (H, W) array representing the board
        # - env_info["last_selected"] = list of length B, each either None or a tuple (r, c)
        # - env_info["pieces"] = array of shape (num_pieces,) or something broadcastable,
        #   containing unique identifiers for each piece type.

        device = obs.device
        B = obs.shape[0]
        H, W = env_info["board_shape"][0]
        num_pieces = env_info["num_pieces"][0]

        # Convert moves_since_attack to tensor
        moves_since_attack = torch.tensor(env_info["moves_since_attack"], dtype=torch.float32, device=device)

        # Stack cur_board into a (B, H, W) tensor
        # Originally cur_board might be a list of np arrays. We stack and then convert to torch.
        cur_board = np.stack(env_info["cur_board"], axis=0)  # shape: (B, H, W)
        cur_board = torch.tensor(cur_board, device=device)

        # last_selected: list of length B, each None or (r, c)
        last_selected = env_info["last_selected"]

        # pieces: ensure it's a tensor of shape (B, num_pieces) if needed.
        # If pieces is just a 1D array of length num_pieces (same for all), broadcast it:
        pieces_list = env_info["pieces"]  # shape: (B,), each element is e.g. (num_pieces,)
        pieces = np.stack(pieces_list, axis=0)  # Now pieces is (B, num_pieces)
        pieces = torch.tensor(pieces, device=device)

        # Pass observation through the pyramid (feature extractor)
        board_embed = self.pyramid(obs)  # (B, C, H, W)

        # Create tiled_no_attack: shape (B, 1, H, W)
        tiled_no_attack = (moves_since_attack / 200.0).view(B, 1, 1, 1).expand(B, 1, H, W)

        # Handle last_selected possibly being None
        valid_mask = np.array([sel is not None for sel in last_selected])
        valid_indices = np.where(valid_mask)[0]  # indices of batch elements with a valid selection

        # Initialize last_piece_type with a default invalid value (e.g., -1)
        last_piece_type = torch.full((B,), fill_value=-1, dtype=cur_board.dtype, device=device)

        if len(valid_indices) > 0:
            # Extract rows and cols for valid entries
            valid_rows = np.array([last_selected[i][0] for i in valid_indices])
            valid_cols = np.array([last_selected[i][1] for i in valid_indices])

            # Convert to torch tensors
            valid_indices_t = torch.tensor(valid_indices, dtype=torch.long, device=device)
            rows_t = torch.tensor(valid_rows, dtype=torch.long, device=device)
            cols_t = torch.tensor(valid_cols, dtype=torch.long, device=device)

            # Index cur_board for valid entries
            selected_pieces = cur_board[valid_indices_t, rows_t, cols_t]
            last_piece_type[valid_indices_t] = selected_pieces

        # Create one_hot_last_selected: (B, num_pieces, H, W)
        one_hot_last_selected = torch.zeros((B, num_pieces, H, W), device=device)

        # Find the piece indices for each last_piece_type
        # Compare pieces and last_piece_type
        # shapes: pieces: (B, num_pieces), last_piece_type: (B,)
        match = (pieces == last_piece_type.unsqueeze(1))
        # nonzero returns (batch_idx, piece_idx) for matches
        batch_idx, piece_idx = match.nonzero(as_tuple=True)

        # For those with valid last_selected, we have their (row, col)
        # Extract valid (row, col) again in torch
        if len(valid_indices) > 0:
            valid_rows_t = torch.tensor([ls[0] for ls in last_selected if ls is not None], device=device,
                                        dtype=torch.long)
            valid_cols_t = torch.tensor([ls[1] for ls in last_selected if ls is not None], device=device,
                                        dtype=torch.long)

            # Now, batch_idx is in ascending order of batches and should match the order of valid entries
            # We can directly use batch_idx to index into valid_rows_t and valid_cols_t
            # Note: match.nonzero() returns results in ascending order, so batch_idx corresponds to the sorted order of those batches
            # valid_indices_t was also sorted. Both should align as long as last_piece_type matches one piece uniquely.
            # To be safe, we can map batch_idx back to the index in valid_indices.
            # We know last_piece_type is unique per environment selection, so there should be a one-to-one mapping.
            # Let's create a dictionary from batch index to position in valid_indices:
            valid_idx_map = {v: i for i, v in enumerate(valid_indices)}
            # Create a mapping tensor for batch_idx to row in valid_rows_t
            map_positions = torch.tensor([valid_idx_map[int(b.item())] for b in batch_idx], device=device,
                                         dtype=torch.long)

            # Now use map_positions to index valid_rows_t and valid_cols_t
            one_hot_last_selected[batch_idx, piece_idx, valid_rows_t[map_positions], valid_cols_t[map_positions]] = 1.0

        # Compute policies
        deployment_policy = self.deployment_head(board_embed).view(B, 1, H, W)
        selection_policy = self.selection_head(torch.cat((board_embed, tiled_no_attack), dim=1)).view(B, 1, H, W)
        movement_policy = self.movement_head(
            torch.cat((board_embed, tiled_no_attack, one_hot_last_selected), dim=1)).view(B, 1, H, W)

        # Compute values
        deployment_value = self.value_head(torch.cat((board_embed,
                                                      torch.zeros_like(tiled_no_attack),
                                                      torch.zeros_like(one_hot_last_selected)), dim=1))
        selection_value = self.value_head(torch.cat((board_embed,
                                                     tiled_no_attack,
                                                     torch.zeros_like(one_hot_last_selected)), dim=1))
        movement_value = self.value_head(torch.cat((board_embed,
                                                    tiled_no_attack,
                                                    one_hot_last_selected), dim=1))

        return (deployment_policy, deployment_value), (selection_policy, selection_value), (
        movement_policy, movement_value)




