import random

import numpy as np
from gymnasium import Env, spaces
import pygame

from stratego_gym.envs.masked_multi_discrete import MaskedMultiDiscrete

'''
Pieces Encodings: (Negated Encodings represent the other player's pieces)
'''
EMPTY = 0
OBSTACLE = 1
FLAG = 2
BOMB = 3
SPY = 4
SCOUT = 5
MINER = 6
SERGEANT = 7
LIEUTENANT = 8
CAPTAIN = 9
MAJOR = 10
COLONEL = 11
GENERAL = 12
MARSHAL = 13

MAP_4x4 = np.array([[0, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])

MAP_6x6 = np.array([[0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]])

MAP_10x10 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
                      [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

LIMITED_PIECE_SET = {FLAG, SPY, SERGEANT, MARSHAL}
LIMITED_PIECE_SET2 = {FLAG, SPY, SCOUT, MARSHAL}
FULL_PIECE_SET = {FLAG, BOMB, SPY, SCOUT, MINER, SERGEANT, LIEUTENANT, CAPTAIN, MAJOR, COLONEL, GENERAL, MARSHAL}

DEPLOYMENT_PHASE = 0
SELECTION_PHASE = 1
MOVEMENT_PHASE = 2
GAME_OVER = 3
GAME_PHASE_DICT = {0: "DEPLOYMENT_PHASE", 1: "SELECTION_PHASE", 2: "MOVEMENT_PHASE"}

GAME_CONFIG_4x4 = {
    "game_map": MAP_4x4,
    "pieces": LIMITED_PIECE_SET2
}

# PyGame Rendering Constants
WINDOW_SIZE = 800


def get_random_choice(valid_items):
    if np.sum(valid_items) != 0:
        probs = (valid_items / valid_items.sum()).flatten()
        flat_index = random.choices(range(len(probs)), weights=probs, k=1)[0]
        return np.unravel_index(flat_index, valid_items.shape)
    else:
        return -1, -1


class StrategoEnv(Env):
    metadata = {"render_modes": [None, "human"]}

    def __init__(self, game_config=None, render_mode=None):

        if game_config is None:
            self.game_config = GAME_CONFIG_4x4

        self.game_map = np.copy(self.game_config["game_map"])
        self.game_phase = DEPLOYMENT_PHASE
        self.render_mode = render_mode

        self.board = None
        self.pieces = None
        self.movable_pieces = None

        self.p1_public_obs_info = None
        self.p2_public_obs_info = None

        self.p1_unrevealed = None
        self.p2_unrevealed = None

        self.p1_observed_moves = None
        self.p2_observed_moves = None

        self.p1_deploy_idx = 0
        self.p2_deploy_idx = 0

        self.p1_last_selected = None
        self.p2_last_selected = None

        self.draw_conditions = {"total_moves": 0, "moves_since_attack": 0}
        self.player = 1  # player1: 1, player2: -1

        self.window = None
        self.clock = None

        self.observation_space = self._get_observation_space()
        self.action_space = MaskedMultiDiscrete(self.game_map.shape, dtype=np.int64)

        if render_mode not in [None, "human", "rgb_array"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")

    def _get_observation_space(self):
        if self.game_map.shape == (4, 4):
            shape = (23, 4, 4)
            mask_shape = (4, 4)
        else:
            shape = (82, 10, 10)
            mask_shape = (10, 10)

        return spaces.Dict({
            "obs": spaces.Box(low=-3, high=1, shape=shape, dtype=np.float64),
            "action_mask": spaces.Box(low=0, high=1, shape=mask_shape, dtype=np.int64)
        })

    def _get_action_space(self):
        return self.action_space
        # return spaces.MultiDiscrete(self.game_map.shape, dtype=np.int64)

    def set_player_pieces(self, player1pieces: np.ndarray, player2pieces: np.ndarray):
        self.p1_pieces = player1pieces
        self.p2_pieces = player2pieces

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.draw_conditions = {"total_moves": 0, "moves_since_attack": 0}
        self.player = 1  # player1: 1, player2: -1

        self.game_phase = DEPLOYMENT_PHASE
        self.p1_deploy_idx = 0
        self.p2_deploy_idx = 0

        self.window = None
        self.clock = None
        self.generate_board()
        return self.generate_env_state(), self.get_info()

    def generate_board(self):
        self.board = np.copy(self.game_map)
        if self.game_map.shape == (4, 4):
            self.pieces = np.array(list(self.game_config["pieces"]))
            self.movable_pieces = self.pieces[~np.isin(self.pieces, [FLAG, BOMB])]

            # 1st channel is unmoved, 2nd channel is moved, 3rd channel is revealed
            self.p1_public_obs_info = np.zeros((3, 4, 4))
            self.p2_public_obs_info = np.zeros((3, 4, 4))
            self.p1_public_obs_info[0, 3, :] = 1
            self.p2_public_obs_info[0, 0, :] = 1

            # TODO Fix the way we generate these arrays
            self.p1_unrevealed = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1])
            self.p2_unrevealed = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1])

            self.p1_observed_moves = np.zeros((5, 4, 4))
            self.p2_observed_moves = np.zeros((5, 4, 4))

    def generate_observation(self):
        if self.game_map.shape == (4, 4):
            obstacles = (np.abs(self.board) == OBSTACLE)[None, :]
            # TODO Fix the way we generate this
            private_obs = np.concatenate(
                (self.board[None, :] == FLAG, self.board[None, :] == SPY,
                 self.board[None, :] == SCOUT, self.board[None, :] == MARSHAL)
            )

            if self.game_phase == DEPLOYMENT_PHASE:
                public_obs = np.zeros((4, 4, 4))
                opp_public_obs = np.zeros((4, 4, 4))
                moves_obs = np.zeros_like(self.p1_observed_moves)
            else:
                public_obs1 = self.get_public_obs(self.p1_public_obs_info, self.p1_unrevealed)
                public_obs2 = self.get_public_obs(self.p2_public_obs_info, self.p2_unrevealed)
                public_obs = public_obs1 if self.player == 1 else public_obs2
                opp_public_obs = public_obs2 if self.player == 1 else public_obs1
                moves_obs = self.p1_observed_moves if self.player == 1 else self.p2_observed_moves

            scalar_obs = np.ones((4, 4, 4))
            scalar_obs[0] *= self.draw_conditions["total_moves"] / 2000
            scalar_obs[1] *= self.draw_conditions["moves_since_attack"] / 200
            scalar_obs[2] *= self.game_phase == DEPLOYMENT_PHASE
            scalar_obs[3] *= self.game_phase == MOVEMENT_PHASE

            last_selected_coord = self.p1_last_selected if self.player == 1 else self.p2_last_selected
            last_selected_obs = np.zeros((1, 4, 4))
            if last_selected_coord is not None:
                last_selected_obs[0][last_selected_coord] = 1

            return np.concatenate((obstacles, private_obs, opp_public_obs, public_obs, moves_obs, scalar_obs, last_selected_obs))

    '''
    Returns both the observation and an action mask in a dictionary
    '''
    def generate_env_state(self):
        obs = self.generate_observation()
        if self.game_phase == DEPLOYMENT_PHASE:
            action_mask = self.valid_spots_to_place()
        elif self.game_phase == SELECTION_PHASE:
            action_mask = self.valid_pieces_to_select()
        else:
            action_mask = self.valid_destinations()
        self.action_space.set_mask(action_mask.astype(bool))
        return {"obs": obs, "action_mask": action_mask}

    def get_public_obs(self, public_obs_info, unrevealed):
        if np.sum(unrevealed[self.pieces]) == 0:
            probs_unmoved = np.zeros_like(unrevealed[self.pieces])
        else:
            probs_unmoved = unrevealed[self.pieces] / np.sum(unrevealed[self.pieces])

        if np.sum(unrevealed[self.movable_pieces]) == 0:
            probs_moved = np.zeros_like(unrevealed[self.pieces])
        else:
            probs_moved = unrevealed[self.pieces] / np.sum(unrevealed[self.movable_pieces])
        probs_moved *= np.isin(self.pieces, self.movable_pieces).astype(np.int32)

        public_obs_unmoved = public_obs_info[0] * probs_unmoved[:, None, None]
        public_obs_moved = public_obs_info[1] * probs_moved[:, None, None]
        public_obs_revealed = np.int32((public_obs_info[2] == self.pieces[:, None, None]))

        return public_obs_unmoved + public_obs_moved + public_obs_revealed

    def encode_move(self, action: np.ndarray):
        selected_piece = np.sum(action[0] * self.board)
        destination = np.sum(action[1] * self.board)
        if destination == EMPTY:
            return action[1] - action[0]
        else:
            return action[1] - (2 + (selected_piece - 1) / 12) * action[0]

    def get_info(self):
        """
        The get_info method returns a dictionary containing the following information: \n
        - The current player (cur_player)
        - The current board state (cur_board)
        - The shape of the board (board_shape)
        - The number of pieces in the game (num_pieces)
        - The total number of moves made (total_moves)
        - The number of moves since the last attack (moves_since_attack)
        - The current game phase (game_phase)
        - The last selected piece (last_selected). This is only valid if the game phase is MOVEMENT_PHASE,
          and it corresponds to the last piece selected by the current player.
        """
        board = np.copy(self.board)
        if self.player == -1:
            board = np.rot90(board, 2) * -1
        return {"cur_player": np.array(self.player), "cur_board": board, "pieces": self.pieces,
                "board_shape": self.board.shape, "num_pieces": len(self.pieces),
                "total_moves": self.draw_conditions["total_moves"],
                "moves_since_attack": self.draw_conditions["moves_since_attack"],
                "game_phase": np.array(self.game_phase),
                "last_selected": None if self.game_phase != MOVEMENT_PHASE else
                self.p1_last_selected if self.player == 1 else self.p2_last_selected}

    def step(self, action: tuple):
        # Convert ndarray action to tuple if necessary
        if isinstance(action, np.ndarray):
            action = tuple(action.squeeze())

        valid, msg = self.validate_coord(action)
        if not valid:
            raise ValueError(msg)

        if self.game_phase == DEPLOYMENT_PHASE:
            if self.valid_spots_to_place()[action] == 0:
                action = tuple(self.action_space.sample())
                # raise ValueError("Invalid Deployment Location")

            if self.player == 1:
                self.board[action] = self.pieces[self.p1_deploy_idx]
                self.p1_deploy_idx += 1
            else:
                self.board[action] = self.pieces[self.p2_deploy_idx]
                self.p2_deploy_idx += 1

            if self.p2_deploy_idx == len(self.pieces):
                self.game_phase = SELECTION_PHASE

            self.board = np.rot90(self.board, 2) * -1
            self.player *= -1

            return self.generate_env_state(), 0, False, False, self.get_info()

        elif self.game_phase == SELECTION_PHASE:
            if self.valid_pieces_to_select()[action] == 0:
                action = tuple(self.action_space.sample())
                # raise ValueError("Invalid Piece Selection")

            if self.player == 1:
                self.p1_last_selected = action
            else:
                self.p2_last_selected = action

            self.game_phase = MOVEMENT_PHASE
            return self.generate_env_state(), 0, False, False, self.get_info()

        else:
            return self.movement_step(action)

    def movement_step(self, action: tuple):
        source = self.p1_last_selected if self.player == 1 else self.p2_last_selected
        dest = action

        # Action is a tuple representing a coordinate on the board
        valid, msg = self.check_action_valid(source, dest)
        if not valid:
            action = tuple(self.action_space.sample())
            dest = action
            # raise ValueError(msg)

        # Get Selected Piece Identity and Destination Identity
        selected_piece = self.board[source]
        destination = self.board[dest]

        action = np.zeros((2,) + self.board.shape, dtype=np.int64)
        action[0][source] = 1
        action[1][dest] = 1

        # Initialize Reward, Termination, and Info
        reward = 0
        terminated = False

        # Check if draw conditions are met
        if self.draw_conditions["total_moves"] >= 2000 or self.draw_conditions["moves_since_attack"] >= 200:
            self.board = np.rot90(self.board, 2) * -1
            self.player *= -1
            self.game_phase = GAME_OVER
            return self.generate_env_state(), 0, True, False, self.get_info()

        # Update Draw conditions
        self.draw_conditions["total_moves"] += 1
        if destination == EMPTY:
            self.draw_conditions["moves_since_attack"] += 1
        else:
            self.draw_conditions["moves_since_attack"] = 0

        # Update Move Histories
        self.p1_observed_moves = np.roll(self.p1_observed_moves, 1, axis=0)
        self.p2_observed_moves = np.roll(self.p2_observed_moves, 1, axis=0)
        cur_player_moves = self.p1_observed_moves if self.player == 1 else self.p2_observed_moves
        other_player_moves = self.p2_observed_moves if self.player == 1 else self.p1_observed_moves

        move = self.encode_move(action)
        cur_player_moves[0] = move
        other_player_moves[0] = np.rot90(move, 2) * -1

        # Perform Move Logic

        cur_player_public_info = self.p1_public_obs_info if self.player == 1 else self.p2_public_obs_info
        cur_player_unrevealed = self.p1_unrevealed if self.player == 1 else self.p2_unrevealed
        other_player_public_info = self.p2_public_obs_info if self.player == 1 else self.p1_public_obs_info
        other_player_unrevealed = self.p2_unrevealed if self.player == 1 else self.p1_unrevealed

        if ((selected_piece != MINER and destination == -BOMB) or  # Bomb
                (selected_piece == -destination)):  # Equal Strength
            # Remove Both Pieces
            self.board *= np.prod(1 - action, axis=0)
            cur_player_public_info *= np.prod(1 - action, axis=0)
            other_player_public_info *= np.rot90(np.prod(1 - action, axis=0), 2)
            cur_player_unrevealed[selected_piece] -= 1
            other_player_unrevealed[destination] -= 1
        elif ((selected_piece == SPY and destination == -MARSHAL) or  # Spy vs Marshal
              (selected_piece > -destination) or  # Attacker is stronger (Bomb case already handled)
              (destination == -FLAG)):  # Enemy Flag Found
            # Remove Enemy Piece
            self.board *= np.prod(1 - action, axis=0)
            self.board += action[1] * selected_piece
            cur_player_public_info *= 1 - action[0]
            if destination != EMPTY:
                cur_player_public_info[2] += action[1] * selected_piece
                other_player_public_info *= np.rot90(1 - action[1], 2)
                cur_player_unrevealed[selected_piece] -= 1
                other_player_unrevealed[destination] -= 1
            else:
                scout_move = np.sum(np.abs(np.argwhere(action[0] == 1)[0] - np.argwhere(action[1] == 1)[0])) > 1
                if scout_move:
                    cur_player_public_info[2] += action[1] * selected_piece
                    cur_player_unrevealed[selected_piece] -= 1
                else:
                    cur_player_public_info[1] += action[1]

            if destination == -FLAG:
                reward = 1
                terminated = True
        elif selected_piece < -destination:
            # Remove Attacker
            self.board *= 1 - action[0]
            cur_player_public_info *= 1 - action[0]
            other_player_public_info *= np.rot90(1 - action[1], 2)
            other_player_public_info[2] += np.rot90(action[1] * destination, 2)
            cur_player_unrevealed[selected_piece] -= 1
            other_player_unrevealed[destination] -= 1

        self.board = np.rot90(self.board, 2) * -1
        self.player *= -1

        # Check if any pieces can be moved. If one player has no movable pieces, the other player wins.
        # If both players have no movable pieces, the game is a draw.
        if not terminated and (np.sum(self.board >= SPY) == 0 or self.valid_pieces_to_select().sum() == 0):
            draw_game = (np.sum(self.board <= -SPY) == 0) or self.valid_pieces_to_select(is_other_player=True).sum() == 0
            self.game_phase = GAME_OVER
            return self.generate_env_state(), 0 if draw_game else 1, True, False, self.get_info()

        self.game_phase = GAME_OVER if terminated else SELECTION_PHASE
        return self.generate_env_state(), reward, terminated, False, self.get_info()

    def validate_coord(self, coord):
        if len(coord) != 2 and all(isinstance(item, int) for item in coord):
            return False, "Source tuple size or type is not as expected"

        if coord[0] < 0 or coord[0] >= self.board.shape[0]:
            return False, "Source row is out of bounds"

        if coord[1] < 0 or coord[1] >= self.board.shape[1]:
            return False, "Source column is out of bounds"

        return True, None

    def check_action_valid(self, src: tuple, dest: tuple):
        valid, msg = self.validate_coord(src)
        if not valid:
            return False, msg

        valid, msg = self.validate_coord(dest)
        if not valid:
            return False, msg

        selected_piece = self.board[src]
        if selected_piece < SPY:
            return False, "Selected piece cannot be moved by player"

        destination = self.board[dest]

        if abs(destination) == OBSTACLE:
            return False, "Destination is an obstacle"

        if destination > OBSTACLE:
            return False, "Destination is already occupied by player's piece"

        if selected_piece != SCOUT:
            selected_piece_coord = np.array(src)
            destination_coord = np.array(dest)
            if np.sum(np.abs(selected_piece_coord - destination_coord)) != 1:
                return False, "Invalid move"
        else:
            selected_piece_coord = np.array(src)
            destination_coord = np.array(dest)

            if selected_piece_coord[0] != destination_coord[0] and selected_piece_coord[1] != destination_coord[1]:
                return False, "Scouts can only move in straight lines"

            path_slice = self.board[
                         min(selected_piece_coord[0], destination_coord[0]) + 1:max(selected_piece_coord[0],
                                                                                    destination_coord[0]),
                         min(selected_piece_coord[1], destination_coord[1]) + 1:max(selected_piece_coord[1],
                                                                                    destination_coord[1])]

            if np.any(path_slice != 0):
                return False, "Pieces in the path of scout"

        return True, "Valid Action"

    def valid_spots_to_place(self) -> np.ndarray:
        if self.game_map.shape == (4, 4):
            mask = np.zeros_like(self.board)
            mask[3, :] = self.board[3, :] == EMPTY
            return mask
        return np.zeros(0)

    def valid_pieces_to_select(self, is_other_player=False) -> np.ndarray:
        padded_board = np.pad(self.board, 1, constant_values=OBSTACLE)
        padded_board[padded_board == -OBSTACLE] = OBSTACLE

        # Shift the padded array in all four directions
        shift_left = np.roll(padded_board, 1, axis=1)[1:-1, 1:-1]
        shift_right = np.roll(padded_board, -1, axis=1)[1:-1, 1:-1]
        shift_up = np.roll(padded_board, 1, axis=0)[1:-1, 1:-1]
        shift_down = np.roll(padded_board, -1, axis=0)[1:-1, 1:-1]

        # Check conditions to create the boolean array
        surrounded = (shift_left >= OBSTACLE) & (shift_right >= OBSTACLE) & (shift_up >= OBSTACLE) & (
                shift_down >= OBSTACLE)

        return np.logical_and((self.board <= -SPY) if is_other_player else (self.board >= SPY), ~surrounded).astype(int)

    def valid_destinations(self):
        if self.game_phase != MOVEMENT_PHASE:
            return np.zeros_like(self.board)

        selected = self.p1_last_selected if self.player == 1 else self.p2_last_selected
        selected_piece_val = self.board[selected]
        board_shape = np.array(self.board.shape)

        directions = np.array([[0, 0, 1, -1], [1, -1, 0, 0]])
        destinations = np.zeros_like(self.board)

        if selected_piece_val == SCOUT:
            for direction in directions.T:
                positions = np.array(selected)[:, None] + direction[:, None]
                encountered_enemy = 0
                while (
                        np.all(positions >= 0, axis=0)
                        and np.all(positions < board_shape[:, None], axis=0)
                        and encountered_enemy < 1
                ):
                    if self.board[positions[0], positions[1]] != EMPTY:
                        if self.board[positions[0], positions[1]] > -FLAG:
                            break
                        encountered_enemy += 1
                    destinations[positions[0], positions[1]] = 1
                    positions += direction[:, None]
            return destinations

        else:
            positions = np.array(selected)[:, None] + directions
            valid_positions = positions[
                              :,
                              (np.all(positions >= 0, axis=0)) &
                              (np.all(positions < board_shape[:, None], axis=0))
                              ]
            mask = (
                    (self.board[valid_positions[0], valid_positions[1]] <= EMPTY) &
                    (self.board[valid_positions[0], valid_positions[1]] != -OBSTACLE)
            )
            valid_positions = valid_positions[:, mask]
            destinations[valid_positions[0], valid_positions[1]] = 1
            return destinations

    def get_random_action(self) -> tuple:
        if self.game_phase == DEPLOYMENT_PHASE:
            valid_spots = self.valid_spots_to_place()
            return get_random_choice(valid_spots)
        elif self.game_phase == SELECTION_PHASE:
            pieces_to_select = self.valid_pieces_to_select()
            return get_random_choice(pieces_to_select)
        elif self.game_phase == MOVEMENT_PHASE:
            destinations = self.valid_destinations()
            return get_random_choice(destinations)
        else:
            return -1, -1

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (WINDOW_SIZE, WINDOW_SIZE)
            )

        cell_size = (WINDOW_SIZE // self.board.shape[0], WINDOW_SIZE // self.board.shape[1])

        # Calculate the number of cells in each direction
        num_cells_r = self.board.shape[0]
        num_cells_c = self.board.shape[1]

        board = np.copy(self.board)
        if self.player == -1:
            board = np.rot90(board, 2) * -1

        font = pygame.font.Font(None, 80)
        # Draw the grid
        for r in range(num_cells_r):
            for c in range(num_cells_c):
                rect = pygame.Rect(c * cell_size[0], r * cell_size[1], cell_size[0], cell_size[1])
                rect_center = rect.center
                if abs(board[r][c]) == OBSTACLE:
                    pygame.draw.rect(self.window, (158, 194, 230), rect)
                    text = None
                elif board[r][c] >= FLAG:
                    pygame.draw.rect(self.window, (217, 55, 58), rect)
                    render_text = 'F' if board[r][c] == FLAG else 'B' if board[r][c] == BOMB else str(board[r][c] - 3)
                    text = font.render(render_text, True, (255, 255, 255))
                elif board[r][c] <= -FLAG:
                    pygame.draw.rect(self.window, (24, 118, 181), rect)
                    render_text = 'F' if board[r][c] == -FLAG else 'B' if board[r][c] == -BOMB else str(-(board[r][c] + 3))
                    text = font.render(render_text, True, (255, 255, 255))
                else:
                    pygame.draw.rect(self.window, (242, 218, 180), rect)
                    text = None

                pygame.draw.rect(self.window, (255, 255, 255), rect, width=3)
                if text is not None:
                    text_rect = text.get_rect(center=rect_center)
                    self.window.blit(text, text_rect)

        pygame.event.pump()
        pygame.display.update()
