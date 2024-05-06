import random
import time

import gymnasium
import numpy as np
from gymnasium import Env, spaces
import pygame

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

# PyGame Rendering Constants
WINDOW_SIZE = 800


class StrategoEnv(Env):
    def __init__(self, game_map=MAP_4x4):
        self.game_map = np.copy(game_map)

        self.board = None
        self.pieces = None
        self.movable_pieces = None

        self.p1_public_obs_info = None
        self.p2_public_obs_info = None

        self.p1_unrevealed = None
        self.p2_unrevealed = None

        self.p1_observed_moves = None
        self.p2_observed_moves = None

        self.p1_pieces = None
        self.p2_pieces = None

        self.draw_conditions = {"total_moves": 0, "moves_since_attack": 0}
        self.player = 1  # player1: 1, player2: -1

        self.window = None
        self.clock = None

        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()

    def _get_observation_space(self):
        if self.game_map.shape == (4, 4):
            shape = (20, 4, 4)
        else:
            shape = (82, 10, 10)

        return spaces.Box(low=-3, high=1, shape=shape, dtype=np.float64)

    def _get_action_space(self):
        if self.game_map.shape == (4, 4):
            shape = (2, 4, 4)
        else:
            shape = (2, 10, 10)

        return spaces.Box(low=0, high=1, shape=shape, dtype=np.int32)

    def set_player_pieces(self, player1pieces: np.ndarray, player2pieces: np.ndarray):
        self.p1_pieces = player1pieces
        self.p2_pieces = player2pieces

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.draw_conditions = {"total_moves": 0, "moves_since_attack": 0}
        self.player = 1  # player1: 1, player2: -1

        self.window = None
        self.clock = None
        self.generate_board()
        return self.generate_observation(), {"cur_player": self.player, "cur_board": np.copy(self.board),
                                             "total_moves": 0, "moves_since_attack": 0}

    def generate_board(self):
        self.board = np.copy(self.game_map)
        if self.game_map.shape == (4, 4):
            if (self.p1_pieces is None) or (self.p2_pieces is None):
                self.p1_pieces = np.array([FLAG, SPY, SCOUT, MARSHAL])
                self.p2_pieces = np.array([FLAG, SPY, SCOUT, MARSHAL])

            self.pieces = np.array(list(LIMITED_PIECE_SET2))
            self.movable_pieces = self.pieces[~np.isin(self.pieces, [FLAG, BOMB])]
            assert self.p1_pieces.shape == (4,) and self.p2_pieces.shape == (4,)
            self.board[3, :] = np.copy(self.p1_pieces)
            self.board[0, :] = np.copy(-self.p2_pieces)

            # 1st channel is unmoved, 2nd channel is moved, 3rd channel is revealed
            self.p1_public_obs_info = np.zeros((3, 4, 4))
            self.p2_public_obs_info = np.zeros((3, 4, 4))
            self.p1_public_obs_info[0, 3, :] = 1
            self.p2_public_obs_info[0, 0, :] = 1

            self.p1_unrevealed = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1])
            self.p2_unrevealed = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1])

            self.p1_observed_moves = np.zeros((5, 4, 4))
            self.p2_observed_moves = np.zeros((5, 4, 4))

        assert np.all(np.isin(self.p1_pieces, self.pieces))
        assert np.all(np.isin(self.p2_pieces, self.pieces))

    def generate_observation(self):
        if self.game_map.shape == (4, 4):
            obstacles = (np.abs(self.board) == OBSTACLE)[None, :]
            private_obs = np.concatenate(
                (self.board[None, :] == FLAG, self.board[None, :] == SPY,
                 self.board[None, :] == SERGEANT, self.board[None, :] == MARSHAL)
            )

            public_obs1 = self.get_public_obs(self.p1_public_obs_info, self.p1_unrevealed)
            public_obs2 = self.get_public_obs(self.p2_public_obs_info, self.p2_unrevealed)

            moves_obs = self.p1_observed_moves if self.player == 1 else self.p2_observed_moves
            scalar_obs = np.ones((2, 4, 4))
            scalar_obs[0] *= self.draw_conditions["total_moves"] / 2000
            scalar_obs[1] *= self.draw_conditions["moves_since_attack"] / 200

            return np.concatenate((obstacles, private_obs, public_obs2, public_obs1, moves_obs, scalar_obs))

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

    def pieces_available_to_deploy(self):
        if self.game_map.shape == (4, 4):
            return {FLAG: 1, SPY: 1, SERGEANT: 1, MARSHAL: 1}

    def step(self, action: np.ndarray):
        # Check if any pieces can be moved
        if np.sum((self.board >= SPY).astype(int)) == 0:
            self.board = np.rot90(self.board, 2) * -1
            self.player *= -1
            info = {"cur_player": self.player,
                    "cur_board": np.copy(self.board),
                    "total_moves": self.draw_conditions["total_moves"],
                    "moves_since_attack": self.draw_conditions["moves_since_attack"]}

            draw_game = np.sum((self.board <= -SPY).astype(int) == 0)
            return self.generate_observation(), 0, draw_game, False, info

        # First channel in action is selected piece, and second channel is the destination
        action = action.astype(np.int64)
        valid, msg = self.check_action_valid(action)
        if not valid:
            raise ValueError(msg)

        # Get Selected Piece Identity and Destination Identity
        selected_piece = np.sum(action[0] * self.board)
        destination = np.sum(action[1] * self.board)

        # Update Draw conditions
        self.draw_conditions["total_moves"] += 1
        if destination == EMPTY:
            self.draw_conditions["moves_since_attack"] += 1
        else:
            self.draw_conditions["moves_since_attack"] = 0

        # Initialize Reward, Termination, and Info
        reward = 0
        terminated = False
        info = {"cur_player": self.player,
                "cur_board": np.copy(self.board),
                "total_moves": self.draw_conditions["total_moves"],
                "moves_since_attack": self.draw_conditions["moves_since_attack"]}

        # Check if draw conditions are met
        if self.draw_conditions["total_moves"] >= 2000 or self.draw_conditions["moves_since_attack"] >= 200:
            self.board = np.rot90(self.board, 2) * -1
            self.player *= -1
            return self.generate_observation(), reward, True, False, info

        # Update Move Histories
        cur_player_moves = self.p1_observed_moves if self.player == 1 else self.p2_observed_moves
        other_player_moves = self.p2_observed_moves if self.player == 1 else self.p1_observed_moves
        cur_player_moves = np.roll(cur_player_moves, 1, axis=0)
        other_player_moves = np.roll(other_player_moves, 1, axis=0)

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
        info.update({"cur_player": self.player, "cur_board": np.copy(self.board)})

        return self.generate_observation(), reward, terminated, False, info

    def check_action_valid(self, action: np.ndarray):
        if action.shape != (2,) + self.board.shape:
            return False, "Action shape is not as expected"

        selected_piece = np.sum(action[0] * self.board)
        if selected_piece < SPY:
            return False, "Selected piece cannot be moved by player"

        destination = np.sum(action[1] * self.board)

        if abs(destination) == OBSTACLE:
            return False, "Destination is an obstacle"

        if destination > OBSTACLE:
            return False, "Destination is already occupied by player's piece"

        if selected_piece != SCOUT:
            selected_piece_coord = np.argwhere(action[0] == 1)[0]
            destination_coord = np.argwhere(action[1] == 1)[0]
            if np.sum(np.abs(selected_piece_coord - destination_coord)) != 1:
                return False, "Invalid move"
        else:
            selected_piece_coord = np.argwhere(action[0] == 1)[0]
            destination_coord = np.argwhere(action[1] == 1)[0]

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

    def valid_pieces_to_select(self) -> np.ndarray:
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

        return np.logical_and(self.board >= SPY, ~surrounded).astype(int)

    def valid_destinations(self, selected_piece: np.ndarray):
        if np.sum(selected_piece) == 0:
            return np.zeros_like(selected_piece)
        position = np.argwhere(selected_piece == 1)[0]
        selected_piece_val = self.board[position[0], position[1]]
        board_shape = self.board.shape

        directions = np.array([[0, 0, 1, -1], [1, -1, 0, 0]])
        destinations = np.zeros_like(selected_piece)
        if selected_piece_val == SCOUT:
            for direction in directions.T:
                positions = position[:, None] + direction[:, None]
                encountered_enemy = 0
                while np.all(positions >= 0) and np.all(positions < board_shape) and encountered_enemy < 1:
                    if self.board[positions[0], positions[1]] != EMPTY:
                        if self.board[positions[0], positions[1]] > -FLAG:
                            break
                        encountered_enemy += 1
                    destinations[positions[0], positions[1]] = 1
                    positions += direction[:, None]
            return destinations
        else:
            positions = position[:, None] + directions
            valid_positions = positions[:,
                              np.all(positions >= 0, axis=0) &
                              np.all(positions < np.array(board_shape)[:, None], axis=0)]
            valid_positions = valid_positions[:,
                              (self.board[valid_positions[0], valid_positions[1]] <= EMPTY) &
                              (self.board[valid_positions[0], valid_positions[1]] != -OBSTACLE)]
            destinations[valid_positions[0], valid_positions[1]] = 1
            return destinations

    def get_random_action(self) -> np.ndarray:
        pieces_to_select = self.valid_pieces_to_select()
        if np.sum(pieces_to_select) != 0:
            probs0 = (pieces_to_select / pieces_to_select.sum()).flatten()
            chosen_piece = random.choices(range(len(probs0)), weights=probs0, k=1)[0]
            action0 = np.zeros_like(probs0)
            action0[chosen_piece] = 1
            action0 = action0.reshape(pieces_to_select.shape)
        else:
            action0 = np.zeros_like(pieces_to_select)

        destinations = self.valid_destinations(action0)
        if np.sum(destinations) != 0:
            probs1 = (destinations / destinations.sum()).flatten()
            chosen_destination = random.choices(range(len(probs1)), weights=probs1, k=1)[0]
            action1 = np.zeros_like(probs1)
            action1[chosen_destination] = 1
            action1 = action1.reshape(pieces_to_select.shape)
        else:
            action1 = np.zeros_like(destinations)

        return np.concatenate((action0[None, ...], action1[None, ...]), axis=0)

    def render(self, player=None):
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
