import numpy as np

from src.game import Agent, GameState
from gymnasium import Env



class RLgameState(GameState):
    def __init__(self, env: Env, observation):
        self._env = env
        self._woodoku_env = self._env.env.env
        self._cur_observation = observation
        self.terminated = False

    def get_legal_actions(self):
        # state is a dict of board, piece1, piece2, piece3
        # board is the value of the board
        # piece1, piece2, piece3 are the 3 pieces that can be placed on the board
        board =  self._cur_observation['board']
        piece1 =  self._cur_observation['block_1']
        piece2 =  self._cur_observation['block_2']
        piece3 =  self._cur_observation['block_3']
        pieces = [piece1, piece2, piece3]
        legal_actions = []

        for piece_index, piece in enumerate(pieces):
            # if piece is all zero matrix then skip
            if np.all(piece == 0):
                continue
            for row in range(9):
                for col in range(9):
                    # Place block_1 at ((action-0) // 9, (action-0) % 9) , Place block_2 at ((action-81) // 9, (action-81) % 9), Place block_3 at ((action-162) // 9, (action-162) % 9)
                    # calculate  Discrete(243) = 81*row + 9*col + piece_index
                    action_helper = 81 * piece_index + 9 * row + col
                    if self._woodoku_env._is_valid_position(action_helper):
                        legal_actions.append((piece_index, row, col))



        legal_actions = [81 * action[0] + 9 * action[1] + action[2] for action in legal_actions]
        return legal_actions

    def apply_action(self, action):
        observation, reward, terminated, _, info = self._env.step(action)
        new_state = RLgameState(self._env, observation)
        self.terminated = terminated
        return new_state, reward, terminated, info

    def __eq__(self, othr):
        return (isinstance(othr, type(self))
                and (self.board, self.block1, self.block2, self.block3) ==
                (othr.board, othr.block1, othr.block2, othr.block3))

    def __hash__(self):
        """Converts a state consisting of numpy arrays to a hashable type (tuple)."""
        return hash((self.board.tostring(), self.block1.tostring(), self.block2.tostring(), self.block3.tostring()))