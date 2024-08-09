from random import random
from src.evaluations import evaluationFunctions

import numpy as np
from tqdm import tqdm

from src.game import Agent


class RandomAgent(Agent):
    def get_action(self, state):
        pos_actions = state.get_legal_actions()

        # Randomly choose an action
        return pos_actions[int(random() * len(pos_actions))]


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """
    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in tqdm(legal_moves)]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """
        # Useful information you can extract from a GameState (game_state.py)
        successor_game_state = current_game_state.generate_successor(action=action)
        block_1 = successor_game_state.block1
        block_2 = successor_game_state.block2
        block_3 = successor_game_state.block3
        board = successor_game_state.board
        score = successor_game_state.score
        moved_block, x, y = action // 81, (action % 81) // 9, (action % 81) % 9

        "*** YOUR CODE HERE ***"
        square_center = ((x//3)*3 + 1, (y//3)*3 + 1)

        square_count = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                square_count += board[square_center[0] + i, square_center[1] + j]

        successor_empty_tiles = np.sum(board == 0)

        # number of legal moves in the successor state
        num_legal_moves_successor = len(successor_game_state.get_legal_actions())

        return score + 1000 * num_legal_moves_successor + square_count ** 2 + 10 * successor_empty_tiles + evaluationFunctions.avoid_jagged_edges(current_game_state, action)