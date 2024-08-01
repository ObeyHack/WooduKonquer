from random import random

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
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
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
        board = current_game_state.board
        score = current_game_state.score
        successor_board = successor_game_state.board
        successor_score = successor_game_state.score
        reward = successor_score - score

        "*** YOUR CODE HERE ***"

        num_legal_moves = len(current_game_state.get_legal_actions())
        num_legal_moves_successor = len(successor_game_state.get_legal_actions())
        num_legal_moves_diff = num_legal_moves_successor - num_legal_moves

        return reward + 100 * num_legal_moves_diff