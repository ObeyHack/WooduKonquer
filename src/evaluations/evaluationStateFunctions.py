import numpy as np
from tqdm import tqdm

from src.game import GameState
import cv2


def score_function(current_game_state):
    """
    This function returns the score given by the game in the current state

    :param board: the board
    :param x: occupied block x coordinate
    :param y: occupied block y coordinate
    :return:  the number of occupied blocks around the given block
    """
    return current_game_state.score


def remaining_possible_moves(current_game_state):
    """
    This function returns the number of available legal actions in the current state

    :param board: the board
    :param x: occupied block x coordinate
    :param y: occupied block y coordinate
    :return:  the number of occupied blocks around the given block
    """
    return len(current_game_state.get_legal_actions())


def connected_components(current_game_state):
    """
    This evaluation function considers both the number and shape of connected components
    on the board after a move. It aims to minimize the number of components and prefers
    convex components.

    :param current_game_state: the current game state
    :return: A measure for the quality of the current game state based on connected components
    """

    def is_component_convex(component):
        """
        Check if the given component (a binary numpy array) is convex.
        The component is assumed to be a binary mask (1s and 0s).


        :param component: the component to check convex for
        :return: True if the component is convex, False otherwise
        """
        # Find contours in the component
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return True  # No contours, trivially convex

        # Check if the largest contour is convex
        return cv2.isContourConvex(contours[0])

    # Extract the board from the successor game state
    board = current_game_state.board

    # Invert the board to find empty space regions
    inverted_board = 1 - board

    # Apply connected components labeling to find isolated empty regions
    num_labels, labels = cv2.connectedComponents(inverted_board, connectivity=8)

    # Initialize the score
    score = 0

    # Set weights for different factors
    component_penalty = -10  # Penalty per component
    convex_reward = 10  # Reward for convex component
    non_convex_penalty = -20 # Additional penalty for non-convex component

    for label in range(1, num_labels):
        component = (labels == label).astype(np.uint8)
        score += component_penalty  # Penalize for having a component
        if is_component_convex(component):
            score += convex_reward  # Reward for convex shape
        else:
            score += non_convex_penalty  # Penalize for non-convex shape

    block_1 = current_game_state.block1
    block_2 = current_game_state.block2
    block_3 = current_game_state.block3
    return score


def num_empty_squares(current_game_state):
    """
    This evaluation function gets a game state and returns the number of 3x3 crushable squares
    that are completely empty on the current game state

    :param current_game_state: the current game state
    :return: the number of 3x3 empty crushable squares on the board
    """
    board = current_game_state.board
    square_count = 0
    for i in range(3):
        for j in range(3):
            square_center = (i * 3 + 1, j * 3 + 1)
            square_occupied_bricks = 0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    square_occupied_bricks += board[square_center[0] + i, square_center[1] + j]
            if square_occupied_bricks == 0:
                square_count += 1
    return square_count


def square_contribution(current_game_state):
    """
    This function calculates the number of bricks in the 3x3 crashable square in which the block will be placed.
    :param current_game_state:  the current game state
    :param action:  the action to evaluate
    :return: the number of bricks  in the 3x3 crashable square in which the block will be placed
    """
    board = current_game_state.board
    max_square_occupied_bricks = 0
    for i in range(3):
        for j in range(3):
            square_center = (i * 3 + 1, j * 3 + 1)
            square_occupied_bricks = 0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    square_occupied_bricks += board[square_center[0] + i, square_center[1] + j]
            max_square_occupied_bricks = max(max_square_occupied_bricks, square_occupied_bricks)
    return max_square_occupied_bricks


def best_evaluation(current_game_state):
    """
    This evaluation function gets a game state and returns a combination of the functions connected_components
    and num_empty_squares

    :param current_game_state: the current game state
    :return: the number of 3x3 empty crushable squares on the board
    """
    return (15 * remaining_possible_moves(current_game_state) +
            connected_components(current_game_state) + square_contribution(current_game_state)**2)
