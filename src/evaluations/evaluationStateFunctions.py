import numpy as np
from tqdm import tqdm

from src.game import GameState
import cv2


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


def num_action_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return len(current_game_state.get_legal_actions())


def is_component_convex(component):
    """
    Check if the given component (a binary numpy array) is convex.
    The component is assumed to be a binary mask (1s and 0s).
    """
    # Find contours in the component
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return True  # No contours, trivially convex

    # Check if the largest contour is convex
    return cv2.isContourConvex(contours[0])


def evaluation_function_4(current_game_state):
    """
    This evaluation function considers both the number and shape of connected components
    on the board after a move. It aims to minimize the number of components and prefers
    convex components.
    """

    # Extract the board from the successor game state
    board = current_game_state.board

    # Invert the board to find empty space regions
    inverted_board = 1 - board

    # Apply connected components labeling to find isolated empty regions
    num_labels, labels = cv2.connectedComponents(inverted_board, connectivity=4)

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
    num_legal_moves = len(current_game_state.get_legal_actions())

    # Factor in the number of legal moves in the successor state

    # Weight the legal moves positively to keep options open
    score += 15 * num_legal_moves

    return score


def evaluation_function_6(current_game_state):
    # Useful information you can extract from a GameState (game_state.py)
    block_1 = current_game_state.block1
    block_2 = current_game_state.block2
    block_3 = current_game_state.block3
    board = current_game_state.board
    score = current_game_state.score

    "*** YOUR CODE HERE ***"
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
    # square_center = ((x // 3) * 3 + 1, (y // 3) * 3 + 1)
    #
    # square_count = 0
    # for i in range(-1, 2):
    #     for j in range(-1, 2):
    #         square_count += board[square_center[0] + i, square_center[1] + j]
    #
    # row_count = 0
    # for i in range(-2, 3):
    #     temp_row_count = 0
    #     if (x + i > 0) and (x + i < len(board) - 1):
    #         temp_row_count = np.sum(board[x + i])
    #     row_count = max(row_count, temp_row_count)
    #
    # col_count = 0
    # for i in range(-2, 3):
    #     temp_col_count = 0
    #     if (y + i > 0) and (y + i < len(board[0]) - 1):
    #         temp_col_count = np.sum(board[:, y + i])
    #     col_count = max(col_count, temp_col_count)

    # successor_empty_tiles = np.sum(board == 0)

    # number of legal moves in the successor state

    # return score + 1000 * num_legal_moves_successor + square_count ** 2 + 10 * successor_empty_tiles + evaluationFunctions.avoid_jagged_edges(current_game_state, action)
    return evaluation_function_4(current_game_state) + square_count ** 2
