import numpy as np
from tqdm import tqdm

from src.game import GameState
import cv2


def check_score(current_game_state, action):
    """
    This function returns the score given by the game after applying the given action
    on the given state.

    :param board: the board
    :param x: occupied block x coordinate
    :param y: occupied block y coordinate
    :return:  the number of occupied blocks around the given block
    """
    successor_game_state = current_game_state.generate_successor(action=action)
    return successor_game_state.score


def occupied_neighbours(board, x: int, y: int):
    """
    This function counts the number of occupied blocks around a given block on the board.
    This is a helper fcuntion for avoid jagged edges

    :param board: the board
    :param x: occupied block x coordinate
    :param y: occupied block y coordinate
    :return:  the number of occupied blocks around the given block
    """
    occupied_blocks_count = 0
    for k in range(-1, 2):
        for l in range(-1, 2):
            if x + k < 0 or x + k > (len(board) - 1) or y + l < 0 or y + l > (len(board[0]) - 1):
                occupied_blocks_count += 1
            if k != 0 and l != 0:
                occupied_blocks_count += board[x + k][y + l]
    return occupied_blocks_count


def avoid_jagged_edges(current_game_state : GameState, action : int):
    """
    This evaluation function aims to minimize the number of jagged edges on the board
    after a move. A jagged edge is defined as a cell that is surrounded by 8 occupied
    cells.

    :param current_game_state: the state
    :param action:
    :return: the of jagged edges on the board after the move
    """
    successor_game_state = current_game_state.generate_successor(action=action)
    block_1 = successor_game_state.block1
    block_2 = successor_game_state.block2
    block_3 = successor_game_state.block3
    blocks = [block_1, block_2, block_3]
    successor_board = successor_game_state.board
    successor_score = successor_game_state.score
    jagged_edges = 0
    for i in range(len(successor_board)):
        for j in range(len(successor_board[i])):
            if occupied_neighbours(successor_board, i, j) == 8:
                jagged_edges += 1
    return jagged_edges
    # counting all trapped blocks


def square_contribution(current_game_state, action):
    """
    This function calculates the number of bricks in the 3x3 crashable square in which the block will be placed.
    :param current_game_state:  the current game state
    :param action:  the action to evaluate
    :return: the number of bricks  in the 3x3 crashable square in which the block will be placed
    """
    successor_game_state = current_game_state.generate_successor(action=action)
    block_1 = successor_game_state.block1
    block_2 = successor_game_state.block2
    block_3 = successor_game_state.block3
    moved_block, x, y = action // 81, (action % 81) // 9, (action % 81) % 9
    square_center = ((x // 3) * 3 + 1, (y // 3) * 3 + 1)
    board = successor_game_state.board

    square_count = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            square_count += board[square_center[0] + i, square_center[1] + j]

    return square_count


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


def remaining_possible_moves(current_game_state, action):
    """
    This evaluation function estimates the number of possible legal actions remaining
    after applying the given action on the given state.

    :param current_game_state: the current game state
    :param action: the action to evaluate
    :return: estimation of the number of possible legal moves after taking said action
    """

    successor_game_state = current_game_state.generate_successor(action=action)
    num_legal_moves = len(successor_game_state.get_legal_actions())
    return num_legal_moves


def connected_components(current_game_state, action):
    """
    This evaluation function considers both the number and shape of connected components
    on the board after a move. It aims to minimize the number of components and prefers
    convex components.

    :param current_game_state: the current game state
    :param action: the action to evaluate
    :return: the score of the action
    """

    # Generate the successor game state based on the action taken
    successor_game_state = current_game_state.generate_successor(action=action)
    combo_number = successor_game_state.combo
    straight_number = successor_game_state.straight
    # Extract the board from the successor game state
    board = successor_game_state.board

    # Invert the board to find empty space regions
    inverted_board = 1 - board

    # Apply connected components labeling to find isolated empty regions
    num_labels, labels = cv2.connectedComponents(inverted_board, connectivity=4)

    # Initialize the score
    score = 0

    # Set weights for different factors
    component_penalty = -10  # Penalty per component
    convex_reward = 5  # Reward for convex component
    non_convex_penalty = -20  # Additional penalty for non-convex component

    for label in range(1, num_labels):
        component = (labels == label).astype(np.uint8)
        score += component_penalty  # Penalize for having a component
        if is_component_convex(component):
            score += convex_reward  # Reward for convex shape
        else:
            score += non_convex_penalty  # Penalize for non-convex shape

        # Factor in the number of legal moves in the successor state
    num_legal_moves = len(successor_game_state.get_legal_actions())

        # Weight the legal moves positively to keep options open
    score += 15 * num_legal_moves

    return score


def best_evaluation(current_game_state, action):
    """
    This evaluation function is an expression made up of various evaluation functions.
    The function gives the best average score for a ReflexAgent

    :param current_game_state: the current game state
    :param action: the action to evaluate
    :return: the score of the action

    .. seealso:: :class:`ReflexAgent`
    For more details, see the ReflexAgent class.
    """
    return connected_components(current_game_state, action) + square_contribution(current_game_state, action) ** 2

