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


def avoid_jagged_edges_diag(current_game_state):
    """
    This evaluation function aims to minimize the number of jagged edges on the board
    after a move. A jagged edge is defined as a cell that is surrounded by 8 occupied
    cells.

    :param current_game_state: the state
    :param action:
    :return: the of jagged edges on the board after the move
    """

    def check_for_jagged(board, x: int, y: int):
        if board[x, y] == board[x + 1, y + 1] and board[x + 1, y] == board[x, y + 1] and board[x, y] != board[x + 1, y]:
            return 1
        return 0

    successor_game_state = current_game_state
    block_1 = successor_game_state.block1
    block_2 = successor_game_state.block2
    block_3 = successor_game_state.block3
    blocks = [block_1, block_2, block_3]
    successor_board = successor_game_state.board
    successor_score = successor_game_state.score
    jagged_edges = 0
    for i in range(len(successor_board) - 1):
        for j in range(len(successor_board[i]) - 1):
            jagged_edges += check_for_jagged(successor_board, i, j)
    return - jagged_edges
    # counting all trapped blocks


def avoid_jagged_edges(current_game_state : GameState):
    """
    This evaluation function aims to minimize the number of jagged edges on the board
    after a move. A jagged edge is defined as a cell that is surrounded by 8 occupied
    cells.

    :param current_game_state: the state
    :param action:
    :return: the of jagged edges on the board after the move
    """

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
                if k != 0 or l != 0:
                    occupied_blocks_count += board[x + k][y + l]
        return occupied_blocks_count

    successor_game_state = current_game_state
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