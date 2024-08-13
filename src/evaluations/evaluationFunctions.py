import numpy as np
from src.game import GameState
import cv2


def base_evaluation_function(current_game_state, action):
    return current_game_state.score


def occupied_neighbours(board, x: int, y: int):
    """Gets a board and a coordinate on it and returns the number of occupied neighbours around the coordinate

    Parameters
    ----------
    board : ?
        A state of a regular woodoku board
    x : int
        The x value of the coordinate
    y : int
        The y value of the coordinate

    Returns
    -------
    int
        The number of occupied blocks around the given coordinate on the given board.
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
    """Given a woodoku game state and an action, measures the number of pointy edges created after applying the action

    Parameters
    ----------
    current_game_state :
        A state of a regular woodoku board
    x : int
        The x value of the coordinate
    y : int
        The y value of the coordinate

    Returns
    -------
    int
        The number of occupied blocks around the given coordinate on the given board.
    """
    successor_game_state = current_game_state.generate_successor(action=action)
    block_1 = successor_game_state.block1
    block_2 = successor_game_state.block2
    block_3 = successor_game_state.block3
    blocks = [block_1, block_2, block_3]
    successor_board = successor_game_state.board
    successor_score = successor_game_state.score
    jagged_edges = 0

    moved_block, x, y = blocks[action // 81], (action % 81) // 9, (action % 81) % 9
    for i in range(len(successor_board)):
        for j in range(len(successor_board[i])):
            if occupied_neighbours(successor_board, i, j) == 8:
                jagged_edges += 1
    return - jagged_edges
    # counting all trapped blocks


def triple_move(current_game_state : GameState, action: int):
    successor_game_state = current_game_state.generate_successor(action=action)
    block_1 = successor_game_state.block1
    block_2 = successor_game_state.block2
    block_3 = successor_game_state.block3
    blocks = [block_1, block_2, block_3]


def matching_blocks(current_game_state : GameState, action: int):
    successor_game_state = current_game_state.generate_successor(action=action)
    block_1 = current_game_state.block1
    block_2 = current_game_state.block2
    block_3 = current_game_state.block3
    blocks = [block_1, block_2, block_3]


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


def board_density(board):
    """
    Calculate the density of filled blocks on the board.
    Density is defined as the ratio of filled blocks to total blocks.
    """
    total_blocks = board.size
    filled_blocks = np.sum(board)
    return filled_blocks / total_blocks


def evaluation_function_3(current_game_state, action):
    """
    This evaluation function considers both the number and shape of connected components
    on the board after a move. It aims to minimize the number of components, prefers
    convex components, and rewards board cleaning.
    """

    # Generate the successor game state based on the action taken
    successor_game_state = current_game_state.generate_successor(action=action)

    # Extract the board from the current and successor game state
    current_board = current_game_state.board
    successor_board = successor_game_state.board

    # Calculate the density of filled blocks
    current_density = board_density(current_board)
    successor_density = board_density(successor_board)

    # Invert the board to find empty space regions
    inverted_board = 1 - successor_board

    # Apply connected components labeling to find isolated empty regions
    num_labels, labels = cv2.connectedComponents(inverted_board, connectivity=4)

    # Initialize the score
    score = 0

    # Set weights for different factors
    component_penalty = -10  # Penalty per component
    convex_reward = 5  # Reward for convex component
    non_convex_penalty = -20  # Additional penalty for non-convex component
    density_reward = 15  # Reward for increased board density
    cleaning_reward = 5  # Reward for significant clearing

    for label in range(1, num_labels):
        component = (labels == label).astype(np.uint8)
        score += component_penalty  # Penalize for having a component
        if is_component_convex(component):
            score += convex_reward  # Reward for convex shape
        else:
            score += non_convex_penalty  # Penalize for non-convex shape

    # Reward for improved board density
    score += -(successor_density - current_density)

    # Reward for clearing blocks
    cleared_blocks = np.sum(current_board) - np.sum(successor_board)
    score += cleaning_reward * cleared_blocks

    # Factor in the number of legal moves in the successor state
    num_legal_moves = len(successor_game_state.get_legal_actions())
    score += 10 * num_legal_moves

    return score


def evaluation_function_4(current_game_state, action):
    """
    This evaluation function considers both the number and shape of connected components
    on the board after a move. It aims to minimize the number of components and prefers
    convex components.
    """

    # Generate the successor game state based on the action taken
    successor_game_state = current_game_state.generate_successor(action=action)

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
    convex_reward = 10  # Reward for convex component
    non_convex_penalty = -20 # Additional penalty for non-convex component

    for label in range(1, num_labels):
        component = (labels == label).astype(np.uint8)
        score += component_penalty  # Penalize for having a component
        if is_component_convex(component):
            score += convex_reward  # Reward for convex shape
        else:
            score += non_convex_penalty  # Penalize for non-convex shape

    block_1 = successor_game_state.block1
    block_2 = successor_game_state.block2
    block_3 = successor_game_state.block3
    num_legal_moves = len(successor_game_state.get_legal_actions())

        # Factor in the number of legal moves in the successor state

        # Weight the legal moves positively to keep options open
    score += 15 * num_legal_moves

    return score


def evaluation_function_5(current_game_state, action):
    #just legal actions and combos
    successor_game_state = current_game_state.generate_successor(action=action)
    block_1 = successor_game_state.block1
    block_2 = successor_game_state.block2
    block_3 = successor_game_state.block3
    board = successor_game_state.board
    score = successor_game_state.score
    # number of legal moves in the successor state
    num_legal_moves_successor = len(successor_game_state.get_legal_actions())
    #get number of combos
    combo = successor_game_state.combo
    stright = successor_game_state.straight
    return num_legal_moves_successor + combo*15 + stright*15


def square_contribution(current_game_state, action):
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


def best_evaluation(current_game_state, action):
    return evaluation_function_4(current_game_state, action) + square_contribution(current_game_state, action) ** 2
