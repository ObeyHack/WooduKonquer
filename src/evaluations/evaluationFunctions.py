import numpy as np
from tqdm import tqdm
from src.evaluations.evaluationStateFunctions import *
from src.game import GameState
import cv2


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
    successor = current_game_state.generate_successor(action=action)
    return (connected_components(successor) + 15 * remaining_possible_moves(successor) +
            square_contribution(current_game_state, action) ** 2)

