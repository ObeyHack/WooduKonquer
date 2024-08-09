import numpy as np
from tqdm import tqdm

from src.game import GameState


def occupied_neighbours(board, x: int, y: int):
    occupied_blocks_count = 0
    for k in range(-1, 2):
        for l in range(-1, 2):
            if x + k < 0 or x + k > (len(board) - 1) or y + l < 0 or y + l > (len(board[0]) - 1):
                occupied_blocks_count += 1
            if k != 0 and l != 0:
                occupied_blocks_count += board[x + k][y + l]
    return occupied_blocks_count


def avoid_jagged_edges(current_game_state : GameState, action : int):
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
    return - jagged_edges * 3
    # counting all trapped blocks


def triple_move(current_game_state : GameState, action: int):
    successor_game_state = current_game_state.generate_successor(action=action)
    block_1 = successor_game_state.block1
    block_2 = successor_game_state.block2
    block_3 = successor_game_state.block3
    blocks = [block_1, block_2, block_3]
