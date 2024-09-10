import numpy as np

from woodokuMind import util


class FeatureExtractor:
  def getFeatures(self, state, action):
    """
      Returns a dict from features to counts
      Usually, the count will just be 1.0 for
      indicator functions.
    """
    util.raiseNotDefined()


class IdentityExtractor(FeatureExtractor):
  def getFeatures(self, state, action):
      feats = util.Counter()
      feats[(state, action)] = 1.0
      return feats


class SmartExtractor(FeatureExtractor):
  def getFeatures(self, state, action):
    features = util.Counter()
    features["bias"] = 1.0
    state2 = state.generate_successor(action)

    # Custom Features
    ### combo
    features["combo"] = state2.combo
    ### score
    features["score"] = state2.score
    ### straight
    features["straight"] = state2.straight
    ### stuck cells (number of single cells that are all surrounded by other cells)
    features["stuck cells"] = 0
    for i in range(len(state2.board)):
        for j in range(len(state2.board[i])):
            if state2.board[i][j] == 0:
                if i == 0 or i == len(state2.board) - 1 or j == 0 or j == len(state2.board[i]) - 1:
                    continue
                if state2.board[i - 1][j] != 0 and state2.board[i + 1][j] != 0 and state2.board[i][j - 1] != 0 and state2.board[i][j + 1] != 0:
                    features["stuck cells"] += 1

    features["cell diff"] = np.sum(state2.board == 0) - np.sum(state.board == 0)
    features.normalize()
    return features


class EstimationExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        features = util.Counter()
        block_1 = state.block1
        block_2 = state.block2
        block_3 = state.block3
        blocks = [block_1, block_2, block_3]
        block, row, column = action // 81, (action % 81) // 9, (action % 81) % 9
        block_shape = blocks[block]
        block_list = state._env.env.env._block_list
        block_num = 0
        for i in range(len(block_list)):
            if np.all((block_list[i] - block_shape) == 0):
                block_num = i
                break
        successor_game_state = state.generate_successor(action=action)


        # interpret an action as choosing a 3X3 square to place the block in rather then a specific location
        square = (row // 3, column // 3)
        square_center = (square[0] * 3 + 1, square[1] * 3 + 1)
        # interpret board as squares with the number of bricks inside each square
        prev_board = state.board
        board = successor_game_state.board
        compressed_board = board[square_center[0] - 1: square_center[0] + 2, square_center[1] - 1: square_center[1] + 2]
        compressed_board_rep = 0
        prev_compressed_board = prev_board[square_center[0] - 1: square_center[0] + 2, square_center[1] - 1: square_center[1] + 2]
        coord = (row % 3, column % 3)
        for i in range(3):
            for j in range(3):
                compressed_board_rep += int(compressed_board[i, j]) * (10 **  (i * 3 + j))
        features[(compressed_board_rep, coord, block_num)] = (np.sum(compressed_board)**2 +
                                                              10 * np.max(np.sum(compressed_board == 0)
                                                                          - np.sum(prev_compressed_board == 0), 0))
        features.normalize()
        return features

class EstimationDifExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        features = util.Counter()
        successor_game_state = state.generate_successor(action=action)
        block, row, column = action // 81, (action % 81) // 9, (action % 81) % 9


        # interpret an action as choosing a 3X3 square to place the block in rather then a specific location
        square = (row // 3, column // 3)
        block_1 = successor_game_state.block1
        block_2 = successor_game_state.block2
        block_3 = successor_game_state.block3
        blocks = [block_1, block_2, block_3]
        block_shape = blocks[block]
        block_list = state._env.env.env._block_list
        block_num = 0
        for i in range(len(block_list)):
            if np.all(np.equal(block_list[i], block_shape)):
                block_num = i
                break
        # interpret board as squares with the number of bricks inside each square
        prev_board = state.board
        board = successor_game_state.board
        compressed_board = np.zeros(shape=(3, 3))
        for i in range(3):
            i_center = i * 3 + 1
            for j in range(3):
                j_center = j * 3 + 1
                compressed_board[i, j] = np.sum(board[i_center - 1: i_center + 2, j_center - 1: j_center + 2])
        compressed_board_rep = 0
        for i in range(3):
            for j in range(3):
                compressed_board_rep += compressed_board[i, j] * (10 ** (i * 3 + j))
        features[(compressed_board_rep, square, block_num)] = (10 * len(successor_game_state.get_legal_actions()) + compressed_board[square[0], square[1]] ** 2)
        features.normalize()
        return features


