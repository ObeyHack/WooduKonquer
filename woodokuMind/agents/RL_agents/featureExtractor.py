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
        """
        gets a state and an action and returns them as features on the feature map.

        :param state: the current game state
        :param action: a legal action in the given state
        :return: feature map with keys of shape (state, action)
        """
        feats = util.Counter()
        feats[(state, action)] = 1.0
        return feats


class SmartExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        """
        gets a state and an action and maps them to several underlying features.
        The features are bias, score, combo, number of stuck cells and difference in cells
        aftert taking the given action in the given state

        :param state: the current game state
        :param action: a legal action in the given state
        :return: feature map with keys of shape (state, action)
        """
        features = util.Counter()
        features["bias"] = 1.0
        state_suc = state.generate_successor(action)

        block_1 = state_suc.block1.astype(np.int32)
        block_2 = state_suc.block2.astype(np.int32)
        block_3 = state_suc.block3.astype(np.int32)
        board = state_suc.board.astype(np.int32)

        # Custom Features
        ### combo
        features["combo"] = state_suc.combo
        ### score
        features["score"] = state_suc.score
        ### straight
        features["straight"] = state_suc.straight
        ### stuck cells (number of single cells that are all surrounded by other cells)
        features["stuck cells"] = 0
        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] == 0:
                    if i == 0 or i == len(board) - 1 or j == 0 or j == len(board[i]) - 1:
                        continue
                    if board[i - 1][j] != 0 and board[i + 1][j] != 0 and board[i][j - 1] != 0 and \
                            board[i][j + 1] != 0:
                        features["stuck cells"] += 1

        features["cell diff"] = np.sum(board == 0) - np.sum(state.board.astype(np.int32) == 0)
        features.normalize()
        return features


class EstimationExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        """
        This function extracts features from the state, for the given state and action, the features are the number of
        bricks in each 3X3 square and the number of bricks in the square where the action is taken.
        maps state and an action to a compressed board state, the 3x3 square where the block is placed and
        the block type. The features are normalized.

        :param state: the current state of the game
        :param action: the action to take
        :return: a dictionary of features and their values for the given state and action
        """
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
        prev_compressed_board = prev_board[square_center[0] - 1: square_center[0] + 2,
                                square_center[1] - 1: square_center[1] + 2]
        coord = (row % 3, column % 3)
        for i in range(3):
            for j in range(3):
                compressed_board_rep += int(compressed_board[i, j]) * (10 ** (i * 3 + j))
        features[(compressed_board_rep, coord, block_num)] = (np.sum(compressed_board) ** 2 +
                                                              10 * np.max(np.sum(compressed_board == 0)
                                                                          - np.sum(prev_compressed_board == 0), 0))
        features.normalize()
        return features


class EstimationDifExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        """
        This function extracts features from the state, for the given state and action, the features are the number of
        bricks in each 3X3 square and the number of bricks in the square where the action is taken with more emphasis on
        remaining legal moves. maps state and an action to a compressed board state, the 3x3 square where the block is
        placed and the block type. The features are normalized.

        :param state: the current state of the game
        :param action: the action to take
        :return: a dictionary of features and their values for the given state and action
        """
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
        features[(compressed_board_rep, square, block_num)] = (
                    10 * len(successor_game_state.get_legal_actions()) + compressed_board[square[0], square[1]] ** 2)
        features.normalize()
        return features

