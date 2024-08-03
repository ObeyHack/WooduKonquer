import numpy as np

from src import util


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