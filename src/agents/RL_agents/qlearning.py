import collections
import numpy as np
from src import util
from src.RLgame import RLgameState, RLAgent
import random


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


class QLearningAgent(RLAgent):
    def __init__(self, **args):
        super().__init__(**args)
        self.q_values = collections.defaultdict(float)

    def get_q_value(self, state: RLgameState, action):
        # convert state to tuple
        return self.q_values[state, action]

    def get_value(self, state: RLgameState):
        if state.is_terminated():
            return 0.0

        possible_actions = state.get_legal_actions()
        return max([self.q_values[state, action] for action in possible_actions])

    def get_policy(self, state: RLgameState):
        if state.is_terminated():
            return 0

        possible_actions = state.get_legal_actions()
        q_values = {action: self.q_values[state, action] for action in possible_actions}
        return max(q_values, key=q_values.get)

    def get_action(self, state: RLgameState):
        if random.random() < self.epsilon:
            action = random.choice(state.get_legal_actions())
            return action

        action = self.get_policy(state)
        return action

    def update(self, state: RLgameState, action, next_state: RLgameState, reward: int):
        q_value = self.get_q_value(state, action)
        next_value = self.get_value(next_state)
        sample = reward + self.discount * next_value
        self.q_values[state, action] = (1 - self.alpha) * q_value + self.alpha * sample
        return self.q_values[state, action]


class ApproximateQAgent(QLearningAgent):
  """
     ApproximateQLearningAgent

     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """
  def __init__(self, extractor='SmartExtractor', **args):
    super().__init__(**args)
    self.featExtractor = util.lookup(extractor, globals())()

    # You might want to initialize weights here.
    "*** YOUR CODE HERE ***"
    self.weights = util.Counter()

  def get_q_value(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    "*** YOUR CODE HERE ***"
    features = self.featExtractor.getFeatures(state, action)
    qValue = 0
    for feature in features:
        qValue += self.weights[feature] * features[feature]
    return qValue

  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    "*** YOUR CODE HERE ***"
    features = self.featExtractor.getFeatures(state, action)
    correction = (reward + self.discount * self.get_value(nextState)) - self.get_q_value(state, action)
    for feature in features:
        self.weights[feature] += self.alpha * correction * features[feature]
    return self.weights
