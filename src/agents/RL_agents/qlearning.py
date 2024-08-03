import collections
from src.RLgame import RLgameState, RLAgent
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


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


class deepQlearningAgent(RLAgent):
    def __init__(self, alpha=1.0, epsilon=0.05, gamma=0.8):
        super().__init__(alpha, epsilon, gamma)
        self.model = DQN(input_size=156, hidden_size=1000, num_classes=243)
        self.target_model = DQN(input_size=156, hidden_size=1000, num_classes=243)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma

    def vectorize_state(self, state):  # 156 size
        board_tensor = torch.tensor(state.board.astype(float)).float()
        block1_tensor = torch.tensor(state.block1.astype(float)).float()
        block2_tensor = torch.tensor(state.block2.astype(float)).float()
        block3_tensor = torch.tensor(state.block3.astype(float)).float()
        state_tensor = torch.cat(
            (board_tensor.flatten(), block1_tensor.flatten(), block2_tensor.flatten(), block3_tensor.flatten()), 0)
        return state_tensor

    def get_q_value(self, state: RLgameState, action):
        state_tensor = self.vectorize_state(state)
        action = torch.tensor(action).float()
        return self.model(state_tensor).flatten()[action]

    def get_value(self, state: RLgameState):
        state_tensor = self.vectorize_state(state)
        return self.model(state_tensor).max().item()

    def get_policy(self, state: RLgameState):
        state_tensor = self.vectorize_state(state)
        return self.model(state_tensor).argmax().item()

    def get_action(self, state: RLgameState):
        state_tensor = self.vectorize_state(state)
        if np.random.rand() < self.epsilon:
            return random.choice(state.get_legal_actions())
        else:
            with torch.no_grad():  # Ensure no gradient is tracked
                q_values = self.model(state_tensor)
                # legel_actions = state.get_legal_actions()
                # take argmax of legal actions
                return q_values.argmax().item()

    def update(self, state, action, nextState, reward):
        # Vectorize the state and next state
        state_tensor = self.vectorize_state(state)
        next_state_tensor = self.vectorize_state(nextState)

        # Convert action to tensor
        action_tensor = torch.tensor([action]).long()

        # Get the current Q-value for the given state and action
        current_q_value = self.model(state_tensor).gather(0, action_tensor)

        # Get the maximum Q-value for the next state
        next_q_value = self.target_model(next_state_tensor).max().item()

        # Calculate the target Q-value
        target_q_value = reward + self.gamma * next_q_value

        # Compute the loss
        target_q_value_tensor = torch.tensor([target_q_value]).float()
        loss = self.loss_fn(current_q_value, target_q_value_tensor)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Optionally update the target network to be closer to the main network
        # This could be done periodically, not every time.
        # make it every 1000 steps

        self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x.flatten()))
        x = self.fc2(x)
        return x
