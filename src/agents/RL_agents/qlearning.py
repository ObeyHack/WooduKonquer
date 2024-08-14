import collections
from tqdm import tqdm
from src import util
from src.RLgame import RLgameState, RLAgent
from src.agents.RL_agents.featureExtractor import *
import random


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
        if (not self.is_trained) and random.random() < self.epsilon:
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

    def train_agent(self, env, max_steps=1000, logger=None):
        rewards = []
        for episode in tqdm(range(self.num_episodes)):
            obs, info = env.reset()
            state = RLgameState(env, obs, info)
            step = 0
            run_reward = 0
            while step < max_steps:
                action = self.get_action(state)
                next_state, reward, terminated, info = state.apply_action(action)
                if state.is_terminated():
                    break
                run_reward += reward
                self.update(state, action, next_state, reward)
                step += 1
                state = next_state
            if self.epsilon > 0.3:
                self.epsilon -= 0.0001

            # print(f'Episode {episode}, Reward {run_reward}, Score {state.score}')
            rewards.append(run_reward)
            if logger:
                logger["rewards"].append(run_reward)

        self.set_trained()
        return rewards


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



