import collections
from src.RLgame import RLgameState, RLAgent
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
        return max([self.get_q_value(state, action) for action in possible_actions])

    def get_policy(self, state: RLgameState):
        if state.is_terminated():
            return 0

        possible_actions = state.get_legal_actions()
        q_values = {action: self.get_q_value(state, action) for action in possible_actions}
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
        self.q_values[(state, action)] = (1 - self.alpha) * q_value + self.alpha * sample
        return self.q_values[(state, action)]


