import collections
from random import random

from src import util
from src.RLgame import RLAgent, RLgameState


class MonteCarloAgent(RLAgent):
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
