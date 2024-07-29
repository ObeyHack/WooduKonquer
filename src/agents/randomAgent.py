from random import random
from src.game import Agent


class RandomAgent(Agent):
    def getAction(self, state):
        pos_actions = state.get_agent_legal_actions()

        # Randomly choose an action
        return pos_actions[int(random() * len(pos_actions))]
