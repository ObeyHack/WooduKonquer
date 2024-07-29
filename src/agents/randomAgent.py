from src.agents.agent import Agent


class RandomAgent(Agent):
    def getAction(self, state):
        return self.action_space.sample()
