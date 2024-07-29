from agents.agent import Agent
from gymnasium import Env


class GameState():
    def __init__(self, env: Env):
        self.env = env
        self

    def initialize(self):
        return self.env.reset()

    def get_observation(self):
        return self.env.get_observation()


class Game():
    def __init__(self, env: Env, agent: Agent):
        """
        environment
        :param env: gym environment
        :param agent: age
        """
        self.env = env
        self.agent = agent

    def run(self):
        """
        Run the game
        :return: score of the game
        """
        observation, info = self.env.reset()
        terminated = False
        score = 0
        while not terminated:
            # render the environment
            self.env.render()
            # take action
            action = self.agent.getAction(observation, self.env)
            observation, reward, terminated, _, info = self.env.step(action)
            score += reward
        return score