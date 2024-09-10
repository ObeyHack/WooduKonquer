from itertools import groupby
import numpy as np
from tqdm import tqdm
from gymnasium import Env
import gymnasium as gym
from woodokuMind import util
from woodokuMind.game import Agent, GameState, Game



class RLgameState(GameState):

    def apply_action(self, action):
        from copy import deepcopy
        new_state = deepcopy(self)
        return new_state.apply_action(action)


class RLAgent(Agent):
    def __init__(self, alpha=0.5, epsilon=0.1, gamma=0.8, num_episodes=10000):
        """
        Sets options, which can be passed in via the Pacman command line using -a alpha=0.5,...
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(gamma)
        self.is_trained = False
        self.num_episodes = num_episodes

    def set_trained(self):
        self.is_trained = True

    def get_q_value(self, state: RLgameState, action):
        """
        Should return Q(state,action)
        """
        util.raiseNotDefined()

    def get_value(self, state: RLgameState):
        """
        What is the value of this state under the best action?
        Concretely, this is given by

        V(s) = max_{a in actions} Q(s,a)
        """
        util.raiseNotDefined()

    def get_policy(self, state: RLgameState):
        """
        What is the best action to take in the state. Note that because
        we might want to explore, this might not coincide with getAction
        Concretely, this is given by

        policy(s) = arg_max_{a in actions} Q(s,a)

        If many actions achieve the maximal Q-value,
        it doesn't matter which is selected.
        """
        util.raiseNotDefined()

    def get_action(self, state: RLgameState):
        """
        state: can call state.getLegalActions()
        Choose an action and return it.
        """
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
    	This class will call this function, which you write, after
    	observing a transition and reward
        """
        util.raiseNotDefined()

    def train_agent(self, env, max_steps=1000, logger=None):
        util.raiseNotDefined()
