import numpy as np
from gymnasium import Env
from src.util import raiseNotDefined
import copy
from gymnasium.utils import seeding

class GameState:
    def __init__(self, env: Env, observation, info):
        """

        :param env:
        :param observation:
                board: a 2D list representing the board state
                block_1: a 2D list representing the block state
                block_2: a 2D list representing the block state
        :param info:
                "action_mask" - legal actions
                "combo" - number of combo
                "straight" - number of straight
                "score" - current score
                "terminated" - is the game terminated
        """
        self._env = env
        self._cur_observation = observation
        self._woodoku_env = self._env.env.env
        if self._env.render_mode == "rgb_array":
            self._woodoku_env = self._env.env.env.env
        self._woodoku_env.__deepcopy__ = self.__copy
        self.terminated = False
        self.legal_action = np.array([i for i in range(len(info["action_mask"])) if info["action_mask"][i] == 1])
        self._combo = info["combo"]
        self._straight = info["straight"]
        self._score = info["score"]

    def __copy(self, memo):
        from copy import deepcopy, copy
        cls = self._woodoku_env.__class__
        result = cls.__new__(cls)
        memo[id(self._woodoku_env)] = result
        for k, v in self._woodoku_env.__dict__.items():
            if (k == "window" or k == "observation_space" or k == "action_space"):
                setattr(result, k, copy(v))
                continue

            if k == "clock" or k == "__deepcopy__":
                setattr(result, k, v)
                continue

            setattr(result, k, deepcopy(v, memo))
        return result

    def __deepcopy__(self, memo):
        from copy import deepcopy
        env2 = deepcopy(self._env)
        info ={
            "action_mask": self.legal_action,
            "combo": self._combo,
            "straight": self._straight,
            "score": self._score,
        }
        return GameState(env2, deepcopy(self._cur_observation), deepcopy(info))

    def __eq__(self, othr):
        return (isinstance(othr, type(self))
                and ((self.board == othr.board).all()) and
                (self.block1 == othr.block1).all() and
                (self.block2 == othr.block2).all() and
                (self.block3 == othr.block3).all())

    def __hash__(self):
        """Converts a state consisting of numpy arrays to a hashable type (tuple)."""
        return hash((self.board.tostring(), self.block1.tostring(), self.block2.tostring(), self.block3.tostring()))

    def is_terminated(self):
        return self.terminated

    @property
    def combo(self):
        return self._combo

    @property
    def straight(self):
        return self._straight

    @property
    def observation(self):
        return self._cur_observation

    @property
    def board(self):
        return self._cur_observation["board"]

    @property
    def block1(self):
        return self._cur_observation["block_1"]

    @property
    def block2(self):
        return self._cur_observation["block_2"]

    @property
    def block3(self):
        return self._cur_observation["block_3"]

    @property
    def score(self):
        return self._score

    @property
    def action_space(self):
        return self._woodoku_env.action_space

    @property
    def observation_space(self):
        return self._woodoku_env.observation_space

    def get_legal_actions(self):
        return self.legal_action

    def apply_action(self, action):
        observation, reward, terminated, _, info = self._env.step(action)
        self._cur_observation = observation
        self.terminated = terminated
        self._combo = info["combo"]
        self._straight = info["straight"]
        self.legal_action = [i for i in range(len(info["action_mask"])) if info["action_mask"][i] == 1]
        self._score = info["score"]
        return self, reward, terminated, info

    def generate_successor(self, action):
        new_state = copy.deepcopy(self)

        # check if 2 of the blocks are 0
        if ((np.all(new_state.block1 == 0) and np.all(new_state.block2 == 0)) or (
                np.all(new_state.block1 == 0) and np.all(new_state.block3 == 0)) or (
                np.all(new_state.block2 == 0) and np.all(new_state.block3 == 0))):
            seed = 1
            rng, np_seed = seeding.np_random(seed)
            new_state._woodoku_env._np_random = rng
            new_state._woodoku_env._np_random_seed = np_seed
            # new_state._woodoku_env.seed(seed)

        # Turn off rendering
        render_mode = new_state._woodoku_env.render_mode
        new_state._woodoku_env.render_mode = None

        # Apply action
        new_state.apply_action(action)

        # Turn on rendering
        new_state._woodoku_env.render_mode = render_mode
        new_state._env.env.env = new_state._woodoku_env

        # Return new state
        return new_state


class Agent:
  """
  An agent must define a getAction method, but may also define the
  following methods which will be called if they exist:
  """
  def __init__(self, index=0):
    self.index = index

  def get_action(self, state: GameState):
    """
    The Agent will receive a state and must return an action
    :param state: GameState object holding observation from the environment and some useful methods.
    The observation is a dictionary containing the following:
    - board: a 2D list representing the board state
    - block_1: a 2D list representing the block state
    - block_2: a 2D list representing the block state
    - block_3: a 2D list representing the block state
    :return: action to take from the action space: Discrete(243). It is an integer as follows:
    - 0~80 : use block_1
    -- Place block_1 at ((action-0) // 9, (action-0) % 9)
    - 81~161 : use block_2
    -- Place block_2 at ((action-81) // 9, (action-81) % 9)
    - 162~242 : use block_3
    -- Place block_3 at ((action-162) // 9, (action-162) % 9)
    """
    raiseNotDefined()


class Game:
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
        obs, info = self.env.reset()
        state = GameState(self.env, obs, info)
        terminated = False
        score = 0
        steps = 0
        while not terminated:
            # render the environment
            self.env.render()
            # take action
            action = self.agent.get_action(state)
            state, reward, terminated, info = state.apply_action(action)
            score += reward
            steps += 1

            if steps % 50 == 0:
                print(f"Step: {steps}, Score: {score}")
        return score