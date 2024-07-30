from gymnasium import Env
from src.util import raiseNotDefined
import copy


class GameState:
    def __init__(self, env: Env):
        self._env = env
        self._cur_observation = self._env.reset()[0]
        self._woodoku_env = self._env.env.env
        self._woodoku_env.__deepcopy__ = self.__copy
        self.terminated = False

    def __copy(self, memo):
        from copy import deepcopy, copy
        cls = self._woodoku_env.__class__
        result = cls.__new__(cls)
        memo[id(self._woodoku_env)] = result
        for k, v in self._woodoku_env.__dict__.items():
            if k == "window":
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
        return GameState(env2)


    def is_terminated(self):
        return self.terminated

    @property
    def observation(self):
        return self._cur_observation

    @property
    def board(self):
        return self._woodoku_env._board

    @property
    def block1(self):
        return self._woodoku_env._block_1

    @property
    def block2(self):
        return self._woodoku_env._block_2

    @property
    def block3(self):
        return self._woodoku_env._block_3

    @property
    def score(self):
        return self._woodoku_env._score

    def get_legal_actions(self):
        actions = self._woodoku_env.legality
        return [i for i in range(len(actions)) if actions[i] == 1]

    def apply_action(self, action):
        observation, reward, terminated, _, info = self._env.step(action)
        self._cur_observation = observation
        self.terminated = terminated
        return reward, terminated, info

    def generate_successor(self, action):
        new_state = copy.deepcopy(self)
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
        state = GameState(self.env)
        terminated = False
        score = 0
        while not terminated:
            # render the environment
            self.env.render()
            # take action
            action = self.agent.get_action(state)
            reward, terminated, info = state.apply_action(action)
            score += reward
        return score