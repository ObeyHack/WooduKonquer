from gymnasium import Env
from src.util import raiseNotDefined
import copy


class GameState:
    def __init__(self, env: Env):
        self._env = env
        self._cur_observation = self._env.reset()[0]

    @property
    def observation(self):
        return self._cur_observation

    def get_agent_legal_actions(self):
        actions = self._env.env.env.legality
        return [i for i in range(len(actions)) if actions[i] == 1]

    def apply_action(self, action):
        observation, reward, terminated, _, info = self._env.step(action)
        self._cur_observation = observation
        return reward, terminated, info


    def generate_successor(self, action):
        env2 = copy.deepcopy(self._env.unwrapped)
        new_state.apply_action(action)
        return new_state

class Agent:
  """
  An agent must define a getAction method, but may also define the
  following methods which will be called if they exist:
  """
  def __init__(self, index=0):
    self.index = index

  def getAction(self, state: GameState):
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
            action = self.agent.getAction(state)
            reward, terminated, info = state.apply_action(action)
            score += reward
        return score