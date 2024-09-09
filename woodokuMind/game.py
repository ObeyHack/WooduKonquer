import random
import numpy as np
from gymnasium import Env
import copy
from gymnasium.utils import seeding
from gym_woodoku.envs.blocks import blocks
from woodokuMind.util import raiseNotDefined

game_block = blocks["woodoku"]
blocks_range = len(game_block)
block_space = []

worst_case_blocks_idx = [0, 1, 6, 7, 8, 9, 10, 11, 12, 15, 16, 45, 46]
worst_case_range = len(worst_case_blocks_idx)
for i in range(worst_case_range):
    for j in range(i+1, worst_case_range):
        for k in range(j+1, worst_case_range):
            block_space.append((i, j, k))

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
        action_mask = [0] * 243
        for i in self.legal_action:
            action_mask[i] = 1
        info ={
            "action_mask": action_mask,
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

    def get_legal_actions(self, agent_index=0):
        if agent_index == 0:
            return self.legal_action

        else:
            return self.get_opponent_legal_actions()

    def get_opponent_legal_actions(self):
        """
        Get legal actions for the opponent.
        its empty action for when there is more than 1 block in the game
        else it returns all the triplets (3) of all possible blocks meaning: [(0-46), (0-46), (0-46)]
        :return:
        """

        # check if all 3 of the blocks are full
        if (not np.all(self.block1 == 0)) and (not np.all(self.block2 == 0)) and (not np.all(self.block3 == 0)):
            return block_space
            # return random.sample(block_space, 200)

        else:
            return [(-1, -1, -1)]


    def apply_action(self, action):
        observation, reward, terminated, _, info = self._env.step(action)
        self._cur_observation = observation
        self.terminated = terminated
        self._combo = info["combo"]
        self._straight = info["straight"]
        self.legal_action = [i for i in range(len(info["action_mask"])) if info["action_mask"][i] == 1]
        self._score = info["score"]

        #  assert reward != 0, "Reward should not be 0"
        return self, reward, terminated, info

    def apply_opponent_action(self, action):
        """
        Apply opponent action
        :param action: either a triplet (3) of (0-46, 0-46, 0-46)
        or empty action for when there is more than 1 block in the game (-1, -1, -1)
        :return:
        """
        if action == (-1, -1, -1):
            return

        # our env.step
        def opponent_step(env, action, block1, block2, block3):
            env._block_1 = block1
            env._block_2 = block2
            env._block_3 = block3
            env._block_valid_pos = []

            def update_block_pos(env, action):
                for i in range(3):
                    valid_list = []
                    for r in range(5):
                        for c in range(5):
                            if env._block_list[action[i]][r][c] == 1:
                                valid_list.append((r, c))
                    env._block_valid_pos.append(valid_list)

            update_block_pos(env, action)
            env._get_legal_actions()
            observation = env._get_obs()
            info = env._get_info()
            terminated = env._is_terminated()
            return observation, 0, terminated, False, info

        block1 = game_block[action[0]]
        block2 = game_block[action[1]]
        block3 = game_block[action[2]]

        observation, reward, terminated, _, info = opponent_step(self._woodoku_env, action, block1, block2, block3)
        self._cur_observation = observation
        self.terminated = terminated
        self._combo = info["combo"]
        self._straight = info["straight"]
        self.legal_action = [i for i in range(len(info["action_mask"])) if info["action_mask"][i] == 1]
        self._score = info["score"]
        return self, reward, terminated, info

    def generate_successor(self, action, agent_index=0):
        new_state = copy.deepcopy(self)

        # check if 2 of the blocks are 0
        if ((np.all(new_state.block1 == 0) and np.all(new_state.block2 == 0)) or (
                np.all(new_state.block1 == 0) and np.all(new_state.block3 == 0)) or (
                np.all(new_state.block2 == 0) and np.all(new_state.block3 == 0))):
            seed = hash(new_state) % 2 ** 32
            # seed = 1
            rng, np_seed = seeding.np_random(seed)
            new_state._woodoku_env._np_random = rng
            new_state._woodoku_env._np_random_seed = np_seed
        #     # new_state._woodoku_env.seed(seed)

        # Turn off rendering
        render_mode = new_state._woodoku_env.render_mode
        new_state._woodoku_env.render_mode = None

        # Apply action
        if agent_index == 0:
            new_state.apply_action(action)

        else:
            new_state.apply_opponent_action(action)


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