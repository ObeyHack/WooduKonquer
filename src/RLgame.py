from itertools import groupby
import numpy as np
from tqdm import tqdm
from src import util
from src.game import Agent, GameState, Game
from gymnasium import Env
import gymnasium as gym


class RLgameState(GameState):

    def apply_action(self, action):
        from copy import deepcopy
        new_state = deepcopy(self)
        return new_state.apply_action(action)

    @staticmethod
    def reward_for_clearing_board(board, previous_board, done, truncated):
        reward = 0

        # Reward for clearing rows, columns, or 3x3 grids
        cleared_cells = np.sum(previous_board) - np.sum(board)
        return cleared_cells * 10  # higher reward for more cleared cells

    @staticmethod
    def calculate_reward(board, previous_board, done, truncated):
        reward = 0

        # Reward for clearing rows, columns, or 3x3 grids
        cleared_cells = np.sum(previous_board) - np.sum(board)
        reward += cleared_cells * 10  # higher reward for more cleared cells

        # Bonus for clearing multiple rows/columns/grids
        reward += (cleared_cells // 9) * 50  # additional bonus for clearing entire rows/columns

        # Small positive reward for a valid move
        reward += 1

        # Penalty for near game over (few empty cells)
        empty_cells = np.sum(board == 0)
        if empty_cells < 10:
            reward -= 10 * (10 - empty_cells)  # larger penalty as the board fills up

        # Penalty for isolated blocks (empty cells surrounded by filled cells)
        isolated_blocks = 0
        for row in range(1, 8):
            for col in range(1, 8):
                if board[row, col] == 0 and board[row - 1, col] == 1 and board[row + 1, col] == 1 and board[
                    row, col - 1] == 1 and board[row, col + 1] == 1:
                    isolated_blocks += 1
        reward -= isolated_blocks * 20  # penalty for isolated empty cells

        # Bonus for larger contiguous open spaces
        reward += np.max([len(list(g)) for k, g in groupby(np.nditer(board.flatten())) if k == 0])

        return reward


class RLAgent(Agent):
    def __init__(self, alpha=1.0, epsilon=0.05, gamma=0.8):
        """
        Sets options, which can be passed in via the Pacman command line using -a alpha=0.5,...
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(gamma)
        self.is_trained = False

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

    @staticmethod
    def train_agent(agent, env, num_episodes=1000, max_steps=1000, plot_rewards=False):
        def plot(rewards):
            import matplotlib.pyplot as plt
            plt.plot(rewards)
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Reward vs Episode')

        rewards = []
        for episode in tqdm(range(num_episodes)):
            obs, info = env.reset()
            state = RLgameState(env, obs, info)
            step = 0
            run_reward = 0
            while step < max_steps:
                action = agent.get_action(state)
                next_state, reward, terminated, info = state.apply_action(action)
                if state.is_terminated():
                    break
                run_reward += reward
                agent.update(state, action, next_state, reward)
                step += 1
                state = next_state

            #print(f'Episode {episode}, Reward {run_reward}, Score {state.score}')
            rewards.append(run_reward)

        if plot_rewards:
            plot(rewards)

        agent.set_trained()
        return rewards