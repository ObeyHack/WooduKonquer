from itertools import groupby
import numpy as np
from tqdm import tqdm
from src import util
from src.game import Agent, GameState, Game
from gymnasium import Env
import gymnasium as gym


class RLgameState(GameState):
    def __init__(self, env: Env, observation):
        self._env = env
        self._woodoku_env = self._env.env.env
        self._cur_observation = observation
        self.terminated = False
        self._combo = 0
        self._straight = 0

    def get_legal_actions(self):
        # state is a dict of board, piece1, piece2, piece3
        # board is the value of the board
        # piece1, piece2, piece3 are the 3 pieces that can be placed on the board
        board = self._cur_observation['board']
        piece1 = self._cur_observation['block_1']
        piece2 = self._cur_observation['block_2']
        piece3 = self._cur_observation['block_3']
        pieces = [piece1, piece2, piece3]
        legal_actions = []

        for piece_index, piece in enumerate(pieces):
            # if piece is all zero matrix then skip
            if np.all(piece == 0):
                continue
            for row in range(9):
                for col in range(9):
                    # Place block_1 at ((action-0) // 9, (action-0) % 9) ,
                    # Place block_2 at ((action-81) // 9, (action-81) % 9),
                    # Place block_3 at ((action-162) // 9, (action-162) % 9)
                    # calculate  Discrete(243) = 81*row + 9*col + piece_index
                    action_helper = 81 * piece_index + 9 * row + col
                    if self._woodoku_env._is_valid_position(action_helper):
                        legal_actions.append((piece_index, row, col))

        legal_actions = [81 * action[0] + 9 * action[1] + action[2] for action in legal_actions]
        return legal_actions

    def apply_action(self, action):
        observation, reward, terminated, _, info = self._env.step(action)
        new_state = RLgameState(self._env, observation)
        self.terminated = terminated
        self._combo = info["combo"]
        self._straight = info["straight"]
        return new_state, reward, terminated, info

    def __eq__(self, othr):
        return (isinstance(othr, type(self))
                and (self.board, self.block1, self.block2, self.block3) ==
                (othr.board, othr.block1, othr.block2, othr.block3))

    def __hash__(self):
        """Converts a state consisting of numpy arrays to a hashable type (tuple)."""
        return hash((self.board.tostring(), self.block1.tostring(), self.block2.tostring(), self.block3.tostring()))

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
            plt.show()

        rewards = []
        for episode in tqdm(range(num_episodes)):
            state = RLgameState(env, env.reset()[0])
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
        return rewards


class RLGame(Game):
    def __init__(self, env: Env, agent: RLAgent):
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
        # Turn off rendering
        render_mode = self.env.env.env.render_mode
        self.env.env.env.render_mode = None

        # Train the agent
        print("Training the agent")
        RLAgent.train_agent(self.agent, self.env, num_episodes=1000, plot_rewards=True)

        # Turn on rendering
        self.env.env.env.render_mode = render_mode

        state = RLgameState(self.env, self.env.reset()[0])
        terminated = False
        score = 0
        while not terminated:
            # render the environment
            self.env.render()
            # take action
            action = self.agent.get_action(state)
            state, reward, terminated, info = state.apply_action(action)
            score = state.score
        return score
