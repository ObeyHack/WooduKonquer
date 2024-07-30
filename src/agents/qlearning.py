from itertools import groupby

import gym_woodoku
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import random
import warnings

warnings.simplefilter('ignore')
import seaborn as sns
import collections

sns.set()


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha #learning rate
        self.gamma = gamma#discount factor
        self.epsilon = epsilon #exploration rate
        self.q_values = collections.defaultdict(float)

    def to_hashable(self,state):
        """Converts a state consisting of numpy arrays to a hashable type (tuple)."""
        return tuple([tuple(state['board'].flatten()), tuple(state['block_1'].flatten()), tuple(state['block_2'].flatten()), tuple(state['block_3'].flatten())])

    def get_q_value(self, state, action):
        # convert state to tuple
        (board, piece1, piece2, piece3) = state
        state_helper = self.to_hashable(state)
        return self.q_values[state_helper, action]



    def get_value(self, state,env=None):
        possible_actions = self.get_legal_actions(state,env)
        if len(possible_actions) == 0:
            return 0.0

        return max([self.get_q_value(state, 81*action[0] + 9*action[1] + action[2]) for action in possible_actions])

    def get_legal_actions(self, state,env=None):
        #state is a dict of board, piece1, piece2, piece3
        #board is the value of the board
        #piece1, piece2, piece3 are the 3 pieces that can be placed on the board
        board=state['board']
        piece1=state['block_1']
        piece2=state['block_2']
        piece3=state['block_3']
        pieces = [piece1, piece2, piece3]
        legal_actions = []

        for piece_index, piece in enumerate(pieces):
            #if piece is all zero matrix then skip
            if np.all(piece == 0):
                continue
            for row in range(9):
                for col in range(9):
                    #Place block_1 at ((action-0) // 9, (action-0) % 9) , Place block_2 at ((action-81) // 9, (action-81) % 9), Place block_3 at ((action-162) // 9, (action-162) % 9)
                    #calculate  Discrete(243) = 81*row + 9*col + piece_index
                    action_helper = 81*piece_index + 9*row + col
                    if env.env.env._is_valid_position(action_helper):
                        legal_actions.append((piece_index, row, col))
        return legal_actions






    def get_policy(self, state,env=None):
        possible_actions = self.get_legal_actions(state,env=env)
        if len(possible_actions) == 0:
            return None
        q_values = {action: self.get_q_value(state, 81*action[0] + 9*action[1] + action[2]) for action in possible_actions}
        return max(q_values, key=q_values.get)

    def get_action(self, state,env=None):
        action = None
        if random.random() < self.epsilon:
            the_action = random.choice(self.get_legal_actions(state,env=env))
            #remove the action from the list of possible actions
            return the_action
        action = self.get_policy(state,env=env)
        #remove the action from the list of possible actions
        return action
        # its a tuple of 3 values, the first value is the index of the piece, the second value is the row, the third value is the column

    def update(self, state, action, next_state, reward,env=None):
        q_value = self.get_q_value(state, 81*action[0] + 9*action[1] + action[2])
        next_q_value = self.get_value(next_state,env)
        new_q_value = q_value + self.alpha * (reward + self.gamma * next_q_value - q_value)
        (board, piece1, piece2, piece3) = state

        self.q_values[self.to_hashable(state), 81*action[0] + 9*action[1] + action[2]] = new_q_value
def reward_for_clearing_board(board, previous_board, done, truncated):
    reward = 0

    # Reward for clearing rows, columns, or 3x3 grids
    cleared_cells = np.sum(previous_board) - np.sum(board)
    return cleared_cells * 10  # higher reward for more cleared cells
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
            if board[row, col] == 0 and board[row-1, col] == 1 and board[row+1, col] == 1 and board[row, col-1] == 1 and board[row, col+1] == 1:
                isolated_blocks += 1
    reward -= isolated_blocks * 20  # penalty for isolated empty cells



    # Bonus for larger contiguous open spaces
    reward += np.max([len(list(g)) for k, g in groupby(np.nditer(board.flatten())) if k == 0])

    return reward
def train_agent(agent, env, num_episodes=1000):
    rewards = []
    steps = 0
    for episode in range(num_episodes):
        state, info = env.reset()
        #reset the agent for the new episode
        steps = 0
        total_reward = 0
        done = False
        truncated = False
        while not done and not truncated:
            action = agent.get_action(state,env=env)
            #get number of ones in the board
            number_of_ones = np.sum(state['board'])
            #if action is None then there are no legal actions
            if action is None:
                print(f'Episode {episode}, Reward {total_reward}')
                break
            helper_action = 81*action[0] + 9*action[1] + action[2]
            next_state, reward, done, truncated, info = env.step(helper_action)
            number_of_ones= np.sum(next_state['board'])-number_of_ones
            total_reward += reward
            reward= calculate_reward(next_state['board'],state['board'],done,truncated)
            #|number of ones| is the reward abs(number_of_ones)
            agent.update(state, action, next_state, reward,env=env)
            steps += 1
            state = next_state

        print (f'Episode {episode}, Reward {total_reward}')
        rewards.append(total_reward)

    return rewards


if __name__ == "__main__":
    #env = gym.make('gym_woodoku/Woodoku-v0', game_mode='woodoku', render_mode='human')
    # env = gym.wrappers.RecordVideo(env, video_folder='./video_folder')

    #observation, info = env.reset()
    #make env without render to run faster
    env= gym.make('gym_woodoku/Woodoku-v0', game_mode='woodoku', render_mode=None)
    observation, info = env.reset()
    # make qlearning agent here
    agent = QLearningAgent()#Reward of end: 812

    rewards = train_agent(agent, env, num_episodes=1000)
    #plot the rewards
    plt.plot(rewards)
    plt.show()
    #let the agent play the game
    env = gym.make('gym_woodoku/Woodoku-v0', game_mode='woodoku', render_mode='human')
    state, info = env.reset()
    done = False
    reward_sum = 0
    while not done:
        action = agent.get_policy(state,env=env)
        action = 81*action[0] + 9*action[1] + action[2]
        state, reward, done, truncated, info = env.step(action)
        reward_sum += reward
        env.render()

    print(f'Reward of end: {reward_sum}')

