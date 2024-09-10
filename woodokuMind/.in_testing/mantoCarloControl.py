import collections
import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
from woodokuMind.RLgame import RLgameState
from woodokuMind.agents.RL_agents.qlearning import QLearningAgent


class MCControlAgent(QLearningAgent):
    def __init__(self, **args):
        super().__init__(**args)
        self.q_values = collections.defaultdict(float)

    def update(self, state: RLgameState, action, returns):
        self.q_values[state, action] = np.mean(returns[state, action])
        return self.q_values[state, action]

    def train_agent(self, env, max_steps=10, logger=None):
        def generate_episode(steps):
            episode = []
            obs, info = env.reset()
            state = RLgameState(env, obs, info)
            step = 0

            # Make the state random by randomly applying 0-2 random actions
            for _ in range(random.randint(0, 2)):
                action = random.choice(state.get_legal_actions())
                state = state.apply_action(action)[0]

            while step < steps and not state.is_terminated():
                action = self.get_action(state)
                next_state, reward, terminated, info = state.apply_action(action)
                episode.append((state, action, reward))
                state = next_state
            return episode

        returns = collections.defaultdict(list)
        for _ in tqdm(range(self.num_episodes)):
            G = 0
            episode = generate_episode(max_steps)

            for i, (state, action, reward) in enumerate(reversed(episode)):
                G = self.discount * G + reward
                returns[state, action].append(G)
                self.update(state, action, returns)

        self.is_trained = True




