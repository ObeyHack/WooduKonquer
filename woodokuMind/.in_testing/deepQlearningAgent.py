import collections
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from woodokuMind.RLgame import RLgameState, RLAgent

class DeepQlearningAgent(RLAgent):
    def __init__(self, alpha=1.0, epsilon=0.05, gamma=0.8):
        super().__init__(alpha, epsilon, gamma)
        self.model = DQN(input_size=156, hidden_size=1000, num_classes=243)
        self.target_model = DQN(input_size=156, hidden_size=1000, num_classes=243)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma

    def vectorize_state(self, state):  # 156 size
        board_tensor = torch.tensor(state.board.astype(float)).float()
        block1_tensor = torch.tensor(state.block1.astype(float)).float()
        block2_tensor = torch.tensor(state.block2.astype(float)).float()
        block3_tensor = torch.tensor(state.block3.astype(float)).float()
        state_tensor = torch.cat(
            (board_tensor.flatten(), block1_tensor.flatten(), block2_tensor.flatten(), block3_tensor.flatten()), 0)
        return state_tensor

    def get_q_value(self, state: RLgameState, action):
        state_tensor = self.vectorize_state(state)
        action = torch.tensor(action).float()
        return self.model(state_tensor).flatten()[action]

    def get_value(self, state: RLgameState):
        state_tensor = self.vectorize_state(state)
        return self.model(state_tensor).max().item()

    def get_policy(self, state: RLgameState):
        state_tensor = self.vectorize_state(state)
        return self.model(state_tensor).argmax().item()

    def get_action(self, state: RLgameState):
        state_tensor = self.vectorize_state(state)
        if np.random.rand() < self.epsilon:
            return random.choice(state.get_legal_actions())
        else:
            with torch.no_grad():  # Ensure no gradient is tracked
                q_values = self.model(state_tensor)
                # legel_actions = state.get_legal_actions()
                # take argmax of legal actions
                return q_values.argmax().item()

    def update(self, state, action, nextState, reward):
        # Vectorize the state and next state
        state_tensor = self.vectorize_state(state)
        next_state_tensor = self.vectorize_state(nextState)

        # Convert action to tensor
        action_tensor = torch.tensor([action]).long()

        # Get the current Q-value for the given state and action
        current_q_value = self.model(state_tensor).gather(0, action_tensor)

        # Get the maximum Q-value for the next state
        next_q_value = self.target_model(next_state_tensor).max().item()

        # Calculate the target Q-value
        target_q_value = reward + self.gamma * next_q_value

        # Compute the loss
        target_q_value_tensor = torch.tensor([target_q_value]).float()
        loss = self.loss_fn(current_q_value, target_q_value_tensor)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Optionally update the target network to be closer to the main network
        # This could be done periodically, not every time.
        # make it every 1000 steps

        self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x.flatten()))
        x = self.fc2(x)
        return x