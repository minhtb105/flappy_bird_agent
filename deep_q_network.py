from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from configs.dqn_configs import GAMMA, LEARNING_RATE, BATCH_SIZE, EPSILON_START, EPSILON_MIN, EPSILON_DECAY, MEMORY_SIZE

class DeepQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        return self.fc4(x)


class FlappyBirdAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = GAMMA
        self.lr = LEARNING_RATE
        self.batch_size = BATCH_SIZE
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY

        self.replay_buffer = deque(maxlen=MEMORY_SIZE)

        self.policy_net = DeepQNetwork(state_dim, action_dim)
        self.target_net = DeepQNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, state, steps_done=0):
        """
        Selects an action using an epsilon-greedy strategy.
        """
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * (self.epsilon_decay ** (1 / (1 + steps_done / 10000))))  # Slower decay

        if random.random() < self.epsilon:
            return torch.tensor([[torch.randint(self.action_dim, (1,))]], dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy_net(state.float()).max(1)[1].view(1, 1)

    def store_transition(self, state, action, reward, next_state):
        self.replay_buffer.append((state, action, reward, next_state))

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)

        # Compute Q_values
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + self.gamma * next_q_values

        # Update policy network
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
