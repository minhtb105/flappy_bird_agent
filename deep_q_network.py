import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import numpy as np
from configs.dqn_configs import *

class DeepQNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))

        return self.fc4(x)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        """
            Implements Prioritized Experience Replay (PER).
            :param capacity: Max size of replay buffer
            :param alpha: Priority exponent (0 = uniform sampling, 1 = full priority)
        """
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)
        self.position = 0  # # Pointer for inserting experiences
        self.priorities = np.zeros((capacity,), dtype=np.float32)  # Stores priority values
        self.alpha = alpha


    def store_transition(self, state, action, reward, next_state):
        """Store transition with maximum priority."""
        max_priority = self.priorities.max() if self.memory else 1.0  # Default priority for new experiences

        if len(self.memory) < self.capacity:
            self.memory.append((state, action, reward, next_state))
        else:
            self.memory[self.position] = (state, action, reward, next_state)

        self.priorities[self.position] = max_priority  # assign max priority
        self.position = (self.position + 1) % self.capacity  # circular buffer

    def sample(self, batch_size, beta=0.4):
        """Samples a batch using priority-based probability distribution."""
        if len(self.memory) == 0:
            return None  # Avoid error if buffer is empty

        priorities = self.priorities[: len(self.memory)] ** self.alpha
        probabilities = priorities / priorities.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)  # sample indices
        experiences = [self.memory[idx] for idx in indices]

        # Compute importance-sampling weights to reduce bias
        weights = (len(self.memory) * probabilities[indices]) ** (-beta)
        weights /= weights.max()  # Normalize weights

        return experiences, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices, td_errors):
        """Updates the priorities of sampled transitions."""
        self.priorities[indices] = td_errors + 0.01  # Avoid zero priority


class FlappyBirdAgent:
    def __init__(self, input_dim, action_dim):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.gamma = GAMMA
        self.lr = LEARNING_RATE
        self.batch_size = BATCH_SIZE
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY

        self.replay_buffer = PrioritizedReplayBuffer(MEMORY_SIZE)

        self.policy_net = DeepQNetwork(input_dim, action_dim)
        self.target_net = DeepQNetwork(input_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, state):
        """
        Selects an action using an epsilon-greedy strategy.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * 0.999)  # More exploration

        if random.random() < self.epsilon:
            return torch.tensor([[torch.randint(self.action_dim, (1,))]], dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy_net(state.float()).max(1)[1].view(1, 1)

    def train(self):
        """
        Trains the model using Double DQN with Prioritized Experience Replay.
        """
        if len(self.replay_buffer.memory) < self.batch_size:
            return   # Skip training if not enough samples

        experiences, indices, weights = self.replay_buffer.sample(BATCH_SIZE)
        states, actions, rewards, next_states = zip(*experiences)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)

        # Compute current Q-values
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        # Compute target Q-values using target network
        with torch.no_grad():
            best_action = self.policy_net(next_states).argmax(dim=-1, keepdim=True)  # Select best action using policy_net
            next_q_values = self.target_net(next_states).gather(1, best_action).squeeze(1)  # Evaluate using target_net
            target_q_values = rewards + self.gamma * next_q_values
            target_q_values = (target_q_values - target_q_values.mean()) / (target_q_values.std() + 1e-5)

        # Compute TD-Error
        td_errors = abs(target_q_values - q_values)

        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors.abs().detach().numpy())

        # Update policy network
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
