import numpy as np
import torch
from configs.dqn_configs import FRAME_STACK, MAX_REPLAY_SIZE, BATCH_SIZE
from collections import deque


class PrioritizedReplayBuffer:
    def __init__(self, capacity=MAX_REPLAY_SIZE, alpha=0.6):
        """
            Implements Prioritized Experience Replay (PER).
            :param capacity: Max size of replay buffer
            :param alpha: Priority exponent (0 = uniform sampling, 1 = full priority)
        """
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)  # Circular buffer for storing experiences
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

    def sample(self, batch_size=BATCH_SIZE, beta=0.4):
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
        self.priorities[indices] = abs(td_errors) + 0.001  # Avoid zero priority
