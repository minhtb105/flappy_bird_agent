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

        states, actions, rewards, next_states = zip(*experiences)

        states = torch.tensor(np.stack(states), dtype=torch.float32)  # (batch_size, FRAME_STACK, state_dim)
        next_states = torch.tensor(np.stack(next_states), dtype=torch.float32)  # (batch_size, FRAME_STACK, state_dim)
        actions = torch.tensor(actions, dtype=torch.long)  # (batch_size,)
        rewards = torch.tensor(rewards, dtype=torch.float32)  # (batch_size,)
        
        # Compute importance-sampling weights to reduce bias
        weights = (len(self.memory) * probabilities[indices]) ** (-beta)
        weights /= weights.max()  # Normalize weights
        weights = torch.tensor(weights, dtype=torch.float32)
        
        return states, actions, rewards, next_states, indices, weights

    def update_priorities(self, indices, td_errors):
        """Updates the priorities of sampled transitions."""
        self.priorities[indices] = abs(td_errors) + 0.001  # Avoid zero priority

    def to_torch_dict(self):
        states, actions, rewards, next_states = zip(*[(s, a, r, ns) for s, a, r, ns in self.memory])
        return {
            'states': torch.tensor(np.stack(states), dtype=torch.float32),
            'actions': torch.tensor(actions, dtype=torch.long),
            'rewards': torch.tensor(rewards, dtype=torch.float32),
            'next_states': torch.tensor(np.stack(next_states), dtype=torch.float32),
            'priorities': torch.tensor(self.priorities, dtype=torch.float32),
            'position': self.position
        }
        
    def load_from_torch_dict(self, data):
        self.memory.clear()
        self.position = data['position']
        self.priorities = data['priorities'].numpy()

        for i in range(len(data['states'])):
            transition = (
                data['states'][i].numpy(),
                data['actions'][i].item(),
                data['rewards'][i].item(),
                data['next_states'][i].numpy(),
                self.priorities
            )
            self.memory.append(transition)
        