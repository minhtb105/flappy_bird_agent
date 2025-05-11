import numpy as np
import torch
from configs.dqn_configs import *
from collections import deque
import logging
import time

# Setup logging
logging.basicConfig(filename='logs/debug_log.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

last_logged_errors = {}

def log_once_per(error_key: str, message: str, interval_seconds: int = 500):
    current_time = time.time()
    if error_key not in last_logged_errors or (current_time - last_logged_errors[error_key]) > interval_seconds:
        logging.error(message)
        last_logged_errors[error_key] = current_time

# Error counter
error_counts = {
    "store_transition": 0,
    "sample": 0,
    "update_priorities": 0,
    "to_torch_dict": 0,
    "load_from_dict": 0,
    "insufficient_memory": 0,
}

def log_error_stats(step, interval=1000):
    if step % interval == 0:
        for key, count in error_counts.items():
            if count > 0:
                logging.warning(f"[{key}] occurred {count} times in last {interval} steps")
                error_counts[key] = 0


class PrioritizedReplayBuffer:
    def __init__(self, capacity=MAX_REPLAY_SIZE, alpha=ALPHA_INIT, beta=BETA):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        self.beta = beta
        logging.debug(f"Initialized PER buffer with capacity {capacity} and alpha {alpha}")

    def store_transition(self, state, action, reward, next_state):
        try:
            max_priority = self.priorities.max() if self.memory else 1.0
            if len(self.memory) < self.capacity:
                self.memory.append((state, action, reward, next_state))
            else:
                self.memory[self.position] = (state, action, reward, next_state)
            self.priorities[self.position] = max_priority
            self.position = (self.position + 1) % self.capacity
        except Exception as e:
            error_counts["store_transition"] += 1
            log_once_per("store_transition", f"Store transition error: {e}")

    def sample(self, batch_size=BATCH_SIZE):
        if len(self.memory) < MIN_REPLAY_SIZE:
            log_once_per("insufficient_memory", "Insufficient replay memory to sample from")
            error_counts["insufficient_memory"] += 1
            
            return None

        try:
            priorities = self.priorities[: len(self.memory)] ** self.alpha
            priorities = np.nan_to_num(priorities, nan=1.0, posinf=1.0, neginf=0.0)
            total = priorities.sum()
            probabilities = np.ones_like(priorities) / len(priorities) if total == 0 or np.isnan(total) else priorities / total
            indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
            experiences = [self.memory[idx] for idx in indices]

            states, actions, rewards, next_states = zip(*experiences)

            states = torch.tensor(np.stack(states), dtype=torch.float32)
            next_states = torch.tensor(np.stack(next_states), dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=torch.float32)

            weights = (len(self.memory) * probabilities[indices]) ** (-self.beta)
            weights /= weights.max()
            weights = torch.tensor(weights, dtype=torch.float32)

            return states, actions, rewards, next_states, indices, weights

        except Exception as e:
            error_counts["sample"] += 1
            log_once_per("sample", f"Sample error: {e}")
            
            return None

    def update_priorities(self, indices, td_errors):
        try:
            td_errors = np.clip(td_errors, -50, 50)
            td_errors = np.nan_to_num(td_errors, nan=1.0, posinf=10.0, neginf=-10.0)
            self.priorities[indices] = np.abs(td_errors) + 1e-6
            std_dev = np.std(td_errors)
            if std_dev > 100:
                logging.warning(f"TD error variance too high: {std_dev:.2f}")
            logging.debug(f"Updated priorities for {len(indices)} indices")
        except Exception as e:
            error_counts["update_priorities"] += 1
            log_once_per("update_priorities", f"Update priorities error: {e}")

    def decay_alpha(self, steps_done):
        """Exponential decay of alpha."""
        old_alpha = self.alpha
        self.alpha = max(ALPHA_FINAL, old_alpha * np.exp(-ALPHA_DECAY * steps_done))
        if steps_done % 10000 == 0:
            logging.info(f"Alpha decayed from {old_alpha:.4f} to {self.alpha:.4f} at step {steps_done}")

    def to_torch_dict(self):
        try:
            states, actions, rewards, next_states = zip(*[(s, a, r, ns) for s, a, r, ns in self.memory])
            return {
                'states': torch.tensor(np.stack(states), dtype=torch.float32),
                'actions': torch.tensor(actions, dtype=torch.long),
                'rewards': torch.tensor(rewards, dtype=torch.float32),
                'next_states': torch.tensor(np.stack(next_states), dtype=torch.float32),
                'priorities': torch.tensor(self.priorities, dtype=torch.float32),
                'position': self.position
            }
        except Exception as e:
            error_counts["to_torch_dict"] += 1
            log_once_per("to_torch_dict", f"to_torch_dict error: {e}")
            
            return {}

    def load_from_torch_dict(self, data):
        try:
            self.memory.clear()
            self.position = data['position']
            self.priorities = data['priorities'].numpy()

            for i in range(len(data['states'])):
                transition = (
                    data['states'][i].numpy(),
                    data['actions'][i].item(),
                    data['rewards'][i].item(),
                    data['next_states'][i].numpy(),
                )
                self.memory.append(transition)

            if len(self.memory) > self.capacity:
                logging.warning("Loaded memory exceeds capacity")
            logging.debug(f"Loaded {len(self.memory)} transitions from saved state")
        except Exception as e:
            error_counts["load_from_dict"] += 1
            log_once_per("load_from_dict", f"Load from torch dict error: {e}")
