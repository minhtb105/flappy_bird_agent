import numpy as np
import torch
from configs.dqn_configs import *
from collections import deque
import logging
import time

# Setup logging
logging.basicConfig(filename='logs/debug_log.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

last_logged_errors = {}

def log_once_per(error_key: str, message: str, interval_seconds: int = 500, level: str = "error"):
    """
    Log a message once every `interval_seconds` seconds per unique error_key.

    Args:
        error_key (str): Unique key to identify the error type.
        message (str): Message to log.
        interval_seconds (int): Cooldown time before re-logging same key.
        level (str): Log level: 'error', 'warning', 'info', or 'debug'.
    """
    current_time = time.time()
    if error_key not in last_logged_errors or (current_time - last_logged_errors[error_key]) > interval_seconds:
        if level == "warning":
            logging.warning(message)
        elif level == "info":
            logging.info(message)
        elif level == "debug":
            logging.debug(message)
        else:
            logging.error(message)
            
        last_logged_errors[error_key] = current_time

# Error counter
error_counts = {
    "high_td_error": 0,
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
            print(e)

    def sample(self, batch_size=BATCH_SIZE):
        if len(self.memory) < MIN_REPLAY_SIZE:
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
            print(e)
            
            return None

    def update_priorities(self, indices, td_errors):
        try:
            td_errors = np.clip(td_errors, -10, 10)
            
            td_errors = np.nan_to_num(td_errors, nan=1.0, posinf=10.0, neginf=-10.0)
            
            smoothed_priorities = np.sqrt(np.abs(td_errors) + 1e-6)
            self.priorities[indices] = smoothed_priorities

            std_dev = np.std(td_errors)
            if std_dev > 100:
                log_once_per("high_td_error", f"High TD error variance: {std_dev:.2f}", interval_seconds=600, level="warning")
        except Exception as e:
            print(e)

    def decay_alpha(self, steps_done):
        """Exponential decay of alpha."""
        old_alpha = self.alpha
        self.alpha = ALPHA_FINAL + (ALPHA_INIT - ALPHA_FINAL) * np.exp(-ALPHA_DECAY * steps_done)
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
            print(e)
            
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
            print(e)
