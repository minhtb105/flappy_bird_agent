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
        self.buffer_len = 0
        self.last_saved_idx = 0
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
            self.buffer_len += 1
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

    def get_new_transitions(self):
        if self.last_saved_idx >= self.buffer_len:
            return None

        new_transitions = list(self.memory)[self.last_saved_idx:]
        priorities = self.priorities[self.last_saved_idx:]

        return new_transitions, priorities

    def filter_buffer(self, episode=0, random_ratio=0.2, reward_ratio=0.4, pr_ratio=0.4, buffer_size=40000):
        if self.last_saved_idx >= self.buffer_len:
            return None

        states, actions, rewards, next_states = zip(*list(self.memory)[self.last_saved_idx:])
        priorities = self.priorities[self.last_saved_idx:]

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        priorities = np.array(priorities)

        n = len(states)
        if buffer_size is None:
            buffer_size = n
            
        n_random = int(buffer_size * random_ratio)
        n_reward = int(buffer_size * reward_ratio)
        n_pr = int(buffer_size * pr_ratio)
        
        all_indices = np.arange(n)
        
        random_indices = np.random.choice(all_indices, n_random, replace=False)
        reward_indices = np.argpartition(-rewards, n_reward)[:n_reward]
        priorities_indices = np.argpartition(-priorities, n_pr)[:n_pr]

        selected_indices = np.unique(np.concatenate([random_indices, reward_indices, priorities_indices]))

        if len(selected_indices) > buffer_size:
            selected_indices = np.random.choice(selected_indices, buffer_size, replace=False)

        filtered_data = {
            'states': states[selected_indices],
            'actions': actions[selected_indices],
            'rewards': rewards[selected_indices],
            'next_states': next_states[selected_indices],
            "priorities": priorities[selected_indices]
        }

        torch.save(filtered_data, f"models/replay_buffer_ep{episode}.pt")

    def load_and_merge_replay_buffers(self, paths=None):
        """
        Load and merge multiple .pt replay buffer files, then assign to replay_buffer.
        """
        import glob
        if paths is None:
            paths = sorted(glob.glob("models/replay_buffer_ep*.pt"))

        if not paths:
            logging.warning("No replay buffer files found to load.")
            return {}

        all_states, all_actions, all_rewards, all_next_states, all_priorities = [], [], [], [], []

        for path in paths:
            data = torch.load(path, weights_only=False)
            all_states.append(data['states'])
            all_actions.append(data['actions'])
            all_rewards.append(data['rewards'])
            all_next_states.append(data['next_states'])
            all_priorities.append(data['priorities'])

        states = np.concatenate(all_states, axis=0)
        actions = np.concatenate(all_actions, axis=0)
        rewards = np.concatenate(all_rewards, axis=0)
        next_states = np.concatenate(all_next_states, axis=0)
        priorities = np.concatenate(all_priorities, axis=0)

        # Clear and refill replay buffer
        self.memory.clear()
        self.position = 0
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)

        n = min(len(states), self.capacity)
        for i in range(n):
            self.memory.append((states[i], actions[i], rewards[i], next_states[i]))
            self.priorities[i] = priorities[i]
        self.position = n % self.capacity
        self.last_saved_idx = n % self.capacity

        logging.info(f"Merged and loaded {n} transitions into replay buffer.")

    def to_torch_dict(self, path, new_transitions=None, priorities=None):
        if new_transitions is None:
            new_transitions = self.memory
        if priorities is None:
            priorities = self.priorities

        try:
            states, actions, rewards, next_states = zip(*[(s, a, r, ns) for s, a, r, ns in new_transitions])
            torch.save({
                'states': torch.tensor(np.stack(states), dtype=torch.float32),
                'actions': torch.tensor(actions, dtype=torch.long),
                'rewards': torch.tensor(rewards, dtype=torch.float32),
                'next_states': torch.tensor(np.stack(next_states), dtype=torch.float32),
                'priorities': torch.tensor(priorities, dtype=torch.float32),
            }, path)
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