import numpy as np
import torch
import logging
from collections import deque
from configs.dqn_configs import *
from configs.game_configs import NUM_RAYS


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
            td_errors = np.clip(td_errors, 1e-6, 5)
            
            td_errors = np.nan_to_num(td_errors, nan=0.0, posinf=5.0, neginf=1e-6)
            
            smoothed_priorities = np.sqrt(np.abs(td_errors) + 1e-6)
            self.priorities[indices] = smoothed_priorities
        except Exception as e:
            print(e)

    def decay_alpha(self, steps_done):
        """Exponential decay of alpha."""
        self.alpha = ALPHA_FINAL + (ALPHA_INIT - ALPHA_FINAL) * np.exp(-ALPHA_DECAY * steps_done)

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

        torch.save(filtered_data, f"models/replay_buffer.pt")

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

    def load_from_torch_dict(self):
        try:
            data = torch.load("models/buffer.pth")
            self.memory.clear()
            self.priorities = data['priorities'].cpu().numpy() if hasattr(data['priorities'], 'cpu') else data['priorities']

            for i in range(min(len(data['states']), MAX_REPLAY_SIZE)):
                transition = (
                    data['states'][i].cpu().numpy(),
                    data['actions'][i].item() if hasattr(data['actions'][i], 'item') else data['actions'][i],
                    data['rewards'][i].item() if hasattr(data['rewards'][i], 'item') else data['rewards'][i],
                    data['next_states'][i].cpu().numpy(),
                )
                self.memory.append(transition)

            if len(self.memory) > self.capacity:
                logging.warning("Loaded memory exceeds capacity")
                
            logging.debug(f"Loaded {len(self.memory)} transitions from saved state")
        except Exception as e:
            print(e)

    def update_beta(self, steps_done, beta_final=1.0,
                    beta_start=0.4, beta_frames=3_000_000):
        """Linearly anneal beta from beta_start to beta_final over
        beta_frames steps.
        """
        self.beta = min(beta_final,
                        beta_start + (beta_final - beta_start) * steps_done / beta_frames)