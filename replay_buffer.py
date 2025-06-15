import numpy as np
import torch
import zarr
from zarr.storage import LocalStore
from zarr.codecs import BloscCodec
import logging
from configs.dqn_configs import FRAME_STACK, ALPHA_INIT, BETA
from configs.game_configs import NUM_RAYS


class ZarrPrioritizedReplayBuffer:
    def __init__(self,
                 path: str,
                 capacity: int,
                 state_shape: int = 180,
                 action_dim: int = 2,
                 chunk_size: int = 1024,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 overwrite: bool = True):
        
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.state_shape = state_shape
        self.chunk_size = chunk_size
        self.state_shape = state_shape
        self.action_dim = action_dim
    
        if overwrite:
            import shutil
            shutil.rmtree(path, ignore_errors=True)
    
        # Setup Zarr store
        store = LocalStore(path, read_only=False)
        mode = 'w' if overwrite else 'a'
        root = zarr.open_group(store=store, mode=mode)
        
        # dataset shapes
        states_shape = (capacity, FRAME_STACK, state_shape)
        actions_shape = (capacity, action_dim)
        rewards_shape = (capacity,)
        dones_shape = (capacity,)
        priorities_shape = (capacity,)
        
        # chunk along first dim
        chunks_states = (chunk_size, FRAME_STACK, state_shape)
        chunks_actions = (chunk_size, action_dim)
        chunks_1d = (chunk_size,)

        # require datasets
        self.states = root.require_dataset(
            name="states",
            shape=states_shape, 
            chunks=chunks_states,
            dtype=np.float32, 
            fill_value=0.0,
        )
        self.next_states = root.require_dataset(
            name="next_states",
            shape=states_shape, 
            chunks=chunks_states,
            dtype=np.float32,
            fill_value=0.0,
        )
        self.actions = root.require_dataset(
            name="actions",
            shape=actions_shape,
            chunks=chunks_actions,
            dtype=np.int64,
            fill_value=0,
        )
        self.rewards = root.require_dataset(
            name="rewards",
            shape=rewards_shape,
            chunks=chunks_1d,
            dtype=np.float32,
            fill_value=0.0,
        )
        self.dones = root.require_dataset(
            name="dones",
            shape=dones_shape,
            chunks=chunks_1d,
            dtype=bool,
            fill_value=False,
        )
        self.priorities = root.require_dataset(
            name="priorities",
            shape=priorities_shape,
            chunks=chunks_1d,
            dtype=np.float32,
            fill_value=0.0,
        )

        # attrs for current position and size
        self.group = root
        attrs = root.attrs
        if overwrite:
            attrs['pos'] = 0
            attrs['size'] = 0
        self.pos = int(attrs.get('pos', 0))
        self.size = int(attrs.get('size', 0))
        
        self._ram_buffer = []

    def store(self, state, action, reward, next_state, done, priority=None):
        try:
            if self.size == 0:
                max_priority = 1.0
            else:
                max_priority = float(self.priorities[:self.size].max())
                
            p = priority if priority is not None else max_priority
            self._ram_buffer.append((state, action, reward, next_state, done, priority))
            if len(self._ram_buffer) >= self.chunk_size:
                self._flush_ram_buffer_to_zarr()
                
        except Exception as e:
            logging.error(f"Error storing transition: {e}")

    def _flush_ram_buffer_to_zarr(self):
        for sample in self._ram_buffer:
            state, action, reward, next_state, done, priority = sample
            idx = self.pos
            self.states[idx] = state
            self.actions[idx] = action
            self.rewards[idx] = reward
            self.next_states[idx] = next_state
            self.dones[idx] = done
            self.priorities[idx] = priority if priority is not None else 1.0
            self.pos = (idx + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
        self._ram_buffer.clear()

    def sample(self, batch_size: int):
        if self.size < batch_size:
            return None
        
        # compute probabilities
        priorities = np.array(self.priorities[:self.size]) ** self.alpha
        probs = priorities / priorities.sum()
        indices = np.random.choice(self.size, batch_size, p=probs)

        # fetch
        states = torch.tensor(self.states[indices], dtype=torch.float32)
        actions = torch.tensor(self.actions[indices], dtype=torch.long)
        rewards = torch.tensor(self.rewards[indices], dtype=torch.float32)
        next_states = torch.tensor(self.next_states[indices], dtype=torch.float32)
        dones = torch.tensor(self.dones[indices], dtype=torch.bool)

        # importance weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, td_errors):
        # smooth
        ps = np.sqrt(np.abs(td_errors) + 1e-6)
        self.priorities[indices] = ps

    def update_beta(self, steps_done, beta_final=1.0,
                    beta_start=0.4, beta_frames=3_000_000):
        """Linearly anneal beta from beta_start to beta_final over
        beta_frames steps.
        """
        old_beta = self.beta
        self.beta = min(beta_final,
                        beta_start + (beta_final - beta_start) * steps_done / beta_frames)
        if steps_done % 10000 == 0:
            logging.info(f"Beta updated from {old_beta:.4f} to {self.beta:.4f} at step {steps_done}")

    def save_checkpoint(self):
        self.group.attrs['pos'] = self.pos
        self.group.attrs['size'] = self.size

    def load_checkpoint(self, buffer_file: str):
        self.group = zarr.open_group(buffer_file, mode='r+')
        self.pos  = self.group.attrs.get('pos', 0)
        self.size = self.group.attrs.get('size', 0)

    def __len__(self):
        return self.size