import random 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from configs.dqn_configs import *
from configs.game_configs import NUM_RAYS
from dueling_motion_transformers import DuelingMotionTransformer
from replay_buffer import PrioritizedReplayBuffer


class FlappyBirdAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = GAMMA
        self.lr = LEARNING_RATE
        self.batch_size = BATCH_SIZE
        self.epsilon = TEMP_INIT
        self.epsilon_min = TEMP_MIN
        self.epsilon_decay = TEMP_DECAY

        self.insert_count = 0
        self.samples_per_insert = SAMPLES_PER_INSERT_RATIO

        self.replay_buffer = PrioritizedReplayBuffer(MAX_REPLAY_SIZE)

        self.policy_net = DuelingMotionTransformer(state_dim=state_dim, action_dim=action_dim)
        self.target_net = DuelingMotionTransformer(state_dim=state_dim, action_dim=action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE,
                                    weight_decay=WEIGHT_DECAY)

    def choose_action(self, state, steps_done=0):
        """
        Selects an action using an epsilon-greedy strategy.
        """
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * (self.epsilon_decay ** (1 / (1 + steps_done / 5000))))  # Slower decay

        if random.random() < self.epsilon:
            return torch.tensor([[torch.randint(self.action_dim, (1,))]], dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].item()

    def train(self):
        """
        Trains the model using Double DQN with Prioritized Experience Replay.
        """
        if self.insert_count % self.samples_per_insert != 0:
            return
        
        if len(self.replay_buffer.memory) < self.batch_size:
            return   # Skip training if not enough samples

        experiences, indices, weights = self.replay_buffer.sample()
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

        # Compute TD-Error
        td_errors = target_q_values - q_values

        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors.abs().detach().numpy())

        # Update policy network
        loss = torch.log(torch.cosh(q_values - target_q_values)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), GLOBAL_CLIP_NORM)
        self.optimizer.step()
