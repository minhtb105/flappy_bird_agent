import random
import torch
import torch.optim as optim
from configs.dqn_configs import *
from dueling_motion_transformers import DuelingMotionTransformer
from replay_buffer import PrioritizedReplayBuffer


class FlappyBirdAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = GAMMA
        self.lr = LEARNING_RATE
        self.batch_size = BATCH_SIZE
        self.temp = TEMP_INIT  # Initial temperature for Boltzmann exploration
        self.temp_min = TEMP_MIN  # Minimal Boltzmann temperature
        self.temp_decay = TEMP_DECAY  # Decay of Boltzmann temperature

        self.insert_count = 0
        self.samples_per_insert = SAMPLES_PER_INSERT_RATIO

        self.replay_buffer = PrioritizedReplayBuffer(MAX_REPLAY_SIZE)

        self.policy_net = DuelingMotionTransformer(state_dim=state_dim, action_dim=action_dim)
        self.target_net = DuelingMotionTransformer(state_dim=state_dim, action_dim=action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE,
                                    weight_decay=WEIGHT_DECAY)

        self.td_errors = []
        self.losses = []
        self.losses.append(0)  # Initialize losses list with a zero value
     
        self.grad_norms = []
        self.max_q_values = []
        self.min_q_values = []
        self.max_q_values.append(0)
        self.min_q_values.append(0)
        self.q_values = []
        self.q_values.append(0)  # Initialize Q-values list with a zero value
     
    def choose_action(self, state, steps_done=0):
        """
        Selects an action using Boltzmann Exploration (softmax over Q-values).
        """
        # Get the Q-values for all actions from the policy network
        with torch.no_grad():
            q_values = self.policy_net(state).squeeze(0)  # shape: [action_dim]

        # Apply Boltzmann exploration
        exp_q_values = torch.exp((q_values / self.temp))
        probs = exp_q_values / exp_q_values.sum() + 1e-6  # Normalize to get a probability distribution

        # Sample action from the distribution
        action = torch.multinomial(probs, num_samples=1).item()  # Sample one action based on the probabilities

        return action

    def update_temperature(self):
        """
        Update the temperature for Boltzmann exploration.
        The temperature decays over time, but never goes below `temp_min`.
        """
        self.temp = max(self.temp_min, self.temp * self.temp_decay)

    def train(self):
        """
        Trains the model using Double DQN with Prioritized Experience Replay.
        """
        if self.insert_count % self.samples_per_insert != 0:
            return

        if len(self.replay_buffer.memory) < self.batch_size:
            return  # Skip training if not enough samples

        states, actions, rewards, next_states, indices, weights = self.replay_buffer.sample()
        
        # Compute current Q-values
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        self.q_values.append(q_values.mean().item())  # Store mean Q-value for logging
        max_q = q_values.max().item()
        min_q = q_values.min().item()
        self.max_q_values.append(max_q)  # Store max Q-value for logging
        self.min_q_values.append(min_q)  # Store min Q-value for logging
        
        # Compute target Q-values using the target network
        with torch.no_grad():
            best_action = self.policy_net(next_states).argmax(dim=-1,
                                                              keepdim=True)  # Select the best action using policy_net
            next_q_values = self.target_net(next_states).gather(1, best_action).squeeze(1)  # Evaluate using target_net
            target_q_values = rewards + self.gamma * next_q_values
    
        # Compute TD-Error
        td_errors = target_q_values - q_values
        self.td_errors.append(td_errors.abs().mean().item())

        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors.abs().detach().numpy())

        # Update policy network
        loss = torch.log(torch.cosh(q_values - target_q_values)).mean()
        self.losses.append(loss)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), GLOBAL_CLIP_NORM)
        
        total_norm = 0
        for p in self.policy_net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        total_norm = total_norm ** 0.5
        self.grad_norms.append(total_norm)
        
        self.optimizer.step()
        
    def count_parameters(self):
        total = sum(p.numel() for p in self.policy_net.parameters())
        trainable = sum(p.numel() for p in self.policy_net.parameters() if p.requires_grad)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
