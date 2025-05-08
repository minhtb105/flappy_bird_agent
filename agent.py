import random
import torch
import torch.optim as optim
import time
import logging
from configs.dqn_configs import *
from dueling_motion_transformers import DuelingMotionTransformer
from replay_buffer import PrioritizedReplayBuffer

# Setup logging
logging.basicConfig(filename='logs/debug_log.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Error tracking structures
last_logged_errors = {}
error_counts = {
    "q_nan": 0,
    "invalid_probs": 0,
    "train_failure": 0,
    "choose_action_failure": 0,
    "store_transition_failure": 0,
    "update_temperature, failure": 0,
}

def log_once_per(error_key: str, message: str, step: int, interval_seconds: int = 500, interval_steps: int = 1000):
    now = time.time()
    if error_key not in last_logged_errors or (now - last_logged_errors[error_key]) > interval_seconds:
        logging.error(message)
        last_logged_errors[error_key] = now

    if step % interval_steps == 0 and error_counts[error_key] > 0:
        logging.info(f"[{error_key}] occurred {error_counts[error_key]} times in last {interval_steps} steps")
        error_counts[error_key] = 0

def log_error_stats(step, interval=1000):
    if step % interval == 0:
        for key, count in error_counts.items():
            if count > 0:
                logging.warning(f"[{key}] occurred {count} times in last {interval} steps")
                error_counts[key] = 0


class FlappyBirdAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = GAMMA
        self.lr = LEARNING_RATE
        self.batch_size = BATCH_SIZE
        self.temp = TEMP_INIT
        self.temp_min = TEMP_MIN
        self.temp_decay = TEMP_DECAY

        self.insert_count = 0
        self.samples_per_insert = SAMPLES_PER_INSERT_RATIO

        self.replay_buffer = PrioritizedReplayBuffer(MAX_REPLAY_SIZE)

        self.policy_net = DuelingMotionTransformer(state_dim=state_dim, action_dim=action_dim)
        self.target_net = DuelingMotionTransformer(state_dim=state_dim, action_dim=action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        self.td_errors = []
        self.losses = [0]
        self.grad_norms = []
        self.max_q_values = [0]
        self.min_q_values = [0]
        self.q_values = [0]
        self.train_steps = 0
        self.action_counter = {0: 0, 1: 0}  # Count action usage for debug

    def choose_action(self, state, explore=True):
        try:
            with torch.no_grad():
                q_values = self.policy_net(state).squeeze(0)

            if not explore:
                return torch.argmax(q_values).item()

            if torch.isnan(q_values).any():
                error_counts["q_nan"] += 1
                log_once_per("q_nan", f"[choose_action] NaN in Q-values: {q_values.tolist()}", self.train_steps)

            q_values = torch.clamp(q_values, -50, 50)
            exp_q = torch.exp(q_values / (self.temp + 1e-6))
            probs = exp_q / (exp_q.sum() + 1e-6)

            if torch.isnan(probs).any() or probs.sum().item() <= 0:
                error_counts["invalid_probs"] += 1
                log_once_per("invalid_probs", f"[choose_action] Invalid probs: {probs.tolist()} | Q: {q_values.tolist()} | temp: {self.temp:.4f}", self.train_steps)
                
                return random.randint(0, self.action_dim - 1)

            action = torch.multinomial(probs, 1).item()
            self.action_counter[action] += 1

            return action

        except Exception as e:
            error_counts["choose_action"] += 1
            log_once_per("choose_action_failure", f"[choose_action] Exception: {str(e)}", self.train_steps)
            
            return random.randint(0, self.action_dim - 1)

    def update_temperature(self):
        try:
            old_temp = self.temp
            self.temp = max(self.temp_min, self.temp * self.temp_decay)
            logging.debug(f"[temperature] Decayed from {old_temp:.6f} to {self.temp:.6f}")
        except Exception as e:
            error_counts["update_temperature"] += 1
            log_once_per("update_temperature", f"[temperature] Exception: {str(e)}", self.train_steps)

    def train(self):
        self.train_steps += 1
        log_error_stats(self.train_steps, interval=1000)
        
        if self.insert_count % self.samples_per_insert != 0:
            return

        if len(self.replay_buffer.memory) < MIN_REPLAY_SIZE:
            return

        try:
            sample = self.replay_buffer.sample()
            if sample is None:
                return

            states, actions, rewards, next_states, indices, weights = sample
            q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            self.q_values.append(q_values.mean().item())
            self.max_q_values.append(q_values.max().item())
            self.min_q_values.append(q_values.min().item())

            with torch.no_grad():
                best_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                target_q = self.target_net(next_states).gather(1, best_actions).squeeze(1)
                target_q_values = rewards + self.gamma * target_q

            td_errors = target_q_values - q_values
            self.td_errors.append(td_errors.abs().mean().item())
            self.replay_buffer.update_priorities(indices, td_errors.abs().detach().cpu().numpy())

            loss = torch.log(torch.cosh(q_values - target_q_values)).mean()
            self.losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), GLOBAL_CLIP_NORM)

            total_norm = 0.0
            for p in self.policy_net.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            self.grad_norms.append(total_norm)

            logging.debug(f"[train] Loss: {loss.item():.6f}, GradNorm: {total_norm:.4f}, TD-error: {td_errors.abs().mean().item():.4f}")
            self.optimizer.step()

        except Exception as e:
            error_counts["train_failure"] += 1
            log_once_per("train_failure", f"[train] Training failed: {str(e)}", self.train_steps)

    def count_parameters(self):
        total = sum(p.numel() for p in self.policy_net.parameters())
        trainable = sum(p.numel() for p in self.policy_net.parameters() if p.requires_grad)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        logging.info(f"Total parameters: {total}, Trainable parameters: {trainable}")

    def analyze_behavior(self):
        try:
            total = sum(self.action_counter.values())
            if total == 0:
                logging.warning("No actions taken yet.")
                return

            ratio_jump = self.action_counter[1] / total
            ratio_nothing = self.action_counter[0] / total

            logging.info(f"Action usage: Jump = {ratio_jump:.2%}, Nothing = {ratio_nothing:.2%}")

            if ratio_jump > 0.95:
                logging.warning("Agent strongly prefers to JUMP. May be stuck in biased behavior.")
            elif ratio_nothing > 0.95:
                logging.warning("Agent avoids jumping almost entirely. Check Q-value balance or reward bias.")

            if self.q_values:
                logging.debug(f"Latest Q-value mean: {self.q_values[-1]:.4f}, max: {self.max_q_values[-1]}, min: {self.min_q_values[-1]}")
        except Exception as e:
            logging.error(f"analyze_behavior failed: {e}")