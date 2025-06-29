import random
import numpy as np
import torch
import torch.optim as optim
import time
import logging
from configs.dqn_configs import *
from configs.game_configs import NUM_RAYS
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
    "store_transition_failure": 0,
    "choose_action_failure": 0,
}

def log_once_per(error_key: str, message: str, step: int, interval_seconds: int = 500, interval_steps: int = 1000, level: str = "error"):
    now = time.time()
    if error_key not in last_logged_errors or (now - last_logged_errors[error_key]) > interval_seconds:
        last_logged_errors[error_key] = now
        
        if level == "warning":
            logging.warning(message)
        elif level == "info":
            logging.info(message)
        elif level == "debug":
            logging.debug(message)
        else:
            logging.error(message)

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
        self.learning_rate = LEARNING_RATE
        self.weight_decay = WEIGHT_DECAY
        self.batch_size = BATCH_SIZE
        self.temperature = TEMP_INIT
        self.temperature_min = TEMP_MIN
        self.temperature_decay = TEMP_DECAY

        self.insert_count = 0
        self.samples_per_insert = SAMPLES_PER_INSERT_RATIO

        self.replay_buffer = PrioritizedReplayBuffer()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DuelingMotionTransformer(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.target_net = DuelingMotionTransformer(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.use_soft_update = True

        self.optimizer = optim.AdamW(self.policy_net.parameters(), 
                                    lr=self.learning_rate, 
                                    weight_decay=self.weight_decay)

        self.td_errors = []
        self.losses = []
        self.q_values = []
        self.train_steps = 0

    def choose_action(self, state, explore=True):
        try:
            with torch.no_grad():
                state = state.to(self.device)
                q_values = self.policy_net(state).squeeze(0)

            if not explore:
                return torch.argmax(q_values).item()

            # Boltzmann exploration
            q_np = q_values.cpu().numpy()
            q_np = q_np - np.max(q_np)
            exp_q = np.exp(q_np / max(self.temperature, 1e-6))
            probs = exp_q / np.sum(exp_q)
            action = np.random.choice(self.action_dim, p=probs)
            
            return int(action)

        except Exception as e:
            error_counts["choose_action"] += 1
            log_once_per("choose_action_failure", f"[choose_action] Exception: {str(e)}", self.train_steps)
            
            return random.randint(0, self.action_dim - 1)

    def update_temperature(self):
        self.temperature = max(self.temperature * TEMP_DECAY, TEMP_MIN)

    def train(self):
        self.train_steps += 1
        log_error_stats(self.train_steps, interval=1000)
        
        if self.insert_count % self.samples_per_insert != 0:
            return

        if len(self.replay_buffer.memory) < MIN_REPLAY_SIZE:
            return

        try:
            sample = self.replay_buffer.sample(self.batch_size)
            if sample is None:
                return

            states, actions, rewards, next_states, indices, weights = sample
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            weights = weights.to(self.device)
            
            q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            self.q_values.append(q_values.mean().item())

            with torch.no_grad():
                best_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                target_q = self.target_net(next_states).gather(1, best_actions).squeeze(1)
                target_q_values = rewards + self.gamma * target_q

            td_errors = target_q_values - q_values
            self.td_errors.append(td_errors.abs().mean().item())
            self.replay_buffer.update_priorities(indices, td_errors.abs().detach().cpu().numpy())
            self.replay_buffer.update_beta(self.train_steps)

            loss = (weights * torch.log(torch.cosh(td_errors))).mean()
            self.losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), GLOBAL_CLIP_NORM)

            self.optimizer.step()
            del loss, q_values, td_errors
            torch.cuda.empty_cache()

        except Exception as e:
            error_counts["train_failure"] += 1
            log_once_per("train_failure", f"[train] Training failed: {str(e)}", self.train_steps)

    def log_to_tensorboard(self, writer, step):
        if self.losses and step % 20 == 0 and step > 0:
            writer.add_scalar(f"Train/Loss/EP_{(step // 1000 + 1) * 1000}", self.losses[-1], step)

        if self.td_errors and step % 20 == 0 and step > 0:
            writer.add_scalar(f"Train/TD_Error/EP_{(step // 1000 + 1) * 1000}", self.td_errors[-1], step)

        if self.q_values and step % 20 == 0 and step > 0:
            writer.add_scalar(f"Train/QValue/EP_{(step // 1000 + 1) * 1000}", self.q_values[-1], step)

        # Attention map as image
        attn = self.policy_net.get_attention_weights()
        if attn is not None and step % 1000 == 0 and step > 0:
            try:
                import matplotlib.pyplot as plt
                import io
                import PIL.Image
                fig, ax = plt.subplots()
                ax.imshow(attn[0][0].cpu().detach().numpy(), cmap='viridis')
                ax.set_title("Attention Heatmap")
                ax.axis("off")

                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                image = PIL.Image.open(buf)
                image = np.array(image).transpose(2, 0, 1)[:3]  # CHW
                writer.add_image("Train/Attention", image, step)
                plt.close()
            except Exception as e:
                print(e)
            
        writer.flush()    

    def soft_update_target(self, tau):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def count_parameters(self):
        total = sum(p.numel() for p in self.policy_net.parameters())
        trainable = sum(p.numel() for p in self.policy_net.parameters() if p.requires_grad)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        logging.info(f"Total parameters: {total}, Trainable parameters: {trainable}")

    def update_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(),
            lr=self.learning_rate,
            weight_decay=getattr(self, "weight_decay", 0.0),
            betas=(getattr(self, "beta1", 0.9), getattr(self, "beta2", 0.999)),
            eps=getattr(self, "adam_epsilon", 1e-8)
        )
        
    def get_positional_embedding(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(0)

        x = x.to(self.device)
        with torch.no_grad():
            emb = self.policy_net.embedding(x)
            pos_enc = self.policy_net.positional_encoding.unsqueeze(0)
            pos_emb = emb + pos_enc
            
        return pos_emb.squeeze(0)
