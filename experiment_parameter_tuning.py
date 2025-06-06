from bayes_opt import BayesianOptimization, acquisition
import numpy as np
import os
from typing import List
from train import train_loop
from configs.pbounds import *
from configs.game_configs import *
from configs.dqn_configs import *


def remove_checkpoint_files():
    for f in ["models/policy_net.pth", "models/target_net.pth", "models/replay_buffer.pt"]:
        if os.path.exists(f):
            os.remove(f)

def compute_objective_from_motion(reward_log):
    episode_lengths = [len(ep) for ep in reward_log]
    episode_rewards = [sum(ep) for ep in reward_log]

    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    score = (
        mean_reward                        
        + 0.01 * mean_length                  
        - 0.05 * std_length             
        - 0.1 * std_reward                    
    )
    
    return score

def optimize_physics_params(gravity, jump_strength, max_downward_speed):
    remove_checkpoint_files()
    config = {
        "gravity": float(gravity),
        "jump_strength": float(jump_strength),
        "max_downward_speed": float(max_downward_speed),
    }
    
    reward_log = train_loop(config, num_episodes=300, visualize=False)
    
    return compute_objective_from_motion(reward_log)

def compute_objective_from_rewards(reward_log: List[List[float]], switch_threshold=200):
    episode_lengths = [len(ep) for ep in reward_log]
    mean_survival = np.mean(episode_lengths) + np.random.normal(0, 0.1)
    
    if mean_survival >= switch_threshold:
        scores = [sum(1 for r in ep if r >= 9.0) for ep in reward_log]
        return np.mean(scores)
    
    return mean_survival

def optimize_reward_shaping(
    reward_pass_pipe,
    penalty_death,
    reward_center_gap,
    penalty_edge_height,
    penalty_high_alt,
    reward_medium_alt,
    reward_alive
    ):
    remove_checkpoint_files()
    config = {
        "reward_pass_pipe": float(reward_pass_pipe),
        "penalty_death": float(penalty_death),
        "reward_center_gap": float(reward_center_gap),
        "penalty_edge_height": float(penalty_edge_height),
        "penalty_high_alt": float(penalty_high_alt),
        "reward_medium_alt": float(reward_medium_alt),
        "reward_alive": float(reward_alive),
    }
    reward_log = train_loop(config, num_episodes=300, visualize=False)

    return compute_objective_from_rewards(reward_log)

def compute_objective_exploration(reward_log, min_required=1.5):
    first_rewards = [np.sum(ep) for ep in reward_log[:50]]
    last_rewards = [np.sum(ep) for ep in reward_log[-50:]]

    gain = np.mean(last_rewards) - np.mean(first_rewards)
    final = np.mean(last_rewards)

    return gain + 0.5 * (final if final >= min_required else 0)

def train_agent_with_exploration_params(temperature, temperature_min, temperature_decay):
    remove_checkpoint_files()
    config = {
        "temperature": temperature,
        "temperature_min": temperature_min,
        "temperature_decay": temperature_decay,
    }
    
    reward_log = train_loop(config, num_episodes=300, visualize=False)

    return compute_objective_exploration(reward_log)

def compute_objective_replay(reward_log):
    last_rewards = [np.sum(ep) for ep in reward_log[-30:]]
    mean_reward = np.mean(last_rewards)
    std_reward = np.std(last_rewards)

    return mean_reward - 0.1 * std_reward

def train_agent_with_replay_params(
    replay_buffer_size, batch_size, alpha_init, beta
):
    remove_checkpoint_files()
    config = {
        "replay_buffer_size": int(replay_buffer_size),
        "batch_size": int(batch_size),
        "alpha_init": float(alpha_init),
        "beta": float(beta),
    }
    
    reward_log = train_loop(config, num_episodes=120, visualize=False)
    
    return compute_objective_replay(reward_log)

def train_agent_with_learning_bounds(learning_rate, weight_decay, beta1, beta2, adam_epsilon):
    remove_checkpoint_files()
    config = {
        "learning_rate": learning_rate,
        "beta1": beta1,
        "beta2": beta2,
        "weight_decay": weight_decay,
        "adam_epsilon": adam_epsilon
    }
    
    reward_log = train_loop(config, num_episodes=300, visualize=False)
    last_rewards = [np.sum(ep) for ep in reward_log[-100:]]
    mean_reward = np.mean(last_rewards)
    std_reward = np.std(last_rewards)
    
    return mean_reward - 0.1 * std_reward

def compute_objective_target_update(reward_log):
    last_rewards = [np.sum(ep) for ep in reward_log[-30:]]
    smoothness = -np.std(last_rewards)

    return last_rewards + smoothness

def train_agent_with_target_update_params(tau):
    remove_checkpoint_files()
    config = {
        "tau": tau
    }

    reward_log = train_loop(config, num_episodes=120, visualize=False)
    
    return compute_objective_target_update(reward_log)

def compute_objective_from_architecture(reward_log):
    total_rewards = [np.sum(ep) for ep in reward_log]
    peak = np.max(total_rewards)
    generalization = np.mean(total_rewards)
    
    return 0.5 * peak + 0.5 * generalization

def train_agent_with_dqn_architecture(embed_dim, num_heads):
    remove_checkpoint_files()
    config = {
        "d_model": embed_dim,
        "num_heads": num_heads
    }
    
    reward_log = train_loop(config, num_episodes=120, visualize=False)
    
    return compute_objective_from_architecture(reward_log)

def narrow_bounds_safe(best_params, margin=0.1, hard_bounds=None):
    narrowed = {}
    for k, v in best_params.items():
        if k not in hard_bounds:
            continue

        low, high = hard_bounds[k]

        if v <= low:
            narrowed[k] = (low, min(low + abs(margin * v), high))
        elif v >= high:
            narrowed[k] = (max(high - abs(margin * v), low), high)
        else:
            delta = abs(margin * v)
            narrowed[k] = (
                max(v - delta, low),
                min(v + delta, high)
            )
            
    return narrowed


def early_stop_bo(optimizer, n_iter=10, patience=5):
    best_value = float('-inf')
    no_improve_steps = 0
    
    for i in range(n_iter):
        optimizer.maximize(init_points=0, n_iter=1)

        current_best = optimizer.max['target']
        if current_best > best_value + 0.01:  # small threshold to detect real improvement
            best_value = current_best
            no_improve_steps = 0
        else:
            no_improve_steps += 1
            
        if no_improve_steps >= patience:
            break
    

if __name__ == "__main__":
    best_motion_dynamic_params = {
        'gravity': GRAVITY,
        'jump_strength': JUMP_STRENGTH,
        'max_downward_speed': MAX_DOWNWARD_SPEED,
    }

    best_reward_params = {
        'reward_pass_pipe': REWARD_PASS_PIPE,
        'penalty_death': PENALTY_DEATH,
        'penalty_edge_height': PENALTY_EDGE_HEIGHT,
        'penalty_high_alt': PENALTY_HIGH_ALT,
        'reward_alive': REWARD_ALIVE
    }


    best_exploration_params = {
        "temperature": TEMP_INIT,
        "temperature_min": TEMP_MIN,
        "temperature_decay": TEMP_DECAY,
    }
    
    best_learning_params = {
        "learning_rate": LEARNING_RATE,
        "beta1": BETA1,
        "beta2": BETA2,
        "weight_decay": WEIGHT_DECAY,
        "adam_epsilon": ADAM_EPSILON
    }
    
    best_replay_params = {
        "replay_buffer_size": MAX_REPLAY_SIZE, # int
        "batch_size": BATCH_SIZE, # int
        "alpha_init": ALPHA_INIT,
        "beta": BETA,
    }
    
    best_target_update_params = {
        "tau": TAU,                
    }
    
    best_architecture_params = {
        "d_model": EMBED_DIM,     
        "num_heads": NUM_HEADS,        
        "ff_hidden_ratio": FF_MULT 
    }
    
    narrowed_motion_dynamic_bounds = narrow_bounds_safe(best_motion_dynamic_params, 0.1, 
                                                        motion_dynamic_bounds)

    acq1 = acquisition.UpperConfidenceBound(kappa=2, exploration_decay=0.9)
    acq2 = acquisition.ProbabilityOfImprovement(xi=0.2, exploration_decay=0.9)
    acq3 = acquisition.ExpectedImprovement(xi=0.2, exploration_decay=0.9)
    acq = acquisition.GPHedge(base_acquisitions=[acq1, acq2, acq3])
    
    optimizer = BayesianOptimization(
        f=optimize_physics_params,
        pbounds=narrowed_motion_dynamic_bounds,
        acquisition_function=acq,
        random_state=72,
    )

    # optimizer.maximize(init_points=12, n_iter=0)
    optimizer.probe(params=best_motion_dynamic_params, lazy=False)

    early_stop_bo(optimizer, n_iter=30, patience=5)
