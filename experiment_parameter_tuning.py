from bayes_opt import BayesianOptimization
import numpy as np
from typing import List
from train import train_loop
from configs.pbounds import *
from configs.game_configs import *
from configs.dqn_configs import *


def compute_objective_from_motion(reward_log):
    episode_lengths = [len(ep) for ep in reward_log]
    std_survival = np.std(episode_lengths)
    mean_survival = np.mean(episode_lengths)

    return mean_survival - 0.1 * std_survival

def optimize_physics_params(gravity, jump_strength, drag, max_downward_speed, max_upward_speed):
    config = {
        "gravity": float(gravity),
        "jump_strength": float(jump_strength),
        "drag": float(drag),
        "max_downward_speed": float(max_downward_speed),
        "max_upward_speed": float(max_upward_speed),
    }
    
    reward_log = train_loop(config, num_episodes=120, visualize=False)
    
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
    config = {
        "reward_pass_pipe": int(round(reward_pass_pipe)),
        "penalty_death": float(penalty_death),
        "reward_center_gap": float(reward_center_gap),
        "penalty_edge_height": float(penalty_edge_height),
        "penalty_high_alt": float(penalty_high_alt),
        "reward_medium_alt": float(reward_medium_alt),
        "reward_alive": float(reward_alive),
    }

    reward_log = train_loop(config, num_episodes=120, visualize=False)

    return compute_objective_from_rewards(reward_log)

def compute_objective_exploration(reward_log):
    first_rewards = [np.sum(ep) for ep in reward_log[:30]]
    last_rewards = [np.sum(ep) for ep in reward_log[-30:]]

    gain = np.mean(last_rewards) - np.mean(first_rewards)

    return gain

def train_agent_with_exploration_params(epsilon_decay, epsilon, epsilon_min):
    config = {
        "epsilon_decay": epsilon_decay,
        "epsilon": epsilon,
        "epsilon_min": epsilon_min,
    }
    
    reward_log = train_loop(config, num_episodes=120, visualize=False)

    return compute_objective_exploration(reward_log)

def compute_objective_replay(reward_log):
    last_rewards = [np.sum(ep) for ep in reward_log[-30:]]
    mean_reward = np.mean(last_rewards)
    std_reward = np.std(last_rewards)

    return mean_reward - 0.1 * std_reward

def train_agent_with_replay_params(
    replay_buffer_size, batch_size, alpha_init, beta
):
    config = {
        "replay_buffer_size": int(replay_buffer_size),
        "batch_size": int(batch_size),
        "alpha_init": float(alpha_init),
        "beta": float(beta),
    }
    
    reward_log = train_loop(config, num_episodes=120, visualize=False)
    
    return compute_objective_replay(reward_log)

def compute_objective_target_update(reward_log):
    last_rewards = [np.sum(ep) for ep in reward_log[-30:]]
    smoothness = -np.std(last_rewards)

    return last_rewards + smoothness

def train_agent_with_target_update_params(tau):
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
    config = {
        "d_model": embed_dim,
        "num_heads": num_heads
    }
    
    reward_log = train_loop(config, num_episodes=120, visualize=False)
    
    return compute_objective_from_architecture(reward_log)

def narrow_bounds(best_params, margin=0.1):
    return {
        key : (
            best_params[key] * (1 - margin),
            best_params[key] * (1 + margin)
        )
        for key in best_params.keys()
    }


if __name__ == "__main__":
    best_motion_dynamic_pbounds = {
        'gravity': GRAVITY,
        'jump_strength': JUMP_STRENGTH,
        'drag': DRAG,
        'max_downward_speed': MAX_DOWNWARD_SPEED,
        'max_upward_speed': MAX_UPWARD_SPEED,
    }

    best_reward_bounds = {
        'reward_pass_pipe': REWARD_PASS_PIPE,
        'penalty_death': PENALTY_DEATH,
        'reward_center_gap': REWARD_CENTER_GAP,
        'penalty_edge_height': PENALTY_EDGE_HEIGHT,
        'penalty_high_alt': PENALTY_HIGH_ALT,
        'reward_medium_alt': REWARD_MEDIUM_ALT,
        'reward_alive': REWARD_ALIVE
    }


    best_exploration_bounds = {
        "epsilon_decay": EPSILON_DECAY,
        "epsilon": EPSILON,
        "epsilon_min": EPSILON_MIN,
    }
    
    best_learning_bounds = {
        "learning_rate": LEARNING_RATE,
        "optimizer": ['Adam'], 
    }
    
    best_replay_bounds = {
        "replay_buffer_size": MAX_REPLAY_SIZE, # int
        "batch_size": BATCH_SIZE, # int
        "alpha_init": ALPHA_INIT,
        "beta": BETA,
    }
    
    best_target_update_bounds = {
        "tau": TAU,                
    }
    
    best_architecture_bounds = {
        "d_model": EMBED_DIM,     
        "num_heads": NUM_HEADS,        
        "ff_hidden_ratio": (2.0, 4.0) 
    }
    
    narrowed_reward_bounds = narrow_bounds(best_reward_bounds, margin=0.1)
    
    optimizer = BayesianOptimization(
        f=optimize_reward_shaping,
        pbounds=narrowed_reward_bounds,
        random_state=0,
    )

    optimizer.maximize(
        init_points=25,    
        n_iter=25,   
        acq='ei'     
    )  
