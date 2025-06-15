motion_dynamic_bounds = {
    'gravity': (0.2, 1.2),
    'jump_strength': (-3, -1),
    'max_downward_speed': (6, 10),
}

reward_bounds = {
    'reward_pass_pipe': (1, 5),
    'penalty_death': (-5, -1),
    'reward_center_gap': (0.1, 0.5),
    'penalty_edge_height': (-0.5, -0.1),
    'penalty_high_alt': (-0.5, -0.1),
    'reward_medium_alt': (0.1, 0.5),
    'reward_alive': (0.01, 0.05)
}


exploration_bounds = {
    "temperature": (0.5, 1.5),
    "temperature_min": (0.01, 0.1),
    "temperature_decay": (0.99, 0.999995),
}
    
learning_bounds = {
    "learning_rate": (1e-5, 1e-3),
    "adam_epsilon": (1e-8, 1e-6),       
    "weight_decay": (0.0, 1e-3),  
}
    
replay_bounds = {
    "replay_buffer_size": (50000, 300000),  # int
    "batch_size": (32, 256),                # int
    "alpha_init": (0.3, 0.7),
    "beta": (0.4, 1.0),
}
    
target_update_bounds = {
    "tau": (0.001, 0.05),                
}
    
architecture_bounds = {
    "embed_dim": (64, 512),     
    "num_heads": (2, 8),        
    "ff_hidden_ratio": (2.0, 4.0) 
}
    