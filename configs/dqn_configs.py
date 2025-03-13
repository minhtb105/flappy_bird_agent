# Hyperparameters for Deep Q-Network
GAMMA = 0.999  # Discount factor
LEARNING_RATE = 0.01
BATCH_SIZE = 32

# Exploration parameters
EPSILON_START = 1  # Initial exploration rate
EPSILON_MIN = 0.01   # Minimum exploration rate
EPSILON_DECAY = 0.999  # Decay rate per step

# Replay Memory
MEMORY_SIZE = 150000
