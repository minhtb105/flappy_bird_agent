# Hyperparameters for Deep Q-Network
WARMUP_STEPS = 1000  # warmup steps for learning rate
TRAIN_STEPS = 1e6
LEARNING_RATE = 0.0003
BATCH_SIZE = 256
EMBED_DIM = 128
GAMMA = 0.99  # Discount factor

TAU = 0.005
NUM_LAYERS = 2
FF_MULT = 4  # Multiplier of MLP block dimension
NUM_HEADS = 6  # num of attention heads
GLOBAL_CLIP_NORM = 1  # Globally normalized clipping of gradient
WEIGHT_DECAY = 0.0001  # Weight decay for AdamW optimizer

FRAME_STACK = 12 # Size of short-term (episodic) memory

# Exploration parameters
TEMP_INIT = 0.5  #  Initial Boltzmann temperature for exploration
TEMP_MIN = 0.01   # Minimum Boltzmann temperature
TEMP_DECAY = 0.999999  # Decay of Boltzmann temperature

# Replay Memory
MAX_REPLAY_SIZE = 1000000
