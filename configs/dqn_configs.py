# Hyperparameters for Deep Q-Network
WARMUP_STEPS = 1000  # warmup steps for learning rate
TRAIN_STEPS = 1e6
LEARNING_RATE = 0.0004
BATCH_SIZE = 256
EMBED_DIM = 128
GAMMA = 0.99  # Discount factor
# Target Network Update Configurations
TARGET_UPDATE = 3000  # Hard update every 3,000 steps

TAU = 0.005
NUM_LAYERS = 2
FF_MULT = 4  # Multiplier of MLP block dimension
NUM_HEADS = 4  # num of attention heads
GLOBAL_CLIP_NORM = 1  # Globally normalized clipping of gradient
WEIGHT_DECAY = 0.0001  # Weight decay for AdamW optimizer

FRAME_STACK = 4  # Size of short-term (episodic) memory

# Exploration parameters
EPSILON = 1 # Epsilon for epsilon-greedy exploration
EPSILON_DECAY = 0.999995  # Decay rate for epsilon
EPSILON_MIN = 0.05  # Minimum epsilon value

# Replay Memory
MAX_REPLAY_SIZE = 300000
MIN_REPLAY_SIZE = 1000
SAMPLES_PER_INSERT_RATIO = 32
ALPHA_INIT = 0.58  # Control the amount of prioritization
ALPHA_FINAL = 0.2 
ALPHA_DECAY = 5e-6 
BETA = 0.4  # Importance sampling exponent
TARGET_UPDATE_FREQ = 1000  # Hard update target network every 1000 steps

# Training configurations
LOAD_MODEL = True  # Load from a saved checkpoint if available
SAVE_INTERVAL = 10000  # Save the model every 10.000 steps
TEST_MODE = False  # If True, AI only plays without training
NUM_EPISODES = 10000  
MAX_STEPS_PER_EPISODE = 10000000  # Maximum steps per episode
CONSECUTIVE_WINS_THRESHOLD = 100  # Stop training if AI wins 100 consecutive episodes
SAVE_REPLAY_BUFFER_INTERVAL = 100  # Save the replay buffer every 100 episodes
VISUALIZATION_INTERVAL = 1000  # Interval for visualization