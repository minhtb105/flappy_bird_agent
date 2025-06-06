# Constants for color
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Pygame setup
WIDTH = 288
HEIGHT = 512
FPS = 30

# Background
BACKGROUND_HEIGHT = 412

# Base
BASE_HEIGHT = HEIGHT - BACKGROUND_HEIGHT
BASE_WIDTH = WIDTH

# Bird properties
BIRD_X = 24
BIRD_Y = BACKGROUND_HEIGHT / 2
BIRD_WIDTH = 34
BIRD_HEIGHT = 24
GRAVITY = 0.9614 
JUMP_STRENGTH = -1.346
MAX_DOWNWARD_SPEED = 6.687 
MAX_UPWARD_SPEED = -8
PLAYER_ROTATION_THRESHOLD = 20  # Bird's rotation threshold.
PLAYER_VEL_ROT = 3

# Pipe properties
PIPE_WIDTH = 52
PIPE_SPEED = -4
PIPE_GAP_SIZE = 100  # Distance between top pipe and bottom pipe
PIPE_SPACING = 130

NUM_RAYS = 180
MAX_RAY_LENGTH = 0.84 * WIDTH - BIRD_X  # Maximum length of the ray
PLAYER_PRIVATE_ZONE = 30  # Size of agent's private zone

PENALTY_DEATH = -1
PENALTY_EDGE_HEIGHT = -0.5
PENALTY_HIGH_ALT = -0.5
REWARD_PASS_PIPE = 1 
REWARD_ALIVE = 0.1
REWARD_CENTER_GAP = 0.1

# Asset paths
ASSETS_PATH = "assets/"
BACKGROUND_IMAGE = ASSETS_PATH + "background.png"
BASE_IMAGE = ASSETS_PATH + "base.png"
BIRD_IMAGE = ASSETS_PATH + "bluebird.png"
PIPE_IMAGE = ASSETS_PATH + "pipe.png"
GAME_OVER_IMAGE = ASSETS_PATH + "gameover.png"

# Start screen images
START_IMAGES = [ASSETS_PATH + f"{i}.png" for i in range(6)]