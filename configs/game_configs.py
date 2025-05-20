# Constants for color
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Pygame setup
WIDTH = 512
HEIGHT = 412
FPS = 60

# Background
BACKGROUND_HEIGHT = 412

# Base
BASE_HEIGHT = HEIGHT - BACKGROUND_HEIGHT

# Bird properties
BIRD_X = 24
BIRD_Y = BACKGROUND_HEIGHT / 2
BIRD_WIDTH = 34
BIRD_HEIGHT = 24
GRAVITY = 1.6
JUMP_STRENGTH = -6  # Apply a jump force to the bird (negative value makes it move upwards)

# Pipe properties
PIPE_WIDTH = 52
PIPE_SPEED = 3
PIPE_GAP_SIZE = 100  # Distance between top pipe and bottom pipe
PIPE_START_OFFSET = 350 # Offset for the first pipe
PIPE_SPACING = 200
NUM_PIPES = 2  # Number of pipes on the screen at a time

NUM_RAYS = 60
MAX_RAY_LENGTH = 0.8 * WIDTH - BIRD_X  # Maximum length of the ray
PLAYER_PRIVATE_ZONE = 30  # Size of agent's private zone

# Asset paths
ASSETS_PATH = "assets/"
BACKGROUND_IMAGE = ASSETS_PATH + "background.png"
BASE_IMAGE = ASSETS_PATH + "base.png"
BIRD_IMAGE = ASSETS_PATH + "bluebird.png"
PIPE_IMAGE = ASSETS_PATH + "pipe.png"
GAME_OVER_IMAGE = ASSETS_PATH + "gameover.png"

# Start screen images
START_IMAGES = [ASSETS_PATH + f"{i}.png" for i in range(6)]