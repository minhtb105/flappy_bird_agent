# Pygame setup
WIDTH = 400
HEIGHT = 600
FPS = 60

# Background
BACKGROUND_HEIGHT = 500

# Base
BASE_HEIGHT = HEIGHT - BACKGROUND_HEIGHT

# Bird properties
BIRD_X = 50
BIRD_Y = 300
GRAVITY = 0.5
JUMP_STRENGTH = -8  # Apply a jump force to the bird (negative value makes it move upwards)

# Pipe properties
PIPE_WIDTH = 50
PIPE_SPEED = 3
PIPE_GAP_SIZE = 100  # Distance between two pipes

# Asset paths
ASSETS_PATH = "assets/"
BACKGROUND_IMAGE = ASSETS_PATH + "background.png"
BASE_IMAGE = ASSETS_PATH + "base.png"
BIRD_IMAGE = ASSETS_PATH + "bluebird.png"
PIPE_IMAGE = ASSETS_PATH + "pipe.png"
GAME_OVER_IMAGE = ASSETS_PATH + "gameover.png"

# Start screen images
START_IMAGES = [ASSETS_PATH + f"{i}.png" for i in range(6)]
