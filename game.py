import random
import time
import pygame
from configs.game_configs import *

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
running = True

# Caption
pygame.display.set_caption("Flappy Bird Agent")

# Function to show start images
def show_start_images():
    for img_path in START_IMAGES:
        img = pygame.image.load(img_path)
        img = pygame.transform.scale(img, (WIDTH, HEIGHT))
        screen.blit(img, (0, 0))
        pygame.display.flip()
        time.sleep(0.5)

# Game over
game_over_icon = pygame.image.load(GAME_OVER_IMAGE)

# Run the start images
show_start_images()

# Background
background = pygame.image.load(BACKGROUND_IMAGE)
resized_background = pygame.transform.scale(background, (WIDTH, BACKGROUND_HEIGHT))

# Base
base = pygame.image.load(BASE_IMAGE)
resized_base = pygame.transform.scale(base, (WIDTH, BASE_HEIGHT))

# Bird
bird = pygame.image.load(BIRD_IMAGE)
bird_x, bird_y = BIRD_X, BIRD_Y
velocity = 0  # bird speed

# Pipe
pipe_top = pygame.image.load(PIPE_IMAGE)
pipe_bottom = pygame.image.load(PIPE_IMAGE)

# Pipe variables
pipe_x = WIDTH - PIPE_WIDTH

# function to generate a new pipe height
def generate_random_pipe():
    top_height = random.randint(50, 350)
    bottom_height = HEIGHT - BASE_HEIGHT - top_height - PIPE_GAP_SIZE

    return top_height, bottom_height


pipe_top_height, pipe_bottom_height = generate_random_pipe()

while running:
    # fill the screen with a color to wipe away anything from last frame
    screen.fill("white")

    screen.blit(resized_background, (0, 0))
    screen.blit(resized_base, (0, 500))
    screen.blit(bird, (bird_x, bird_y))

    # Rescale image to random height
    resized_pipe_top = pygame.transform.scale(pipe_top, (PIPE_WIDTH, pipe_top_height))
    resized_pipe_bottom = pygame.transform.scale(pipe_bottom, (PIPE_WIDTH, pipe_bottom_height))

    screen.blit(resized_pipe_top, (pipe_x, 0))
    screen.blit(resized_pipe_bottom, (pipe_x, BACKGROUND_HEIGHT - pipe_bottom_height))

    # Move pipes
    pipe_x -= PIPE_SPEED

    # Check if pipes go off the screen and reset them
    if pipe_x < -PIPE_WIDTH:
        pipe_x = WIDTH
        pipe_top_height, pipe_bottom_height = generate_random_pipe()

    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            velocity = JUMP_STRENGTH

    # Apply gravity
    velocity += GRAVITY
    bird_y += velocity

    # Prevent bird from falling below base
    if bird_y > HEIGHT - BASE_HEIGHT - bird.get_height():
        bird_y = HEIGHT - BASE_HEIGHT - bird.get_height()
        velocity = 0

    # Prevent bird flying too high
    if bird_y < 0:
        bird_y = 0
        velocity = 0

    # Collision detection
    bird_rect = pygame.Rect(bird_x, bird_y, bird.get_width(), bird.get_height())
    pipe_top_rect = pygame.Rect(pipe_x, 0, PIPE_WIDTH, pipe_top.get_height())
    pipe_bottom_rect = pygame.Rect(pipe_x, BACKGROUND_HEIGHT - pipe_bottom_height, PIPE_WIDTH, pipe_bottom_height)

    if bird_rect.colliderect(pipe_top_rect) or bird_rect.colliderect(pipe_bottom_rect):
        running = False
        screen.blit(game_over_icon, (100, 100))

    # flip() the display to put your work on screen
    pygame.display.flip()
    clock.tick(60)  # limits FPS to 60

pygame.quit()
