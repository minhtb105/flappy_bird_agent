import random
import time
import pygame
import numpy as np
from configs.game_configs import *


class FlappyBirdPygame:
    def __init__(self):
        # Pygame setup
        pygame.init()
        self.is_game_over = False
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("Flappy Bird Agent")
        self.font = pygame.font.Font('arial.ttf', 25)

        # Load assets
        self.game_over_icon = pygame.image.load(GAME_OVER_IMAGE)
        self.background = pygame.transform.scale(pygame.image.load(BACKGROUND_IMAGE), (WIDTH, BACKGROUND_HEIGHT))
        self.base = pygame.transform.scale(pygame.image.load(BASE_IMAGE), (WIDTH, BASE_HEIGHT))
        self.bird = pygame.image.load(BIRD_IMAGE)
        self.pipe_top = pygame.image.load(PIPE_IMAGE)
        self.pipe_bottom = pygame.image.load(PIPE_IMAGE)

        self.show_start_images()
        self.reset()
        
    # Function to show start images
    def show_start_images(self):
        for img_path in START_IMAGES:
            img = pygame.image.load(img_path)
            img = pygame.transform.scale(img, (WIDTH, HEIGHT))
            self.screen.blit(img, (0, 0))
            pygame.display.flip()
            time.sleep(0.5)

    def reset(self):
        # Reset game to original state
        self.bird_x, self.bird_y = BIRD_X, BIRD_Y
        self.velocity = 0
        self.pipe_x = WIDTH
        self.generate_random_pipe()
        self.is_game_over = False
        self.score = 0
        self.frame_count = 0

    # function to generate a new pipe height
    def generate_random_pipe(self):
        """Randomizes pipe height and gap size while ensuring a valid gap."""
        min_pipe_height = 50  # Minimum pipe height (prevents pipes from covering the whole screen)
        max_pipe_height = BACKGROUND_HEIGHT - 200  # Ensure there's enough space for the gap

        min_gap_size = 50  # Prevents gaps that are too small
        max_gap_size = 200  # Prevents gaps that are too large

        self.pipe_gap = random.randint(min_gap_size, max_gap_size)  # Enforce safe gap size
        self.pipe_top_height = random.randint(min_pipe_height, max_pipe_height - self.pipe_gap)
        self.pipe_bottom_height = BACKGROUND_HEIGHT - self.pipe_top_height - self.pipe_gap

    def is_collision(self):
        # Create mask for bird and pipe
        bird_mask = pygame.mask.from_surface(self.bird)
        resized_pipe_top = pygame.transform.scale(self.pipe_top, (PIPE_WIDTH, self.pipe_top_height))
        resized_pipe_bottom = pygame.transform.scale(self.pipe_bottom, (PIPE_WIDTH, self.pipe_bottom_height))
        pipe_top_mask = pygame.mask.from_surface(resized_pipe_top)
        pipe_bottom_mask = pygame.mask.from_surface(resized_pipe_bottom)

        # Calculate offset between bird and pipe
        offset_top = (self.pipe_x - self.bird_x, 0 - self.bird_y)
        offset_bottom = (self.pipe_x - self.bird_x, BACKGROUND_HEIGHT - self.pipe_bottom_height - self.bird_y)

        return bird_mask.overlap(pipe_top_mask, offset_top) or bird_mask.overlap(pipe_bottom_mask, offset_bottom)

    def get_state(self):
        """
            Returns the normalized game state as a feature vector.
            """
        # Normalize bird position (0 = top, 1 = bottom)
        bird_y_norm = self.bird_y / HEIGHT

        # Normalize velocity (assuming max speed is ±10)
        velocity_norm = self.velocity / 10

        # Normalize pipe distance (0 = bird at pipe, 1 = farthest away)
        pipe_x_norm = self.pipe_x / WIDTH

        # Normalize distance to gap (0 = bird at gap center, values around -1 to 1)
        gap_center = self.pipe_top_height + (PIPE_GAP_SIZE // 2)
        distance_to_gap_norm = (self.bird_y - gap_center) / HEIGHT

        # Return normalized state
        return np.array([bird_y_norm, velocity_norm, pipe_x_norm, distance_to_gap_norm])

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_game_over = True
                pygame.quit()

    def step(self, action=None):
        """
        Executes one step in the Flappy Bird game environment.
        Returns: reward, done (whether game ended), score
        """
        self.handle_events()

        self.frame_count += 1  # Count frames for training
        reward = 0.1  # Small reward for surviving

        # Move bird
        self.move(action)

        # Penalize staying at the top too long
        if self.bird_y < 50 or self.bird_y > HEIGHT - BASE_HEIGHT - 50:
            reward -= 0.05  # Small penalty for staying too high or too low

        # Move pipes
        self.pipe_x -= PIPE_SPEED
        if self.pipe_x < -PIPE_WIDTH:
            self.pipe_x = WIDTH
            self.generate_random_pipe()
            self.score += 1  # Gain points when passing pipe
            reward += 1  # Big reward when passing a pipe

        # Collision detection
        if self.is_collision():
            self.is_game_over = True
            self.screen.blit(self.game_over_icon, (WIDTH / 2 - 100, HEIGHT / 2 - 100))
            pygame.display.flip()
            time.sleep(1)
            reward = -1  # Heavy penalty for crashing

            return reward, self.is_game_over, self.score  # Game over

        self.update_ui()
        self.clock.tick(FPS)

        return reward, self.is_game_over, self.score

    def move(self, action):
        if np.array_equal(action, [0, 1]):
            self.velocity = JUMP_STRENGTH

        # Apply gravity
        self.velocity += GRAVITY
        self.bird_y += self.velocity

        # Prevent bird flying too high or falling below base
        if self.bird_y > HEIGHT - BASE_HEIGHT - self.bird.get_height():
            self.bird_y = HEIGHT - BASE_HEIGHT - self.bird.get_height()
            self.velocity = 0

        if self.bird_y < 0:
            self.bird_y = 0
            self.velocity = 0

    def update_ui(self):
        self.screen.fill("white")
        self.screen.blit(self.background, (0, 0))
        self.screen.blit(self.base, (0, 500))
        self.screen.blit(self.bird, (self.bird_x, self.bird_y))

        resized_pipe_top = pygame.transform.scale(self.pipe_top, (PIPE_WIDTH, self.pipe_top_height))
        resized_pipe_bottom = pygame.transform.scale(self.pipe_bottom, (PIPE_WIDTH, self.pipe_bottom_height))
        self.screen.blit(resized_pipe_top, (self.pipe_x, 0))
        self.screen.blit(resized_pipe_bottom, (self.pipe_x, BACKGROUND_HEIGHT - self.pipe_bottom_height))

        text = self.font.render("Score: " + str(self.score), True, (255, 255, 255))
        self.screen.blit(text, [0, 0])
        pygame.display.flip()
