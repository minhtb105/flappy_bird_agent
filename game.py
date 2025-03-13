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

        # Initialize pipes
        self.pipes = []
        self.create_initial_pipes()
        
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
        self.pipes = []
        self.create_initial_pipes()
        self.is_game_over = False
        self.score = 0

    def create_initial_pipes(self):
        """Create the initial columns with equal spacing."""
        for i in range(4):  # Assume creating 4 initial columns.
            self.create_new_pipe((i + 1.5) * PIPE_SPACING)

    # function to generate a new pipe height
    def create_new_pipe(self, x_position):
        """Randomizes pipe height and gap size while ensuring a valid gap."""
        min_pipe_height = 50  # Minimum pipe height (prevents pipes from covering the whole screen)
        max_pipe_height = BACKGROUND_HEIGHT - 200  # Ensure there's enough space for the gap

        min_gap_size = 100  # Prevents gaps that are too small
        max_gap_size = 150  # Prevents gaps that are too large

        pipe_gap = random.randint(min_gap_size, max_gap_size)  # Enforce safe gap size
        pipe_top_height = random.randint(min_pipe_height, max_pipe_height - pipe_gap)
        pipe_bottom_height = BACKGROUND_HEIGHT - pipe_top_height - pipe_gap

        new_pipe = {
            'x': x_position,
            'top_height': pipe_top_height,
            'bottom_height': pipe_bottom_height,
            'gap': pipe_gap,
            "passed": False,
        }

        self.pipes.append(new_pipe)

    def is_collision(self):
        # Create mask for bird and pipe
        bird_mask = pygame.mask.from_surface(self.bird)

        for pipe in self.pipes:
            resized_pipe_top = pygame.transform.scale(self.pipe_top, (PIPE_WIDTH, pipe['top_height']))
            resized_pipe_bottom = pygame.transform.scale(self.pipe_bottom, (PIPE_WIDTH, pipe['bottom_height']))
            pipe_top_mask = pygame.mask.from_surface(resized_pipe_top)
            pipe_bottom_mask = pygame.mask.from_surface(resized_pipe_bottom)

            # Calculate offset between bird and pipe
            offset_top = (pipe['x'] - self.bird_x, 0 - self.bird_y)
            offset_bottom = (pipe['x'] - self.bird_x, BACKGROUND_HEIGHT - pipe['bottom_height'] - self.bird_y)

            if bird_mask.overlap(pipe_top_mask, offset_top) or bird_mask.overlap(pipe_bottom_mask, offset_bottom):
                return True

        return False

    def get_state(self):
        """
            Returns the normalized game state as a feature vector.
            """
        # Normalize bird position
        bird_y_normalized = self.bird_y / HEIGHT

        # Normalize velocity (assuming max speed is ±10)
        velocity_normalized = self.velocity / 10

        last_pipe = self.pipes[-1] if self.pipes else None
        next_pipe = None
        next_next_pipe = None

        for i, pipe in enumerate(self.pipes):
            if pipe['x'] > self.bird_x:
                next_pipe = pipe
                if i + 1 < len(self.pipes):
                    next_next_pipe = self.pipes[i + 1]
                break

        # If there is no column in front, use the default value.
        last_pipe_x = last_top_pipe_y = last_bottom_pipe_y = 1.0
        next_pipe_x = next_top_pipe_y = next_bottom_pipe_y = 1.0
        next_next_pipe_x = next_next_top_pipe_y = next_next_bottom_pipe_y = 1.0

        if last_pipe:
            last_pipe_x = last_pipe['x'] / WIDTH
            last_top_pipe_y = last_pipe['top_height'] / HEIGHT
            last_bottom_pipe_y = (HEIGHT - last_pipe['bottom_height']) / HEIGHT

        if next_pipe:
            next_pipe_x = next_pipe['x'] / WIDTH
            next_top_pipe_y = next_pipe['top_height'] / HEIGHT
            next_bottom_pipe_y = (HEIGHT - next_pipe['bottom_height']) / HEIGHT

        if next_next_pipe:
            next_next_pipe_x = next_next_pipe['x'] / WIDTH
            next_next_top_pipe_y = next_next_pipe['top_height'] / HEIGHT
            next_next_bottom_pipe_y = (HEIGHT - next_next_pipe['bottom_height']) / HEIGHT

        state = [
            last_pipe_x,
            last_top_pipe_y,
            last_bottom_pipe_y,
            next_pipe_x,
            next_top_pipe_y,
            next_bottom_pipe_y,
            next_next_pipe_x,
            next_next_top_pipe_y,
            next_next_bottom_pipe_y,
            bird_y_normalized,
            velocity_normalized,
        ]

        return np.array(state)

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

        reward = 0

        # Move bird
        self.move(action)

        # Penalize staying at the top or the bottom too long
        if self.bird_y < 50 :
            reward -= 0.05
        elif self.bird_y > HEIGHT - BASE_HEIGHT - 50:
            reward -= 0.04

        # Move pipes
        for pipe in self.pipes:
            pipe['x'] -= PIPE_SPEED

        # Check if bird has passed a pipe
        for pipe in self.pipes:
            if pipe['x'] + PIPE_WIDTH < self.bird_x and not pipe.get('passed', False):
                self.score += 1  # Gain points when passing pipe
                reward += 1  # Big reward when passing a pipe
                pipe['passed'] = True  # Mark this pipe as passed

        # Remove pipes that have moved out of the screen and create new ones
        if self.pipes and self.pipes[0]['x'] < -PIPE_WIDTH:
            self.pipes.pop(0)
            self.create_new_pipe(self.pipes[-1]['x'] + PIPE_SPACING)

        # Collision detection
        if self.is_collision():
            self.is_game_over = True
            self.screen.blit(self.game_over_icon, (WIDTH / 2 - 100, HEIGHT / 2 - 100))
            pygame.display.flip()
            time.sleep(1)
            reward = -1  # Heavy penalty for crashing

            return reward, self.is_game_over, self.score  # Game over

        reward += 0.1
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

        for pipe in self.pipes:
            resized_pipe_top = pygame.transform.scale(self.pipe_top, (PIPE_WIDTH, pipe['top_height']))
            resized_pipe_bottom = pygame.transform.scale(self.pipe_bottom, (PIPE_WIDTH, pipe['bottom_height']))
            self.screen.blit(resized_pipe_top, (pipe['x'], 0))
            self.screen.blit(resized_pipe_bottom, (pipe['x'], BACKGROUND_HEIGHT - pipe['bottom_height']))

        text = self.font.render("Score: " + str(self.score), True, (255, 255, 255))
        self.screen.blit(text, [0, 0])
        pygame.display.flip()
