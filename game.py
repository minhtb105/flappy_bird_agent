import math
import random
import time
import numpy as np
import pygame
from configs.game_configs import *


class Pipe:
    def __init__(self, x):
        self.x = x
        self.height = 0
        self.top_pipe = pygame.image.load(PIPE_IMAGE)
        self.bottom_pipe = pygame.image.load(PIPE_IMAGE)

        """Randomizes pipe height and gap size while ensuring a valid gap."""
        min_pipe_height = 50  # Minimum pipe height (prevents pipes from covering the whole screen)
        max_pipe_height = BACKGROUND_HEIGHT - 200  # Ensure there's enough space for the gap

        self.top_pipe_height = random.randint(min_pipe_height, max_pipe_height - PIPE_GAP_SIZE)
        self.bottom_pipe_height = BACKGROUND_HEIGHT - self.top_pipe_height - PIPE_GAP_SIZE

        self.top_rect = pygame.Rect(self.x, 0, PIPE_WIDTH, self.top_pipe_height)
        self.bottom_rect = pygame.Rect(self.x, self.top_pipe_height + PIPE_GAP_SIZE, PIPE_WIDTH,
                                       self.bottom_pipe_height)

        self.passed = False

    def move(self):
        self.x -= PIPE_SPEED
        self.top_pipe.x = self.x
        self.bottom_pipe.x = self.x

    def off_screen(self):
        return self.x + PIPE_WIDTH < 0

class FlappyBirdPygame:
    def __init__(self):
        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("Flappy Bird Agent")
        self.font = pygame.font.Font('arial.ttf', 25)
        self.is_game_over = False

        # Load assets
        self.game_over_icon = pygame.image.load(GAME_OVER_IMAGE)
        self.background = pygame.transform.scale(pygame.image.load(BACKGROUND_IMAGE), (WIDTH, BACKGROUND_HEIGHT))
        self.base = pygame.transform.scale(pygame.image.load(BASE_IMAGE), (WIDTH, BASE_HEIGHT))
        self.bird = pygame.image.load(BIRD_IMAGE)

        self.velocity = 0
        self.gravity = GRAVITY
        self.bird_angle = BIRD_ANGLE
        self.pipes = []

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
        self.pipes = [Pipe(PIPE_SPACING + (i + 1.5)) for i in range(3)]
        self.is_game_over = False
        self.score = 0

    def update_bird(self):
        self.velocity += self.gravity
        self.bird_y += self.velocity

        # Pitch angle logic
        if self.velocity < 0:
            self.bird_angle = math.radians(45)
        else:
            self.bird_angle = max(math.radians(-90), self.bird_angle - 0.05)

    def cast_rays(self):
        ray_distances = []
        angle_step = RAY_SPREAD / (NUM_RAYS - 1)

        for i in range(NUM_RAYS):
            angle_deg = -RAY_SPREAD / 2 + i * angle_step
            angle_rad = math.radians(angle_deg)
            dx = math.cos(angle_rad)
            dy = math.sin(angle_rad)

            # Loop to simulate the ray extending outward up to MAX_RAY_LENGTH pixels
            for length in range(1, MAX_RAY_LENGTH + 1):
                # Calculate the current point of the ray based on bird's position and direction (dx, dy)
                x = int(self.bird_x + dx * length)
                y = int(self.bird_y + dy * length)

                # If the point is outside the screen boundary, stop the ray
                if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
                    break

                # Get the color of the pixel at the current (x, y) position on the screen
                pixel = self.screen.get_at((x, y))

                # If the pixel is estimated to be a green obstacle (like a pipe), stop the ray
                r, g, b, _ = pixel
                if g > 150 and r < 100 and b < 100:
                    break

                ray_distances.append(length / MAX_RAY_LENGTH)

        return ray_distances

    def get_state(self):
        """
            Returns the normalized game state as a feature vector.
            """

        ray_inputs = self.cast_rays()
        velocity_norm = self.velocity / 10

        return np.array(ray_inputs + [velocity_norm], dtype=np.float32)

    def draw_rays(self):
        angle_step = RAY_SPREAD / (NUM_RAYS - 1)
        for i in range(NUM_RAYS):
            angle_deg = -RAY_SPREAD / 2 + i * angle_step
            angle_rad = math.radians(angle_deg)
            dx = math.cos(angle_rad)
            dy = math.sin(angle_rad)

            end_x = int(self.bird_x + dx * MAX_RAY_LENGTH)
            end_y = int(self.bird_y + dy * MAX_RAY_LENGTH)
            pygame.draw.line(self.screen, (255, 0, 0), (self.bird_x, self.bird_y), (end_x, end_y), 1)

    def is_collision(self):
        # Create mask for bird and pipe
        bird_mask = pygame.mask.from_surface(self.bird)

        for pipe in self.pipes:
            resized_top_pipe = pygame.transform.scale(pipe.top_pipe, (PIPE_WIDTH, pipe['top_height']))
            resized_bottom_pipe = pygame.transform.scale(pipe.bottom_pipe, (PIPE_WIDTH, pipe['bottom_height']))
            top_pipe_mask = pygame.mask.from_surface(resized_top_pipe)
            bottom_pipe_mask = pygame.mask.from_surface(resized_bottom_pipe)

            # Calculate offset between bird and pipe
            offset_top = (pipe['x'] - self.bird_x, 0 - self.bird_y)
            offset_bottom = (pipe['x'] - self.bird_x, BACKGROUND_HEIGHT - pipe['bottom_height'] - self.bird_y)

            if bird_mask.overlap(top_pipe_mask, offset_top) or bird_mask.overlap(bottom_pipe_mask, offset_bottom):
                return True

        return False

    @staticmethod
    def compute_private_zone_radius():
        """
        Compute the radius of the bird's private zone based on bird size and hyperparameter PLAYER_PRIVATE_ZONE.
        Returns:
            float: radius of the private zone
        """
        private_zone_radius = (max(BIRD_WIDTH, BIRD_HEIGHT) + PLAYER_PRIVATE_ZONE) / 2

        return private_zone_radius

    def draw_private_zone(self):
        """
        Draw the private zone of the bird as a transparent circle (for debugging).
        """
        radius = self.compute_private_zone_radius()
        pygame.draw.circle(self.screen, (255, 0, 0), (int(self.bird_x), int(self.bird_y)), int(radius), 1)

    def check_private_zone_reward(self):
        """
        Reward or penalize based on whether any obstacle is inside the bird's private zone.

        Returns:
            float: reward (positive if safe, negative if too close)
        """
        radius = self.compute_private_zone_radius()
        bird_center = pygame.Vector2(self.bird_x, self.bird_y)

        for pipe in self.pipes:
            # Top pipe
            if bird_center.distance_to(pipe.top_rect.center) < radius:
                return -0.5

            # Bottom pipe
            if bird_center.distance_to(pipe.bottom_rect.center) < radius:
                return -0.5

        return 0.1

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

        # Penalize staying at the top too long
        if self.bird_y < 50:
            reward -= 0.5

        # Move pipes
        for pipe in self.pipes:
            pipe['x'] -= PIPE_SPEED

        reward += self.check_private_zone_reward()

        # Check if bird has passed a pipe
        for pipe in self.pipes:
            if pipe['x'] + PIPE_WIDTH < self.bird_x and not pipe.get('passed', False):
                self.score += 1  # Gain points when passing pipe
                reward += 1  # Big reward when passing a pipe
                pipe['passed'] = True  # Mark this pipe as passed

        # Remove pipes that have moved out of the screen and create new ones
        if self.pipes and self.pipes[0]['x'] < -PIPE_WIDTH:
            self.pipes.pop(0)
            self.pipes.append(Pipe(self.pipes[-1]['x'] + PIPE_SPACING))

        # Collision detection
        if self.is_collision():
            self.is_game_over = True
            self.screen.blit(self.game_over_icon, (WIDTH / 2 - 100, HEIGHT / 2 - 100))
            pygame.display.flip()
            time.sleep(1)
            reward = -1  # Heavy penalty for crashing

            return reward, self.is_game_over, self.score  # Game over

        reward += 0.1  # Small reward for "still alive"

        self.update_ui()
        self.clock.tick(FPS)

        return reward, self.is_game_over, self.score

    def move(self, action):
        if np.array_equal(action, [0, 1]):
            self.velocity = JUMP_STRENGTH
            self.bird = pygame.transform.rotate(pygame.image.load(BIRD_IMAGE), 45)
        elif np.array_equal(action, [1, 0]):
            self.bird = pygame.transform.rotate(pygame.image.load(BIRD_IMAGE), -90)

        self.update_bird()

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
        self.draw_private_zone()

        for pipe in self.pipes:
            resized_top_pipe = pygame.transform.scale(pipe.top_pipe, (PIPE_WIDTH, pipe.pipe_top_height))
            resized_bottom_pipe = pygame.transform.scale(pipe.bottom_pipe, (PIPE_WIDTH, pipe.pipe_bottom_height))
            self.screen.blit(resized_top_pipe, (pipe['x'], 0))
            self.screen.blit(resized_bottom_pipe, (pipe['x'], BACKGROUND_HEIGHT - pipe['bottom_height']))

        self.draw_rays()
        text = self.font.render("Score: " + str(self.score), True, (255, 255, 255))
        self.screen.blit(text, [0, 0])
        pygame.display.flip()
