import math
import random
import time
import torch
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
        max_pipe_height = BACKGROUND_HEIGHT - 100  # Ensure there's enough space for the gap

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
        self.pipes = []
        self.ray_distances = np.zeros(NUM_RAYS)

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
        self.pipes = [Pipe(PIPE_START_OFFSET + i * PIPE_SPACING) for i in range(NUM_PIPES)]
        self.is_game_over = False
        self.score = 0

    def cast_rays(self, fov=np.pi/2, max_distance=MAX_RAY_LENGTH):
        angles = np.linspace(-fov/2, fov/2, NUM_RAYS)

        self.ray_distances = np.ones(NUM_RAYS) * max_distance
        
        for i, angle in enumerate(angles):
            ray_dx = np.cos(angle)
            ray_dy = np.sin(angle)
            
            for d in np.linspace(0, max_distance, int(max_distance)):
                test_x = self.bird_x + ray_dx * d
                test_y = self.bird_y + ray_dy * d
                
                if test_y <= 0 or test_y >= BACKGROUND_HEIGHT:
                    self.ray_distances[i] = d
                    break
                
                hit_pipe = False
                for pipe in self.pipes:
                    pipe_x = pipe.x
                    gap_top = pipe.top_pipe_height
                    gap_bottom = BACKGROUND_HEIGHT - pipe.bottom_pipe_height
                    pipe_width = PIPE_WIDTH
                    
                    if pipe_x <= test_x <= pipe_x + pipe_width:
                        if test_y <= gap_top or test_y >= gap_bottom:
                            self.ray_distances[i] = d
                            hit_pipe = True
                            break
            
                if hit_pipe:
                    break
        
        self.ray_distances /= max_distance
        self.ray_distances = np.clip(self.ray_distances, 0, 1)
        
        return self.ray_distances

    def get_state(self):
        """
            Returns the normalized game state as a feature vector.
            """

        ray_inputs = self.cast_rays()
        
        v_min = JUMP_STRENGTH
        v_max = 10
        velocity_norm = np.clip((self.velocity - v_min) / (v_max - v_min))

        return np.append(ray_inputs, velocity_norm).astype(np.float32)

    def is_collision(self):
        # Create mask for bird and pipe
        bird_mask = pygame.mask.from_surface(self.bird)

        for pipe in self.pipes:
            resized_top_pipe = pygame.transform.scale(pipe.top_pipe, (PIPE_WIDTH, pipe.top_pipe_height))
            resized_bottom_pipe = pygame.transform.scale(pipe.bottom_pipe, (PIPE_WIDTH, pipe.bottom_pipe_height))
            top_pipe_mask = pygame.mask.from_surface(resized_top_pipe)
            bottom_pipe_mask = pygame.mask.from_surface(resized_bottom_pipe)

            # Calculate offset between bird and pipe
            offset_top = (pipe.x - self.bird_x, 0 - self.bird_y)
            offset_bottom = (pipe.x - self.bird_x, BACKGROUND_HEIGHT - pipe.bottom_pipe_height - self.bird_y)

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

    def check_private_zone_reward(self):
        """
        Reward or penalize based on whether any obstacle is inside the bird's private zone.

        Returns:
            float: reward (positive if safe, negative if too close)
        """
        radius = self.compute_private_zone_radius()
        bird_center = pygame.Vector2(self.bird_x + BIRD_WIDTH / 2, self.bird_y + BIRD_HEIGHT / 2)

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
        if self.bird_y < 100:
            reward -= 0.5

        # Move pipes
        for pipe in self.pipes:
            pipe.x -= PIPE_SPEED

        reward += self.check_private_zone_reward()

        # Check if bird has passed a pipe
        for pipe in self.pipes:
            if pipe.x + PIPE_WIDTH < self.bird_x and not pipe.passed == False:
                self.score += 1  # Gain points when passing pipe
                reward += 1  # Big reward when passing a pipe
                pipe.passed = True  # Mark this pipe as passed

        # Remove pipes that have moved out of the screen and create new ones
        if self.pipes and self.pipes[0].x < -PIPE_WIDTH:
            self.pipes.pop(0)
            self.pipes.append(Pipe(self.pipes[-1].x + PIPE_SPACING))

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

        self.velocity += self.gravity
        self.bird_y += self.velocity

        # Prevent bird flying too high or falling below base
        if self.bird_y > HEIGHT - BASE_HEIGHT - self.bird.get_height():
            self.bird_y = HEIGHT - BASE_HEIGHT - self.bird.get_height()
            self.velocity = 0

        if self.bird_y < 0:
            self.bird_y = 0
            self.velocity = 0

    def draw_rays(self, fov=np.pi/2, max_distance=MAX_RAY_LENGTH):
        num_rays = len(self.ray_distances)
        angles = np.linspace(-fov/2, fov/2, num_rays)
        
        for i, angle in enumerate(angles):
            ray_length = self.ray_distances[i] * max_distance

            ray_dx = np.cos(angle)
            ray_dy = np.sin(angle)

            end_x = self.bird_x + ray_dx * ray_length + BIRD_WIDTH / 2
            end_y = self.bird_y + ray_dy * ray_length + BIRD_HEIGHT / 2

            pygame.draw.line(self.screen, (0, 255, 0), (self.bird_x + BIRD_WIDTH / 2, self.bird_y + BIRD_HEIGHT / 2), (end_x, end_y), 1)

    def draw_private_zone(self):
        """
        Draw the private zone of the bird as a transparent circle (for debugging).
        """
        radius = self.compute_private_zone_radius()
        center_x = self.bird_x + BIRD_WIDTH // 2
        center_y = self.bird_y + BIRD_HEIGHT // 2
        pygame.draw.circle(self.screen, (255, 0, 0), (int(center_x), int(center_y)), int(radius), 1)

    def update_ui(self):
        self.screen.fill("white")
        self.screen.blit(self.background, (0, 0))
        self.screen.blit(self.base, (0, 500))
        self.screen.blit(self.bird, (self.bird_x, self.bird_y))
        self.draw_private_zone()

        for pipe in self.pipes:
            resized_top_pipe = pygame.transform.scale(pipe.top_pipe, (PIPE_WIDTH, pipe.top_pipe_height))
            resized_bottom_pipe = pygame.transform.scale(pipe.bottom_pipe, (PIPE_WIDTH, pipe.bottom_pipe_height))
            self.screen.blit(resized_top_pipe, (pipe.x, 0))
            self.screen.blit(resized_bottom_pipe, (pipe.x, BACKGROUND_HEIGHT - pipe.bottom_pipe_height))

        self.draw_rays()
        text = self.font.render("Score: " + str(self.score), True, (255, 255, 255))
        self.screen.blit(text, [0, 0])
        pygame.display.flip()
