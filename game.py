import random
import time
import numpy as np
import pygame
import numba
from configs.game_configs import *


class Pipe:
    def __init__(self, x):
        self.x = x
        self.height = 0
        self.top_pipe = Pipe.pipe_image
        self.bottom_pipe = Pipe.pipe_image
        self.speed = PIPE_SPEED

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
        self.x += self.speed

    def off_screen(self):
        return self.x + PIPE_WIDTH < 0

class LIDAR:
    def __init__(self, max_distance, num_rays=180):
        self._max_distance = max_distance
        self.num_rays = num_rays
        
    def scan(self, player_x, player_y, player_rot, pipes, ground_y):
        collisions = np.zeros((self.num_rays, 2))
        result = np.empty([self.num_rays])
        offset_x = player_x + BIRD_WIDTH
        offset_y = player_y + (BIRD_HEIGHT / 2)

        visible_rot = PLAYER_ROTATION_THRESHOLD
        if player_rot <= PLAYER_ROTATION_THRESHOLD:
            visible_rot = player_rot

        pipes_sorted = sorted(pipes, key=lambda p: p.x)

        for i, angle in enumerate(np.linspace(0, 180, self.num_rays, endpoint=False)):
            rad = np.radians(angle - 90 - visible_rot)
            x = self._max_distance * np.cos(rad) + offset_x
            y = self._max_distance * np.sin(rad) + offset_y
            line = (offset_x, offset_y, x, y)
            collisions[i] = (x, y)

            # Check ground collision
            ground_rect = pygame.Rect(0, BACKGROUND_HEIGHT, BASE_WIDTH, BASE_HEIGHT)
            collision_ground = ground_rect.clipline(line)
            if collision_ground:
                collisions[i] = collision_ground[0]

            # Check pipe collision
            for pipe in pipes_sorted:
                top_rect = pygame.Rect(pipe.x, 0, PIPE_WIDTH, pipe.top_pipe_height)
                bottom_rect = pygame.Rect(pipe.x, BACKGROUND_HEIGHT - pipe.bottom_pipe_height, PIPE_WIDTH, pipe.bottom_pipe_height)
                
                hit = top_rect.clipline(line)
                if hit:
                    collisions[i] = hit[0]
                    break
                
                hit = bottom_rect.clipline(line)
                if hit:
                    collisions[i] = hit[0]
                    break

            if collisions[i][1] > ground_y:
                collisions[i][1] = ground_y

            result[i] = np.sqrt(
                (offset_x - collisions[i][0]) ** 2 +
                (offset_y - collisions[i][1]) ** 2
            )

        norm_result = result / self._max_distance  
        
        return norm_result

class FlappyBirdPygame:
    def __init__(self, visualize=True):
        self.visualize = visualize
        self.screen = None
        self.clock = None
        self.font = None
        
        if self.visualize:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Flappy Bird Agent")
            self.font = pygame.font.Font(f'assets/arial.ttf', 25)
        else:
            self.screen = None
            self.clock = None
            self.font = None
            
        self.is_game_over = False

        # Load assets
        self.game_over_icon = pygame.image.load(GAME_OVER_IMAGE)
        self.background = pygame.transform.scale(pygame.image.load(BACKGROUND_IMAGE), (WIDTH, BACKGROUND_HEIGHT))
        self.base = pygame.transform.scale(pygame.image.load(BASE_IMAGE), (WIDTH, BASE_HEIGHT))
        self.bird = pygame.image.load(BIRD_IMAGE)
        self.pipe_image = pygame.image.load(PIPE_IMAGE)
        self.bird_mask = pygame.mask.from_surface(self.bird)
        
        Pipe.pipe_image = self.pipe_image
        
        self.steps = 0  # total steps survived in an episode
        self.jump_strength = JUMP_STRENGTH
        self.gravity = GRAVITY
        self.jump_strength = JUMP_STRENGTH
        self.max_downward_speed = MAX_DOWNWARD_SPEED
        self.max_upward_speed = MAX_UPWARD_SPEED
        self.flapped = False
        self.bird_rot = 45
        
        self.pipes = []
        self.ray_distances = np.ones(int(NUM_RAYS)) 

        self.reward_pass_pipe = REWARD_PASS_PIPE
        self.penalty_death = PENALTY_DEATH
        self.reward_alive = REWARD_ALIVE
        self.penalty_edge_height = PENALTY_EDGE_HEIGHT
        self.penalty_high_alt = PENALTY_HIGH_ALT
        self.reward_center_gap = REWARD_CENTER_GAP

        self.rewards = []

        self.lidar = LIDAR(MAX_RAY_LENGTH, num_rays=NUM_RAYS)

        if self.visualize:
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
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.bird_x, self.bird_y = BIRD_X, BIRD_Y
        self.velocity = 0
        self.bird_rot = 45
        self.pipe_x = WIDTH
        self.pipes = [Pipe(WIDTH), Pipe(WIDTH + WIDTH / 2), Pipe(WIDTH + WIDTH)]
        self.is_game_over = False
        self.score = 0
        self.episode_rewards_record = []

    def cast_rays(self):
        ray_inputs = self.lidar.scan(
            self.bird_x, self.bird_y, self.bird_rot, self.pipes, BACKGROUND_HEIGHT
        )
        self.ray_distances[:] = ray_inputs
        
        return self.ray_distances

    def get_state(self):
        """
            Returns the normalized game state as a feature vector.
            """

        ray_inputs = self.cast_rays()
        
        return np.array(ray_inputs).astype(np.float32)

    def is_collision(self, pipe):
        # Create mask for pipe
        resized_top_pipe = pygame.transform.scale(pipe.top_pipe, (PIPE_WIDTH, pipe.top_pipe_height))
        resized_bottom_pipe = pygame.transform.scale(pipe.bottom_pipe, (PIPE_WIDTH, pipe.bottom_pipe_height))
        top_pipe_mask = pygame.mask.from_surface(resized_top_pipe)
        bottom_pipe_mask = pygame.mask.from_surface(resized_bottom_pipe)

        # Calculate offset between bird and pipe
        offset_top = (pipe.x - self.bird_x, 0 - self.bird_y)
        offset_bottom = (pipe.x - self.bird_x, BACKGROUND_HEIGHT - pipe.bottom_pipe_height - self.bird_y)

        if self.bird_mask.overlap(top_pipe_mask, offset_top) or self.bird_mask.overlap(bottom_pipe_mask, offset_bottom):
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
        ray_distances_px = self.ray_distances * MAX_RAY_LENGTH

        if np.any(ray_distances_px < radius):
            return self.penalty_edge_height

        next_pipes = [pipe for pipe in self.pipes if pipe.x + PIPE_WIDTH > self.bird_x]
        if next_pipes:
            next_pipe = min(next_pipes, key=lambda p: p.x)
            gap_top = next_pipe.top_pipe_height
            gap_bottom = next_pipe.bottom_pipe_height
            bird_mid_y = self.bird_y + BIRD_HEIGHT / 2
            
            if gap_top < bird_mid_y < gap_bottom:
                return self.reward_center_gap
        
        return 0

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
        self.steps += 1
        reward = 0

        # Move bird
        self.move(action)
        # Reward for being alive
        reward += self.reward_alive

        if self.bird_y < 2 * BIRD_HEIGHT or self.bird_y >= BACKGROUND_HEIGHT - BIRD_HEIGHT * 2:
            reward += self.penalty_high_alt

        # Penalty if too close to obstacle (private zone)
        reward += self.check_private_zone_reward()

        # Move pipes
        for pipe in self.pipes:
            pipe.move()

        # Check if bird has passed a pipe
        for pipe in self.pipes:
            if pipe.x + PIPE_WIDTH < self.bird_x and not pipe.passed:
                self.score += 1  # Gain points when passing pipe
                reward += self.reward_pass_pipe  # Big reward when passing a pipe
                self.episode_rewards_record.append(reward)
                self.rewards.append(self.episode_rewards_record)
                pipe.passed = True  # Mark this pipe as passed
                break

        # Remove pipes that have moved out of the screen and create new ones
        if self.pipes and self.pipes[0].x < -PIPE_WIDTH:
            last_pipe_x = self.pipes[-1].x
            self.pipes.pop(0)
            self.pipes.append(Pipe(last_pipe_x + PIPE_WIDTH + PIPE_SPACING))

        # Collision detection
        for pipe in self.pipes:
            if self.is_collision(pipe):
                self.is_game_over = True
                self.screen.blit(self.game_over_icon, (WIDTH / 2 - 100, HEIGHT / 2 - 100))
                pygame.display.flip()
                time.sleep(1)
                reward += self.penalty_death  # Heavy penalty for crashing
                self.episode_rewards_record.append(reward)
                self.rewards.append(self.episode_rewards_record)

                return reward, self.is_game_over, self.score  # Game over

        self.update_ui()
        if self.visualize:
            self.clock.tick(FPS)

        return reward, self.is_game_over, self.score

    def move(self, action):
        if (isinstance(action, (list, np.ndarray)) and action[1] == 1) or (isinstance(action, int) and action == 1):
            if self.bird_y > -2 * BIRD_HEIGHT:
                self.velocity = self.jump_strength
                self.flapped = True
        else:
            self.flapped = False

        if self.velocity < self.max_downward_speed and not self.flapped:
            self.velocity += self.gravity 
        
        if self.flapped:
            self.flapped = False
            self.bird_rot = 45
        else:
            if self.bird_rot > -90:
                self.bird_rot -= PLAYER_VEL_ROT
        
        max_fall = BACKGROUND_HEIGHT - self.bird.get_height() - self.bird_y
        self.bird_y += min(self.velocity, max_fall)
        
        # Prevent bird flying too high or falling below base
        if self.bird_y > HEIGHT - BASE_HEIGHT - self.bird.get_height():
            self.bird_y = HEIGHT - BASE_HEIGHT - self.bird.get_height()

        if self.bird_y < 0:
            self.bird_y = 0

    def draw_rays(self, fov=np.pi/2, max_distance=MAX_RAY_LENGTH):
        if not self.visualize:
            return
        
        visible_rotation = PLAYER_ROTATION_THRESHOLD
        if visible_rotation <= PLAYER_ROTATION_THRESHOLD:
            visible_rotation = PLAYER_ROTATION_THRESHOLD

        offset_x = self.bird_x + BIRD_WIDTH
        offset_y = self.bird_y + BIRD_HEIGHT / 2

        angles = np.linspace(0- 90 - visible_rotation, 180 - 90 - visible_rotation, NUM_RAYS)
        radians = np.radians(angles)
        sin_angles = np.sin(radians)
        cos_angles = np.cos(radians)

        for i, rad in enumerate(radians):
            ray_length = self.ray_distances[i] * max_distance

            end_x = offset_x + cos_angles[i] * ray_length
            end_y = offset_y + sin_angles[i] * ray_length

            color = (0, 255, 0) if self.ray_distances[i] < 1 else (255, 0, 0)  # Green for hits, red for max distance
            pygame.draw.line(self.screen, color, (offset_x, offset_y), (end_x, end_y), 1)

    def draw_private_zone(self):
        """
        Draw the private zone of the bird as a transparent circle (for debugging).
        """
        if not self.visualize:
            return
        
        radius = self.compute_private_zone_radius()
        center_x = self.bird_x + BIRD_WIDTH // 2
        center_y = self.bird_y + BIRD_HEIGHT // 2
        pygame.draw.circle(self.screen, (255, 0, 0), (int(center_x), int(center_y)), int(radius), 1)

    def update_ui(self):
        if not self.visualize:
            return 
        
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
