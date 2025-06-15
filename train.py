import pygame
import os
import sys
import cv2
import logging
import imageio
from collections import deque
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from agent import FlappyBirdAgent 
from configs.dqn_configs import *
from configs.game_configs import *
from game import FlappyBirdPygame
from visualization_utils import *


# Initialize game and agent
game = FlappyBirdPygame(visualize=True)
state_dim = NUM_RAYS 
num_actions = 2  # [Do nothing, Jump]
agent = FlappyBirdAgent(state_dim, num_actions)

agent.count_parameters()

writer = SummaryWriter("plots/")

def train_loop(config=None, 
               num_episodes=NUM_EPISODES,
               visualize=False,
               record_video=True, 
               video_path="plots/training_record.mp4"):
    
    load_model()
    agent.replay_buffer.load_checkpoint("models/buffer.zarr")
    if config:
        for param, value in config.items():
            if hasattr(game, param):
                game.__dict__[param] = value
            elif hasattr(agent, param):
                agent.__dict__[param] = value
            elif hasattr(agent.replay_buffer, param):
                agent.replay_buffer.__dict__[param] = value
        
        learning_keys = {"learning_rate", "weight_decay", "beta1", "beta2"}
        if any(k in config for k in learning_keys):
            agent.update_optimizer()

    max_score, consecutive_wins, steps = 0, 0, 0

    if record_video:
        video_writer = imageio.get_writer(video_path, fps=FPS, codec="libx264")
    else:
        video_writer = None
    
    for episode in range(num_episodes):
        game.reset()
        state = game.get_state()
        state_seq = deque([state] * FRAME_STACK, maxlen=FRAME_STACK)
        is_winning_episode = True

        while not game.is_game_over and steps < MAX_STEPS_PER_EPISODE:
            if record_video and hasattr(game, "screen") and game.score > max_score:
                frame = pygame.surfarray.array3d(game.screen)
                frame = frame.transpose(1, 0, 2)  # (width, height, 3) -> (height, width, 3)
                if frame.shape[0] != HEIGHT or frame.shape[1] != WIDTH:
                    frame = cv2.resize(frame, (WIDTH, HEIGHT))
                video_writer.append_data(frame)

            state_array = np.array(state_seq)
            state_tensor = torch.tensor(state_array, dtype=torch.float32).unsqueeze(0).to(agent.device)

            with torch.no_grad():
                action = agent.choose_action(state_tensor)

            action_one_hot = [0, 1] if action == 1 else [1, 0]

            reward, game_over, _ = game.step(action_one_hot)
            next_state = game.get_state()
            state_seq.append(next_state)
            next_state_array = np.array(state_seq)

            agent.update_temperature()
            agent.replay_buffer.store(state_array, action, reward, next_state_array, game.is_game_over)
            agent.insert_count += 1
            agent.train()

            steps += 1
            agent.soft_update_target(TAU)

            if game_over:
                break

        max_score = max(max_score, game.score)

        if visualize:
            agent.log_to_tensorboard(writer, episode)

        if game.is_game_over or game.score == 0:
            is_winning_episode = False

        consecutive_wins = consecutive_wins + 1 if is_winning_episode else 0
            
        if episode % SAVE_INTERVAL == 0 and episode != 0:
            async_save("models/policy.pth", agent.policy_net.state_dict())
            async_save("models/target.pth", agent.target_net.state_dict())
            async_save("models/optimizer.pth", agent.optimizer.state_dict())
            agent.replay_buffer.save_checkpoint()

        if consecutive_wins >= CONSECUTIVE_WINS_THRESHOLD:
            logging.info(f"AI has won {CONSECUTIVE_WINS_THRESHOLD} times in a row. Stopping training.")
            break
        
    if video_writer is not None:
        video_writer.close()
        logging.info(f"Video saved to {video_path}")
        
    logging.info(f"Training completed. Max test score: {max_score}")

    return game.rewards

def test_loop(num_episodes=100, max_steps_per_episode=1000):
    load_model()
    agent.policy_net.eval()
    agent.target_net.eval()
    
    max_score = 0
    test_scores = []

    for _ in range(num_episodes):
        game.reset()
        state = game.get_state()
        state_seq = deque([state] * FRAME_STACK, maxlen=FRAME_STACK)
        total_reward, total_score, steps = 0, 0, 0

        while not game.is_game_over and steps < max_steps_per_episode:
            state_tensor = torch.tensor(state_seq, dtype=torch.float32).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action = agent.choose_action(state_tensor, explore=False)
            action_one_hot = [0, 1] if action == 1 else [1, 0]
            reward, game_over, score = game.step(action_one_hot)
            next_state = game.get_state()
            state_seq.append(next_state)
            total_reward += reward
            steps += 1
            total_score += score

            if game_over:
                break

        test_scores.append(total_score)
        max_score = max(max_score, total_score)
        game.reset()

    logging.info(f"Testing completed. Max test score: {max_score}")
    pygame.display.quit()
    pygame.quit()

def load_model():
    if os.path.exists("models/policy_net.pth"):
        agent.policy_net.load_state_dict(torch.load("models/policy_net.pth"))
        logging.info("Policy network loaded successfully.")
    else:
        logging.warning("No model found to load.")

    if os.path.exists("models/target_net.pth"):
        agent.target_net.load_state_dict(torch.load("models/target_net.pth"))
        logging.info("Target network loaded successfully.")
    else:
        logging.warning("No target network found to load.")

    if os.path.exists("models/optimizer.pth"):
        agent.optimizer.load_state_dict(torch.load("models/optimizer.pth"))
        logging.info("Optimizer state loaded successfully.")
    else:
        logging.warning("No optimizer state found to load.")
         
def async_save(path, obj):
    import threading
    t = threading.Thread(target=torch.save, args=(obj, path))
    t.start()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train Flappy Bird AI")
    parser.add_argument("--mode", type=str, choices=['train', 'test'], default='train')
    parser.add_argument("--log-level", type=str, choices=['debug', 'info', 'warning', 'error'], default='info', help="Logging level")
    args = parser.parse_args()
    
    log_levels = {'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING, 'error': logging.ERROR}
    logging.basicConfig(filename='logs/debug_log.txt', level=log_levels[args.log_level], format='%(asctime)s - %(levelname)s - %(message)s')

    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('font_manager').setLevel(logging.ERROR)

    if args.mode == 'train':
        train_loop(record_video=True)
    else:
        test_loop()

    pygame.display.quit()
    pygame.quit()
    logging.shutdown() # Ensure all logs are flushed
    sys.exit(0) # Exit the entire program safely
