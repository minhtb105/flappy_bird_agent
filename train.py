import pygame
import os
import numpy as np
import sys
import logging
import imageio
from collections import deque
import torch
from torch.utils.tensorboard import SummaryWriter
from agent import FlappyBirdAgent 
from configs.dqn_configs import (NUM_EPISODES, TAU, FRAME_STACK,
                                 MAX_STEPS_PER_EPISODE, SAVE_INTERVAL,
                                 CONSECUTIVE_WINS_THRESHOLD)
from configs.game_configs import NUM_RAYS, FPS
from game import FlappyBirdPygame


# Initialize game and agent
game = FlappyBirdPygame(visualize=True)
state_dim = NUM_RAYS 
num_actions = 2  # [Do nothing, Jump]
agent = FlappyBirdAgent(state_dim, num_actions)

agent.count_parameters()

writer = SummaryWriter("plots/")
video_path="plots/training_record.mp4"
video_writer = imageio.get_writer(video_path, fps=FPS, codec="libx264")

def train_loop(config=None, 
               num_episodes=NUM_EPISODES,
               visualize=False,
               record_video=True):
    max_score = 0
    try:
        load_model()
        agent.replay_buffer.load_from_torch_dict()
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

        consecutive_wins, steps = 0, 0
        for episode in range(num_episodes):
            game.reset()
            state = game.get_state()
            state_seq = deque([state] * FRAME_STACK, maxlen=FRAME_STACK)
            is_winning_episode = True

            while (not game.is_game_over) and (steps < MAX_STEPS_PER_EPISODE):
                if record_video and hasattr(game, "screen") and game.score > max_score:
                    frame = pygame.surfarray.array3d(game.screen)
                    frame = frame.transpose(1, 0, 2) 
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
                agent.replay_buffer.store_transition(state_array, action, reward, next_state_array)
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
                torch.save(agent.policy_net.state_dict(), "models/policy.pth")
                torch.save(agent.target_net.state_dict(), "models/target.pth")
                torch.save(agent.optimizer.state_dict(), "models/optimizer.pth")

            if consecutive_wins >= CONSECUTIVE_WINS_THRESHOLD:
                logging.info(f"AI has won {CONSECUTIVE_WINS_THRESHOLD} times in a row. Stopping training.")
                break
        
        return game.rewards
    finally:
        agent.replay_buffer.to_torch_dict(path="models/buffer.pth")
        if video_writer is None:
            pass
        else:
            video_writer.close()
            logging.info(f"Video saved to {video_path}")

        logging.info(f"Training completed. Max test score: {max_score}")

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
         

if __name__ == "__main__":
    train_loop(record_video=True)
    video_writer.close()
    pygame.display.quit()
    pygame.quit()
    logging.shutdown() # Ensure all logs are flushed
    sys.exit(0) # Exit the entire program safely
