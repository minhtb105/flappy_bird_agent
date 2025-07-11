import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

from agent import FlappyBirdAgent
from game import FlappyBirdPygame
from configs.dqn_configs import FRAME_STACK
from configs.game_configs import NUM_RAYS
import torch
from collections import deque
import pygame
import logging

MODEL_DIR = "/app/models"

game = FlappyBirdPygame(visualize=False) 
state_dim = NUM_RAYS
num_actions = 2
agent = FlappyBirdAgent(state_dim, num_actions)

def load_model():
    agent.policy_net.load_state_dict(torch.load(f"{MODEL_DIR}/policy.pth", map_location=agent.device))
    agent.target_net.load_state_dict(torch.load(f"{MODEL_DIR}/target.pth", map_location=agent.device))
    agent.optimizer.load_state_dict(torch.load(f"{MODEL_DIR}/optimizer.pth", map_location=agent.device))
    agent.replay_buffer.load(f"{MODEL_DIR}/buffer.pth")

def test_loop(num_episodes=2, max_steps_per_episode=10000000):
    load_model()
    agent.policy_net.eval()
    agent.target_net.eval()

    max_score = 0
    test_scores = []

    for _ in range(num_episodes):
        game.reset()
        state = game.get_state()
        state_seq = deque([state] * FRAME_STACK, maxlen=FRAME_STACK)
        total_score, steps = 0, 0

        while not game.is_game_over and steps < max_steps_per_episode:
            state_tensor = torch.tensor(state_seq, dtype=torch.float32).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action = agent.choose_action(state_tensor, explore=False)
            action_one_hot = [0, 1] if action == 1 else [1, 0]
            reward, game_over, score = game.step(action_one_hot)
            next_state = game.get_state()
            state_seq.append(next_state)
            steps += 1
            total_score += score

            if game_over:
                break

        test_scores.append(game.score)
        max_score = max(max_score, game.score)
        game.reset()

    logging.info(f"Testing completed. Max test score: {max_score}")
    pygame.display.quit()
    pygame.quit()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_loop()
