import numpy as np
import torch
import pygame
import os
import pickle
from collections import deque
from agent import FlappyBirdAgent 
from configs.dqn_configs import *
from configs.game_configs import NUM_RAYS
from game import FlappyBirdPygame
from visualize_training import plot


# Training configurations
LOAD_MODEL = True  # Load from a saved checkpoint if available
SAVE_INTERVAL = 10000  # Save the model every 10000 steps
TEST_MODE = False  # If True, AI only plays without training
NUM_EPISODES = 10000  # Increased number of episodes for longer training
MAX_STEPS_PER_EPISODE = 10000000  # Maximum steps per episode
CONSECUTIVE_WINS_THRESHOLD = 100  # Stop training if AI wins 100 consecutive episodes

# Initialize game and agent
game = FlappyBirdPygame()
state_dim = NUM_RAYS + 1
num_actions = 2  # [Do nothing, Jump]
agent = FlappyBirdAgent(state_dim, num_actions)

# Target Network Update Configurations
TARGET_UPDATE = 3000  # Hard update every 3,000 steps
steps_done = 0  # Step counter

# Load model if available
if LOAD_MODEL and os.path.exists("policy_net.pth"):
    agent.policy_net.load_state_dict(torch.load("policy_net.pth"))
    agent.target_net.load_state_dict(torch.load("target_net.pth"))
    print("Loaded saved model.")

    # Reset epsilon so AI continues exploring instead of only exploiting past actions
    agent.epsilon = max(agent.epsilon_min, agent.epsilon * 0.99)  # Ensure some exploration

if os.path.exists("replay_buffer.pkl"):
    with open("replay_buffer.pkl", "rb") as f:
        agent.memory = pickle.load(f)

    print(f"Loaded replay buffer with {len(agent.memory)} experiences.")

scores = []
mean_scores = []
window_size = 10  # Moving average window for mean score
consecutive_wins = 0  # Counter for consecutive wins

for episode in range(NUM_EPISODES):
    state = game.get_state()
    state_seq = deque([state] * FRAME_STACK, maxlen=FRAME_STACK)
    total_reward = 0
    steps = 0
    is_winning_episode = True  # Assume the AI wins unless it loses

    # game loop
    while not game.is_game_over and steps < MAX_STEPS_PER_EPISODE:
        state_array = np.array(state_seq)
        state_tensor = torch.tensor(state_array, dtype=torch.float32).unsqueeze(0) # (1, seq_length, state_dim)
        
        action = agent.choose_action(state_tensor)
        action_one_hot = [0, 1] if action == 1 else [1, 0]

        reward, game_over, score = game.step(action_one_hot)
        next_state = game.get_state()
        state_seq.append(next_state)
        next_state_array = np.array(state_seq)

        if not TEST_MODE:
            agent.replay_buffer.store_transition(state_array, action, reward, next_state_array)
            agent.insert_count += 1
            agent.train()

        state = next_state
        total_reward += reward
        steps += 1
        steps_done += 1

        # Soft Update: Gradually update target network weights
        for target_param, policy_param in zip(agent.target_net.parameters(), agent.policy_net.parameters()):
            target_param.data.copy_(TAU * policy_param.data + (1 - TAU) * target_param.data)

        # Save model every `SAVE_INTERVAL` steps
        if steps_done % SAVE_INTERVAL == 0:
            torch.save(agent.policy_net.state_dict(), "policy_net.pth")
            torch.save(agent.target_net.state_dict(), "target_net.pth")
            print(f"Saved model at step {steps_done}")

        if game_over or steps >= MAX_STEPS_PER_EPISODE:
            scores.append(score)
            mean_scores.append(np.mean(scores[-window_size:]))

            # Check if AI lost (score = 0)
            if game_over or score == 0:
                is_winning_episode = False  # AI lost

            print(f"Episode {episode + 1}: Score = {score}, Mean Score = {mean_scores[-1]:.2f}, Consecutive Wins = {consecutive_wins}")

            game.reset()

            break

    # Update consecutive win counter
    if is_winning_episode:
        consecutive_wins += 1
    else:
        consecutive_wins = 0  # Reset if AI loses

    # Stop training if AI wins `CONSECUTIVE_WINS_THRESHOLD` times in a row
    if consecutive_wins >= CONSECUTIVE_WINS_THRESHOLD:
        print(f"AI has won {CONSECUTIVE_WINS_THRESHOLD} consecutive times! Training stopped.")
        break

    if episode % 1000 == 0:
        with open("replay_buffer.pkl", "wb") as f:
            pickle.dump(list(agent.replay_buffer.memory), f)

print("Training Completed!")

plot(scores, mean_scores)

pygame.quit()
