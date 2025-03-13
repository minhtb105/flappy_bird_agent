import numpy as np
import torch
import pygame
import os
import pickle
from deep_q_network import FlappyBirdAgent
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
input_shape = 11
num_actions = 2  # [Do nothing, Jump]
agent = FlappyBirdAgent(input_shape, num_actions)

# Target Network Update Configurations
TARGET_UPDATE = 3000  # Hard update every3,000 steps
tau = 0.05  # Soft update factor
steps_done = 0  # Step counter

# Load model if available
if LOAD_MODEL and os.path.exists("policy_net.pth"):
    agent.policy_net.load_state_dict(torch.load("policy_net.pth"))
    agent.target_net.load_state_dict(torch.load("target_net.pth"))
    print("Loaded saved model.")

    # Reset epsilon so AI continues exploring instead of only exploiting past actions
    agent.epsilon = max(agent.epsilon_min, 0.1)  # Ensure some exploration

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
    total_reward = 0
    steps = 0
    is_winning_episode = True  # Assume the AI wins unless it loses

    # game loop
    while not game.is_game_over and steps < MAX_STEPS_PER_EPISODE:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        action = agent.choose_action(state_tensor).item()
        action_one_hot = [0, 1] if action == 1 else [1, 0]

        reward, game_over, score = game.step(action_one_hot)
        next_state = game.get_state()

        if not TEST_MODE:
            agent.replay_buffer.store_transition(state, action, reward, next_state)
            agent.train()

        state = next_state
        total_reward += reward
        steps += 1
        steps_done += 1

        # Hard Update: Update target network every `TARGET_UPDATE` steps
        if steps_done % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        # Soft Update: Gradually update target network weights
        for target_param, policy_param in zip(agent.target_net.parameters(), agent.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)

        # Save model every `SAVE_INTERVAL` steps
        if steps_done % SAVE_INTERVAL == 0:
            torch.save(agent.policy_net.state_dict(), "policy_net.pth")
            torch.save(agent.target_net.state_dict(), "target_net.pth")
            print(f"Saved model at step {steps_done}")

        if game_over or steps >= MAX_STEPS_PER_EPISODE:
            scores.append(score)
            mean_scores.append(np.mean(scores[-window_size:]))

            # Check if AI lost (score = 0)
            if score == 0:
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

with open("replay_buffer.pkl", "wb") as f:
    pickle.dump(agent.replay_buffer, f)

print("Training Completed!")

plot(scores, mean_scores)

pygame.quit()
