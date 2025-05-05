import numpy as np
import torch
import pygame
import os
from collections import deque
from agent import FlappyBirdAgent 
from configs.dqn_configs import *
from configs.game_configs import NUM_RAYS
from game import FlappyBirdPygame
from visualization_utils import *

# Initialize game and agent
game = FlappyBirdPygame()
state_dim = NUM_RAYS + 1
num_actions = 2  # [Do nothing, Jump]
agent = FlappyBirdAgent(state_dim, num_actions)

agent.count_parameters()

steps_done = 0  # Step counter

# Load model if available
if LOAD_MODEL and os.path.exists("models/policy_net.pth"):
    agent.policy_net.load_state_dict(torch.load("models/policy_net.pth"))
    agent.target_net.load_state_dict(torch.load("models/target_net.pth"))
    print("Loaded saved model.")

    # Reset epsilon so AI continues exploring instead of only exploiting past actions
    agent.epsilon = max(agent.epsilon_min, agent.epsilon * TEMP_DECAY_RESET)  # Ensure some exploration

if os.path.exists("replay_buffer.pt"):
    buffer_data = torch.load("models/replay_buffer.pt")
    agent.replay_buffer.load_from_torch_dict(buffer_data)
    print(f"Loaded replay buffer with {len(agent.replay_buffer.memory)} experiences.")

scores = []
mean_scores = []
max_score = 0
window_size = 50  # Moving average window for mean score
consecutive_wins = 0  # Counter for consecutive wins
epsilons = []  # Store epsilon values for plotting

for episode in range(NUM_EPISODES):
    state = game.get_state()
    state_seq = deque([state] * FRAME_STACK, maxlen=FRAME_STACK)
    total_reward = 0
    steps = 0
    is_winning_episode = True  # Assume the AI wins unless it loses

    # game loop
    while not game.is_game_over and steps < MAX_STEPS_PER_EPISODE:
        state_array = np.array(state_seq)
        state_tensor = torch.tensor(state_array, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        
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
            torch.save(agent.policy_net.state_dict(), "models/policy_net.pth")
            torch.save(agent.target_net.state_dict(), "models/target_net.pth")
            print(f"Saved model at step {steps_done}")

        if game_over or steps >= MAX_STEPS_PER_EPISODE:
            max_score = max(max_score, score)  # Update max score   
            scores.append(score)
            if (episode + 1) % window_size == 0:
                mean_scores.append(np.mean(scores[-window_size:]))

            # Check if AI lost (score = 0)
            if game_over or score == 0:
                is_winning_episode = False  # AI lost

            if (episode + 1) % window_size == 0:
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

    if episode % SAVE_REPLAY_BUFFER_INTERVAL == 0 and episode != 0:
        # Save replay buffer every 1000 episodes
        buffer_dict = agent.replay_buffer.to_torch_dict()
        torch.save(buffer_dict, "models/replay_buffer.pt")
        print("Replay buffer saved.")

    if episode % TRACK_EPSILON_DECAY_INTERVAL == 0:
        epsilons.append(agent.epsilon)

    if episode % VISUALIZATION_INTERVAL == 0 and episode != 0:
        print(max_score)
        plot(scores[::window_size], save_path="plots/scores.png", label="Score", title="Score per Episode")
        plot(mean_scores, save_path="plots/mean_scores.png", label="Mean Score", title="Mean Score per Episode")
        plot_epsilon_decay(epsilons) 
        plot_attention_heatmap(agent.policy_net.get_attention_weights()) 
        plot_histogram_td_errors(agent.td_errors)
        plot_losses(agent.losses, window_size=window_size)
        plot_grad_norms(agent.grad_norms)
        plot_q_values_distribution(agent.q_values)
        plot_q_stats(agent.max_q_values, agent.min_q_values)

print("Training Completed!")
pygame.quit()
