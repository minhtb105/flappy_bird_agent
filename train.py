import pygame
import os
import logging
from collections import deque
import argparse
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

def log_error_stats(step, message, interval=1000):
    if step % interval == 0:
        for key, count in error_counts.items():
            if count > 0:
                logging.warning(f"[{key}] occurred {count} times in last {interval} steps at step {step}: {message}")
                error_counts[key] = 0

# Error counter
error_counts = {
    "action_selection_failure": 0,
    "game_step": 0,
    "training_step": 0,
}


# Load model if available
if LOAD_MODEL and os.path.exists("models/policy_net.pth"):
    agent.policy_net.load_state_dict(torch.load("models/policy_net.pth"))
    agent.target_net.load_state_dict(torch.load("models/target_net.pth"))
    logging.info("Loaded saved model.")

    # agent.temp = max(agent.temp_min, agent.temp * TEMP_DECAY_RESET)  # Reset temperature

if os.path.exists("replay_buffer.pt"):
    buffer_data = torch.load("models/replay_buffer.pt")
    agent.replay_buffer.load_from_torch_dict(buffer_data)
    logging.info(f"Loaded replay buffer with {len(agent.replay_buffer.memory)} experiences.")

def train_loop():
    scores, mean_scores, temperatures = [], [], []
    max_score, consecutive_wins, steps_done = 0, 0, 0

    for episode in range(NUM_EPISODES):
        state = game.get_state()
        state_seq = deque([state] * FRAME_STACK, maxlen=FRAME_STACK)
        total_reward, steps, total_score = 0, 0, 0
        is_winning_episode = True

        while not game.is_game_over and steps < MAX_STEPS_PER_EPISODE:
            state_array = np.array(state_seq)
            state_tensor = torch.tensor(state_array, dtype=torch.float32).unsqueeze(0)

            try:
                with torch.no_grad():
                    action = agent.choose_action(state_tensor)
            except Exception as e:
                error_counts["action_selection_failure"] += 1
                log_error_stats(steps_done, f"Action selection failed: {e}")
                break

            action_one_hot = [0, 1] if action == 1 else [1, 0]

            try:
                reward, game_over, score = game.step(action_one_hot)
                total_score += score
            except Exception as e:
                error_counts["game_step"] += 1
                log_error_stats("game_step", f"Game step failed: {e}")
                break

            next_state = game.get_state()
            state_seq.append(next_state)
            next_state_array = np.array(state_seq)

            try:
                agent.update_temperature()
                agent.replay_buffer.store_transition(state_array, action, reward, next_state_array)
                agent.insert_count += 1
                agent.train()
            except Exception as e:
                error_counts["training_step"] += 1
                log_error_stats(steps_done, f"Training step failed: {e}")
                break

            game.is_game_over = game_over
            total_reward += reward
            steps_done += 1

            # Soft Update: Gradually update target network weights
            for target_param, policy_param in zip(agent.target_net.parameters(), agent.policy_net.parameters()):
                target_param.data.copy_(TAU * policy_param.data + (1 - TAU) * target_param.data)

            if steps_done % SAVE_INTERVAL == 0 and steps_done != 0:
                torch.save(agent.policy_net.state_dict(), "models/policy_net.pth")
                torch.save(agent.target_net.state_dict(), "models/target_net.pth")
                logging.info(f"Saved model at step {steps_done}")

            if game_over:
                break

        max_score = max(max_score, total_score)
        scores.append(total_score)

        if (episode + 1) % 50 == 0:
            mean_scores.append(np.mean(scores[-50:]))
            logging.info(f"Episode {episode+1}: Score = {total_score}, Mean Score = {mean_scores[-1]:.2f}, Wins = {consecutive_wins}")

        if game.is_game_over or total_score == 0:
            is_winning_episode = False

        consecutive_wins = consecutive_wins + 1 if is_winning_episode else 0

        if consecutive_wins >= CONSECUTIVE_WINS_THRESHOLD:
            logging.info(f"AI has won {CONSECUTIVE_WINS_THRESHOLD} times in a row. Stopping training.")
            break

        if episode % SAVE_REPLAY_BUFFER_INTERVAL == 0 and episode != 0:
            buffer_dict = agent.replay_buffer.to_torch_dict()
            torch.save(buffer_dict, "models/replay_buffer.pt")
            logging.info("Saved replay buffer.")

        if episode % TRACK_TEMPERATURE_DECAY_INTERVAL == 0:
            temperatures.append(agent.temp)

        if episode % VISUALIZATION_INTERVAL == 0 and episode != 0:
            vis_folder = f"plots/episode_{episode}"
            os.makedirs(vis_folder, exist_ok=True)
            plot(scores[::50], save_path=f"{vis_folder}/scores.png", label="Score", title="Score per Episode")
            plot(mean_scores, save_path=f"{vis_folder}/mean_scores.png", label="Mean Score", title="Mean Score per Episode")
            plot_temperature_decay(temperatures, save_path=f"{vis_folder}/temperature_decay.png")
            plot_attention_heatmap(agent.policy_net.get_attention_weights(), save_path=f"{vis_folder}/attention_weights.png")
            plot_histogram_td_errors(agent.td_errors, save_path=f"{vis_folder}/td_errors.png")
            plot_losses(agent.losses, 50, save_path=f"{vis_folder}/losses.png")
            plot_grad_norms(agent.grad_norms, save_path=f"{vis_folder}/grad_norms.png")
            plot_q_values_distribution(agent.q_values, save_path=f"{vis_folder}/q_values_dist.png")
            plot_q_stats(agent.max_q_values, agent.min_q_values, save_path=f"{vis_folder}/q_stats.png")

        game.reset()
        
    logging.info("Training Completed")
    pygame.quit()

def test_loop(num_episodes=100, max_steps_per_episode=1000, window_size=10):
    max_score = 0
    test_scores, mean_scores = [], []

    for episode in range(num_episodes):
        game.reset()
        state = game.get_state()
        total_reward, total_score, steps = 0, 0, 0

        while not game.is_game_over and steps < max_steps_per_episode:
            try:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action = agent.choose_action(state_tensor, explore=False)
            except Exception as e:
                logging.error(f"Test action error at episode {episode}: {e}")
                break

            action_one_hot = [0, 1] if action == 1 else [1, 0]
            reward, game_over, score = game.step(action_one_hot)
            next_state = game.get_state()
            state = next_state
            total_reward += reward
            steps += 1
            total_score += score

            if game_over:
                break

        test_scores.append(total_score)
        max_score = max(max_score, total_score)

        if (episode + 1) % window_size == 0:
            mean_score = np.mean(test_scores[-window_size:])
            mean_scores.append(mean_score)
            logging.info(f"[TEST] Ep {episode+1} - Score: {total_score}, Mean Score: {mean_score:.2f}, Max Score: {max_score}")

    logging.info(f"Testing completed. Max test score: {max_score}")
    pygame.quit()

def extract_log_message(line):
    parts = line.split(" - ", maxsplit=2)
    
    return ': '.join(parts[1:]).strip() if len(parts) == 3 else line.strip()

def clean_log(input_path='logs/debug_log.txt', output_path='logs/clean_log.txt'):
    try:
        with open(input_path, 'r') as f:
            lines = f.readlines()

        cleaned_lines = []
        last_msg = None

        for line in lines:
            msg = extract_log_message(line)
            if msg != last_msg:
                cleaned_lines.append(line)
                last_msg = msg

        with open(output_path, 'w') as f:
            f.writelines(cleaned_lines)

        logging.info(f"Cleaned debug log: {len(lines)} → {len(cleaned_lines)} lines written to {output_path}")
    except Exception as e:
        logging.error(f"[clean_log] Failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Flappy Bird AI")
    parser.add_argument("--mode", type=str, choices=['train', 'test'], default='train')
    parser.add_argument("--log-level", type=str, choices=['debug', 'info', 'warning', 'error'], default='info', help="Logging level")
    args = parser.parse_args()
    
    log_levels = {'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING, 'error': logging.ERROR}
    logging.basicConfig(filename='logs/debug_log.txt', level=log_levels[args.log_level], format='%(asctime)s - %(levelname)s - %(message)s')

    if args.mode == 'train':
        train_loop()
    else:
        test_loop()
        
    clean_log()
    agent.analyze_behavior()
