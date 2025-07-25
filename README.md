# 🐦 Flappy Bird AI – Reinforcement Learning Agent

🚀 **Flappy Bird AI** is a deep reinforcement learning (RL) agent trained using **Motion Transformers** and **Prioritized Experience Replay (PER)**. The AI learns to **play Flappy Bird autonomously**, adapting to dynamic environments with **stochastic pipes**.

![Flappy Bird AI Demo](assets/demo.gif)  

---

## 🚀 Features
- 🧠 **Dueling Motion Transformers** – Reduces overestimation of Q-values for stable learning.
- 🎯 **Prioritized Experience Replay (PER)** – Learns faster by focusing on high-error transitions.
- 📡 **LIDAR-based Observations** – Bird sees the world via 180-degree raycasting for better perception.
- 🌪️ **Stochastic Pipe Generation** – Forces the agent to generalize by varying obstacle positions.
- 🧱 **Temporal State Stacking** – Uses `FRAME_STACK=12` to encode time into input.
- 💾 **Model Checkpointing** – Saves model every N episodes for resumable training or evaluation.
- 📈 **TensorBoard Logging** – Visualize loss, rewards, Q-values, TD errors and more.
---

## 📥 Install Dependencies
Ensure you have Python 3.8+ installed, then run:
```bash
git clone https://github.com/minhtb105/flappy-bird-agent.git
cd flappy-bird-agent
---

# 🤖 Training the AI

Train the **Flappy Bird agent** using Motion Transformers:

```bash
pip install -r requirements.txt
```

Run the training script:

```bash
python train.py
```

- The AI will start learning from scratch!  
- Model checkpoints are saved every 1000 episodes.  
- Training metrics are logged in plots/ for TensorBoard.
---

Launch TensorBoard:

```bash
tensorboard --logdir=plots
```

# 🔬 How the AI Works

Flappy Bird AI is trained using Dueling Motion Transformers with several optimizations:

### 🧠 Neural Network Architecture
- **Input**: The LIDAR sensor 180.
- **Hidden Layers**: Fully connected deep neural network.
- **Output**: Q-values for jump or no jump decisions.

---

🗂 Directory Structure
```bash
📦 your-project/
├── train.py                        # Main training loop
├── agent.py                        # DQN agent logic
├── replay_buffer.py                # Prioritized Experience Replay + Filtering + PER
├── game.py                         # Flappy Bird environment using pygame
├── dueling_motion_transformers.py # Transformer-based Dueling Q-Network
├── inspect_buffer.py              # Analyze and visualize replay buffer content
├── experiment_parameter_tuning.py # Bayesian optimization for RL hyperparameters
│
├── configs/
│   ├── dqn_configs.py              # Hyperparameters for DQN training
│   ├── game_configs.py             # Physics and game-specific settings
│   └── pbounds.py                  # Parameter bounds for Bayesian optimization
│
├── models/                         # Saved models and replay buffer chunks
├── logs/                           # TensorBoard logs
├── plots/                          # Buffer analysis plots (PCA, rewards, priorities)
```

# 🚀 Future Improvements

- 🔹 **NeuroEvolution** – Train the AI using genetic algorithms instead of backpropagation.
- 🔹 **Self-Play** – Train the AI by competing against itself.
- 🔹 **Multi-Agent Training** – Create multiple birds learning in parallel.
- 🔹 **Imitation Learning** – Train the AI using human gameplay data.
