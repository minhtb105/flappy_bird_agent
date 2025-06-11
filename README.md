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

### 🔁 Replay Buffer Design (Prioritized + Filtered)
The replay buffer is designed for efficiency and focus by selectively retaining high-quality transitions. It includes:

🧠 Filtering Strategy
Before saving the buffer to disk, we filter transitions from memory using the following weighted sampling criteria:

Type	Description	Ratio
🔄 random_ratio	Uniformly random transitions (ensure diversity)	20%
💰 reward_ratio	Transitions with high absolute reward values	40%
🔥 priority_ratio	Transitions with high TD-error priorities (PER)	40%

This ensures the agent learns from:

both mistakes and successes (reward_ratio),

surprising/difficult transitions (priority_ratio),

and a small set of random noise for generalization (random_ratio).

These filtered samples are saved periodically to disk.

💾 Efficient Batch-Saving & Loading
Instead of saving the entire replay buffer at once (which is large), the filtered samples are:

✅ Split into batches (e.g., 50,000 samples per chunk)

✅ Saved as .pt files (e.g., replay_buffer_epXXX_chunk0.pt)

✅ Reloaded later and merged during fine-tuning or evaluation

This helps reduce memory usage and improves I/O performance during long training runs.

---

🗂 Directory Structure
```bash
📦 your-project/
├── train.py # Main training loop
├── agent.py # DQN agent logic
├── game.py # Flappy Bird environment
├── replay_buffer.py # Replay buffer with PER
│ ├── configs/
│ ├── dqn_configs.py # DQN settings
│ └── game_configs.py # Game settings
│ ├── models/ # Saved models & checkpoints
├── plots/ # Reward, Q-value, loss, TD-error, ... graphs
├── logs/ # logs 
```

# 🚀 Future Improvements

- 🔹 **NeuroEvolution** – Train the AI using genetic algorithms instead of backpropagation.
- 🔹 **Self-Play** – Train the AI by competing against itself.
- 🔹 **Multi-Agent Training** – Create multiple birds learning in parallel.
- 🔹 **Imitation Learning** – Train the AI using human gameplay data.
