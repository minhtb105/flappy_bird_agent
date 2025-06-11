# 🐦 Flappy Bird AI – Reinforcement Learning Agent

🚀 **Flappy Bird AI** is a deep reinforcement learning (RL) agent trained using **Double Deep Q-Networks (DDQN)** and **Prioritized Experience Replay (PER)**. The AI learns to **play Flappy Bird autonomously**, adapting to dynamic environments with **stochastic pipes**.

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

Train the **Flappy Bird agent** using Deep Q-Learning:

```bash
pip install -r requirements.txt
```

Run the training script:

```bash
python train.py
```

The AI will start learning from scratch!  
Model checkpoints are saved every 1000 episodes.  
Training graphs are automatically generated to visualize progress.

---

# 🔬 How the AI Works

Flappy Bird AI is trained using Deep Q-Learning (DQN) with several optimizations:

### 🧠 Neural Network Architecture
- **Input**: The LIDAR sensor 180.
- **Hidden Layers**: Fully connected deep neural network.
- **Output**: Q-values for jump or no jump decisions.

### 🏗️ Reinforcement Learning Enhancements

| **Feature**                     | **Purpose**                                         |
|----------------------------------|----------------------------------------------------|
| **Dueling Motion Transformers**            | Prevents Q-value overestimation                    |
| **Prioritized Experience Replay (PER)** | Speeds up learning by focusing on important experiences |
| **Stochastic Pipes**             | Forces AI to adapt to random environments          |
| **Soft Target Network Updates** | Improves stability during training              |

---

# 🚀 Future Improvements

- 🔹 **NeuroEvolution** – Train the AI using genetic algorithms instead of backpropagation.
- 🔹 **Self-Play** – Train the AI by competing against itself.
- 🔹 **Multi-Agent Training** – Create multiple birds learning in parallel.
- 🔹 **Imitation Learning** – Train the AI using human gameplay data.
