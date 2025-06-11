# 🐦 Flappy Bird AI – Reinforcement Learning Agent

🚀 **Flappy Bird AI** is a deep reinforcement learning (RL) agent trained using **Double Deep Q-Networks (DDQN)** and **Prioritized Experience Replay (PER)**. The AI learns to **play Flappy Bird autonomously**, adapting to dynamic environments with **stochastic pipes**.

![Flappy Bird AI Demo](assets/demo.gif)  

---

## 🚀 Features
- **🧠 Double Deep Q-Networks (DDQN)** – More stable training by reducing Q-value overestimation.
- **🎯 Prioritized Experience Replay (PER)** – AI learns faster by focusing on important mistakes.
- **🌎 Stochastic Pipes (Randomized Levels)** – Prevents overfitting by forcing AI to adapt.
- **💾 Model Checkpointing** – Saves progress every 10000 steps for later training or testing.

---

## 📥 Install Dependencies
Ensure you have Python 3.8+ installed, then run:
```bash
pip install -r requirements.txt

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
Model checkpoints are saved every 500 episodes.  
Training graphs are automatically generated to visualize progress.

---

# 🔬 How the AI Works

Flappy Bird AI is trained using Deep Q-Learning (DQN) with several optimizations:

### 🧠 Neural Network Architecture
- **Input**: Bird's `y-position`, `velocity`, `distance to next pipe`, and `gap center`.
- **Hidden Layers**: Fully connected deep neural network.
- **Output**: Q-values for jump or no jump decisions.

### 🏗️ Reinforcement Learning Enhancements

| **Feature**                     | **Purpose**                                         |
|----------------------------------|----------------------------------------------------|
| **Double DQN (DDQN)**            | Prevents Q-value overestimation                    |
| **Prioritized Experience Replay (PER)** | Speeds up learning by focusing on important experiences |
| **Stochastic Pipes**             | Forces AI to adapt to random environments          |
| **Soft & Hard Target Network Updates** | Improves stability during training              |

---

# 🚀 Future Improvements

- 🔹 **NeuroEvolution** – Train the AI using genetic algorithms instead of backpropagation.
- 🔹 **Self-Play** – Train the AI by competing against itself.
- 🔹 **Multi-Agent Training** – Create multiple birds learning in parallel.
- 🔹 **Imitation Learning** – Train the AI using human gameplay data.
