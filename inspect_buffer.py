import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os


def load_buffer(path="models/replay_buffer.pt"):
    if not os.path.exists(path):
        print(f"File {path} does not exist.")
        return None

    try:
        data = torch.load(path)
        
        return data
    except Exception as e:
        print(f"Error loading buffer: {e}")

        return None

def inspect_buffer(data):
    states = data['states'].numpy()
    actions = data['actions'].numpy()
    rewards = data['rewards'].numpy()
    next_states = data['next_states'].numpy()
    priorities = data['priorities'].numpy()

    print(f"\nTotal number of samples: {len(states)}")
    print("—" * 40)

    # 1. NaN check
    nan_state = np.isnan(states).any()
    nan_next = np.isnan(next_states).any()
    print(f"There are NaN in state: {nan_state} | next_state: {nan_next}")

    # 2. Reward check
    print("\nReward:")
    print(f"Min: {np.min(rewards):.2f}")
    print(f"Max: {np.max(rewards):.2f}")
    print(f"Mean: {np.mean(rewards):.2f}")
    outliers = np.sum((rewards < -20) | (rewards > 20))
    print(f"Outliers (< -20 or > 20): {outliers} sample(s)")

    # 3. Action distribution
    unique, counts = np.unique(actions, return_counts=True)
    print("\nAction distribution:")
    for u, c in zip(unique, counts):
        pct = c / len(actions) * 100
        print(f"Action {u}: {c} ({pct:.2f}%)")
        if pct > 95:
            print("Bias action!")

    # 4. Priorities
    print("\nPriorities:")
    print(f"Min: {np.min(priorities):.4f}")
    print(f"Max: {np.max(priorities):.4f}")
    print(f"Mean: {np.mean(priorities):.4f}")
    if np.max(priorities) > 100 or np.isnan(priorities).any():
        print("Priorities contain NaN or too large values")

    # 5. States
    print("\nStates:")
    print("State shape:", states.shape)
    print("Min:", states.min())
    print("Max:", states.max())
    print("Mean:", states.mean())
    print("Std dev:", states.std())
    print("NaN:", np.isnan(states).any())
    print("State 0:\n", states[0])
    print("State 1:\n", states[1])
    print("State diff:", np.abs(states[1] - states[0]).mean())

    states_flat = states.reshape(-1, states.shape[-1])  
    actions_repeated = np.repeat(actions, states.shape[1]) 

    pca = PCA(n_components=2)
    states_pca = pca.fit_transform(states_flat)

    plt.figure(figsize=(8,6))
    plt.scatter(states_pca[:, 0], states_pca[:, 1], c=actions_repeated, cmap='coolwarm', s=2)
    plt.colorbar(label='Action')
    plt.title("PCA projection of Full State Space")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.savefig("plots/state_space_pca.png")

    # 5. Plot reward distribution
    plt.hist(rewards, bins=50, color='skyblue', edgecolor='black')
    plt.title("Reward Distribution")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/reward_distribution.png")
    

if __name__ == "__main__":
    buffer_data = load_buffer("models/replay_buffer.pt")
    if buffer_data:
        inspect_buffer(buffer_data)
