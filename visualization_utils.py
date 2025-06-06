import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity


def plot(scores, save_path="plots/scores.png", label="Score", title="Training Progress"):
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel(label)
    plt.plot(scores, label=label)
    plt.tight_layout()
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
        
    plt.close()


def plot_attention_heatmap(attn_weights, title="Attention Heatmap", save_path="plots/attention_weights.png"):
    if isinstance(attn_weights, torch.Tensor):
        attn_weights = attn_weights.detach().cpu().numpy()
        
    # Remove the batch dimension if present
    if attn_weights.ndim == 4:  # (batch_size, num_heads, seq_len, seq_len)
        attn_weights = attn_weights[0]  
        
    assert attn_weights.ndim == 3, "Attention weights should be 3D (num_heads, seq_len, seq_len) or 2D (seq_len, seq_len)."
    if attn_weights.ndim == 2:
        attn_weights = attn_weights[np.newaxis, ...]
        
    num_heads = attn_weights.shape[0] 

    fig, axs = plt.subplots(1, num_heads, figsize=(5 * num_heads, 5))

    if num_heads == 1:
        axs = [axs]  # make iterable

    for i in range(num_heads):
        ax = axs[i]
        assert attn_weights[i].ndim == 2, "Each attention weight should be 2D (seq_len, seq_len)"
        im = ax.imshow(attn_weights[i], cmap='viridis')
        ax.set_title(f"{title} - Head {i}")
        ax.set_xlabel("Input timestep")
        ax.set_ylabel("Output timestep")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()

def plot_losses(losses, window_size=100, title="Loss Visualization", save_path="plots/losses_separated.png"):
    losses = [l.detach().cpu().item() if torch.is_tensor(l) else l for l in losses]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    plt.suptitle(title, fontsize=16)

    # Subplot 1: Raw Loss
    ax1.plot(losses, color='gray', alpha=0.7)
    ax1.axhline(y=0, color='red', linestyle='--', label='Zero Loss')
    ax1.set_ylabel("Raw Loss")
    ax1.set_title("Raw Loss Over Episodes")
    ax1.legend()
    
    # Subplot 2: Smoothed Loss
    avg_losses = [np.mean(losses[i:i+window_size]) for i in range(0, len(losses), window_size)]
    avg_episodes = list(range(0, len(losses), window_size))
    ax2.plot(avg_episodes, avg_losses, color='blue', label=f'Average Loss (per {window_size} episodes)')
    ax2.axhline(y=0, color='red', linestyle='--')
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Average Loss")
    ax2.set_title(f"Smoothed Loss (Every {window_size} Episodes)")
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
        
    plt.close()
        
def plot_histogram_td_errors(td_errors, title="TD Error Distribution", save_path="plots/td_errors.png"):
    if isinstance(td_errors, torch.Tensor):
        td_errors = td_errors.detach().cpu().numpy()
        
    plt.figure(figsize=(10, 5))
    plt.hist(td_errors, bins=50, color='blue', alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', label='Zero TD Error')
    plt.title(title)
    plt.xlabel("TD Error")
    plt.ylabel("Frequency")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
        
    plt.close()
        
def plot_log_td_error(td_errors, title="Log TD Error Distribution", save_path=None):
    if isinstance(td_errors, torch.Tensor):
        td_errors = td_errors.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 5))
    plt.hist(np.log(np.abs(td_errors) + 1e-10), bins=50, color='blue', alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', label='Zero Log TD Error')
    plt.title(title)
    plt.xlabel("Log TD Error")
    plt.ylabel("Frequency")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
        
    plt.close()
        
def plot_grad_norms(grad_norms, title="Gradient Norms Over Episodes", save_path="plots/grad_norms.png"):
    if isinstance(grad_norms, torch.Tensor):    
        grad_norms = grad_norms.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 5))
    plt.plot(grad_norms)
    plt.yscale('log')
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Gradient Norm")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
        
    plt.close()  
        
def plot_q_values_distribution(q_values, title="Q-values Distribution", save_path="plots/q_values_distribution.png"):
    if isinstance(q_values, torch.Tensor):
        q_values = q_values.detach().cpu().numpy()
        
    plt.figure(figsize=(10, 5))
    plt.hist(q_values, bins=50, color='blue', alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', label='Zero Q-value')
    plt.title(title)
    plt.xlabel("Q-value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
        
    plt.close()
        
def plot_q_stats(max_qs, min_qs, save_path="plots/q_stats_plot.png"):
    if isinstance(max_qs, torch.Tensor):
        max_qs = max_qs.detach().cpu().numpy()
    
    if isinstance(min_qs, torch.Tensor):
        min_qs = min_qs.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 5))
    plt.plot(max_qs, label="Max Q")
    plt.plot(min_qs, label="Min Q")
    
    plt.title("Max and Min Q-values")
    
    plt.xlabel("Training Steps")
    plt.ylabel("Q-value")
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
def plot_positional_embedding_similarity(positional_embeddings: torch.Tensor):
    """
    Draws a heatmap of cosine similarity between timestep positional embeddings.
    
    Args:
        positional_embeddings (Tensor): Shape (seq_len, d_model)
    """
    if not isinstance(positional_embeddings, torch.Tensor):
        raise ValueError("positional_embeddings must be a torch.Tensor")
    
    with torch.no_grad():
        # Normalize each embedding to the unit vector
        norm_embeddings = F.normalize(positional_embeddings, p=2, dim=-1)
        
        # Compute cosine similarity matrix
        similarity_matrix = torch.matmul(norm_embeddings, norm_embeddings.T)
        
        # Convert to numpy for visualization
        sim_np = similarity_matrix.detach().cpu().numpy()
        
        # Plot heatmap
        plt.figure(figsize=(10, 5))
        sns.heatmap(sim_np, xticklabels=True, yticklabels=True, cmap='viridis', square=True)
        plt.title("Cosine Similarity between Positional Embeddings")
        plt.xlabel("Timestep")
        plt.ylabel("Timestep")
        plt.tight_layout()
        plt.show()


def plot_timestep_similarity(memory_tensor, title="Cosine Similarity between Timesteps"):
    if memory_tensor.ndim != 2:
        raise ValueError("memory_tensor must be 2D (timesteps x features)")

    sim_matrix = cosine_similarity(memory_tensor)

    plt.figure(figsize=(8, 6))
    sns.heatmap(sim_matrix, annot=False, cmap='viridis', square=True, cbar=True)
    plt.title(title)
    plt.xlabel("Timestep")
    plt.ylabel("Timestep")
    plt.tight_layout()
    plt.show()