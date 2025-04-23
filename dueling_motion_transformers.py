import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from configs.dqn_configs import *
from configs.game_configs import NUM_RAYS


class LightweightAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(LightweightAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.query = nn.Linear(d_model, d_model // 2)
        self.key = nn.Linear(d_model, d_model // 2)
        self.value = nn.Linear(d_model, d_model // 2)
        
        self.out_proj = nn.Linear(d_model // 2, d_model)

        # Add LayerNorm inside attention block
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x  # Save input for skip connection
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        attn_output = torch.matmul(attn_probs, V)
        attn_output = self.out_proj(attn_output)
        
        # Add & Norm
        output = self.norm(attn_output + residual)
        
        return output


class DuelingMotionTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, seq_length=FRAME_STACK, d_model=EMBED_DIM, n_heads=NUM_HEADS):
        super(DuelingMotionTransformer, self).__init__()
        self.seq_length = seq_length
        self.d_model = d_model
        
        self.embedding = nn.Linear(state_dim, d_model)
        
        # Learnable positional embeddings
        self.positional_encoding = nn.Parameter(torch.randn(seq_length, d_model))
        
        self.attention_layer = LightweightAttention(d_model, n_heads)

        # Feedforward block + LayerNorm
        self.feedforward = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.LayerNorm(d_model)    
        )

        # Dueling head
        self.value_stream = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 1) 
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, action_dim) 
        )
        
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, state_dim)
        batch_size = x.shape[0]
        
        x = self.embedding(x)  # (batch_size, seq_length, d_model)
        
        # Add positional encoding
        pos_enc = self.positional_encoding.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, seq_length, d_model)
        x = x + pos_enc 
        
        x = self.attention_layer.forward(x)  # (batch_size, seq_length, d_model)
        x = self.feedforward(x)  # (batch_size, seq_length, d_model)
        
        # Global Average Pooling over sequence length
        x = x.mean(dim=1)  # (batch_size, d_model)

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values
    