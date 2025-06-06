import torch
import torch.nn as nn
from configs.dqn_configs import *


class Attention(nn.Module):
    def __init__(self, d_model, num_heads=NUM_HEADS):
        super(Attention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        
        self.last_attention_probs = None

    def forward(self, x):
        residual = x  # Save input for skip connection
        x = self.norm(x)  # LayerNorm before attention
        
        attn_output, attn_weights = self.attn(x, x, x, need_weights=True)
        self.last_attention_probs = attn_weights.detach()
        
        return attn_output + residual

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_mult):
        super().__init__()
        self.attention = Attention(d_model, n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
        )

    def forward(self, x):
        x = self.attention(x) 
        x = x + self.feedforward(self.norm(x))  # Feedforward with residual
        return x
    
class DuelingMotionTransformer(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        seq_length=FRAME_STACK,
        d_model=EMBED_DIM,
        n_heads=NUM_HEADS,
        ff_mult=FF_MULT,
        num_layers=NUM_LAYERS
    ):
        super(DuelingMotionTransformer, self).__init__()
        self.training = True
        self.seq_length = seq_length
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_mult = ff_mult
        self.num_layers = num_layers

        self.embedding = nn.Linear(state_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(seq_length, d_model))
        self.input_dropout = nn.Dropout(p=0.05)

        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, ff_mult) for _ in range(num_layers)
        ])

        # Dueling head
        self.value_stream = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Tanh()  
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, action_dim) 
        )
        
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, state_dim)
        batch_size = x.shape[0]
        
        x = self.embedding(x)  # (batch_size, seq_length, d_model)
        if self.training:
            x = self.input_dropout(x)  # Apply dropout only during training
        pos_enc = self.positional_encoding.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, seq_length, d_model)
        x = x + pos_enc

        # Encoder stack
        for layer in self.encoder_layers:
            x = layer(x)  # Shape: (B, S, D)

        # Global Average Pooling over sequence length
        x = x.mean(dim=1)  # (B, D)
        
        # Dueling heads
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine to get Q-values (dueling formula with zero-mean advantage)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values
    
    def get_attention_weights(self, layer_idx=0):
        return self.encoder_layers[layer_idx].attention.last_attention_probs
