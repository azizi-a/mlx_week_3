import torch
import torch.nn as nn
from .attention import Attention
from config import BATCH_SIZE

class Encoder(nn.Module):
  def __init__(self, num_attention_blocks):
    super().__init__()

    self.encode = nn.Linear(14 * 14, 64)
    self.positional_encoding = nn.Embedding(16, 64)

    self.attention_blocks = nn.ModuleList([
      Attention() for _ in range(num_attention_blocks)
    ])

    self.attention = Attention()
    self.norm = nn.LayerNorm(64)
    self.pool = nn.AdaptiveAvgPool1d(1)
    self.classifier = nn.Linear(64, 10)
    self.softmax = nn.Softmax(dim=-1)

  def forward(self, x):
    # Take 16 image patches of 14x14 pixels each and encode them
    batch_size = x.shape[0]
    assert x.shape == (batch_size, 16, 196), f"x.shape: {x.shape}"
    x = self.encode(x)
    assert x.shape == (batch_size, 16, 64), f"encoded x.shape: {x.shape}"
    
    # Create positional encodings for each position in sequence
    position = torch.arange(x.size(1))
    pe = self.positional_encoding(position.expand(batch_size, -1))

    # Add positional encodings to each item in batch
    x = x + pe
    assert x.shape == (batch_size, 16, 64), f"positional encoded x.shape: {x.shape}"

    # Apply multiple combined attention and MLP to encoded patches
    for attention in self.attention_blocks:
      x, _ = attention(x, x)
    assert x.shape == (batch_size, 16, 64), f"attention output shape: {x.shape}"

    # Pool and classify the output
    x = self.norm(x)
    x = x.mean(dim=1)
    assert x.shape == (batch_size, 64), f"pooled output shape: {x.shape}"
    x = self.classifier(x)
    assert x.shape == (batch_size, 10), f"classifier output shape: {x.shape}"
    x = self.softmax(x)
    return x
