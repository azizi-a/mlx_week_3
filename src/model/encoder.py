import torch
import torch.nn as nn
from .attention import Attention
from config import EMBEDDING_DIM, PATCH_SIZE, NUM_PATCHES, ENCODER_ATTENTION_BLOCKS

class Encoder(nn.Module):
  def __init__(self ):
    super().__init__()

    self.encode = nn.Linear(PATCH_SIZE ** 2, EMBEDDING_DIM)
    self.positional_encoding = nn.Embedding(NUM_PATCHES, EMBEDDING_DIM)

    self.attention_blocks = nn.ModuleList([
      Attention() for _ in range(ENCODER_ATTENTION_BLOCKS)
    ])

    self.attention = Attention()
    self.norm = nn.LayerNorm(EMBEDDING_DIM)
    self.pool = nn.AdaptiveAvgPool1d(1)
    self.classifier = nn.Linear(EMBEDDING_DIM, 10)
    self.softmax = nn.Softmax(dim=-1)

  def forward(self, x):
    # Take 16 image patches of 14x14 pixels each and encode them
    batch_size = x.shape[0]
    assert x.shape == (batch_size, NUM_PATCHES, PATCH_SIZE ** 2), f"x.shape: {x.shape}"
    x = self.encode(x)
    assert x.shape == (batch_size, NUM_PATCHES, EMBEDDING_DIM), f"encoded x.shape: {x.shape}"
    
    # Create positional encodings for each position in sequence
    position = torch.arange(x.size(1))
    pe = self.positional_encoding(position.expand(batch_size, -1))

    # Add positional encodings to each item in batch
    x = x + pe
    assert x.shape == (batch_size, NUM_PATCHES, EMBEDDING_DIM), f"positional encoded x.shape: {x.shape}"

    # Apply multiple combined attention and MLP to encoded patches
    for attention in self.attention_blocks:
      x, _ = attention(x, x)
    assert x.shape == (batch_size, NUM_PATCHES, EMBEDDING_DIM), f"attention output shape: {x.shape}"

    # Pool and classify the output
    x = self.norm(x)
    x = x.mean(dim=1)
    assert x.shape == (batch_size, EMBEDDING_DIM), f"pooled output shape: {x.shape}"
    x = self.classifier(x)
    assert x.shape == (batch_size, 10), f"classifier output shape: {x.shape}"
    x = self.softmax(x)
    return x
