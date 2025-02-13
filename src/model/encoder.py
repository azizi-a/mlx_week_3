import torch
import torch.nn as nn
from .attention import Attention
from config import BATCH_SIZE

class Encoder(nn.Module):
  def __init__(self, num_attention_blocks):
    super().__init__()

    self.encode = nn.Linear(14 * 14, 64)
    self.norm = nn.LayerNorm(64)
    self.positional_encoding = nn.Embedding(BATCH_SIZE, 16, 64)

    self.attention_blocks = nn.ModuleList([
      Attention() for _ in range(num_attention_blocks)
    ])

    self.attention = Attention()
    self.pool = nn.AdaptiveAvgPool1d(1)
    self.classifier = nn.Linear(64, 10)
    self.softmax = nn.Softmax(dim=-1)

  def forward(self, x):
    # Take 16 image patches of 14x14 pixels each and encode them
    assert x.shape[1:3] == (16, 196), f"x.shape: {x.shape[1:3]}"
    x = self.encode(x)
    assert x.shape[1:3] == (16, 64), f"encoded x.shape: {x.shape[1:3]}"
    # position = torch.arange(x.size(0))
    # print('position.shape', position.shape)
    # pe = self.positional_encoding(position)
    # print('pe.shape', pe.shape)
    # x = x + pe
    # assert x.shape == (128, 16, 64), f"positional encoded x.shape: {x.shape}"
    x = self.norm(x)
    assert x.shape[1:3] == (16, 64), f"normalised x.shape: {x.shape}"

    # Apply multiple combined attention and MLP to encoded patches
    for attention in self.attention_blocks:
      x = attention(x)
    assert x.shape[1:3] == (16, 64), f"attention output shape: {x.shape[1:3]}"

    # Pool and classify the output
    x = x.mean(dim=1)
    assert x.shape[1:2] == (64,), f"pooled output shape: {x.shape[1:2]}"
    x = self.classifier(x)
    assert x.shape[1:2] == (10,), f"classifier output shape: {x.shape[1:2]}"
    x = self.softmax(x)
    return x
