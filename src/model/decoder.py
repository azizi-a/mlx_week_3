import torch
import torch.nn as nn
from .attention import Attention

class Decoder(nn.Module):
  def __init__(self, num_cross_attention_blocks):
    super().__init__()
    
    self.self_attention = nn.Attention()
    self.cross_attention_blocks = nn.ModuleList([
      nn.Attention() for _ in range(num_cross_attention_blocks)
    ])

  def forward(self, x, y):
    x = self.self_attention(x, x)
    for cross_attention in self.cross_attention_blocks:
      x = cross_attention(x, y)
    return x