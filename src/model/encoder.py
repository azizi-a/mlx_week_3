import torch
import torch.nn as nn
from .attention import Attention

class Encoder(nn.Module):
  def __init__(self):
    super().__init__()

    self.encode = nn.Linear(14 * 14, 64)
    self.norm = nn.LayerNorm(64)
    self.positional_encoding = nn.Embedding(16, 64)
    self.attention = Attention()
    self.pool = nn.AdaptiveAvgPool1d(1)
    self.classifier = nn.Linear(64, 10)
    self.softmax = nn.Softmax(dim=-1)

  def forward(self, x):
    # Take 16 image patches of 14x14 pixels each and encode them
    assert x.shape == (16, 196), f"x.shape: {x.shape}"
    x = self.encode(x)
    assert x.shape == (16, 64), f"encoded x.shape: {x.shape}"
    position = torch.arange(x.size(0))
    pe = self.positional_encoding(position)
    x = x + pe
    assert x.shape == (16, 64), f"positional encoded x.shape: {x.shape}"
    x = self.norm(x)
    assert x.shape == (16, 64), f"normalised x.shape: {x.shape}"

    # Apply combined attention and MLP to encoded patches
    x = self.attention(x)
    assert x.shape == (16, 64), f"attention output shape: {x.shape}"

    # Pool and classify the output
    x = x.mean(dim=0)
    assert x.shape == (64,), f"pooled output shape: {x.shape}"
    x = self.classifier(x)
    assert x.shape == (10,), f"classifier output shape: {x.shape}"
    x = self.softmax(x)
    return x
