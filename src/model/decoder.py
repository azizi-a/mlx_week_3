import torch
import torch.nn as nn
from .attention import Attention
from config import EMBEDDING_DIM, DECODER_ATTENTION_BLOCKS

class Decoder(nn.Module):
  def __init__(self):
    super().__init__()

    self.label_embedding = nn.Embedding(12, EMBEDDING_DIM)
    self.norm_1 = nn.LayerNorm(EMBEDDING_DIM)
    self.self_attention = Attention()
    self.cross_attention_blocks = nn.ModuleList([
      Attention() for _ in range(DECODER_ATTENTION_BLOCKS)
    ])
    self.norm_2 = nn.LayerNorm(EMBEDDING_DIM)
    self.output_layer = nn.Linear(EMBEDDING_DIM, 12)
    self.softmax = nn.Softmax(dim=2)

  def forward(self, x_labels, y_patches):
    x_labels = self.label_embedding(x_labels)
    x_labels = self.norm_1(x_labels)
    mask = torch.triu(torch.ones(x_labels.size(1), x_labels.size(1)), diagonal=1)
    x_labels, _ = self.self_attention(x_labels, x_labels, mask)
    for cross_attention in self.cross_attention_blocks:
      x_labels, _ = cross_attention(x_labels, y_patches)
    x_labels = self.norm_2(x_labels)
    x_labels = self.output_layer(x_labels)
    x_labels = self.softmax(x_labels)
    return x_labels