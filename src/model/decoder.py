import torch
import torch.nn as nn
from .attention import Attention
from config import EMBEDDING_DIM, DECODER_ATTENTION_BLOCKS

class Decoder(nn.Module):
  def __init__(self):
    super().__init__()

    self.label_embedding = nn.Embedding(5, EMBEDDING_DIM)
    self.self_attention = nn.Attention()
    self.cross_attention_blocks = nn.ModuleList([
      nn.Attention() for _ in range(DECODER_ATTENTION_BLOCKS)
    ])

  def forward(self, labels, patches):
    labels = self.label_embedding(labels)
    mask = torch.triu(torch.ones(labels.size(1), labels.size(1)), diagonal=1)
    print('mask', mask)
    labels = self.self_attention(labels, labels, mask)
    for cross_attention in self.cross_attention_blocks:
      labels = cross_attention(labels, patches)
    return labels