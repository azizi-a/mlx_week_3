import torch
import torch.nn as nn

class Attention(nn.Module):
  def __init__(self):
    super().__init__()

    self.query = nn.Linear(64, 32)
    self.key = nn.Linear(64, 32)
    self.value = nn.Linear(64, 64)
    self.linear_in = nn.Linear(64, 32)
    self.relu = nn.ReLU()
    self.linear_out = nn.Linear(32, 64)

  def forward(self, x):
    # Attention mechanism
    q = self.query(x)
    k = self.key(x)
    v = self.value(x)
    a = torch.matmul(q, k.transpose(0, 1)) / torch.math.sqrt(k.size(-1))
    a = torch.softmax(a, dim=-1)
    x = torch.matmul(a, v)

    # MLP
    x = self.linear_in(x)
    x = self.relu(x)
    x = self.linear_out(x)
    return x