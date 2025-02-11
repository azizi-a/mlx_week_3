import torch
import torch.nn as nn

class Encoder(nn.Module):
  def __init__(self):
    super().__init__()

    self.encode = nn.Linear(14 * 14, 64)
    self.positional_encoding = nn.Embedding(4, 64)
    self.query = nn.Linear(64, 32)
    self.key = nn.Linear(64, 32)
    self.value = nn.Linear(64, 64)

  def forward(self, x):
    assert x.shape == (4, 196), f"x.shape: {x.shape}"
    x = self.encode(x)
    assert x.shape == (4, 64), f"encoded x.shape: {x.shape}"
    position = torch.arange(x.size(0))
    print('position', position)
    pe = self.positional_encoding(position)
    print('pe shape', pe.shape)
    x = x + pe
    assert x.shape == (4, 64), f"positional encoded x.shape: {x.shape}"
    q = self.query(x)
    assert q.shape == (4, 32), f"query shape: {q.shape}"
    k = self.key(x)
    assert k.shape == (4, 32), f"key shape: {k.shape}"
    v = self.value(x)
    assert v.shape == (4, 64), f"value shape: {v.shape}"
    a = torch.matmul(q, k.transpose(0, 1)) / torch.math.sqrt(k.size(-1))
    assert a.shape == (4, 4), f"attention shape: {a.shape}"
    a = torch.softmax(a, dim=-1)
    assert a.shape == (4, 4), f"softmaxed attention shape: {a.shape}"
    x = torch.matmul(a, v)
    assert x.shape == (4, 64), f"output shape: {x.shape}"

    return x

