import torch
import torch.nn as nn

class Encoder(nn.Module):
  def __init__(self):
    super().__init__()

    self.encode = nn.Linear(14 * 14, 64)
    self.norm = nn.LayerNorm(64)
    self.positional_encoding = nn.Embedding(16, 64)
    self.query = nn.Linear(64, 32)
    self.key = nn.Linear(64, 32)
    self.value = nn.Linear(64, 64)
    self.mlp = nn.Linear(64, 64)
    self.pool = nn.AdaptiveAvgPool1d(1)
    self.classifier = nn.Linear(64, 10)
    self.softmax = nn.Softmax(dim=-1)

  def forward(self, x, iterations=1):
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

    # Apply attention to encoded patches
    q = self.query(x)
    assert q.shape == (16, 32), f"query shape: {q.shape}"
    k = self.key(x)
    assert k.shape == (16, 32), f"key shape: {k.shape}"
    v = self.value(x)
    assert v.shape == (16, 64), f"value shape: {v.shape}"
    a = torch.matmul(q, k.transpose(0, 1)) / torch.math.sqrt(k.size(-1))
    assert a.shape == (16, 16), f"attention shape: {a.shape}"
    a = torch.softmax(a, dim=-1)
    assert a.shape == (16, 16), f"softmaxed attention shape: {a.shape}"
    x = torch.matmul(a, v)
    assert x.shape == (16, 64), f"output shape: {x.shape}"

    # Add MLP to the output
    x = self.mlp(x)
    assert x.shape == (16, 64), f"mlp output shape: {x.shape}"

    # Pool and classify the output
    x = x.mean(dim=0)
    assert x.shape == (64,), f"pooled output shape: {x.shape}"
    x = self.classifier(x)
    assert x.shape == (10,), f"classifier output shape: {x.shape}"
    x = self.softmax(x)
    return x

