import torch
from encoder import Encoder

encoder = Encoder()

x = torch.randn(4, 196)

encoder(x)
