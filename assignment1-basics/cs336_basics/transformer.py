

import torch.nn as nn
from einops import rearrange, einsum


class TransformerLLM(nn.Module):

    def __init__(self, vocab_size: int, emb_dim: int, hiden_dim: int, num_heads: int, num_layer: int):
        super().__init__()
        pass

    def _rms_norm_layer(self, x):
        pass

    def _feed_forward_layer(self, x):
        pass

    def attn_layer(self, x):
        pass

    def init_model(self):
        pass

    def forward(self, x):
        pass

    def generate(self, x, max_len: int):
        pass

