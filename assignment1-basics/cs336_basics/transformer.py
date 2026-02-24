
import math

import torch
import torch.nn as nn

from jaxtyping import Float, Int, Bool
from einops import rearrange, einsum


class LN(nn.Module):
    def __init__(self, in_features: int, 
                    out_features: int, 
                    device: str = None, 
                    dtype: torch.dtype = None) -> None:
        super().__init__()
        # Initialize weight
        delta = 2.0 / math.sqrt(in_features + out_features)
        weight = torch.empty(out_features, in_features, device=device, dtype=dtype)
        nn.init.trunc_normal_(weight, mean=0.0, std=delta, a=-3.0*delta, b=3.0*delta)
        self.weight = nn.Parameter(weight)

    def forward(self, x: Float[torch.Tensor, "... in_features"]) -> Float[torch.Tensor, "... out_features"]:
        return x @ self.weight.T

class EMB(nn.Module):
    def __init__(self, num_embeddings: int, 
            embedding_dim: int, 
            device: str=None, 
            dtype: torch.dtype=None) -> None:
        super().__init__()
        embedding_weight = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        delta = 1
        nn.init.trunc_normal_(embedding_weight, mean=0.0, std=delta, a=-3.0, b=3.0)
        self.embedding_weight = nn.Parameter(embedding_weight)

    def forward(self, x: Int[torch.LongTensor, "... seq_length"],
                ) -> Float[torch.Tensor, "... seq_length embedding_dim"]:
        return self.embedding_weight[x]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int,
            eps: float = 1e-5,
            device: str=None,
            dtype: torch.dtype=None) -> None:
        super().__init__()
        self.eps = torch.tensor(eps, device=device, dtype=dtype)
        
        g = torch.ones(d_model, device=device, dtype=dtype)
        self.g = nn.Parameter(g)


    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        # Calculate in float32 in case of overflow
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms2 = (x*x).mean(dim=-1, keepdim=True)
        rms = torch.rsqrt(rms2 + self.eps)
        out = (x * rms) * self.g
        
        return out.to(in_dtype)


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int,
            d_ff: int,
            device: str=None,
            dtype: torch.dtype=None) -> None:
        super().__init__()
        self.w1 = LN(d_model, d_ff, device, dtype)
        self.w2 = LN(d_ff, d_model, device, dtype)
        self.w3 = LN(d_model, d_ff, device, dtype)

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        p1 = self.w1(x)
        p2 = self.w3(x)
        silu = p1 * torch.sigmoid(p1)
        out = self.w2 (silu * p2)
        return out


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: str=None) -> None:
        super().__init__()
        assert d_k % 2 == 0

        cosines = torch.empty(max_seq_len, d_k, device=device, dtype=torch.float32)
        sines = torch.empty(max_seq_len, d_k, device=device, dtype=torch.float32)

        pos = torch.arange(max_seq_len, device=device, dtype=torch.float32).unsqueeze(-1).repeat(1, d_k//2)
        k = torch.arange(d_k//2, device=device, dtype=torch.float32)
        inv_k = theta ** (-2*k/d_k)
        angles = pos * inv_k
        angles = torch.repeat_interleave(angles, repeats=2, dim=-1)

        sines = torch.sin(angles)
        cosines = torch.cos(angles)
        
        self.register_buffer("cosines", cosines, persistent=False)
        self.register_buffer("sines", sines, persistent=False)
        
    def forward(self, x: Float[torch.Tensor, "... seq_length d_k"], token_positions: Int[torch.LongTensor, "... seq_length"]) -> Float[torch.Tensor, "... seq_length d_k"]:
        cur_cosines = self.cosines[token_positions]
        cur_sines = self.sines[token_positions]

        assert x.shape[-1] % 2 == 0
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        r_x = torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

        out = x * cur_cosines + r_x * cur_sines
        return out


class SoftMax(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Float[torch.Tensor, "... seq_length d_k"], dim: int) -> Float[torch.Tensor, "... seq_length d_k"]:
        x = x - torch.max(x, dim=dim, keepdim=True).values
        exp_x = torch.exp(x)
        sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
        prob = exp_x / sum_exp_x
        return prob


class ScaledDotProductAttn(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, q: Float[torch.Tensor, "... s_q d_q"], 
                    k: Float[torch.Tensor, "... s_k d_k"], 
                    v: Float[torch.Tensor, "... s_v d_v"],
                    softmax: SoftMax,
                    mask: Bool[torch.Tensor, "... s_q s_k"] = None) -> Float[torch.Tensor, "... queries d_v"]:
        d_q = q.shape[-1]
        q_k = einsum(q, k, "... s_q d, ... s_k d -> ... s_q s_k") / math.sqrt(d_q)
        if mask is not None:
            q_k = q_k.masked_fill(mask == False, -float("inf"))
        prob = softmax(q_k, dim=-1)
        out = einsum(prob, v, "... s_q s_k, ... s_k d_v -> ... s_q d_v")
        return out


class MHA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device: str=None, dtype: torch.dtype=None) -> None:
        super().__init__()
        self.sdpa = ScaledDotProductAttn()
        self.num_heads = num_heads

        d_k = d_model // num_heads
        self.q_proj = LN(d_model, d_k*num_heads, device, dtype)
        self.k_proj = LN(d_model, d_k*num_heads, device, dtype)
        self.v_proj = LN(d_model, d_k*num_heads, device, dtype)
        self.o_proj = LN(d_model, d_k*num_heads, device, dtype)

    def forward(self, x: Float[torch.Tensor, "... s d_model"],
                softmax: SoftMax,
                rope: RoPE=None,
                token_positions: Int[torch.LongTensor, "... s"] = None) -> Float[torch.Tensor, "... h_out s_out d_out"]:
        seq_len = x.shape[-2]
        future_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        mask = ~future_mask
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = rearrange(q, "... s (h d_k) -> ... h s d_k", h=self.num_heads)
        k = rearrange(k, "... s (h d_k) -> ... h s d_k", h=self.num_heads)
        v = rearrange(v, "... s (h d_k) -> ... h s d_k", h=self.num_heads)
        if rope is not None:
            q = rope(q, token_positions)
            k = rope(k, token_positions)
        out = rearrange(self.sdpa(q, k, v, softmax, mask=mask), "... h s d_k -> ... s (h d_k)")
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, 
                eps: float=1e-5,
                device: str=None, dtype: torch.dtype=None) -> None:
        super().__init__()
        # sublayer 1
        self.ln1 = RMSNorm(d_model, eps, device, dtype)
        self.mha = MHA(d_model, num_heads, device, dtype)
        # sublayer 2
        self.ln2 = RMSNorm(d_model, eps, device, dtype)
        self.ffn = SwiGLUFFN(d_model, d_ff, device, dtype)

    def forward(self, x: Float[torch.Tensor, "... s d_model"],
                softmax: SoftMax,
                rope: RoPE=None,
                token_positions: Int[torch.LongTensor, "... s"] = None) -> Float[torch.Tensor, "... s d_model"]:
        # sublayer 1
        x = self.mha(self.ln1(x), softmax, rope, token_positions) + x
        # sublayer 2
        x = self.ffn(self.ln2(x)) + x
        return x

class TransformerLLM(nn.Module):

    def __init__(self, vocab_size: int, 
                        d_model: int, 
                        num_heads: int, 
                        num_layer: int,
                        d_ff: int,
                        theta: float=10000,
                        max_seq_len: int=2048,
                        eps: float=1e-5,
                        device: str=None, 
                        dtype: torch.dtype=None) -> None:
        super().__init__()
        d_k = d_model // num_heads
        self.softmax = SoftMax()

        self.emb = EMB(vocab_size, d_model, device, dtype)
        self.rope = RoPE(theta, d_k, max_seq_len, device)
        self.transformer_block = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, eps, device, dtype) for _ in range(num_layer)])
        self.ln3 = RMSNorm(d_model, eps, device, dtype)
        self.proj = LN(d_model, vocab_size, device, dtype)


    def forward(self, x: Int[torch.LongTensor, "... s"]) -> Float[torch.Tensor, "... s vocab_size"]:
        x_emb = self.emb(x)
        token_positions = torch.arange(x.shape[-1], device=x.device)
        for block in self.transformer_block:
            x_emb = block(x_emb, self.softmax, self.rope, token_positions)
        x_emb = self.ln3(x_emb)
        logits = self.proj(x_emb)
        return logits
