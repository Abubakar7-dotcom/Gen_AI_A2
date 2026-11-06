"""E1: reference scaled dot-product attention (materialises the full score matrix).

    softmax( Q K^T / sqrt(d) + mask ) V

Shapes: q [B, H, Nq, d]; k, v [B, Hkv, Nk, d] with H % Hkv == 0 (grouped-query attention, as in
Qwen2.5: each KV head is shared by H/Hkv query heads).

Causal masking with an offset: the Nq queries are the LAST Nq positions of the Nk-long sequence
(offset = Nk - Nq). Query i (absolute position offset+i) may attend key j iff j <= offset+i.
Nq == Nk is ordinary causal self-attention; Nq == 1 is a decode step against a KV cache.
(torch SDPA's is_causal=True aligns the mask to the top-left instead, which is only the same
thing when Nq == Nk.)

Numerical stability: scores are computed in fp32 and the row maximum is subtracted before exp(),
so exp() never overflows (exp(x - max) <= 1) and the result equals the exact softmax.
"""
from __future__ import annotations

import math

import torch


def expand_kv(k: torch.Tensor, v: torch.Tensor, n_heads: int):
    hkv = k.shape[1]
    if hkv == n_heads:
        return k, v
    assert n_heads % hkv == 0, "query heads must be a multiple of kv heads"
    g = n_heads // hkv
    return k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1)


def causal_keep(nq: int, nk: int, device) -> torch.Tensor:
    """Boolean [Nq, Nk]; True where query may attend key (bottom-right aligned causal mask)."""
    off = nk - nq
    return torch.arange(nk, device=device)[None, :] <= (torch.arange(nq, device=device)[:, None] + off)


def reference_attention(q, k, v, causal: bool = True, scale: float | None = None,
                        compute_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    B, H, Nq, d = q.shape
    Nk = k.shape[2]
    assert Nk >= Nq or not causal, "causal attention needs Nk >= Nq"
    k, v = expand_kv(k, v, H)
    scale = scale if scale is not None else 1.0 / math.sqrt(d)
    s = torch.matmul(q.to(compute_dtype), k.to(compute_dtype).transpose(-1, -2)) * scale   # [B,H,Nq,Nk]
    if causal:
        s = s.masked_fill(~causal_keep(Nq, Nk, q.device), float("-inf"))
    m = s.amax(dim=-1, keepdim=True)                     # row max (finite: key 0 is always visible)
    p = torch.exp(s - m)                                 # in (0, 1], masked entries -> 0
    out = torch.matmul(p, v.to(compute_dtype)) / p.sum(dim=-1, keepdim=True)
    return out.to(q.dtype)


def sdpa_attention(q, k, v, causal: bool = True, scale: float | None = None) -> torch.Tensor:
    """PyTorch's fused scaled_dot_product_attention with the same (bottom-right) causal convention."""
    B, H, Nq, d = q.shape
    Nk = k.shape[2]
    k, v = expand_kv(k, v, H)
    if not causal:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)
    if Nq == Nk:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=causal_keep(Nq, Nk, q.device), scale=scale)
