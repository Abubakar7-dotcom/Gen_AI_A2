"""E2: tiled attention with online softmax (the FlashAttention forward algorithm, in plain PyTorch).

The Nq x Nk score matrix is never materialised. Queries are processed in blocks of `block_q`
rows; for each query block we stream over key/value blocks of `block_k` columns and keep, per
query row, three running quantities:

    m    running maximum of the scores seen so far
    l    running softmax normaliser  sum_j exp(s_j - m)
    acc  running un-normalised output sum_j exp(s_j - m) v_j

When a new key block arrives with scores S (block_q x block_k):

    m_new = max(m, rowmax(S))
    alpha = exp(m - m_new)                 # rescales everything accumulated under the old maximum
    P     = exp(S - m_new)
    l     = alpha * l   + rowsum(P)
    acc   = alpha * acc + P @ V_block
    m     = m_new

and after the last block  out = acc / l.  This is exactly softmax(S) V because every term was
accumulated relative to some maximum and then rescaled to the final one (Milakov & Gimelshein 2018;
Dao et al. 2022). Memory for intermediates is O(block_q * block_k) per head instead of O(Nq * Nk).

Causal masking (bottom-right aligned, offset = Nk - Nq; see reference.py):
  * key blocks lying entirely in the future of every query in the block are skipped (never computed);
  * only blocks that straddle the diagonal get an explicit mask;
  * the first key block always contains key 0, which every query may see, so m is finite after it
    and later fully-masked rows contribute exp(-inf) = 0 without producing NaNs.
"""
from __future__ import annotations

import math

import torch

from src.attention.reference import expand_kv


def tiled_attention(q, k, v, causal: bool = True, scale: float | None = None,
                    block_q: int = 64, block_k: int = 64, return_stats: bool = False):
    B, H, Nq, d = q.shape
    Nk = k.shape[2]
    assert Nk >= Nq or not causal, "causal attention needs Nk >= Nq"
    k, v = expand_kv(k, v, H)
    scale = scale if scale is not None else 1.0 / math.sqrt(d)
    off = Nk - Nq
    out = torch.empty(B, H, Nq, d, dtype=torch.float32, device=q.device)
    blocks_computed = blocks_skipped = 0
    for qs in range(0, Nq, block_q):
        qe = min(qs + block_q, Nq)
        qb = q[:, :, qs:qe].float() * scale                                  # [B,H,bq,d]
        m = torch.full((B, H, qe - qs, 1), float("-inf"), device=q.device)
        l = torch.zeros((B, H, qe - qs, 1), device=q.device)
        acc = torch.zeros((B, H, qe - qs, d), device=q.device)
        # last key visible to the last query of this block is (qe-1)+off -> stop there
        k_stop = min(Nk, qe + off) if causal else Nk
        if causal:
            blocks_skipped += math.ceil(Nk / block_k) - math.ceil(k_stop / block_k)
        for ks in range(0, k_stop, block_k):
            ke = min(ks + block_k, k_stop)
            s = torch.matmul(qb, k[:, :, ks:ke].float().transpose(-1, -2))  # [B,H,bq,bk]
            if causal and ke - 1 > qs + off:                                 # block straddles the diagonal
                keep = (torch.arange(ks, ke, device=q.device)[None, :]
                        <= torch.arange(qs, qe, device=q.device)[:, None] + off)
                s = s.masked_fill(~keep, float("-inf"))
            m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
            alpha = torch.exp(m - m_new)
            p = torch.exp(s - m_new)
            l = alpha * l + p.sum(dim=-1, keepdim=True)
            acc = alpha * acc + torch.matmul(p, v[:, :, ks:ke].float())
            m = m_new
            blocks_computed += 1
        out[:, :, qs:qe] = acc / l
    out = out.to(q.dtype)
    if return_stats:
        return out, {"blocks_computed": blocks_computed, "blocks_skipped": blocks_skipped}
    return out
