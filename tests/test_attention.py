"""E1/E2/E3: tiled online-softmax attention == reference == torch SDPA (CPU, small controlled shapes)."""
import math

import pytest
import torch

from src.attention.reference import causal_keep, reference_attention, sdpa_attention
from src.attention.tiled import tiled_attention

# (B, H, Hkv, Nq, Nk, d, causal, block)
SHAPES = [
    (1, 2, 2, 16, 16, 8, True, 4),        # tiny, divisible
    (2, 4, 4, 100, 100, 64, True, 32),    # length not divisible by block
    (1, 4, 4, 1, 77, 64, True, 16),       # decode step (Nq=1) against 77 cached keys
    (1, 4, 4, 37, 130, 32, True, 16),     # chunk of queries at the end of a longer sequence
    (2, 2, 2, 50, 70, 16, False, 16),     # non-causal, Nq != Nk
    (1, 12, 2, 64, 64, 128, True, 32),    # grouped-query attention, Qwen2.5-1.5B head layout
    (1, 1, 1, 5, 5, 4, True, 64),         # block larger than sequence
]


def _qkv(B, H, Hkv, Nq, Nk, d, dtype=torch.float32, seed=0, mag=1.0):
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(B, H, Nq, d, generator=g) * mag).to(dtype)
    k = (torch.randn(B, Hkv, Nk, d, generator=g) * mag).to(dtype)
    v = torch.randn(B, Hkv, Nk, d, generator=g).to(dtype)
    return q, k, v


def _naive(q, k, v, causal):
    """Textbook formula with no max-subtraction, for the stability test."""
    s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if causal:
        s = s.masked_fill(~causal_keep(q.shape[2], k.shape[2], q.device), float("-inf"))
    e = torch.exp(s)
    return (e / e.sum(-1, keepdim=True)) @ v


@pytest.mark.parametrize("B,H,Hkv,Nq,Nk,d,causal,block", SHAPES)
def test_tiled_matches_reference_fp32(B, H, Hkv, Nq, Nk, d, causal, block):
    q, k, v = _qkv(B, H, Hkv, Nq, Nk, d)
    ref = reference_attention(q, k, v, causal)
    til = tiled_attention(q, k, v, causal, block_q=block, block_k=block)
    sdp = sdpa_attention(q, k, v, causal)
    assert (til - ref).abs().max() < 1e-5
    assert (sdp - ref).abs().max() < 1e-5


@pytest.mark.parametrize("B,H,Hkv,Nq,Nk,d,causal,block", SHAPES[:4])
def test_tiled_matches_float64_truth_in_fp16(B, H, Hkv, Nq, Nk, d, causal, block):
    q, k, v = _qkv(B, H, Hkv, Nq, Nk, d, dtype=torch.float16)
    truth = reference_attention(q.double(), k.double(), v.double(), causal, compute_dtype=torch.float64)
    til = tiled_attention(q, k, v, causal, block_q=block, block_k=block).double()
    assert (til - truth).abs().max() < 2e-3   # fp16 output rounding (~1e-3 relative)


def test_block_size_does_not_change_result():
    q, k, v = _qkv(1, 2, 2, 90, 90, 32)
    outs = [tiled_attention(q, k, v, True, block_q=b, block_k=c) for b, c in [(8, 8), (16, 64), (90, 7), (128, 128)]]
    for o in outs[1:]:
        assert (o - outs[0]).abs().max() < 1e-5


def test_large_logits_stable_where_naive_overflows():
    q, k, v = _qkv(1, 2, 2, 32, 32, 16, mag=12.0)   # scores of order 10^2..10^3 -> exp overflows in fp32
    naive = _naive(q, k, v, True)
    assert not torch.isfinite(naive).all()
    truth = reference_attention(q.double(), k.double(), v.double(), True, compute_dtype=torch.float64)
    ref = reference_attention(q, k, v, True)
    til = tiled_attention(q, k, v, True, block_q=8, block_k=8)
    assert torch.isfinite(til).all()
    # With |scores| ~ 1e2-1e3 the fp32 rounding of the exponent alone is ~|s|*6e-8, so compare both
    # fp32 implementations to a float64 ground truth instead of to each other with a fixed 1e-5.
    err_ref = (ref.double() - truth).abs().max().item()
    err_til = (til.double() - truth).abs().max().item()
    assert err_ref < 1e-3 and err_til < 1e-3
    assert err_til <= 4 * err_ref + 1e-6


def test_causal_blocks_are_skipped():
    q, k, v = _qkv(1, 1, 1, 128, 128, 16)
    _, st = tiled_attention(q, k, v, True, block_q=32, block_k=32, return_stats=True)
    assert st == {"blocks_computed": 10, "blocks_skipped": 6}   # lower triangle incl. diagonal: 4+3+2+1
