"""Inference Lab back-end: live decoder and attention comparisons on the loaded models."""
from __future__ import annotations

import time

import pandas as pd
import torch

from src.attention.reference import reference_attention, sdpa_attention
from src.attention.tiled import tiled_attention
from src.data import load_split
from src.decoding.inputs import make_input_ids, record_input_ids
from src.decoding.kv_cache import eos_set, first_mismatch, greedy_kv, greedy_no_cache, sync
from src.decoding.speculative import speculative_greedy

N_TASK = 6
SYNTH = "synthetic text (length from slider)"


def lab_prompt_choices() -> list[str]:
    recs = load_split("val")[:N_TASK]
    return [f"val #{i}: {r['criterion_type']} — {r['criterion_text'][:60]}" for i, r in enumerate(recs)] + [SYNTH]


def _mb(x):
    return None if x is None else round(x / 2**20, 1)


def generation_lab(screener, src, n_in, n_out, ignore_eos, k, methods):
    model, tok, dev = screener.model, screener.tok, screener.device
    if src == SYNTH:
        ids = make_input_ids(tok, int(n_in), dev)
    else:
        ids = record_input_ids(tok, load_split("val")[int(src.split("#")[1].split(":")[0])], dev)
    eos = None if ignore_eos else eos_set(tok)
    n_out = int(n_out)
    runs = {}
    greedy_kv(model, ids[:, :16], 2)   # warm-up (kernel compilation / allocator)
    if "no cache" in methods:
        runs["no cache"] = greedy_no_cache(model, ids, n_out, eos)
    if "KV cache" in methods or not runs:
        runs["KV cache"] = greedy_kv(model, ids, n_out, eos)
    if "speculative" in methods and screener.draft is not None:
        runs["speculative"] = speculative_greedy(model, screener.draft, ids, n_out, k=int(k), eos_ids=eos)
    ref_name = "KV cache" if "KV cache" in runs else next(iter(runs))
    ref = runs[ref_name]
    rows = []
    for name, r in runs.items():
        mm = first_mismatch(ref.tokens, r.tokens)
        rows.append({
            "method": name, "prompt tokens": ids.shape[1], "new tokens": len(r.tokens),
            "prefill ms": round(1e3 * r.prefill_s, 1), "decode ms": round(1e3 * r.decode_s, 1),
            "total ms": round(1e3 * r.total_s, 1), "tokens/s": round(len(r.tokens) / r.total_s, 1),
            "peak memory MiB": _mb(r.peak_mem),
            "KV cache MiB": _mb(getattr(r, "cache_bytes", None)),
            "acceptance rate": round(r.acceptance_rate, 3) if hasattr(r, "acceptance_rate") else None,
            "tokens / target pass": round(r.tokens_per_round, 2) if hasattr(r, "tokens_per_round") else None,
            f"identical to {ref_name}": "✅" if mm is None else f"❌ (first diff at step {mm})",
        })
    note = "" if dev.type == "cuda" else " (MPS/CPU: peak memory is sampled after each forward pass, a lower bound)"
    return pd.DataFrame(rows), tok.decode(ref.tokens, skip_special_tokens=False) + note


def _time(fn, dev, reps=3):
    fn(); sync(dev)
    ts = []
    for _ in range(reps):
        sync(dev); t0 = time.perf_counter(); out = fn(); sync(dev); ts.append(time.perf_counter() - t0)
    return out, sorted(ts)[len(ts) // 2]


def _truth(q, k, v, causal, chunk=512):
    """float64 ground truth on CPU, computed over query chunks so N=4096 does not need a 1.6 GB score matrix.
    Chunk [s, e) attends keys [0, e) (causal, bottom-right aligned) or all keys (non-causal)."""
    q, k, v = q.double(), k.double(), v.double()
    outs = []
    for s in range(0, q.shape[2], chunk):
        e = min(s + chunk, q.shape[2])
        kk, vv = (k[:, :, :e], v[:, :, :e]) if causal else (k, v)
        outs.append(reference_attention(q[:, :, s:e], kk, vv, causal, compute_dtype=torch.float64))
    return torch.cat(outs, dim=2)


def attention_lab(dev, n, d, block, causal, dtype):
    n, d, block = int(n), int(d), int(block)
    dt = torch.float16 if dtype == "fp16" else torch.float32
    if dev.type == "cpu" and dt == torch.float16:
        dt = torch.float32
    B, H, Hkv = 1, 12, 2
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(B, h, n, d, generator=g) for h in (H, Hkv, Hkv))
    truth = _truth(q, k, v, causal)
    qd, kd, vd = (t.to(dt).to(dev) for t in (q, k, v))
    impls = [("reference (full N×N scores)", lambda: reference_attention(qd, kd, vd, causal), B * H * n * n * 4),
             (f"tiled online softmax (tile {block})", lambda: tiled_attention(qd, kd, vd, causal, block_q=block, block_k=block),
              B * H * block * block * 4),
             ("PyTorch SDPA (fused)", lambda: sdpa_attention(qd, kd, vd, causal), None)]
    rows = []
    for name, fn, inter in impls:
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev); base = torch.cuda.memory_allocated(dev)
        out, t = _time(fn, dev)
        err = (out.cpu().double() - truth).abs()
        rows.append({"implementation": name, "N": n, "dtype": str(dt).split(".")[-1], "time ms": round(1e3 * t, 2),
                     "max abs error": f"{err.max().item():.2e}", "mean abs error": f"{err.mean().item():.2e}",
                     "largest intermediate MiB (analytic)": None if inter is None else round(inter / 2**20, 2),
                     "measured peak extra MiB": round((torch.cuda.max_memory_allocated(dev) - base) / 2**20, 1)
                     if dev.type == "cuda" else None})
    if causal:
        _, st = tiled_attention(qd, kd, vd, True, block_q=block, block_k=block, return_stats=True)
        rows[1]["implementation"] += f" — {st['blocks_computed']} tiles computed, {st['blocks_skipped']} skipped (causal)"
    return pd.DataFrame(rows)
