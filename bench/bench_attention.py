"""E3: numerical and performance validation of reference vs tiled online-softmax vs torch SDPA.

    python -m bench.bench_attention --tag mac_m5
    python -m bench.bench_attention --tag t4 --device cuda

Part A (numerics, CPU): several shapes, causal and non-causal, fp32 and fp16 inputs; max / mean
absolute error of each implementation against a float64 ground truth.
Part B (speed/memory, chosen device): Qwen2.5-1.5B head layout (12 query heads, 2 KV heads,
head_dim 128), causal, sequence length N; median of synchronised repeats after warm-up.
Memory: measured allocator peak above the inputs on CUDA; on MPS/CPU there is no peak counter, so
the analytic size of the largest intermediate (score matrix vs one score tile) is reported for all
devices.
Part C: tile-size sweep for the tiled kernel at fixed N.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics as st
import time
from pathlib import Path

import torch

from src.attention.reference import reference_attention, sdpa_attention
from src.attention.tiled import tiled_attention
from src.decoding.kv_cache import sync
from src.modeling import pick_device

ROOT = Path(__file__).resolve().parents[1]

NUM_SHAPES = [  # (B, H, Hkv, Nq, Nk, d)
    (1, 4, 4, 64, 64, 32), (2, 4, 4, 100, 100, 64), (1, 4, 4, 1, 300, 64),
    (1, 4, 4, 37, 130, 32), (1, 12, 2, 256, 256, 128), (1, 12, 2, 1, 512, 128),
]


def qkv(B, H, Hkv, Nq, Nk, d, dtype, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    mk = lambda h, n: torch.randn(B, h, n, d, generator=g).to(dtype).to(device)
    return mk(H, Nq), mk(Hkv, Nk), mk(Hkv, Nk)


def part_a(block: int):
    rows = []
    for shape in NUM_SHAPES:
        for causal in (True, False):
            for dt in (torch.float32, torch.float16):
                q, k, v = qkv(*shape, dt, "cpu")
                truth = reference_attention(q.double(), k.double(), v.double(), causal, compute_dtype=torch.float64)
                for name, fn in (("reference", lambda: reference_attention(q, k, v, causal)),
                                 ("tiled", lambda: tiled_attention(q, k, v, causal, block_q=block, block_k=block)),
                                 ("torch_sdpa", lambda: sdpa_attention(q.float(), k.float(), v.float(), causal).to(dt))):
                    err = (fn().double() - truth).abs()
                    rows.append({"B,H,Hkv,Nq,Nk,d": "x".join(map(str, shape)), "causal": causal, "dtype": str(dt).split(".")[-1],
                                 "impl": name, "max_abs_err": err.max().item(), "mean_abs_err": err.mean().item()})
    return rows


def time_fn(fn, dev, warmup, repeats):
    for _ in range(warmup):
        fn()
    sync(dev)
    ts = []
    for _ in range(repeats):
        sync(dev); t0 = time.perf_counter(); fn(); sync(dev); ts.append(time.perf_counter() - t0)
    return st.median(ts)


def peak_extra_mb(fn, dev):
    if dev.type != "cuda":
        return None
    sync(dev); torch.cuda.reset_peak_memory_stats(dev); base = torch.cuda.memory_allocated(dev)
    fn(); sync(dev)
    return (torch.cuda.max_memory_allocated(dev) - base) / 2**20


def part_b(dev, dtype, ns, blocks, warmup, repeats):
    B, H, Hkv, d = 1, 12, 2, 128
    rows = []
    for n in ns:
        q, k, v = qkv(B, H, Hkv, n, n, d, dtype, dev)
        impls = [("reference", lambda: reference_attention(q, k, v, True), B * H * n * n * 4)]
        impls += [(f"tiled_b{b}", (lambda b=b: tiled_attention(q, k, v, True, block_q=b, block_k=b)), B * H * b * b * 4) for b in blocks]
        impls += [("torch_sdpa", lambda: sdpa_attention(q, k, v, True), None)]
        for name, fn, inter in impls:
            try:
                t = time_fn(fn, dev, warmup, repeats)
                mem = peak_extra_mb(fn, dev)
                err = None
            except RuntimeError as e:  # e.g. out of memory for the reference at large N
                t, mem, err = None, None, str(e).split("\n")[0][:120]
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
            rows.append({"N": n, "impl": name, "ms": None if t is None else 1e3 * t,
                         "largest_intermediate_mb": None if inter is None else inter / 2**20,
                         "measured_peak_extra_mb": mem, "error": err})
            print(f"N={n:5d} {name:12s} " + (f"{1e3*t:9.2f} ms" if t else f"FAILED {err}")
                  + (f"  peak+{mem:8.1f} MiB" if mem is not None else ""), flush=True)
    return rows


def part_c(dev, dtype, n, blocks, warmup, repeats):
    q, k, v = qkv(1, 12, 2, n, n, 128, dtype, dev)
    rows = []
    for b in blocks:
        t = time_fn(lambda: tiled_attention(q, k, v, True, block_q=b, block_k=b), dev, warmup, repeats)
        _, stats = tiled_attention(q, k, v, True, block_q=b, block_k=b, return_stats=True)
        rows.append({"N": n, "block": b, "ms": 1e3 * t, **stats})
        print(f"block={b:4d} {1e3*t:9.2f} ms  {stats}", flush=True)
    return rows


def plot(b_rows, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    impls = sorted({r["impl"] for r in b_rows}, key=lambda s: (s != "reference", s))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for name in impls:
        rr = [r for r in b_rows if r["impl"] == name and r["ms"] is not None]
        ax[0].plot([r["N"] for r in rr], [r["ms"] for r in rr], marker="o", label=name)
        key = "measured_peak_extra_mb" if any(r["measured_peak_extra_mb"] is not None for r in rr) else "largest_intermediate_mb"
        mm = [r for r in rr if r[key] is not None]
        if mm:
            ax[1].plot([r["N"] for r in mm], [r[key] for r in mm], marker="o", label=name)
    ax[0].set(xlabel="sequence length N", ylabel="time (ms)", xscale="log", yscale="log")
    ax[1].set(xlabel="sequence length N", xscale="log", yscale="log",
              ylabel="measured peak extra (MiB)" if any(r["measured_peak_extra_mb"] for r in b_rows) else "largest intermediate (MiB, analytic)")
    ax[0].legend(fontsize=7); fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16"])
    ap.add_argument("--ns", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    ap.add_argument("--blocks", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--sweep-n", type=int, default=2048)
    ap.add_argument("--sweep-blocks", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    dev = pick_device(a.device)
    dtype = {"fp32": torch.float32, "fp16": torch.float16}.get(a.dtype) or (torch.float16 if dev.type != "cpu" else torch.float32)
    env = {"device": str(dev), "dtype": str(dtype), "torch": torch.__version__, "platform": platform.platform(),
           "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None, "warmup": a.warmup, "repeats": a.repeats}
    print(json.dumps(env))
    print("== Part A: numerical error vs float64 truth (CPU)")
    A = part_a(block=32)
    worst = {}
    for r in A:
        key = (r["impl"], r["dtype"])
        worst[key] = max(worst.get(key, 0), r["max_abs_err"])
    for key, v in sorted(worst.items()):
        print(f"  {key[0]:10s} {key[1]:8s} worst max-abs-err {v:.2e}")
    print(f"== Part B: speed / memory on {dev} ({dtype})")
    B = part_b(dev, dtype, a.ns, a.blocks, a.warmup, a.repeats)
    print(f"== Part C: tile-size sweep at N={a.sweep_n}")
    C = part_c(dev, dtype, a.sweep_n, a.sweep_blocks, a.warmup, a.repeats)
    out = ROOT / "results" / "bench"; out.mkdir(parents=True, exist_ok=True)
    (out / f"attention_{a.tag}.json").write_text(json.dumps({"env": env, "numerics": A, "speed": B, "block_sweep": C}, indent=2))
    for name, rows in (("numerics", A), ("speed", B), ("block_sweep", C)):
        with open(out / f"attention_{a.tag}_{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    plot(B, out / f"attention_{a.tag}.png", f"Attention: reference vs tiled vs SDPA — {env['gpu'] or env['device']} {env['dtype']}")
    print(f"-> {out}/attention_{a.tag}*")


if __name__ == "__main__":
    main()
