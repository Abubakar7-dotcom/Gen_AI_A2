"""D2/D3: benchmark explicit KV-cache greedy decoding against the uncached reference.

    python -m bench.bench_kv --model Qwen/Qwen2.5-1.5B-Instruct --tag t4_1p5b
    python -m bench.bench_kv --model Qwen/Qwen2.5-0.5B-Instruct --dtype fp32 --tag mac_0p5b_fp32

For each (input length, output length): same model, same input ids, EOS ignored (fixed output
length); `--warmup` untimed runs, then `--repeats` timed runs per method with device
synchronisation around every timer; medians are reported. Token equality between the two paths is
checked on every configuration; on a mismatch the step and the top-1/top-2 logit margins are logged.
Writes results/bench/kv_<tag>.csv/.json and kv_<tag>.png.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics as st
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from src.decoding.inputs import make_input_ids
from src.decoding.kv_cache import first_mismatch, greedy_kv, greedy_no_cache
from src.modeling import default_dtype, load_tokenizer, pick_device

ROOT = Path(__file__).resolve().parents[1]
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def run(args):
    dev = pick_device(args.device)
    dtype = DTYPES[args.dtype] if args.dtype != "auto" else default_dtype(dev)
    tok = load_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa").to(dev).eval()
    env = {"model": args.model, "dtype": str(dtype), "device": str(dev), "torch": torch.__version__,
           "platform": platform.platform(), "warmup": args.warmup, "repeats": args.repeats,
           "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None}
    print(json.dumps(env))
    rows = []
    for n_in in args.inputs:
        ids = make_input_ids(tok, n_in, dev)
        for n_out in args.outputs:
            res = {}
            for name, fn in (("no_cache", greedy_no_cache), ("kv_cache", greedy_kv)):
                for _ in range(args.warmup):
                    fn(model, ids, n_out)
                res[name] = [fn(model, ids, n_out) for _ in range(args.repeats)]
            a, b = res["no_cache"][0], res["kv_cache"][0]
            mm = first_mismatch(a.tokens, b.tokens)
            for name, rs in res.items():
                peaks = [r.peak_mem for r in rs if r.peak_mem is not None]
                rows.append({
                    "input_tokens": n_in, "output_tokens": n_out, "method": name,
                    "prefill_ms": 1e3 * st.median(r.prefill_s for r in rs),
                    "decode_ms": 1e3 * st.median(r.decode_s for r in rs),
                    "total_ms": 1e3 * st.median(r.total_s for r in rs),
                    "decode_tok_s": st.median(r.decode_tok_per_s for r in rs),
                    "overall_tok_s": st.median(r.tok_per_s for r in rs),
                    "peak_mem_mb": max(peaks) / 2**20 if peaks else None,
                    "kv_cache_mb": (b.cache_bytes / 2**20) if name == "kv_cache" else None,
                    "tokens_equal": mm is None,
                    "first_mismatch_step": mm,
                    "margin_at_mismatch_ref": a.margins[mm] if mm is not None and mm < len(a.margins) else None,
                    "margin_at_mismatch_kv": b.margins[mm] if mm is not None and mm < len(b.margins) else None,
                    "min_margin_ref": min(a.margins),
                })
            r_nc, r_kv = rows[-2], rows[-1]
            print(f"in={n_in:5d} out={n_out:4d} | no-cache {r_nc['total_ms']:8.0f} ms | kv {r_kv['total_ms']:8.0f} ms "
                  f"| speedup {r_nc['total_ms'] / r_kv['total_ms']:5.2f}x | equal={mm is None}"
                  + (f" (first diff step {mm}, margins {r_nc['margin_at_mismatch_ref']:.3f}/{r_kv['margin_at_mismatch_kv']:.3f})" if mm is not None else ""),
                  flush=True)
    return env, rows


def plot(rows, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ins = sorted({r["input_tokens"] for r in rows})
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for n_in in ins:
        for meth, ls in (("no_cache", "--"), ("kv_cache", "-")):
            rr = sorted([r for r in rows if r["input_tokens"] == n_in and r["method"] == meth], key=lambda r: r["output_tokens"])
            x = [r["output_tokens"] for r in rr]
            ax[0].plot(x, [r["total_ms"] / 1e3 for r in rr], ls, marker="o", label=f"{meth} in={n_in}")
            ax[1].plot(x, [r["decode_tok_s"] for r in rr], ls, marker="o", label=f"{meth} in={n_in}")
            if rr and rr[0]["peak_mem_mb"] is not None:
                ax[2].plot(x, [r["peak_mem_mb"] for r in rr], ls, marker="o", label=f"{meth} in={n_in}")
    ax[0].set(xlabel="output tokens", ylabel="end-to-end latency (s)")
    ax[1].set(xlabel="output tokens", ylabel="decode tokens / s")
    ax[2].set(xlabel="output tokens", ylabel="peak memory (MiB)")
    ax[0].legend(fontsize=7)
    fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dtype", default="auto", choices=["auto", *DTYPES])
    ap.add_argument("--device", default=None)
    ap.add_argument("--inputs", type=int, nargs="+", default=[128, 512, 1024])
    ap.add_argument("--outputs", type=int, nargs="+", default=[32, 128, 256])
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    env, rows = run(a)
    out = ROOT / "results" / "bench"; out.mkdir(parents=True, exist_ok=True)
    with open(out / f"kv_{a.tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / f"kv_{a.tag}.json").write_text(json.dumps({"env": env, "rows": rows}, indent=2))
    plot(rows, out / f"kv_{a.tag}.png", f"KV cache vs no cache — {env['model']} {env['dtype']} on {env['gpu'] or env['device']}")
    n_eq = sum(r["tokens_equal"] for r in rows) // 2
    print(f"token-equal configurations: {n_eq}/{len(rows)//2} -> {out/('kv_'+a.tag+'.csv')}")


if __name__ == "__main__":
    main()
