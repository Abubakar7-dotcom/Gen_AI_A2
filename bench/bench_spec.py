"""F1/F3: benchmark greedy speculative decoding against target-only greedy KV decoding.

    python -m bench.bench_spec --tag gpu            # T4: fine-tuned 1.5B target, base vs fine-tuned 0.5B drafts
    python -m bench.bench_spec --tag mac_m5 --n-records 12

Two workloads, same target, same inputs:
  task  real validation prompts (first --n-records), EOS on, max 64 new tokens: the application's
        actual output (a short JSON answer).
  long  one 512-token input, 128 new tokens with EOS ignored: a longer free-text continuation.
For each draft x k: acceptance rate, mean tokens per verification round, latency, tokens/s, peak
memory and speedup over the target-only KV loop. Every output is checked token-for-token against
the target-only greedy output. Writes results/bench/spec_<tag>.csv/.json/.png.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics as st
from pathlib import Path

import torch

from src.data import load_split
from src.decoding.inputs import make_input_ids, record_input_ids
from src.decoding.kv_cache import eos_set, greedy_kv
from src.decoding.speculative import assert_same_tokenizer, speculative_greedy
from src.modeling import default_dtype, load_model, load_tokenizer, pick_device

ROOT = Path(__file__).resolve().parents[1]
B15, B05 = "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def workloads(tok, dev, n_records, long_in, long_out):
    eos = eos_set(tok)
    task = [(record_input_ids(tok, r, dev), 64, eos) for r in load_split("val")[:n_records]]
    long = [(make_input_ids(tok, long_in, dev), long_out, None)]
    return {"task": task, "long": long}


def run_many(fn, items, repeats):
    """Runs fn over all items `repeats` times; returns (per-item results of the last repeat, median total s)."""
    totals, last = [], None
    for _ in range(repeats):
        last = [fn(ids, n, eos) for ids, n, eos in items]
        totals.append(sum(r.total_s for r in last))
    return last, st.median(totals)


def peak_mb(results):
    p = [r.peak_mem for r in results if r.peak_mem is not None]
    return max(p) / 2**20 if p else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=B15)
    ap.add_argument("--target-adapter", default="outputs/ft_1p5b_r16/best")
    ap.add_argument("--draft-base", default=B05)
    ap.add_argument("--draft-adapter", default="outputs/ft_0p5b_r16/best")
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--n-records", type=int, default=24)
    ap.add_argument("--long-in", type=int, default=512)
    ap.add_argument("--long-out", type=int, default=128)
    ap.add_argument("--dtype", default="auto", choices=["auto", *DTYPES])
    ap.add_argument("--device", default=None)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    dev = pick_device(a.device)
    dtype = DTYPES[a.dtype] if a.dtype != "auto" else default_dtype(dev)
    t_adapter = a.target_adapter if a.target_adapter and Path(ROOT / a.target_adapter).exists() else None
    d_adapter = a.draft_adapter if a.draft_adapter and Path(ROOT / a.draft_adapter).exists() else None
    target, tok = load_model(a.target, t_adapter, device=dev, dtype=dtype)
    drafts = {"base_0p5b": load_model(a.draft_base, None, device=dev, dtype=dtype)}
    if d_adapter:
        drafts["ft_0p5b"] = load_model(a.draft_base, d_adapter, device=dev, dtype=dtype)
    for name, (_, dtok) in drafts.items():
        assert_same_tokenizer(tok, dtok)
    env = {"target": a.target, "target_adapter": t_adapter, "draft_base": a.draft_base, "draft_adapter": d_adapter,
           "dtype": str(dtype), "device": str(dev), "torch": torch.__version__, "platform": platform.platform(),
           "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
           "n_records": a.n_records, "long": [a.long_in, a.long_out], "warmup": a.warmup, "repeats": a.repeats}
    print(json.dumps(env), flush=True)

    rows = []
    for wname, items in workloads(tok, dev, a.n_records, a.long_in, a.long_out).items():
        base_fn = lambda ids, n, eos: greedy_kv(target, ids, n, eos_ids=eos)
        for _ in range(a.warmup):
            base_fn(*items[0])
        base_res, base_s = run_many(base_fn, items, a.repeats)
        n_tok = sum(len(r.tokens) for r in base_res)
        rows.append({"workload": wname, "method": "target_only_kv", "draft": None, "k": None, "items": len(items),
                     "tokens": n_tok, "total_s": base_s, "tok_s": n_tok / base_s, "speedup": 1.0,
                     "acceptance_rate": None, "tokens_per_round": None, "target_forwards": n_tok,
                     "draft_forwards": 0, "draft_share": None, "peak_mem_mb": peak_mb(base_res), "tokens_equal": True})
        print(f"[{wname}] target-only KV: {n_tok} tokens in {base_s:.2f}s ({n_tok/base_s:.1f} tok/s)", flush=True)
        for dname, (dmodel, _) in drafts.items():
            for k in a.ks:
                fn = lambda ids, n, eos: speculative_greedy(target, dmodel, ids, n, k=k, eos_ids=eos)
                for _ in range(a.warmup):
                    fn(*items[0])
                res, tot = run_many(fn, items, a.repeats)
                equal = all(r.tokens == b.tokens for r, b in zip(res, base_res))
                prop, acc = sum(r.proposed for r in res), sum(r.accepted for r in res)
                rounds = sum(r.rounds for r in res)
                dshare = sum(r.draft_s for r in res) / max(1e-9, sum(r.draft_s + r.verify_s for r in res))
                rows.append({"workload": wname, "method": "speculative", "draft": dname, "k": k, "items": len(items),
                             "tokens": n_tok, "total_s": tot, "tok_s": n_tok / tot, "speedup": base_s / tot,
                             "acceptance_rate": acc / prop if prop else None,
                             "tokens_per_round": sum(len(r.tokens) - 1 for r in res) / rounds if rounds else None,
                             "target_forwards": sum(r.target_forwards for r in res),
                             "draft_forwards": sum(r.draft_forwards for r in res),
                             "draft_share": dshare if dev.type != "cpu" else None,
                             "peak_mem_mb": peak_mb(res), "tokens_equal": equal})
                r = rows[-1]
                print(f"[{wname}] {dname:9s} k={k}: {tot:7.2f}s  speedup {r['speedup']:.2f}x  accept {r['acceptance_rate']:.2f}  "
                      f"tok/round {r['tokens_per_round']:.2f}  target fwd {r['target_forwards']}  equal={equal}", flush=True)

    out = ROOT / "results" / "bench"; out.mkdir(parents=True, exist_ok=True)
    with open(out / f"spec_{a.tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / f"spec_{a.tag}.json").write_text(json.dumps({"env": env, "rows": rows}, indent=2))
    plot(rows, out / f"spec_{a.tag}.png", f"Speculative decoding — {env['gpu'] or env['device']} {env['dtype']}")
    print(f"all outputs token-identical: {all(r['tokens_equal'] for r in rows)} -> {out}/spec_{a.tag}.*")


def plot(rows, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for w, ls in (("task", "-"), ("long", "--")):
        for d in sorted({r["draft"] for r in rows if r["draft"]}):
            rr = sorted([r for r in rows if r["workload"] == w and r["draft"] == d], key=lambda r: r["k"])
            if rr:
                ax[0].plot([r["k"] for r in rr], [r["speedup"] for r in rr], ls, marker="o", label=f"{w}, draft {d}")
                ax[1].plot([r["k"] for r in rr], [r["acceptance_rate"] for r in rr], ls, marker="o", label=f"{w}, draft {d}")
    ax[0].axhline(1.0, color="grey", lw=0.8)
    ax[0].set(xlabel="draft length k", ylabel="speedup vs target-only KV")
    ax[1].set(xlabel="draft length k", ylabel="acceptance rate", ylim=(0, 1.05))
    ax[0].legend(fontsize=7); fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
