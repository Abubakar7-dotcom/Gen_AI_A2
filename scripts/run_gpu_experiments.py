"""Run every GPU experiment of Phase 2 in order (works on Windows and Linux).

    python scripts/run_gpu_experiments.py              # run all steps, skipping finished ones
    python scripts/run_gpu_experiments.py --force      # re-run everything
    python scripts/run_gpu_experiments.py --batch-size 2   # if 6 GB runs out of memory

Each step's console output is saved to results/logs/<step>.log. A step is skipped when its
output already exists, so the script can be restarted after a crash. The test split is NOT
evaluated here (that happens once, after all choices are frozen on validation).
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
B15, B05 = "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"


def steps(batch_size: int | None):
    bs = ["--batch-size", str(batch_size)] if batch_size else []
    m = "results/metrics"
    return [
        ("data", ["-m", "src.data", "--check", "--token-stats"], None),  # cheap: always re-verify the split
        ("tests", ["-m", "pytest", "-q", "tests/test_split.py", "tests/test_schema.py", "tests/test_metrics.py",
                   "tests/test_train_loss.py"], None),
        ("eval_base_1p5b", ["-m", "src.evaluate", "--name", "base_1p5b", "--base", B15, "--split", "val"], f"{m}/base_1p5b_val.json"),
        ("eval_base_0p5b", ["-m", "src.evaluate", "--name", "base_0p5b", "--base", B05, "--split", "val"], f"{m}/base_0p5b_val.json"),
        ("train_1p5b_r16", ["-m", "src.train", "--config", "configs/train_1p5b.yaml", *bs], "results/training/ft_1p5b_r16/summary.json"),
        ("train_1p5b_r8", ["-m", "src.train", "--config", "configs/train_1p5b_r8.yaml", *bs], "results/training/ft_1p5b_r8/summary.json"),
        ("train_0p5b_r16", ["-m", "src.train", "--config", "configs/train_0p5b.yaml", *bs], "results/training/ft_0p5b_r16/summary.json"),
        ("eval_ft_1p5b_r16", ["-m", "src.evaluate", "--name", "ft_1p5b_r16", "--base", B15, "--adapter", "outputs/ft_1p5b_r16/best", "--split", "val"], f"{m}/ft_1p5b_r16_val.json"),
        ("eval_ft_1p5b_r8", ["-m", "src.evaluate", "--name", "ft_1p5b_r8", "--base", B15, "--adapter", "outputs/ft_1p5b_r8/best", "--split", "val"], f"{m}/ft_1p5b_r8_val.json"),
        ("eval_ft_0p5b_r16", ["-m", "src.evaluate", "--name", "ft_0p5b_r16", "--base", B05, "--adapter", "outputs/ft_0p5b_r16/best", "--split", "val"], f"{m}/ft_0p5b_r16_val.json"),
    ]


def env_report() -> dict:
    import torch
    info = {"python": platform.python_version(), "os": platform.platform(), "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda}
    import transformers, peft
    info.update(transformers=transformers.__version__, peft=peft.__version__)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, vram_gb=round(p.total_memory / 1e9, 2), bf16=torch.cuda.is_bf16_supported(),
                    compute_capability=f"{p.major}.{p.minor}")
    return info


def run(name, args, log_dir: Path) -> float:
    log = log_dir / f"{name}.log"
    print(f"\n=== {name}: python {' '.join(args)}\n    log -> {log.relative_to(ROOT)}", flush=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.Popen([PY, *args], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line); f.write(line)
        p.wait()
    if p.returncode != 0:
        raise SystemExit(f"step '{name}' failed (exit {p.returncode}); see {log}")
    return time.time() - t0


def summary_table() -> str:
    rows = ["| run | split | accuracy | macro-F1 | incl. macro-F1 | excl. macro-F1 | evidence micro-F1 | evidence macro-F1 | parse fail | invalid label |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for f in sorted((ROOT / "results" / "metrics").glob("*_val.json")):
        r = json.loads(f.read_text()); o = r["overall"]; v = r["output_validity"]
        rows.append(f"| {f.stem[:-4]} | val | {o['accuracy']:.3f} | {o['macro_f1']:.3f} | {r['inclusion']['macro_f1']:.3f} | "
                    f"{r['exclusion']['macro_f1']:.3f} | {r['evidence_all']['micro']['f1']:.3f} | {r['evidence_all']['macro']['f1']:.3f} | "
                    f"{v['parse_failures']} | {v['invalid_labels']} |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--only", nargs="*", help="run only these step names")
    a = ap.parse_args()
    log_dir = ROOT / "results" / "logs"; log_dir.mkdir(parents=True, exist_ok=True)
    env = env_report()
    print(json.dumps(env, indent=2))
    (ROOT / "results" / "env_gpu.json").write_text(json.dumps(env, indent=2))
    if not env["cuda_available"]:
        print("WARNING: CUDA not available - this will be very slow. Check the torch install (README).")
    timings = {}
    for name, args, out in steps(a.batch_size):
        if a.only and name not in a.only:
            continue
        if out and (ROOT / out).exists() and not a.force:
            print(f"--- skip {name} (found {out})")
            continue
        timings[name] = round(run(name, args, log_dir), 1)
    (log_dir / "timings.json").write_text(json.dumps(timings, indent=2))
    table = summary_table()
    (ROOT / "results" / "val_summary.md").write_text(table + "\n")
    print("\n" + table + f"\n\nstep timings (s): {timings}")


if __name__ == "__main__":
    main()
