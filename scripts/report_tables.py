"""Print LaTeX table rows for the report from the result files, and copy the report figures.

    python scripts/report_tables.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def f3(x):
    return "--" if x is None else f"{x:.3f}"


def test_table():
    runs = {"Base 1.5B": R / "metrics/base_1p5b_test.json", "LoRA 1.5B r16": R / "metrics/ft_1p5b_r16_test.json"}
    if not all(p.exists() for p in runs.values()):
        print("% test metrics not available yet"); return
    m = {k: json.loads(p.read_text()) for k, p in runs.items()}
    rows = [
        ("Accuracy", lambda r: f3(r["overall"]["accuracy"])),
        ("Macro-F1 (overall)", lambda r: f3(r["overall"]["macro_f1"])),
        ("Macro-F1 inclusion / exclusion", lambda r: f"{f3(r['inclusion']['macro_f1'])} / {f3(r['exclusion']['macro_f1'])}"),
        ("Evidence micro P / R / F1", lambda r: " / ".join(f3(r["evidence_all"]["micro"][k]) for k in ("precision", "recall", "f1"))),
        ("Evidence micro-F1, non-empty gold", lambda r: f3(r["evidence_nonempty_gold"]["micro"]["f1"])),
        ("Parse failures / invalid labels", lambda r: f"{r['output_validity']['parse_failures']} / {r['output_validity']['invalid_labels']}"),
    ]
    print("% --- Table: test results")
    for name, fn in rows:
        print(f"{name} & {fn(m['Base 1.5B'])} & {fn(m['LoRA 1.5B r16'])} \\\\")
    ft = m["LoRA 1.5B r16"]["overall"]["per_class"]
    print("% per-class F1 (fine-tuned): " + ", ".join(f"{k}={f3(v['f1'])} (n={v['support']})" for k, v in ft.items()))


def spec_table():
    p = R / "bench/spec_gpu.json"
    if not p.exists():
        print("% spec_gpu.json not available yet"); return
    d = json.loads(p.read_text())
    print("% --- Table: speculative decoding (workload, draft, k, accept, tok/round, s, tok/s, speedup, equal)")
    for r in d["rows"]:
        tpr = "--" if r["tokens_per_round"] is None else f"{r['tokens_per_round']:.2f}"
        print(f"{r['workload']} & {r['draft'] or 'target only'} & {r['k'] or '--'} & {f3(r['acceptance_rate'])} & {tpr} & "
              f"{r['total_s']:.2f} & {r['tok_s']:.1f} & {r['speedup']:.2f}$\\times$ & {'yes' if r['tokens_equal'] else 'NO'} \\\\")


def copy_figures():
    fig = ROOT / "report" / "figures"
    pairs = {
        R / "training/ft_1p5b_r16/curves.png": "curves_ft_1p5b_r16.png",
        R / "metrics/ft_1p5b_r16_test_cm_overall.png": "cm_ft_1p5b_r16_test.png",
        R / "bench/kv_gpu_1p5b_half.png": "kv_gpu_1p5b_half.png",
        R / "bench/attention_gpu.png": "attention_gpu.png",
        R / "bench/spec_gpu.png": "spec_gpu.png",
    }
    for src, dst in pairs.items():
        if src.exists():
            shutil.copy(src, fig / dst)
            print(f"% copied {src.relative_to(ROOT)} -> report/figures/{dst}")


if __name__ == "__main__":
    test_table()
    spec_table()
    copy_figures()
