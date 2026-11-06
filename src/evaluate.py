"""Evaluate a (base or fine-tuned) model on val or test with greedy decoding.

    python -m src.evaluate --name base_1p5b --split val
    python -m src.evaluate --name ft_1p5b --adapter outputs/ft_1p5b_r16/best --split val
    python -m src.evaluate --name ft_1p5b --adapter <hf-repo> --split test --confirm-test

Writes results/metrics/<name>_<split>.json and results/predictions/<name>_<split>.jsonl.
The test split requires --confirm-test: it is run once, after all choices are frozen on val.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.data import ROOT, load_split
from src.metrics import full_report
from src.modeling import GREEDY_SETTINGS, generate_batch, load_model
from src.schema import parse_output


def evaluate_outputs(records, texts):
    preds = [parse_output(t, r["criterion_type"], len(r["sentences"])) for r, t in zip(records, texts)]
    return preds, full_report(records, preds)


def save_confusion_pngs(report: dict, path_prefix: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for key in ("overall", "inclusion", "exclusion"):
        cm = report[key]["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.imshow(cm["matrix"], cmap="Blues")
        ax.set_xticks(range(len(cm["cols_pred"])), cm["cols_pred"], rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(cm["rows_gold"])), cm["rows_gold"], fontsize=8)
        for i, row in enumerate(cm["matrix"]):
            for j, v in enumerate(row):
                ax.text(j, i, v, ha="center", va="center", fontsize=8)
        ax.set_xlabel("predicted"); ax.set_ylabel("gold"); ax.set_title(f"{path_prefix.name} — {key}")
        fig.tight_layout()
        fig.savefig(f"{path_prefix}_cm_{key}.png", dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--confirm-test", action="store_true")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="debug: evaluate only the first N rows")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    if a.split == "test" and not a.confirm_test:
        raise SystemExit("Refusing to touch the test split without --confirm-test (freeze choices on val first).")

    records = load_split(a.split)[: a.limit] if a.limit else load_split(a.split)
    model, tok = load_model(a.base, a.adapter, device=a.device)
    t0 = time.time()
    texts = generate_batch(model, tok, records, a.batch_size, a.max_new_tokens, progress=True)
    secs = time.time() - t0
    preds, report = evaluate_outputs(records, texts)
    report["settings"] = {"base": a.base, "adapter": a.adapter, "split": a.split, "n": len(records),
                          "decoding": {**GREEDY_SETTINGS, "max_new_tokens": a.max_new_tokens},
                          "device": str(next(model.parameters()).device),
                          "dtype": str(next(model.parameters()).dtype), "seconds": round(secs, 1)}

    mdir, pdir = ROOT / "results" / "metrics", ROOT / "results" / "predictions"
    mdir.mkdir(parents=True, exist_ok=True); pdir.mkdir(parents=True, exist_ok=True)
    stem = f"{a.name}_{a.split}"
    (mdir / f"{stem}.json").write_text(json.dumps(report, indent=2))
    with open(pdir / f"{stem}.jsonl", "w") as f:
        for r, p in zip(records, preds):
            f.write(json.dumps({"annotation_id": r["annotation_id"], "patient_id": r["patient_id"],
                                "criterion_type": r["criterion_type"], "criterion_text": r["criterion_text"],
                                "gold_label": r["gold_label"], "gold_evidence": r["gold_evidence"],
                                "pred_label": p["label"], "pred_evidence": p["evidence"],
                                "label_valid": p["label_valid"], "parse_ok": p["parse_ok"],
                                "unsupported_ids": p["unsupported_ids"], "raw": p["raw"]}) + "\n")
    save_confusion_pngs(report, mdir / stem)
    o, ev = report["overall"], report["evidence_all"]
    print(f"[{stem}] acc={o['accuracy']:.3f} macroF1={o['macro_f1']:.3f} "
          f"incl macroF1={report['inclusion']['macro_f1']:.3f} excl macroF1={report['exclusion']['macro_f1']:.3f} "
          f"evid microF1={ev['micro']['f1']:.3f} macroF1={ev['macro']['f1']:.3f} "
          f"invalid={report['output_validity']} ({secs:.0f}s)")


if __name__ == "__main__":
    main()
