"""Criterion classification metrics and evidence-sentence retrieval metrics.

Classification
  * Invalid / unparseable predictions are mapped to the pseudo-label "invalid". They count as
    wrong (they lower recall of the gold class) but "invalid" is not itself a class in macro-F1.
  * Macro-F1 averages over labels that occur in the gold set or in valid predictions of the
    evaluated subset. A class with zero gold support AND zero predictions (e.g. "excluded" on the
    test split) is reported as n/a and does not enter the macro average.

Evidence (per record, G = gold ids, P = predicted ids)
  * G = {} and P = {}  -> P = R = F1 = 1   (correctly cites nothing)
  * G = {} and P != {} -> P = 0, R undefined, F1 = 0
  * G != {} and P = {} -> P undefined, R = 0, F1 = 0
  * Macro P / R average over records where they are defined; macro F1 averages over all records.
  * Micro P / R / F1 pool TP, FP, FN over all records (empty/empty records contribute nothing).
  * Reported for all records and for the subset with non-empty gold evidence.
"""
from __future__ import annotations

import numpy as np

from src.schema import ALL_LABELS, LABELS_BY_TYPE

INVALID = "invalid"


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p is not None and r is not None and (p + r) > 0 else 0.0
    return p, r, f


def classification_metrics(gold: list[str], pred: list[str | None], label_order: list[str]) -> dict:
    pred = [p if p in label_order else INVALID for p in pred]
    n = len(gold)
    acc = sum(g == p for g, p in zip(gold, pred)) / n if n else 0.0
    per_class = {}
    macro_labels = [l for l in label_order if l in set(gold) or l in set(pred)]
    for l in label_order:
        tp = sum(g == l and p == l for g, p in zip(gold, pred))
        fp = sum(g != l and p == l for g, p in zip(gold, pred))
        fn = sum(g == l and p != l for g, p in zip(gold, pred))
        support = sum(g == l for g in gold)
        if l not in macro_labels:
            per_class[l] = {"precision": None, "recall": None, "f1": None, "support": 0, "note": "n/a: no gold or predictions"}
            continue
        p, r, f = _prf(tp, fp, fn)
        per_class[l] = {"precision": p if p is not None else 0.0, "recall": r if r is not None else 0.0,
                        "f1": f, "support": support}
    macro_f1 = float(np.mean([per_class[l]["f1"] for l in macro_labels])) if macro_labels else 0.0
    cols = label_order + [INVALID]
    cm = [[sum(g == a and p == b for g, p in zip(gold, pred)) for b in cols] for a in label_order]
    return {"n": n, "accuracy": acc, "macro_f1": macro_f1, "macro_labels": macro_labels,
            "invalid_predictions": sum(p == INVALID for p in pred), "per_class": per_class,
            "confusion_matrix": {"rows_gold": label_order, "cols_pred": cols, "matrix": cm}}


def evidence_metrics(gold_sets: list[list[int]], pred_sets: list[list[int]]) -> dict:
    tp = fp = fn = 0
    ps, rs, fs = [], [], []
    for g, p in zip(gold_sets, pred_sets):
        g, p = set(g), set(p)
        t = len(g & p)
        tp, fp, fn = tp + t, fp + len(p - g), fn + len(g - p)
        if not g and not p:
            ps.append(1.0); rs.append(1.0); fs.append(1.0)
            continue
        prec = t / len(p) if p else None
        rec = t / len(g) if g else None
        f1 = 2 * prec * rec / (prec + rec) if prec is not None and rec is not None and prec + rec > 0 else 0.0
        if prec is not None:
            ps.append(prec)
        if rec is not None:
            rs.append(rec)
        fs.append(f1)
    mp, mr, mf = _prf(tp, fp, fn)
    return {
        "n": len(gold_sets),
        "micro": {"precision": mp or 0.0, "recall": mr or 0.0, "f1": mf, "tp": tp, "fp": fp, "fn": fn},
        "macro": {"precision": float(np.mean(ps)) if ps else 0.0, "recall": float(np.mean(rs)) if rs else 0.0,
                  "f1": float(np.mean(fs)) if fs else 0.0},
    }


def full_report(records: list[dict], preds: list[dict]) -> dict:
    """records: gold records from src.data; preds: parse_output() dicts in the same order."""
    def label_of(p, ctype):
        return p["label"] if p["label_valid"] else INVALID

    out = {"overall": classification_metrics([r["gold_label"] for r in records],
                                             [label_of(p, r["criterion_type"]) for r, p in zip(records, preds)],
                                             ALL_LABELS)}
    for t, labels in LABELS_BY_TYPE.items():
        idx = [i for i, r in enumerate(records) if r["criterion_type"] == t]
        out[t] = classification_metrics([records[i]["gold_label"] for i in idx],
                                        [label_of(preds[i], t) for i in idx], labels)
    g = [r["gold_evidence"] for r in records]
    p = [x["evidence"] for x in preds]
    nz = [i for i, x in enumerate(g) if x]
    out["evidence_all"] = evidence_metrics(g, p)
    out["evidence_nonempty_gold"] = evidence_metrics([g[i] for i in nz], [p[i] for i in nz])
    for t in LABELS_BY_TYPE:
        idx = [i for i, r in enumerate(records) if r["criterion_type"] == t]
        out[f"evidence_{t}"] = evidence_metrics([g[i] for i in idx], [p[i] for i in idx])
    out["output_validity"] = {
        "parse_failures": sum(not x["parse_ok"] for x in preds),
        "invalid_labels": sum(not x["label_valid"] for x in preds),
        "records_with_unsupported_ids": sum(bool(x["unsupported_ids"]) for x in preds),
    }
    return out
