"""C4: sort the errors of one evaluated run into categories and dump them with the note sentences,
as raw material for the written error analysis (results/error_analysis.md).

    python scripts/error_analysis.py --run ft_1p5b_r16 --split test

Categories (one record can fall into several):
  wrong_label_right_evidence   label wrong, cited ids == gold ids (non-empty)
  right_label_wrong_evidence   label right, cited ids != gold ids
  missing_info_confusion       one side is "not enough information", the other a decided label
  not_applicable               gold or prediction is "not applicable" and they differ
  polarity_flip                included<->not included or excluded<->not excluded
  unsupported_citation         cited ids that are not sentences of the note
  output_invalid               unparseable output or label not allowed for the criterion type
  truncation                   note sentences dropped to fit the prompt budget (none expected)
Writes results/error_candidates_<run>_<split>.md and prints counts.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FLIPS = {frozenset({"included", "not included"}), frozenset({"excluded", "not excluded"})}
NEI, NA = "not enough information", "not applicable"


def categories(p: dict, n_trunc: int = 0) -> list[str]:
    g, y = p["gold_label"], p["pred_label"]
    ge, pe = set(p["gold_evidence"]), set(p["pred_evidence"])
    cats = []
    if not p["parse_ok"] or not p["label_valid"]:
        cats.append("output_invalid")
    if p["unsupported_ids"]:
        cats.append("unsupported_citation")
    if n_trunc:
        cats.append("truncation")
    if y != g:
        if ge and pe == ge:
            cats.append("wrong_label_right_evidence")
        if (g == NEI) != (y == NEI) and NA not in (g, y):
            cats.append("missing_info_confusion")
        if NA in (g, y):
            cats.append("not_applicable")
        if frozenset({g, y}) in FLIPS:
            cats.append("polarity_flip")
    elif pe != ge:
        cats.append("right_label_wrong_evidence")
    return cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ft_1p5b_r16")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-per-category", type=int, default=12)
    a = ap.parse_args()
    from src.data import load_split
    recs = {r["annotation_id"]: r for r in load_split(a.split)}
    preds = [json.loads(l) for l in open(ROOT / "results" / "predictions" / f"{a.run}_{a.split}.jsonl")]
    by_cat: dict[str, list] = {}
    for p in preds:
        for c in categories(p):
            by_cat.setdefault(c, []).append(p)
    counts = Counter({c: len(v) for c, v in by_cat.items()})
    n_wrong = sum(p["pred_label"] != p["gold_label"] for p in preds)
    lines = [f"# Error candidates: {a.run} on {a.split} ({len(preds)} records, {n_wrong} wrong labels)", "",
             "| category | records |", "|---|---|"] + [f"| {c} | {n} |" for c, n in counts.most_common()]
    conf = Counter((p["criterion_type"], p["gold_label"], p["pred_label"]) for p in preds if p["pred_label"] != p["gold_label"])
    lines += ["", "## Label confusions (type, gold -> predicted)", ""] + [f"- {t}: {g} -> {y}: {n}" for (t, g, y), n in conf.most_common()]
    for c, ps in sorted(by_cat.items()):
        lines += ["", f"## {c} ({len(ps)})"]
        for p in ps[: a.max_per_category]:
            s = recs[p["annotation_id"]]["sentences"]
            show = sorted(set(p["gold_evidence"]) | set(p["pred_evidence"]))
            lines += ["", f"### annotation {p['annotation_id']} — patient {p['patient_id']}, {p['criterion_type']}",
                      f"- criterion: {p['criterion_text'].strip()}",
                      f"- gold: **{p['gold_label']}** {p['gold_evidence']} | predicted: **{p['pred_label']}** {p['pred_evidence']}"
                      + (f" | unsupported ids {p['unsupported_ids']}" if p["unsupported_ids"] else ""),
                      f"- raw output: `{p['raw'].strip()[:200]}`"]
            lines += [f"  - [{i}]{' (gold)' if i in p['gold_evidence'] else ''}{' (cited)' if i in p['pred_evidence'] else ''} {s[i]}"
                      for i in show if i < len(s)]
    out = ROOT / "results" / f"error_candidates_{a.run}_{a.split}.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:4 + len(counts)]))
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
