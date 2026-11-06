"""Dataset loading, the prescribed patient-grouped split, leakage checks and record building.

Split protocol (assignment): sort unique patient IDs, shuffle with numpy default_rng(42),
first 37 -> train, next 8 -> val, last 8 -> test. All rows of a patient stay in one split.

Usage:
    python -m src.data --check            # build split, write manifest + processed jsonl, run checks
    python -m src.data --token-stats      # tokenizer length / truncation statistics
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.schema import ALL_LABELS, LABELS_BY_TYPE, build_messages, format_target

DATASET_ID = "ncbi/TrialGPT-Criterion-Annotations"
DATASET_REVISION = "1cfcafde94a1560a33b4addc1638664fe28fc059"  # pinned HF commit
SEED = 42
SPLIT_SIZES = {"train": 37, "val": 8, "test": 8}

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "split_manifest.json"
PROCESSED_DIR = ROOT / "data" / "processed"


def load_raw() -> pd.DataFrame:
    from datasets import load_dataset
    ds = load_dataset(DATASET_ID, split="train", revision=DATASET_REVISION)
    return ds.to_pandas()


def make_split(patient_ids) -> dict[str, list[str]]:
    ids = np.array(sorted(set(patient_ids)))
    assert len(ids) == sum(SPLIT_SIZES.values()), f"expected 53 patients, got {len(ids)}"
    rng = np.random.default_rng(SEED)
    rng.shuffle(ids)
    ids = ids.tolist()
    a, b = SPLIT_SIZES["train"], SPLIT_SIZES["train"] + SPLIT_SIZES["val"]
    return {"train": ids[:a], "val": ids[a:b], "test": ids[b:]}


def split_note(note: str) -> list[str]:
    """Dataset notes are 'i. sentence' lines numbered 0..n-1. Returns sentence texts in order."""
    sents = []
    for i, line in enumerate(note.split("\n")):
        prefix = f"{i}. "
        if not line.startswith(prefix):
            raise ValueError(f"unexpected note numbering at line {i}: {line[:40]!r}")
        sents.append(line[len(prefix):].strip())
    return sents


def build_records(df: pd.DataFrame, split: dict[str, list[str]]) -> tuple[dict[str, list[dict]], list[dict]]:
    """Builds clean records. Only note, criterion type/text and gold fields are kept;
    GPT-4 prediction columns are never copied. Returns (records per split, dropped rows)."""
    pid2split = {p: s for s, ps in split.items() for p in ps}
    out = {s: [] for s in split}
    dropped = []
    for r in df.itertuples(index=False):
        crit = r.criterion_text if isinstance(r.criterion_text, str) else ""
        if not crit.strip():
            dropped.append({"annotation_id": int(r.annotation_id), "patient_id": r.patient_id,
                            "reason": "empty criterion_text"})
            continue
        ctype = r.criterion_type.strip().lower()
        label = r.expert_eligibility.strip().lower()
        assert label in LABELS_BY_TYPE[ctype], (r.annotation_id, ctype, label)
        sents = split_note(r.note)
        gold_ev = sorted(set(json.loads(r.expert_sentences)))
        assert all(0 <= i < len(sents) for i in gold_ev)
        out[pid2split[r.patient_id]].append({
            "annotation_id": int(r.annotation_id),
            "patient_id": r.patient_id,
            "trial_id": r.trial_id,
            "criterion_type": ctype,
            "criterion_text": crit.strip(),
            "sentences": sents,
            "gold_label": label,
            "gold_evidence": gold_ev,
        })
    return out, dropped


def check_split(df: pd.DataFrame, split: dict[str, list[str]], records: dict[str, list[dict]], dropped) -> None:
    sets = {k: set(v) for k, v in split.items()}
    assert {k: len(v) for k, v in sets.items()} == SPLIT_SIZES
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"])
    assert set().union(*sets.values()) == set(df.patient_id.unique())
    # every kept row in exactly one split, and in the split of its patient
    ids = [r["annotation_id"] for rs in records.values() for r in rs]
    assert len(ids) == len(set(ids)) == len(df) - len(dropped)
    for s, rs in records.items():
        assert all(r["patient_id"] in sets[s] for r in rs)


def make_manifest(df, split, records, dropped) -> dict:
    trials = {s: {r["trial_id"] for r in rs} for s, rs in records.items()}
    man = {
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "protocol": "sorted(unique patient_id) -> numpy.random.default_rng(42).shuffle -> 37/8/8",
        "seed": SEED,
        "total_rows": int(len(df)),
        "total_patients": int(df.patient_id.nunique()),
        "dropped_rows": dropped,
        "splits": {},
        "trial_overlap": {
            "train&val": len(trials["train"] & trials["val"]),
            "train&test": len(trials["train"] & trials["test"]),
            "val&test": len(trials["val"] & trials["test"]),
        },
    }
    for s in split:
        rs = records[s]
        man["splits"][s] = {
            "patient_ids": split[s],
            "n_patients": len(split[s]),
            "n_rows": len(rs),
            "n_rows_by_type": dict(Counter(r["criterion_type"] for r in rs)),
            "label_counts": {t: {l: sum(1 for r in rs if r["criterion_type"] == t and r["gold_label"] == l)
                                 for l in LABELS_BY_TYPE[t]} for t in LABELS_BY_TYPE},
            "rows_with_empty_gold_evidence": sum(1 for r in rs if not r["gold_evidence"]),
            "n_trials": len(trials[s]),
        }
    return man


def prepare(write: bool = True) -> tuple[dict, dict[str, list[dict]]]:
    df = load_raw()
    split = make_split(df.patient_id.unique())
    records, dropped = build_records(df, split)
    check_split(df, split, records, dropped)
    man = make_manifest(df, split, records, dropped)
    if write:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(json.dumps(man, indent=2))
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        for s, rs in records.items():
            with open(PROCESSED_DIR / f"{s}.jsonl", "w") as f:
                for r in rs:
                    f.write(json.dumps(r) + "\n")
    return man, records


def load_split(name: str) -> list[dict]:
    path = PROCESSED_DIR / f"{name}.jsonl"
    if not path.exists():
        prepare(write=True)
    return [json.loads(l) for l in open(path)]


# ---------------------------------------------------------------- tokenization / truncation
def prompt_text(tokenizer, sentences, ctype, criterion) -> str:
    return tokenizer.apply_chat_template(build_messages(sentences, ctype, criterion),
                                         tokenize=False, add_generation_prompt=True)


def fit_to_budget(tokenizer, rec: dict, max_prompt_tokens: int) -> tuple[list[str], list[int]]:
    """Drop whole trailing note sentences until the prompt fits. Returns (kept sentences, dropped ids).
    The criterion text and instructions are never truncated."""
    sents = list(rec["sentences"])
    dropped = []
    while sents:
        n = len(tokenizer(prompt_text(tokenizer, sents, rec["criterion_type"], rec["criterion_text"]),
                          add_special_tokens=False)["input_ids"])
        if n <= max_prompt_tokens:
            break
        dropped.insert(0, len(sents) - 1)
        sents.pop()
    return sents, dropped


def token_stats(model_id: str, max_prompt_tokens: int) -> dict:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    stats = {"model": model_id, "max_prompt_tokens": max_prompt_tokens, "splits": {}}
    for s in SPLIT_SIZES:
        recs = load_split(s)
        lens, tlens, truncated, lost_gold = [], [], 0, 0
        for r in recs:
            p = prompt_text(tok, r["sentences"], r["criterion_type"], r["criterion_text"])
            lens.append(len(tok(p, add_special_tokens=False)["input_ids"]))
            tlens.append(len(tok(format_target(r["gold_label"], r["gold_evidence"]),
                                 add_special_tokens=False)["input_ids"]))
            if lens[-1] > max_prompt_tokens:
                _, dropped = fit_to_budget(tok, r, max_prompt_tokens)
                truncated += 1
                lost_gold += len(set(dropped) & set(r["gold_evidence"]))
        stats["splits"][s] = {
            "prompt_tokens": {"min": int(np.min(lens)), "median": float(np.median(lens)),
                              "p95": float(np.percentile(lens, 95)), "max": int(np.max(lens))},
            "target_tokens": {"min": int(np.min(tlens)), "max": int(np.max(tlens))},
            "n_truncated": truncated,
            "gold_evidence_ids_lost_to_truncation": lost_gold,
        }
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--token-stats", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--max-prompt-tokens", type=int, default=1536)
    a = ap.parse_args()
    if a.check or not a.token_stats:
        man, _ = prepare(write=True)
        print(f"Split OK (no patient leakage). Manifest -> {MANIFEST_PATH.relative_to(ROOT)}")
        for s, info in man["splits"].items():
            print(f"  {s:5s}: {info['n_patients']} patients, {info['n_rows']} rows, "
                  f"{info['rows_with_empty_gold_evidence']} with empty gold evidence")
        print(f"  dropped rows: {man['dropped_rows']}")
        print(f"  trial overlap: {man['trial_overlap']}")
    if a.token_stats:
        st = token_stats(a.model, a.max_prompt_tokens)
        out = ROOT / "results" / "token_stats.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(st, indent=2))
        print(json.dumps(st, indent=2))


if __name__ == "__main__":
    main()
