"""Screening pipeline: note + criteria list -> per-criterion label, cited evidence, explanation,
and a rule-based overall preliminary outcome. Used by the app and by the error analysis.

Per criterion the fine-tuned model returns {"label", "evidence"}; everything shown to the user
beyond that is derived deterministically:
  * evidence text  = the cited sentences copied verbatim from the note (ids outside the note are
                     reported as unsupported citations, never shown as evidence);
  * explanation    = a template filled with the label and the cited sentences. The dataset has no
                     expert explanations and GPT-4 explanations must not be used, so the system does
                     not generate free-text rationales that could state things the note does not say;
  * missing info   = criteria labelled "not enough information" (or unresolved outputs);
  * overall        = rule-based aggregation (see `aggregate`), labelled as preliminary in the UI.
Missing evidence is never treated as proof of eligibility or ineligibility.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.data import fit_to_budget, prompt_text
from src.schema import LABELS_BY_TYPE, parse_output

ROOT = Path(__file__).resolve().parents[1]
APP_VERSION = "1.0.0"
MAX_PROMPT_TOKENS = 1536     # longest training prompt was 621 tokens; longer notes are truncated by whole sentences
MAX_NEW_TOKENS = 64
DISCLAIMER = "Educational screening prototype — not a clinical enrollment decision."


# ------------------------------------------------------------------ input handling
class InputError(ValueError):
    """Raised for user-fixable input problems; the message is shown in the UI."""


_NUMBERED = re.compile(r"^\s*(\d+)[.)]\s+(.*\S)\s*$")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def segment_note(text: str) -> list[str]:
    """Return the note as a list of sentences.
    Already-numbered notes ("0. ...", "1. ..." one per line, starting at 0 or 1) keep their sentences;
    otherwise the text is split on line breaks and sentence-final punctuation."""
    if not text or not text.strip():
        raise InputError("The patient note is empty.")
    lines = [l for l in text.strip().splitlines() if l.strip()]
    m = [_NUMBERED.match(l) for l in lines]
    if all(m):
        nums = [int(x.group(1)) for x in m]
        if nums == list(range(nums[0], nums[0] + len(nums))) and nums[0] in (0, 1):
            return [x.group(2) for x in m]
    sents = []
    for l in lines:
        sents += [s.strip() for s in _SENT_SPLIT.split(l.strip()) if s.strip()]
    return sents


_HEADER = re.compile(r"^\s*(inclusion|exclusion)(\s+criteria)?\s*:?\s*(.*)$", re.I)
_BULLET = re.compile(r"^\s*(?:[-*•·]|\d+[.)])\s*")


def parse_criteria(text: str) -> list[tuple[str, str]]:
    """'Inclusion:' / 'Exclusion:' sections, one criterion per line (bullets/numbers optional).
    Returns [(criterion_type, criterion_text), ...] in order."""
    if not text or not text.strip():
        raise InputError("The criteria box is empty. Add an 'Inclusion:' and/or 'Exclusion:' section.")
    out, cur = [], None
    for raw in text.splitlines():
        if not raw.strip():
            continue
        h = _HEADER.match(raw)
        if h and (h.group(3) == "" or raw.strip().lower().startswith(("inclusion:", "exclusion:", "inclusion criteria:", "exclusion criteria:"))):
            cur = h.group(1).lower()
            rest = h.group(3).strip()
            if rest:
                out.append((cur, _BULLET.sub("", rest).strip()))
            continue
        if cur is None:
            raise InputError(f"Criterion before any section header: {raw.strip()[:60]!r}. "
                             "Start the list with a line 'Inclusion:' or 'Exclusion:'.")
        crit = _BULLET.sub("", raw).strip()
        if crit:
            out.append((cur, crit))
    if not out:
        raise InputError("No criteria found under the 'Inclusion:' / 'Exclusion:' headers.")
    return out


def criteria_to_text(criteria: list[tuple[str, str]]) -> str:
    parts = []
    for ctype in ("inclusion", "exclusion"):
        items = [c for t, c in criteria if t == ctype]
        if items:
            parts.append(f"{ctype.capitalize()}:\n" + "\n".join(f"- {c}" for c in items))
    return "\n\n".join(parts)


# ------------------------------------------------------------------ explanations / aggregation
def _quote(ids: list[int], sents: list[str]) -> str:
    return "; ".join(f"[{i}] “{sents[i]}”" for i in ids)


def explain(label: str | None, ctype: str, criterion: str, ids: list[int], sents: list[str], status: str) -> str:
    if status == "unparseable":
        return "The model output could not be read as a label, so this criterion is unresolved. Check it manually."
    if status == "invalid_label":
        return f"The model returned “{label}”, which is not an allowed label for a {ctype} criterion, so this criterion is unresolved."
    ev = f" Evidence: {_quote(ids, sents)}." if ids else ""
    no_ev = " No sentence was cited, so verify this manually." if not ids else ""
    if label == "not enough information":
        return ("The note does not contain enough information to decide this criterion; the missing information "
                f"must be obtained before screening it.{(' Related sentences: ' + _quote(ids, sents) + '.') if ids else ''}")
    if label == "not applicable":
        return f"The criterion does not apply to this patient.{ev}"
    text = {"included": "The patient appears to MEET this inclusion criterion.",
            "not included": "The patient appears NOT to meet this inclusion criterion.",
            "excluded": "The patient appears to MEET this exclusion criterion, which would exclude them.",
            "not excluded": "The patient does not appear to meet this exclusion criterion."}[label]
    return text + ev + no_ev


AGGREGATION_RULE = ("Likely ineligible if any inclusion criterion is 'not included' or any exclusion criterion is "
                    "'excluded'; otherwise Needs more information if any criterion is 'not enough information' or "
                    "unresolved; otherwise Potentially eligible. 'Not applicable' criteria are ignored.")


def aggregate(rows: list[dict]) -> dict:
    """Rule-based preliminary outcome (not a clinical decision):
        any inclusion 'not included' or exclusion 'excluded'          -> Likely ineligible
        else any 'not enough information' or unresolved output        -> Needs more information
        else                                                          -> Potentially eligible
    'not applicable' criteria are ignored."""
    failing = [r["index"] for r in rows if r["label"] in ("not included", "excluded")]
    missing = [r["index"] for r in rows if r["label"] == "not enough information" or r["status"] != "ok"]
    if failing:
        outcome, why = "Likely ineligible", f"criteria {', '.join(map(str, failing))} indicate ineligibility"
    elif missing:
        outcome, why = "Needs more information", f"criteria {', '.join(map(str, missing))} cannot be decided from the note"
    else:
        outcome, why = "Potentially eligible", "no criterion indicates ineligibility and none is missing information"
    return {"outcome": outcome, "reason": why, "failing": failing, "missing": missing,
            "rule": AGGREGATION_RULE}


# ------------------------------------------------------------------ screening
def git_version() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


@dataclass
class Screener:
    model: object
    tok: object
    model_id: str
    adapter: str | None
    device: torch.device
    draft: object | None = None
    draft_id: str | None = None
    meta: dict = field(default_factory=dict)

    def predict(self, sents: list[str], ctype: str, criterion: str, mode: str = "kv", k: int = 4) -> dict:
        from src.decoding.kv_cache import eos_set, greedy_kv
        from src.decoding.speculative import speculative_greedy
        kept, dropped = fit_to_budget(self.tok, {"sentences": sents, "criterion_type": ctype, "criterion_text": criterion},
                                      MAX_PROMPT_TOKENS)
        p = prompt_text(self.tok, kept, ctype, criterion)
        ids = torch.tensor([self.tok(p, add_special_tokens=False)["input_ids"]], device=self.device)
        eos = eos_set(self.tok)
        if mode == "speculative":
            if self.draft is None:
                raise InputError("Speculative mode needs a draft model, but none is loaded.")
            r = speculative_greedy(self.model, self.draft, ids, MAX_NEW_TOKENS, k=k, eos_ids=eos)
            stats = {"acceptance_rate": r.acceptance_rate, "tokens_per_round": r.tokens_per_round}
        else:
            r = greedy_kv(self.model, ids, MAX_NEW_TOKENS, eos_ids=eos)
            stats = {}
        text = self.tok.decode(r.tokens, skip_special_tokens=True)
        parsed = parse_output(text, ctype, len(kept))
        return {"parsed": parsed, "raw": text, "prompt_tokens": ids.shape[1], "new_tokens": len(r.tokens),
                "seconds": r.total_s, "truncated_ids": dropped, **stats}

    def screen(self, note: str, criteria_text: str, mode: str = "kv", k: int = 4, progress=None) -> dict:
        t0 = time.perf_counter()
        sents = segment_note(note)
        criteria = parse_criteria(criteria_text)
        rows, warnings = [], []
        for i, (ctype, crit) in enumerate(criteria, 1):
            if progress:
                progress(i - 1, len(criteria), crit)
            pr = self.predict(sents, ctype, crit, mode, k)
            p = pr["parsed"]
            status = "ok" if p["label_valid"] else ("unparseable" if p["label"] is None else "invalid_label")
            label = p["label"] if status == "ok" else None
            ids = p["evidence"]
            if pr["truncated_ids"]:
                warnings.append(f"Criterion {i}: the note was too long; sentences {pr['truncated_ids']} were not shown to the model.")
            if p["unsupported_ids"]:
                warnings.append(f"Criterion {i}: the model cited sentence ids {p['unsupported_ids']} that do not exist; they were ignored.")
            rows.append({"index": i, "criterion_type": ctype, "criterion": crit, "label": label, "status": status,
                         "raw_label": p["label"], "evidence_ids": ids, "evidence_text": [sents[j] for j in ids],
                         "unsupported_ids": p["unsupported_ids"],
                         "explanation": explain(p["label"], ctype, crit, ids, sents, status),
                         "missing_information": label == "not enough information" or status != "ok",
                         "raw_output": pr["raw"], "prompt_tokens": pr["prompt_tokens"], "new_tokens": pr["new_tokens"],
                         "seconds": round(pr["seconds"], 3),
                         **{k2: pr[k2] for k2 in ("acceptance_rate", "tokens_per_round") if k2 in pr}})
        overall = aggregate(rows)
        prov = {"app_version": APP_VERSION, "code_version": git_version(), "model": self.model_id,
                "adapter": self.adapter, "draft_model": self.draft_id if mode == "speculative" else None,
                "decoding": {"mode": mode, "strategy": "greedy", "k": k if mode == "speculative" else None,
                             "max_new_tokens": MAX_NEW_TOKENS, "max_prompt_tokens": MAX_PROMPT_TOKENS},
                "device": str(self.device), "dtype": str(next(self.model.parameters()).dtype),
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "input_sha256": hashlib.sha256((note + "\x00" + criteria_text).encode()).hexdigest(),
                "runtime_s": round(time.perf_counter() - t0, 2), **self.meta}
        return {"disclaimer": DISCLAIMER, "sentences": sents, "criteria": rows, "overall": overall,
                "warnings": warnings, "provenance": prov}


def load_screener(model_id: str, adapter: str | None, draft_id: str | None = None, draft_adapter: str | None = None,
                  device: str | None = None) -> Screener:
    from src.modeling import load_model, pick_device
    from src.decoding.speculative import assert_same_tokenizer
    dev = pick_device(device)
    model, tok = load_model(model_id, adapter, device=dev)
    draft = None
    if draft_id:
        draft, dtok = load_model(draft_id, draft_adapter, device=dev)
        assert_same_tokenizer(tok, dtok)
    def label(a):  # repo-relative path (no user home directory in exported reports) or HF repo id
        try:
            return str(Path(a).resolve().relative_to(ROOT)) if a and Path(a).exists() else a
        except ValueError:
            return a
    return Screener(model, tok, model_id, label(adapter), dev, draft,
                    f"{draft_id} + {label(draft_adapter)}" if draft_adapter else draft_id)


# ------------------------------------------------------------------ export
_BADGE = {"included": "#1a7f37", "not excluded": "#1a7f37", "not included": "#cf222e", "excluded": "#cf222e",
          "not enough information": "#9a6700", "not applicable": "#57606a", None: "#8250df"}


def report_html(rep: dict) -> str:
    e = html.escape
    rows = []
    for r in rep["criteria"]:
        lab = r["label"] or ("unparseable" if r["status"] == "unparseable" else f"invalid: {r['raw_label']}")
        ev = "<br>".join(f"<b>[{i}]</b> {e(t)}" for i, t in zip(r["evidence_ids"], r["evidence_text"])) or "<i>none cited</i>"
        rows.append(f"<tr><td>{r['index']}</td><td>{e(r['criterion_type'])}</td><td>{e(r['criterion'])}</td>"
                    f"<td><span style='background:{_BADGE.get(r['label'], '#8250df')};color:#fff;padding:2px 6px;border-radius:4px'>{e(lab)}</span></td>"
                    f"<td>{ev}</td><td>{e(r['explanation'])}</td></tr>")
    note = "".join(f"<li value='{i}'>{e(s)}</li>" for i, s in enumerate(rep["sentences"]))
    warn = "".join(f"<li>{e(w)}</li>" for w in rep["warnings"]) or "<li>none</li>"
    o = rep["overall"]
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Screening report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#1f2328}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #d0d7de;padding:6px;vertical-align:top;font-size:14px}}
th{{background:#f6f8fa}}.disc{{background:#fff8c5;border:1px solid #d4a72c;padding:8px;border-radius:6px}}
ol{{font-size:14px}}pre{{background:#f6f8fa;padding:8px;font-size:12px;overflow:auto}}</style></head><body>
<p class="disc"><b>{e(rep['disclaimer'])}</b></p>
<h1>Trial screening report</h1>
<h2>Preliminary outcome: {e(o['outcome'])}</h2><p>{e(o['reason'])}. <i>Rule:</i> {e(o['rule'])}</p>
<h3>Criteria</h3><table><tr><th>#</th><th>Type</th><th>Criterion</th><th>Label</th><th>Evidence (verbatim)</th><th>Explanation</th></tr>
{''.join(rows)}</table>
<h3>Warnings</h3><ul>{warn}</ul>
<h3>Patient note (numbered)</h3><ol start="0">{note}</ol>
<h3>Provenance</h3><pre>{e(json.dumps(rep['provenance'], indent=2))}</pre></body></html>"""


def save_report(rep: dict, out_dir: Path | None = None) -> tuple[Path, Path]:
    out_dir = out_dir or ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{rep['provenance']['input_sha256'][:8]}"
    jp, hp = out_dir / f"{stem}.json", out_dir / f"{stem}.html"
    jp.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    hp.write_text(report_html(rep), encoding="utf-8")
    return jp, hp
