"""Label sets, prompt template and output parsing/validation.

Model input  = numbered patient note + criterion type + criterion text (nothing else).
Model output = compact JSON: {"label": "<one allowed label>", "evidence": [<sentence ids>]}
"""
from __future__ import annotations

import json
import re

INCLUSION_LABELS = ["included", "not included", "not enough information", "not applicable"]
EXCLUSION_LABELS = ["excluded", "not excluded", "not enough information", "not applicable"]
ALL_LABELS = ["included", "not included", "excluded", "not excluded",
              "not enough information", "not applicable"]
LABELS_BY_TYPE = {"inclusion": INCLUSION_LABELS, "exclusion": EXCLUSION_LABELS}

# Dataset columns that must never reach the model input (model predictions / grading / gold).
FORBIDDEN_INPUT_COLUMNS = ["gpt4_explanation", "gpt4_sentences", "gpt4_eligibility",
                           "explanation_correctness", "expert_eligibility", "expert_sentences",
                           "training"]

_LABEL_HELP = {
    "inclusion": (
        '- "included": the note shows the patient meets this inclusion criterion.\n'
        '- "not included": the note shows the patient does NOT meet it.\n'
        '- "not enough information": the note does not contain enough information to decide.\n'
        '- "not applicable": the criterion does not apply to this patient.'
    ),
    "exclusion": (
        '- "excluded": the note shows the patient meets this exclusion criterion.\n'
        '- "not excluded": the note shows the patient does NOT meet this exclusion criterion.\n'
        '- "not enough information": the note does not contain enough information to decide.\n'
        '- "not applicable": the criterion does not apply to this patient.'
    ),
}

SYSTEM_TEMPLATE = (
    "You are a clinical trial eligibility screener. You are given a patient note whose "
    "sentences are numbered, and one {ctype} criterion of a clinical trial.\n"
    "Choose exactly one label:\n{label_help}\n"
    "Also cite the IDs of the patient sentences that support your decision "
    "(use [] if no sentence is relevant). Base the decision only on the note; "
    "absence of information is not evidence.\n"
    'Answer with JSON only: {{"label": "<label>", "evidence": [<ids>]}}'
)

USER_TEMPLATE = "Patient note:\n{note}\n\nCriterion type: {ctype}\nCriterion: {criterion}"


def number_sentences(sentences: list[str]) -> str:
    return "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))


def build_messages(sentences: list[str], criterion_type: str, criterion_text: str) -> list[dict]:
    """Chat messages for one (patient, criterion) pair. Only these three inputs are used."""
    ctype = criterion_type.strip().lower()
    if ctype not in LABELS_BY_TYPE:
        raise ValueError(f"criterion_type must be 'inclusion' or 'exclusion', got {criterion_type!r}")
    return [
        {"role": "system", "content": SYSTEM_TEMPLATE.format(ctype=ctype, label_help=_LABEL_HELP[ctype])},
        {"role": "user", "content": USER_TEMPLATE.format(
            note=number_sentences(sentences), ctype=ctype, criterion=criterion_text.strip())},
    ]


def format_target(label: str, evidence: list[int]) -> str:
    return json.dumps({"label": label, "evidence": sorted(set(int(i) for i in evidence))})


def parse_output(text: str, criterion_type: str, n_sentences: int) -> dict:
    """Parse and validate model output. Never silently repairs an invalid label.

    Returns: label (str|None), evidence (valid ids), parse_ok, label_valid,
    unsupported_ids (cited ids outside the note), raw.
    """
    out = {"label": None, "evidence": [], "parse_ok": False, "label_valid": False,
           "unsupported_ids": [], "raw": text}
    obj = None
    m = re.search(r"\{.*?\}", text, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = None
    if isinstance(obj, dict):
        out["parse_ok"] = True
        label = obj.get("label")
        ev = obj.get("evidence", [])
    else:
        # Fallback: look for a quoted label; evidence unknown.
        lm = re.search(r'"label"\s*:\s*"([^"]+)"', text)
        label, ev = (lm.group(1) if lm else None), []
    if isinstance(label, str):
        label = label.strip().lower()
        out["label"] = label
        out["label_valid"] = label in LABELS_BY_TYPE.get(criterion_type, [])
    ids = []
    if isinstance(ev, list):
        for x in ev:
            try:
                ids.append(int(x))
            except (TypeError, ValueError):
                continue
    valid = sorted({i for i in ids if 0 <= i < n_sentences})
    out["evidence"] = valid
    out["unsupported_ids"] = sorted({i for i in ids if not (0 <= i < n_sentences)})
    return out
