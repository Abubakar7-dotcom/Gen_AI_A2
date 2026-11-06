"""Deterministic benchmark/test inputs of an exact token length, built from real dataset text."""
from __future__ import annotations

import torch

from src.data import load_split, prompt_text


def make_input_ids(tok, n_tokens: int, device, split: str = "val") -> torch.Tensor:
    """Concatenate chat-formatted prompts of the split (fixed order) and cut to exactly n_tokens."""
    ids: list[int] = []
    for r in load_split(split):
        ids += tok(prompt_text(tok, r["sentences"], r["criterion_type"], r["criterion_text"]),
                   add_special_tokens=False)["input_ids"]
        if len(ids) >= n_tokens:
            break
    if len(ids) < n_tokens:
        raise ValueError(f"not enough text for {n_tokens} tokens")
    return torch.tensor([ids[:n_tokens]], device=device)


def record_input_ids(tok, rec: dict, device) -> torch.Tensor:
    p = prompt_text(tok, rec["sentences"], rec["criterion_type"], rec["criterion_text"])
    return torch.tensor([tok(p, add_special_tokens=False)["input_ids"]], device=device)
