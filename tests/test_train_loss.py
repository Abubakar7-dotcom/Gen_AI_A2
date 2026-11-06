"""B3: the memory-optimised loss (left padding + position_ids + logits_to_keep) must equal the
standard full-logits causal-LM loss computed on each example separately (no padding at all)."""
import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from src.data import load_split
from src.modeling import load_tokenizer
from src.train import collate, encode, length_grouped_batches, target_loss

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def setup():
    tok = load_tokenizer(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="sdpa")
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=4, target_modules=["q_proj", "v_proj"]))
    model.eval()
    recs = load_split("train")
    # pick examples with different prompt and target lengths so padding really happens
    data = [encode(tok, r, 1024) for r in recs[:40]]
    data = sorted(data, key=lambda d: len(d["input_ids"]))
    batch = [data[0], data[len(data) // 2], data[-1]]
    return tok, model, batch


@torch.no_grad()
def test_target_only_loss_matches_full_loss(setup):
    tok, model, batch = setup
    # reference: each example alone, full logits, HF's built-in shifted loss, token-weighted mean
    tot, n = 0.0, 0
    for ex in batch:
        ids = torch.tensor([ex["input_ids"]]); lab = torch.tensor([ex["labels"]])
        out = model(input_ids=ids, labels=lab)
        k = (lab[:, 1:] != -100).sum().item()
        tot += out.loss.item() * k; n += k
    ref = tot / n
    b = collate(batch, tok.pad_token_id)
    assert len({len(x["input_ids"]) for x in batch}) > 1  # padding is exercised
    got = target_loss(model, b).item()
    assert got == pytest.approx(ref, rel=1e-4, abs=1e-5)


def test_collate_left_pads_and_positions():
    batch = [{"input_ids": [5, 6, 7], "labels": [-100, 6, 7]}, {"input_ids": [8], "labels": [8]}]
    b = collate(batch, pad_id=0)
    assert b["input_ids"].tolist() == [[5, 6, 7], [0, 0, 8]]
    assert b["position_ids"].tolist() == [[0, 1, 2], [0, 0, 0]]
    assert b["labels"].tolist() == [[-100, 6, 7], [-100, -100, 8]]


def test_length_grouped_batches_cover_everything_once():
    lens = list(range(50))
    bs = length_grouped_batches(lens, 4, torch.Generator().manual_seed(0))
    flat = [i for b in bs for i in b]
    assert sorted(flat) == lens and all(len(b) <= 4 for b in bs)
