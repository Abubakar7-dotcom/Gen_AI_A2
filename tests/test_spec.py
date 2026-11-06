"""F2/F3: greedy speculative decoding must reproduce the target-only greedy KV output exactly.

Target: Qwen2.5-0.5B-Instruct (fp32, fast enough for a laptop test). Drafts:
  * the fine-tuned 0.5B adapter merged into the base   -> realistic partial acceptance
  * the target itself                                  -> every draft accepted (bonus-token path)
  * a tiny randomly initialised Qwen2 with the same vocab -> almost every draft rejected
Together these exercise every branch of the accept / crop logic.
"""
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

from src.data import load_split
from src.decoding.inputs import make_input_ids, record_input_ids
from src.decoding.kv_cache import eos_set, first_mismatch, greedy_kv
from src.decoding.speculative import assert_same_tokenizer, speculative_greedy
from src.modeling import load_model, load_tokenizer, pick_device

ROOT = Path(__file__).resolve().parents[1]
BASE = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = next((p for p in (ROOT / "adapters" / "ft_0p5b_r16", ROOT / "outputs" / "ft_0p5b_r16" / "best") if p.exists()),
               ROOT / "adapters" / "ft_0p5b_r16")


@pytest.fixture(scope="module")
def setup():
    dev = pick_device()
    tok = load_tokenizer(BASE)
    target = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32, attn_implementation="sdpa").to(dev).eval()
    drafts = {"self": target}
    if ADAPTER.exists():
        drafts["ft_0p5b"], _ = load_model(BASE, str(ADAPTER), device=dev, dtype=torch.float32)
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=target.config.vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=4096)
    drafts["random_tiny"] = Qwen2ForCausalLM(cfg).to(dev).eval()
    return target, drafts, tok, dev


def test_tokenizers_match():
    assert_same_tokenizer(load_tokenizer(BASE), load_tokenizer("Qwen/Qwen2.5-1.5B-Instruct"))


@pytest.mark.parametrize("draft_name", ["self", "ft_0p5b", "random_tiny"])
@pytest.mark.parametrize("k", [1, 3, 5])
def test_spec_equals_target_greedy_fixed_length(setup, draft_name, k):
    target, drafts, tok, dev = setup
    if draft_name not in drafts:
        pytest.skip("fine-tuned adapter not present (outputs/ft_0p5b_r16/best)")
    ids = make_input_ids(tok, 200, dev)
    ref = greedy_kv(target, ids, 40).tokens                       # EOS ignored -> exactly 40 tokens
    spec = speculative_greedy(target, drafts[draft_name], ids, 40, k=k)
    assert len(spec.tokens) == 40
    assert first_mismatch(ref, spec.tokens) is None
    assert 0 <= spec.accepted <= spec.proposed
    if draft_name == "self":
        assert spec.acceptance_rate == 1.0
        assert spec.target_forwards < 40                              # fewer target passes than plain greedy
    if draft_name == "random_tiny":
        assert spec.acceptance_rate < 0.2


@pytest.mark.parametrize("draft_name", ["ft_0p5b", "random_tiny"])
def test_spec_equals_target_greedy_on_task_prompts_with_eos(setup, draft_name):
    target, drafts, tok, dev = setup
    if draft_name not in drafts:
        pytest.skip("fine-tuned adapter not present")
    eos = eos_set(tok)
    for rec in load_split("val")[:4]:
        ids = record_input_ids(tok, rec, dev)
        ref = greedy_kv(target, ids, 64, eos_ids=eos).tokens
        spec = speculative_greedy(target, drafts[draft_name], ids, 64, k=4, eos_ids=eos)
        assert spec.tokens == ref


def test_round_accounting(setup):
    target, drafts, tok, dev = setup
    ids = make_input_ids(tok, 100, dev)
    r = speculative_greedy(target, drafts["self"], ids, 21, k=4)
    # first token from prefill, then 4 full rounds of 4 accepted + 1 bonus = 20 more
    assert r.rounds == 4 and r.accepted_per_round == [4, 4, 4, 4] and len(r.tokens) == 21
    assert r.target_forwards == 1 + 4 and r.tokens_per_round == 5.0
