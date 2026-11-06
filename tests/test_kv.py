"""D1/D2: explicit KV-cache decoding vs the uncached reference (Qwen2.5-0.5B, fp32)."""
import pytest
import torch
from transformers import AutoModelForCausalLM, DynamicCache

from src.data import load_split
from src.decoding.inputs import make_input_ids, record_input_ids
from src.decoding.kv_cache import (causal_mask, eos_set, first_mismatch, forward_step, greedy_kv,
                                   greedy_no_cache)
from src.modeling import greedy_generation_config, load_tokenizer, pick_device

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def mt():
    dev = pick_device()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="sdpa").to(dev).eval()
    return model, load_tokenizer(MODEL), dev


def test_causal_mask():
    m = causal_mask(3, 0, "cpu")[0, 0]
    assert m.tolist() == [[True, False, False], [True, True, False], [True, True, True]]
    d = causal_mask(1, 4, "cpu")[0, 0]          # decode step: one query sees all 5 keys
    assert d.shape == (1, 5) and d.all()
    v = causal_mask(2, 3, "cpu")[0, 0]          # verification block (speculative decoding)
    assert v.tolist() == [[True] * 4 + [False], [True] * 5]


@torch.no_grad()
def test_explicit_mask_and_positions_match_library_forward(mt):
    model, tok, dev = mt
    ids = make_input_ids(tok, 64, dev)
    ref = model(input_ids=ids).logits[0].float()
    got = forward_step(model, ids, None, 0, logits_to_keep=0)  # 0 = keep all positions
    assert torch.allclose(ref, got, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("n_in,n_out", [(64, 16), (256, 32), (512, 48)])
def test_kv_tokens_equal_uncached(mt, n_in, n_out):
    model, tok, dev = mt
    ids = make_input_ids(tok, n_in, dev)
    ref = greedy_no_cache(model, ids, n_out)   # EOS ignored -> fixed output length
    kv = greedy_kv(model, ids, n_out)
    assert len(kv.tokens) == n_out
    i = first_mismatch(ref.tokens, kv.tokens)
    assert i is None, f"diverged at step {i}; top-2 margins ref={ref.margins[i]:.4f} kv={kv.margins[i]:.4f}"


def test_kv_matches_hf_generate_on_task_prompt(mt):
    model, tok, dev = mt
    ids = record_input_ids(tok, load_split("val")[0], dev)
    model.generation_config = greedy_generation_config(tok, 40)
    hf = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids))[0, ids.shape[1]:].tolist()
    kv = greedy_kv(model, ids, 40, eos_ids=eos_set(tok)).tokens
    assert kv == hf


def test_cache_growth_and_size(mt):
    model, tok, dev = mt
    ids = make_input_ids(tok, 100, dev)
    r = greedy_kv(model, ids, 10)
    # after prefill: 100 cached; each of the 9 decode steps feeds one token (+1); the 10th token is never fed
    assert r.cache_len == list(range(100, 110))
    c = model.config
    per_tok = 2 * c.num_hidden_layers * c.num_key_value_heads * (c.hidden_size // c.num_attention_heads) * 4  # fp32
    assert r.cache_bytes == per_tok * 109


@torch.no_grad()
def test_wrong_position_changes_logits(mt):
    """Positional handling matters: feeding the next token with position 0 instead of past_len
    gives different logits (RoPE of the query no longer matches the cached keys)."""
    model, tok, dev = mt
    ids = make_input_ids(tok, 50, dev)
    cache = DynamicCache(config=model.config)
    forward_step(model, ids[:, :-1], cache, 0)
    right = forward_step(model, ids[:, -1:], cache, past_len=49)[-1]
    cache.crop(-1)
    masks = {t: causal_mask(1, 49, dev) for t in set(model.config.layer_types)}
    wrong = model(input_ids=ids[:, -1:], position_ids=torch.zeros(1, 1, dtype=torch.long, device=dev),
                  attention_mask=masks, past_key_values=cache, use_cache=True).logits[0, -1].float()
    assert (right - wrong).abs().max() > 1.0
