"""Greedy autoregressive decoding written against raw model forward passes (no `generate()`).

Two paths, same model, same inputs:
  * greedy_no_cache : reference. Every step re-runs the WHOLE sequence (prompt + generated so far)
                      with use_cache=False and takes argmax of the last position. Cost per step grows
                      with sequence length, total cost ~ O(n^2) token-forwards.
  * greedy_kv       : prefill once over the prompt into an explicit DynamicCache, then feed ONLY the
                      newest token each step; keys/values of earlier tokens are read from the cache.

What this file makes explicit (instead of `use_cache=True` inside generate):
  Positions  Qwen2 applies RoPE to q and k *before* k is written to the cache, so cached keys are
             already rotated with their own absolute positions. The new token must therefore get
             position_ids = past_len (number of cached tokens), not 0. Wrong positions do not crash;
             they silently change the outputs (tests/test_kv.py checks this).
  Masking    We build the boolean attention mask ourselves (True = may attend) and pass it to the model
             as {"full_attention": mask}, bypassing the library's mask builder. Query i at absolute
             position past_len+i may attend key j iff j <= past_len+i. During prefill this is the usual
             lower-triangular mask; in a decode step q_len=1 and the single query may attend every
             cached key, so the mask row is all True: causality is enforced by the cache containing
             only the past.
  Cache      DynamicCache.update concatenates new K/V per layer along the sequence axis. After each
             step we assert the cached length grew by exactly the number of fed tokens. Memory per
             cached token = 2 (K,V) x layers x kv_heads x head_dim x bytes/elem
             (Qwen2.5-1.5B, fp16: 2*28*2*128*2 = 28,672 B ~ 28 KB/token; GQA keeps it small).
  Numerics   Cached and uncached paths compute the same math but with different matrix shapes, so
             kernels may sum in a different order. In fp32 the tokens agree; in fp16/bf16 a step whose
             top-2 logits are nearly tied can flip. We record the top-1/top-2 margin per step so that
             any mismatch can be explained.

Batch size 1 (one patient-criterion pair at a time), which is how the application uses it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from transformers import DynamicCache


# ------------------------------------------------------------------ device helpers
def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def reset_peak(device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def mem_now(device: torch.device) -> int | None:
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device)
    if device.type == "mps":
        return torch.mps.current_allocated_memory()
    return None


def mem_peak(device: torch.device, sampled: int | None) -> int | None:
    """CUDA: true allocator peak. MPS: max of samples taken after each forward pass (misses
    transient buffers inside a forward, so it is a lower bound). CPU: not measured."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device)
    return sampled


# ------------------------------------------------------------------ core pieces
def causal_mask(q_len: int, past_len: int, device) -> torch.Tensor:
    """Boolean [1, 1, q_len, past_len + q_len]; True where query may attend key."""
    q_pos = torch.arange(q_len, device=device)[:, None] + past_len
    k_pos = torch.arange(past_len + q_len, device=device)[None, :]
    return (k_pos <= q_pos)[None, None]


def forward_step(model, input_ids: torch.Tensor, cache: DynamicCache | None, past_len: int,
                 logits_to_keep: int = 1) -> torch.Tensor:
    """One forward pass over `input_ids` (shape [1, q_len]) whose first token sits at absolute
    position `past_len`. Uses/updates `cache` if given. Returns logits of the last
    `logits_to_keep` positions, shape [q_len_kept, vocab] (float32)."""
    q_len = input_ids.shape[1]
    dev = input_ids.device
    position_ids = torch.arange(past_len, past_len + q_len, device=dev)[None]
    mask = causal_mask(q_len, past_len, dev)
    masks = {t: mask for t in set(model.config.layer_types)}
    before = cache.get_seq_length() if cache is not None else None
    out = model(input_ids=input_ids, position_ids=position_ids, attention_mask=masks,
                past_key_values=cache, use_cache=cache is not None, logits_to_keep=logits_to_keep)
    if cache is not None:
        assert cache.get_seq_length() == before + q_len, "cache did not grow by the number of fed tokens"
    return out.logits[0].float()


def cache_nbytes(cache: DynamicCache) -> int:
    return sum(l.keys.numel() * l.keys.element_size() + l.values.numel() * l.values.element_size()
               for l in cache.layers if l.is_initialized)


@dataclass
class GenResult:
    tokens: list[int]
    prefill_s: float
    decode_s: float
    peak_mem: int | None
    margins: list[float] = field(default_factory=list)   # top1 - top2 logit per generated token
    cache_len: list[int] = field(default_factory=list)   # cached tokens after each step (KV path)
    cache_bytes: int | None = None

    @property
    def total_s(self) -> float:
        return self.prefill_s + self.decode_s

    @property
    def decode_tok_per_s(self) -> float:
        n = len(self.tokens) - 1  # first token comes out of prefill
        return n / self.decode_s if self.decode_s > 0 and n > 0 else float("nan")

    @property
    def tok_per_s(self) -> float:
        return len(self.tokens) / self.total_s if self.total_s > 0 else float("nan")


def _pick(logits_row: torch.Tensor) -> tuple[int, float]:
    top = torch.topk(logits_row, 2)
    return int(top.indices[0]), float(top.values[0] - top.values[1])


# ------------------------------------------------------------------ decoders
@torch.no_grad()
def greedy_no_cache(model, prompt_ids: torch.Tensor, max_new_tokens: int,
                    eos_ids: set[int] | None = None) -> GenResult:
    """Reference: recompute the full sequence every step. 'prefill' = the first step."""
    dev = prompt_ids.device
    reset_peak(dev); sync(dev)
    ids, toks, margins, peak = prompt_ids, [], [], mem_now(dev)
    t0 = time.perf_counter(); t_first = None
    for step in range(max_new_tokens):
        logits = forward_step(model, ids, None, 0)
        nxt, m = _pick(logits[-1])
        toks.append(nxt); margins.append(m)
        if step == 0:
            sync(dev); t_first = time.perf_counter()
        cur = mem_now(dev)
        peak = max(peak, cur) if cur is not None else None
        if eos_ids and nxt in eos_ids:
            break
        ids = torch.cat([ids, torch.tensor([[nxt]], device=dev)], dim=1)
    sync(dev); t1 = time.perf_counter()
    return GenResult(toks, t_first - t0, t1 - t_first, mem_peak(dev, peak), margins)


@torch.no_grad()
def greedy_kv(model, prompt_ids: torch.Tensor, max_new_tokens: int,
              eos_ids: set[int] | None = None, cache: DynamicCache | None = None) -> GenResult:
    """Prefill once, then feed one token per step and reuse the cache."""
    dev = prompt_ids.device
    reset_peak(dev); sync(dev)
    cache = cache if cache is not None else DynamicCache(config=model.config)
    peak = mem_now(dev)
    t0 = time.perf_counter()
    # prefill: whole prompt, causal mask, positions 0..P-1; only the last position's logits are needed
    logits = forward_step(model, prompt_ids, cache, past_len=0)
    nxt, m = _pick(logits[-1])
    sync(dev); t_first = time.perf_counter()
    toks, margins, lens = [nxt], [m], [cache.get_seq_length()]
    cur = mem_now(dev)
    peak = max(peak, cur) if cur is not None else None
    while len(toks) < max_new_tokens and not (eos_ids and nxt in eos_ids):
        past = cache.get_seq_length()
        logits = forward_step(model, torch.tensor([[nxt]], device=dev), cache, past_len=past)
        nxt, m = _pick(logits[-1])
        toks.append(nxt); margins.append(m); lens.append(cache.get_seq_length())
        cur = mem_now(dev)
        peak = max(peak, cur) if cur is not None else None
    sync(dev); t1 = time.perf_counter()
    return GenResult(toks, t_first - t0, t1 - t_first, mem_peak(dev, peak), margins, lens, cache_nbytes(cache))


def first_mismatch(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def eos_set(tok) -> set[int]:
    return {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
