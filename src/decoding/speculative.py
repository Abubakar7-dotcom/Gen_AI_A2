"""Greedy speculative decoding written against raw forward passes (no `generate()`).

A small draft model proposes k tokens; the target model checks all of them in ONE forward pass and
keeps the longest correct prefix. Because acceptance is "draft token == target argmax", the output
is exactly the target's own greedy output; the draft only changes how many target forward passes
are needed.

Invariant kept between rounds (both caches):
    cache holds every token of (prompt + output) EXCEPT the last emitted token `last`.
The draft cache may lag by one more token (the k-th draft token after a fully accepted round);
those not-yet-cached tokens are kept in `pending` and fed at the start of the next draft call.

One round, with L = target cache length and drafts d1..dk:
  1. Draft:  feed pending (= [last] or [dk_prev, last]) then d1..d(k-1), one token at a time,
             taking argmax each step -> d1..dk.                                (k draft forwards)
  2. Verify: target forward over [last, d1, ..., dk] at positions L..L+k with a causal mask;
             logits at offset i give g_i = target argmax after seeing last, d1..d_i.
                                                                               (1 target forward)
  3. Accept: n = length of the longest prefix with d_(i+1) == g_i. Emit d1..dn and g_n
             (g_n is the correction if n < k, a free "bonus" token if n == k). n+1 tokens/round.
  4. Sync:   target cache now holds L + k + 1 entries; keep L + 1 + n  -> crop(-(k - n)).
             draft cache holds L + k (dk was never fed); keep L + 1 + n:
               n < k  -> crop(-(k - 1 - n));  n == k -> dk becomes pending for the next round.
EOS: tokens after an emitted EOS are discarded and decoding stops.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from transformers import DynamicCache

from src.decoding.kv_cache import forward_step, mem_now, mem_peak, reset_peak, sync


def assert_same_tokenizer(tok_target, tok_draft):
    """Speculative decoding compares token ids, so both models must share one vocabulary."""
    assert tok_target.get_vocab() == tok_draft.get_vocab(), "draft and target vocabularies differ"
    assert tok_target.all_special_tokens == tok_draft.all_special_tokens, "special tokens differ"
    assert tok_target.eos_token_id == tok_draft.eos_token_id and tok_target.pad_token_id == tok_draft.pad_token_id


@dataclass
class SpecResult:
    tokens: list[int]
    prefill_s: float
    decode_s: float
    peak_mem: int | None
    k: int
    rounds: int = 0
    proposed: int = 0                       # draft tokens offered to the target
    accepted: int = 0                       # draft tokens the target agreed with
    accepted_per_round: list[int] = field(default_factory=list)
    target_forwards: int = 0                # incl. prefill
    draft_forwards: int = 0                 # incl. prefill
    draft_s: float = 0.0
    verify_s: float = 0.0

    @property
    def total_s(self) -> float:
        return self.prefill_s + self.decode_s

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else float("nan")

    @property
    def tokens_per_round(self) -> float:
        """Mean tokens emitted per verification round (accepted + 1 correction/bonus)."""
        return (len(self.tokens) - 1) / self.rounds if self.rounds else float("nan")

    @property
    def tok_per_s(self) -> float:
        return len(self.tokens) / self.total_s if self.total_s > 0 else float("nan")


def _argmax(logits: torch.Tensor) -> list[int]:
    return logits.argmax(-1).tolist()


@torch.no_grad()
def speculative_greedy(target, draft, prompt_ids: torch.Tensor, max_new_tokens: int, k: int = 4,
                       eos_ids: set[int] | None = None) -> SpecResult:
    assert k >= 1
    dev = prompt_ids.device
    timed = dev.type in ("cuda", "mps")   # per-phase timers need syncs; skip them on CPU
    reset_peak(dev); sync(dev)
    tc, dc = DynamicCache(config=target.config), DynamicCache(config=draft.config)
    peak = mem_now(dev)

    def track():
        nonlocal peak
        cur = mem_now(dev)
        peak = max(peak, cur) if cur is not None else None

    t0 = time.perf_counter()
    # Prefill. The target's prefill already yields the first output token. The draft only needs the
    # prompt in its cache (all but the last prompt token would also do; we keep the invariant simple).
    first = _argmax(forward_step(target, prompt_ids, tc, 0)[-1:])[0]
    forward_step(draft, prompt_ids, dc, 0)
    sync(dev); t_first = time.perf_counter()
    track()
    res = SpecResult([first], t_first - t0, 0.0, None, k, target_forwards=1, draft_forwards=1)
    out, pending = res.tokens, [first]
    done = bool(eos_ids and first in eos_ids)

    while not done and len(out) < max_new_tokens:
        remaining = max_new_tokens - len(out)
        kk = min(k, remaining - 1)          # a round emits up to kk+1 tokens
        L = tc.get_seq_length()             # positions 0..L-1 cached; `last` goes to position L
        last = out[-1]
        if kk == 0:                         # one token left: plain target step
            g = _argmax(forward_step(target, torch.tensor([[last]], device=dev), tc, L)[-1:])[0]
            res.target_forwards += 1
            out.append(g); pending.append(g)
            done = bool(eos_ids and g in eos_ids)
            continue

        # 1. draft kk tokens
        ts = time.perf_counter()
        drafts, feed = [], pending
        for _ in range(kk):
            d = _argmax(forward_step(draft, torch.tensor([feed], device=dev), dc, dc.get_seq_length())[-1:])[0]
            res.draft_forwards += 1
            drafts.append(d); feed = [d]
        if timed:
            sync(dev)
        tv = time.perf_counter(); res.draft_s += tv - ts

        # 2. verify all of them with one target forward
        block = torch.tensor([[last] + drafts], device=dev)
        g = _argmax(forward_step(target, block, tc, L, logits_to_keep=kk + 1))
        res.target_forwards += 1
        if timed:
            sync(dev)
        res.verify_s += time.perf_counter() - tv
        track()

        # 3. accept the longest matching prefix, then the target's own next token
        n = 0
        while n < kk and drafts[n] == g[n]:
            n += 1
        new = drafts[:n] + [g[n]]
        res.rounds += 1; res.proposed += kk; res.accepted += n; res.accepted_per_round.append(n)

        # 4. roll both caches back to "everything except the new last token"
        tc.crop(-(kk - n))                  # crop(0) is a no-op
        if n < kk:
            dc.crop(-(kk - 1 - n))
            pending = [new[-1]]
        else:
            pending = [drafts[-1], new[-1]]   # dk was proposed but never fed to the draft
        assert tc.get_seq_length() == L + 1 + n
        assert dc.get_seq_length() + len(pending) - 1 == L + 1 + n

        if eos_ids:
            for i, t in enumerate(new):
                if t in eos_ids:
                    new, done = new[:i + 1], True
                    break
        out.extend(new)

    sync(dev)
    res.decode_s = time.perf_counter() - t_first
    res.peak_mem = mem_peak(dev, peak)
    return res
