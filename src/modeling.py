"""Model / tokenizer loading and batched greedy generation used for training-time validation
and for the before/after evaluation.

NOTE: Qwen2.5-Instruct ships a generation_config with do_sample=True, temperature=0.7,
top_p=0.8, top_k=20 and repetition_penalty=1.1. All evaluation here overrides it with plain
greedy decoding (GREEDY_SETTINGS) so results are deterministic and comparable.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from src.data import prompt_text

GREEDY_SETTINGS = {"do_sample": False, "num_beams": 1, "repetition_penalty": 1.0}


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def native_bf16() -> bool:
    """True only for GPUs with hardware bf16 (compute capability >= 8: Ampere, Ada, Hopper...).
    torch.cuda.is_bf16_supported() defaults to including_emulation=True and reports True on a T4,
    where bf16 is emulated and slow."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def default_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16 if native_bf16() else torch.float16
    return torch.float16 if device.type == "mps" else torch.float32


def load_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = "<|endoftext|>"
    return tok


def load_model(base_id: str, adapter: str | None = None, device=None, dtype=None, merge: bool = True):
    """Loads base model (+ optional LoRA adapter, local dir or HF repo id). Merges the adapter
    into the weights by default so downstream decoders see a plain causal LM."""
    device = pick_device(device) if not isinstance(device, torch.device) else device
    dtype = dtype or default_dtype(device)
    tok = load_tokenizer(base_id)
    model = AutoModelForCausalLM.from_pretrained(base_id, dtype=dtype)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        if merge:
            model = model.merge_and_unload()
    model.to(device).eval()
    return model, tok


def greedy_generation_config(tok, max_new_tokens: int) -> GenerationConfig:
    return GenerationConfig(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
                            eos_token_id=[tok.convert_tokens_to_ids("<|im_end|>"), tok.pad_token_id],
                            **GREEDY_SETTINGS)


@torch.no_grad()
def generate_batch(model, tok, records: list[dict], batch_size: int = 16, max_new_tokens: int = 64,
                   progress: bool = False) -> list[str]:
    """Batched greedy generation (left padding). Returns decoded completions."""
    tok.padding_side = "left"
    gen_cfg = greedy_generation_config(tok, max_new_tokens)
    # Replace (not merge with) Qwen's sampling defaults so nothing leaks in (e.g. repetition_penalty=1.1).
    model.generation_config = gen_cfg
    device = next(model.parameters()).device
    prompts = [prompt_text(tok, r["sentences"], r["criterion_type"], r["criterion_text"]) for r in records]
    # sort by length to reduce padding, restore order afterwards
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outs = [None] * len(prompts)
    rng = range(0, len(order), batch_size)
    if progress:
        from tqdm.auto import tqdm
        rng = tqdm(rng, desc="generate")
    for s in rng:
        idx = order[s:s + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen = model.generate(**enc, generation_config=gen_cfg)
        new = gen[:, enc["input_ids"].shape[1]:]
        for i, seq in zip(idx, new):
            outs[i] = tok.decode(seq, skip_special_tokens=True)
    return outs
