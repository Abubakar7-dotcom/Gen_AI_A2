"""LoRA supervised fine-tuning with a completion-only loss (plain PyTorch loop).

    python -m src.train --config configs/train_1p5b.yaml

Input  : chat prompt (system instructions + numbered note + criterion type + criterion text)
Target : '{"label": ..., "evidence": [...]}' + <|im_end|>
Loss   : causal-LM cross-entropy on target tokens only (prompt tokens masked with -100).
After each epoch: validation loss (teacher forcing) and validation generation metrics
(greedy); the adapter with the best validation macro-F1 is kept as <output_dir>/best.

Memory optimisations for small GPUs (tested target: RTX 4050 laptop, 6 GB):
  * Left padding + explicit position_ids, so every target sits in the last K positions of its row.
    The LM head is then evaluated only there (logits_to_keep=K+1). Full logits would be
    B x L x 151,936 values (~1.6 GB in fp32 for B=4, L=650); target-only logits are ~K/L of that.
  * bf16 autocast when the GPU supports it (Ampere/Ada); fp16 + GradScaler otherwise (T4).
  * Gradient checkpointing, LoRA (only ~1% of parameters get optimizer state).
  * Length-grouped micro-batches to reduce padding.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup

from src.data import ROOT, load_split, prompt_text
from src.evaluate import evaluate_outputs
from src.modeling import generate_batch, load_tokenizer, native_bf16, pick_device
from src.schema import format_target


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode(tok, rec: dict, max_len: int) -> dict:
    prompt = prompt_text(tok, rec["sentences"], rec["criterion_type"], rec["criterion_text"])
    target = format_target(rec["gold_label"], rec["gold_evidence"]) + "<|im_end|>"
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    t = tok(target, add_special_tokens=False)["input_ids"]
    if len(p) + len(t) > max_len:  # never happens with max_len>=700 on this dataset (results/token_stats.json)
        raise ValueError(f"example {rec['annotation_id']} exceeds max_len={max_len}; raise max_len")
    return {"input_ids": p + t, "labels": [-100] * len(p) + t}


def collate(batch: list[dict], pad_id: int) -> dict:
    """LEFT padding. position_ids count only real tokens, so positions match unpadded inference."""
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, n - L:] = torch.tensor(b["input_ids"]); lab[i, n - L:] = torch.tensor(b["labels"]); att[i, n - L:] = 1
    pos = (att.cumsum(-1) - 1).clamp(min=0)
    return {"input_ids": ids, "labels": lab, "attention_mask": att, "position_ids": pos}


def target_loss(model, b: dict, reduction: str = "mean") -> torch.Tensor:
    """Cross-entropy over target tokens only, computing logits only for the last K+1 positions.
    logits[t] predicts token t+1, so logits at positions n-K-1..n-2 predict labels n-K..n-1."""
    labels = b["labels"]
    K = int((labels != -100).sum(-1).max())
    out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                position_ids=b["position_ids"], logits_to_keep=K + 1)
    logits = out.logits[:, :-1].float()
    tgt = labels[:, -K:]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                           ignore_index=-100, reduction=reduction)


def length_grouped_batches(lengths: list[int], bs: int, gen: torch.Generator, mega: int = 16) -> list[list[int]]:
    """Shuffle, then sort inside chunks of bs*mega by length and cut into micro-batches; shuffle batches."""
    order = torch.randperm(len(lengths), generator=gen).tolist()
    batches = []
    for s in range(0, len(order), bs * mega):
        chunk = sorted(order[s:s + bs * mega], key=lambda i: -lengths[i])
        batches += [chunk[j:j + bs] for j in range(0, len(chunk), bs)]
    perm = torch.randperm(len(batches), generator=gen).tolist()
    return [batches[p] for p in perm]


@torch.no_grad()
def val_loss(model, data, bs, pad_id, device, amp):
    model.eval()
    tot, ntok = 0.0, 0
    for s in range(0, len(data), bs):
        b = {k: v.to(device) for k, v in collate(data[s:s + bs], pad_id).items()}
        with amp():
            tot += target_loss(model, b, reduction="sum").item()
        ntok += (b["labels"] != -100).sum().item()
    return tot / max(ntok, 1)


def plot_curves(steps_csv: Path, epochs: list[dict], out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(steps_csv)))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot([float(r["epoch_float"]) for r in rows], [float(r["train_loss"]) for r in rows], label="train loss (per optimizer step)", alpha=.7)
    ax[0].plot([e["epoch"] for e in epochs], [e["val_loss"] for e in epochs], "o-", label="val loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss (target tokens)"); ax[0].legend()
    ax[1].plot([e["epoch"] for e in epochs], [e["val_macro_f1"] for e in epochs], "o-", label="val label macro-F1")
    ax[1].plot([e["epoch"] for e in epochs], [e["val_accuracy"] for e in epochs], "s-", label="val label accuracy")
    ax[1].plot([e["epoch"] for e in epochs], [e["val_evidence_micro_f1"] for e in epochs], "^-", label="val evidence micro-F1")
    ax[1].set_xlabel("epoch"); ax[1].set_ylim(0, 1); ax[1].legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None, help="debug: stop after N optimizer steps")
    ap.add_argument("--val-limit", type=int, default=None, help="debug: use only N validation rows")
    ap.add_argument("--train-limit", type=int, default=None, help="debug: use only N training rows")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override micro-batch; grad_accum is rescaled to keep the effective batch")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.batch_size:
        eff = cfg["batch_size"] * cfg["grad_accum"]
        cfg["batch_size"], cfg["grad_accum"] = a.batch_size, max(1, eff // a.batch_size)
    set_seed(cfg["seed"])
    device = pick_device()
    use_amp = device.type == "cuda"
    if use_amp:
        torch.backends.cuda.matmul.allow_tf32 = True
        amp_dtype = torch.bfloat16 if native_bf16() else torch.float16
    else:
        amp_dtype = torch.float32  # MPS/CPU: plain fp32 (debug runs only)
    amp = (lambda: torch.autocast("cuda", dtype=amp_dtype)) if use_amp else nullcontext

    out_dir = ROOT / cfg["output_dir"]; out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / "results" / "training" / cfg["name"]; log_dir.mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer(cfg["base_model"])
    train_recs = load_split("train")[: a.train_limit] if a.train_limit else load_split("train")
    val_recs = load_split("val")[: a.val_limit] if a.val_limit else load_split("val")
    train = [encode(tok, r, cfg["max_len"]) for r in train_recs]
    val = [encode(tok, r, cfg["max_len"]) for r in val_recs]

    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], dtype=amp_dtype, attn_implementation="sdpa")
    lc = cfg["lora"]
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=lc["r"], lora_alpha=lc["alpha"],
                                             lora_dropout=lc["dropout"], target_modules=lc["target_modules"]))
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"device={device} dtype={amp_dtype} micro-batch={cfg['batch_size']} x accum={cfg['grad_accum']} "
          f"trainable={n_train:,} / {n_total:,} ({100*n_train/n_total:.2f}%)")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    bs, accum = cfg["batch_size"], cfg["grad_accum"]
    steps_per_epoch = math.ceil(len(train) / bs / accum)
    total = steps_per_epoch * cfg["epochs"] if not a.max_steps else a.max_steps
    sched = get_cosine_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)  # bf16 needs no scaling

    steps_csv = log_dir / "train_steps.csv"
    fcsv = open(steps_csv, "w", newline=""); w = csv.writer(fcsv)
    w.writerow(["step", "epoch_float", "train_loss", "lr"])
    history, best, step, t0 = [], -1.0, 0, time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        g = torch.Generator().manual_seed(cfg["seed"] + epoch)
        if cfg.get("group_by_length", True):
            micro = length_grouped_batches([len(x["input_ids"]) for x in train], bs, g)
        else:
            order = torch.randperm(len(train), generator=g).tolist()
            micro = [order[i:i + bs] for i in range(0, len(order), bs)]
        run_loss, run_n = 0.0, 0
        for mi, idx in enumerate(micro):
            b = {k: v.to(device) for k, v in collate([train[i] for i in idx], tok.pad_token_id).items()}
            with amp():
                loss = target_loss(model, b) / accum
            scaler.scale(loss).backward()
            if step == 0 and run_n == 0 and device.type == "cuda":
                print(f"peak GPU memory after first micro-batch: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
            run_loss += loss.item(); run_n += 1
            if run_n % accum == 0 or mi == len(micro) - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
                scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                ep_f = epoch - 1 + (mi + 1) / len(micro)
                w.writerow([step, round(ep_f, 4), round(run_loss * accum / run_n, 5), sched.get_last_lr()[0]]); fcsv.flush()
                if step % 5 == 0:
                    print(f"ep {epoch} step {step}/{total} loss {run_loss * accum / run_n:.4f} ({time.time()-t0:.0f}s)")
                run_loss, run_n = 0.0, 0
                if a.max_steps and step >= a.max_steps:
                    break

        vl = val_loss(model, val, cfg["eval_batch_size"], tok.pad_token_id, device, amp)
        with amp():
            texts = generate_batch(model, tok, val_recs, cfg["eval_batch_size"], cfg["eval_max_new_tokens"])
        _, rep = evaluate_outputs(val_recs, texts)
        rec = {"epoch": epoch, "step": step, "val_loss": vl, "val_accuracy": rep["overall"]["accuracy"],
               "val_macro_f1": rep["overall"]["macro_f1"], "val_evidence_micro_f1": rep["evidence_all"]["micro"]["f1"],
               "val_output_validity": rep["output_validity"], "elapsed_s": round(time.time() - t0, 1)}
        history.append(rec)
        print(json.dumps(rec))
        model.save_pretrained(out_dir / f"epoch{epoch}")
        if rec["val_macro_f1"] > best:
            best = rec["val_macro_f1"]
            if (out_dir / "best").exists():
                shutil.rmtree(out_dir / "best")
            model.save_pretrained(out_dir / "best")
            (out_dir / "best" / "selection.json").write_text(json.dumps({"selected_epoch": epoch, **rec, "config": cfg}, indent=2))
        if a.max_steps and step >= a.max_steps:
            break
    fcsv.close()

    summary = {"config": cfg, "trainable_params": n_train, "total_params": n_total, "device": str(device),
               "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
               "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else None,
               "epochs": history, "best_val_macro_f1": best,
               "best_epoch": max(history, key=lambda e: e["val_macro_f1"])["epoch"]}
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    shutil.copy(a.config, log_dir / "config.yaml")
    plot_curves(steps_csv, history, log_dir / "curves.png")
    print(f"done: best val macro-F1 {best:.3f} at epoch {summary['best_epoch']} -> {out_dir/'best'}")


if __name__ == "__main__":
    main()
