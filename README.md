# TrialMatch — criterion-level clinical-trial eligibility with evidence

> **Educational screening prototype — not a clinical enrollment decision tool.**

A fine-tuned Qwen2.5-1.5B-Instruct (LoRA) that, given a patient note, a criterion type and a criterion
text, returns an eligibility label and the IDs of the supporting patient sentences. Dataset:
[`ncbi/TrialGPT-Criterion-Annotations`](https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations)
(pinned revision in `src/data.py`).

*(Work in progress — full documentation is added as the project is completed.)*

## Setup (macOS / Linux)
```bash
uv venv --python 3.11 .venv          # or: python3.11 -m venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## Data and the fixed split
```bash
.venv/bin/python -m src.data --check --token-stats
```
Sorted unique patient IDs → `numpy.random.default_rng(42).shuffle` → 37 train / 8 val / 8 test patients.
Writes `data/split_manifest.json` (patient IDs, row and label counts), `data/processed/*.jsonl`
and `results/token_stats.json`. One row (annotation 883) has an empty criterion and is excluded
(listed in the manifest). Only `note`, `criterion_type` and `criterion_text` are used as model input;
GPT-4 prediction columns are never read into records (`tests/test_schema.py`).

## Training and evaluation (any CUDA GPU: Kaggle/Colab T4 or a local RTX card)
Online: open `notebooks/train_online.ipynb` on Kaggle or Colab (GPU on), set `REPO_URL`, run all.
Local GPU: `python scripts/run_gpu_experiments.py` (add `--batch-size 2` if a 6 GB card runs out of memory).
Both run the scripts below in order, write one log per step to `results/logs/`, record measured
step timings in `results/logs/timings.json`, and skip steps that already finished.
Training uses bf16 autocast where supported (Ampere/Ada) and fp16 + GradScaler otherwise (T4),
computes logits only for the answer tokens (`logits_to_keep`, left padding), and uses gradient checkpointing.
```bash
python -m src.evaluate --name base_1p5b --split val                     # before fine-tuning
python -m src.train --config configs/train_1p5b.yaml                    # LoRA SFT
python -m src.evaluate --name ft_1p5b_r16 --adapter outputs/ft_1p5b_r16/best --split val
python -m src.evaluate --name ft_1p5b_r16 --adapter <adapter> --split test --confirm-test   # once, at the end
```
Decoding for all evaluations: greedy (`do_sample=False`, `repetition_penalty=1.0`, `max_new_tokens=64`).
Qwen's default sampling config is explicitly replaced.

## Tests
```bash
.venv/bin/python -m pytest -q tests
```
