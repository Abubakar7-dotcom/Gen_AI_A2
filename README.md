# TrialMatch — criterion-level clinical-trial eligibility with evidence

> **Educational screening prototype — not a clinical enrollment decision tool.**

**Demo video:** https://youtu.be/3Ysb4MRhmlE

Given a patient note and one inclusion or exclusion criterion, a LoRA-fine-tuned
**Qwen2.5-1.5B-Instruct** returns an eligibility label and the IDs of the patient sentences that
support it. A local browser app screens a note against a whole criteria list, shows the cited
evidence verbatim, and exports a report. The project also implements, from raw forward passes, an
explicit **KV-cache decoder**, a **tiled online-softmax attention** kernel and **greedy speculative
decoding**, each verified against a reference and benchmarked.

Dataset: [`ncbi/TrialGPT-Criterion-Annotations`](https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations),
pinned to revision `1cfcafde94a1560a33b4addc1638664fe28fc059` (`src/data.py`).

## Contents
| path | what |
|---|---|
| `src/data.py` | loading, the fixed patient split, leakage checks, prompt building, token statistics |
| `src/schema.py` | label sets, prompt template, JSON output parsing / validation |
| `src/train.py`, `configs/` | LoRA SFT with completion-only loss |
| `src/evaluate.py`, `src/metrics.py` | batched greedy evaluation; classification + evidence metrics |
| `src/pipeline.py` | screening: per-criterion prediction → verbatim evidence → template explanation → rule-based outcome → export |
| `src/decoding/kv_cache.py` | uncached reference loop and explicit KV-cache loop |
| `src/decoding/speculative.py` | greedy speculative decoding (draft / verify / cache crop) |
| `src/attention/` | reference attention and tiled online-softmax attention |
| `bench/` | KV-cache, attention and speculative-decoding benchmarks → `results/bench/` |
| `app/` | Gradio app (Screening tab, Inference Lab tab) |
| `tests/` | 68 tests: split, schema, metrics, loss masking, KV equality, attention equality, speculative equality, pipeline |
| `scripts/run_gpu_experiments.py`, `notebooks/train_online.ipynb` | all GPU experiments in one command / on Colab or Kaggle |
| `data/split_manifest.json` | the split: patient IDs, row and label counts per split |
| `results/` | metrics, predictions, confusion matrices, training curves, benchmark CSV/JSON/PNG, logs |
| `report/` | IEEE paper (LaTeX) and figures |

## 1. Setup (macOS / Linux, Python 3.11+)
```bash
git clone https://github.com/Abubakar7-dotcom/Gen_AI_A2.git trialmatch && cd trialmatch
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
# or: python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
```
All versions are pinned in `requirements.txt` (torch 2.15, transformers 5.22.1, peft 0.21.2, gradio 6.31).
The base models (`Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-0.5B-Instruct`, ~4 GB together) are
downloaded from Hugging Face automatically on first use.

**Fine-tuned weights (LoRA adapters)** are included in the repository: `adapters/ft_1p5b_r16`
(screening model, 74 MB) and `adapters/ft_0p5b_r16` (speculative draft, 35 MB); see
`adapters/README.md`. Cloning the repository is enough; retraining (Section 4) writes new adapters
to `outputs/<run>/best`. Any adapter folder or Hugging Face repo id can be passed with `--adapter`.

## 2. Start the app
```bash
./run.sh                 # creates .venv on first run, starts the app, opens http://127.0.0.1:7860
```
On macOS you can also double-click **`start.command`** in Finder. Options: `--no-draft` (skip the
draft model, less memory), `--port`, `--device cpu|mps|cuda`, `--adapter <dir or HF repo>`.

* **Screening tab** — paste a patient note (free text or one numbered sentence per line) and the
  criteria (`Inclusion:` / `Exclusion:` sections, one criterion per line), or load a held-out test
  example. Choose KV-cache greedy or speculative decoding. Output per criterion: label, cited
  sentence IDs, the evidence sentences verbatim (also highlighted in the numbered note), a
  template explanation and a missing-information flag; plus a **rule-based preliminary outcome**
  (any failing criterion → *Likely ineligible*; else any undecidable criterion → *Needs more
  information*; else *Potentially eligible*). The report is downloadable as HTML and JSON (a copy
  is saved in `reports/`) with provenance: model, adapter, decoding settings, device, timestamp,
  input hash, code version. Empty or malformed input gives an inline message; the app keeps running.
* **Inference Lab tab** — runs no-cache, KV-cache and speculative decoding live on the loaded
  model and checks the outputs are token-identical; runs reference, tiled and SDPA attention on
  random inputs and reports error against float64, time and memory.

## 3. Data and the fixed split
```bash
.venv/bin/python -m src.data --check --token-stats
```
Sorted unique patient IDs → `numpy.random.default_rng(42).shuffle` → **37 train / 8 val / 8 test
patients** (602 / 206 / 206 rows). Writes `data/split_manifest.json`, `data/processed/*.jsonl`
and `results/token_stats.json`. Annotation 883 has an empty criterion and is excluded (listed in
the manifest). The longest prompt is 621 tokens, so no note is truncated. Only `note`,
`criterion_type` and `criterion_text` reach the model; GPT-4 columns are never read into the
records (`tests/test_schema.py`). The split is by patient as prescribed; one trial appears in both
train and validation (reported in the manifest).

## 4. Training, evaluation and benchmarks (CUDA GPU)
Everything GPU-side runs with one command, on Colab/Kaggle via `notebooks/train_online.ipynb` or locally:
```bash
python scripts/run_gpu_experiments.py           # data check, tests, base eval, 3 LoRA runs, fine-tuned eval, benchmarks
python scripts/run_gpu_experiments.py --final   # + the one-time test-split evaluation (after all choices are frozen)
```
Each step logs to `results/logs/<step>.log`; measured durations go to `results/logs/timings.json`;
finished steps are skipped on restart. The individual commands:
```bash
python -m src.evaluate --name base_1p5b --split val                                   # before fine-tuning
python -m src.train --config configs/train_1p5b.yaml                                  # LoRA r=16 (also: _r8, train_0p5b)
python -m src.evaluate --name ft_1p5b_r16 --adapter outputs/ft_1p5b_r16/best --split val
python -m src.evaluate --name ft_1p5b_r16 --adapter outputs/ft_1p5b_r16/best --split test --confirm-test
python -m bench.bench_kv --tag gpu_1p5b_half
python -m bench.bench_attention --device cuda --tag gpu
python -m bench.bench_spec --tag gpu        # reported run: --tag mps on the Mac
python scripts/error_analysis.py --run ft_1p5b_r16 --split test
```
Training: LoRA r=16, α=32 on all attention and MLP projections, lr 2e-4, 3 epochs, effective batch
16, loss on answer tokens only, fp16 + GradScaler on T4 (bf16 on Ampere+), gradient checkpointing.
The checkpoint with the best **validation macro-F1** is kept. Evaluation decoding: greedy,
`max_new_tokens=64`; Qwen's default sampling configuration is replaced, not merged.

### Validation results (T4)
| run | accuracy | macro-F1 | evidence micro-F1 | parse failures |
|---|---|---|---|---|
| base Qwen2.5-1.5B | 0.068 | 0.082 | 0.091 | 0 |
| base Qwen2.5-0.5B | 0.194 | 0.133 | 0.000 | 136 |
| **LoRA 1.5B r=16 (selected)** | **0.743** | **0.552** | **0.535** | 0 |
| LoRA 1.5B r=8 | 0.733 | 0.471 | 0.522 | 0 |
| LoRA 0.5B r=16 | 0.743 | 0.433 | 0.387 | 0 |

Full tables: `results/val_summary.md`, per-run JSON in `results/metrics/`.

### Test results (evaluated after all choices were frozen on validation)
| run | accuracy | macro-F1 | evidence micro-F1 | evidence micro-F1 (non-empty gold) | parse failures |
|---|---|---|---|---|---|
| base Qwen2.5-1.5B | 0.087 | 0.130 | 0.111 | 0.323 | 0 |
| **LoRA 1.5B r=16** | **0.704** | **0.369** | **0.594** | **0.708** | 0 |

The test set has no "excluded" rows; the model's single false "excluded" prediction adds that class
with F1 = 0 to the six-label macro average (0.443 over the five labels present in test). Error
analysis of 11 test errors: `results/error_analysis.md`. The test evaluation and the speculative
benchmark below ran on the Mac (MPS, fp16): Colab's free GPU quota was used up, and the files from
an earlier T4 test run (same macro-F1, 0.369) were lost when the Colab session reset.

### Speculative decoding (fine-tuned 1.5B target, Apple M5, fp16)
All 12 configurations produce output token-identical to the target-only KV decoder. On task prompts
the fine-tuned 0.5B draft has acceptance 0.88–0.93 (base draft 0.65–0.72) and gives a 1.24–1.27×
speedup at k = 4–6; the base draft breaks even (0.92–1.01×). Full table:
`results/bench/spec_mps.csv`.

## 5. Tests
```bash
.venv/bin/python -m pytest -q tests          # 68 tests, ~3 min on an M-series Mac
```

## AI-use acknowledgement
Code, tests and documentation were written with the assistance of an AI coding assistant
(Claude, Anthropic). All design decisions, results and the report were reviewed by the author;
every number reported comes from the logged runs in `results/`.
