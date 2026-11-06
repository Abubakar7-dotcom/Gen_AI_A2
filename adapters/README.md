# LoRA adapters (trained on a Colab Tesla T4, see `results/training/`)

| folder | base model | LoRA | selected epoch | val macro-F1 | use |
|---|---|---|---|---|---|
| `ft_1p5b_r16` | Qwen/Qwen2.5-1.5B-Instruct | r=16, α=32, all attn+MLP proj. | 2 | 0.552 | screening model (target) |
| `ft_0p5b_r16` | Qwen/Qwen2.5-0.5B-Instruct | r=16, α=32, all attn+MLP proj. | 3 | 0.433 | speculative-decoding draft |

`selection.json` records the checkpoint selection (best validation macro-F1) and the training config.
Load with PEFT: `PeftModel.from_pretrained(base_model, "adapters/ft_1p5b_r16")`, or via
`src.modeling.load_model(base_id, "adapters/ft_1p5b_r16")` (merges the adapter into the weights).
