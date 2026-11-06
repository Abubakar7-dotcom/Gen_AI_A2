# Demo video script (target 6 minutes)

Before recording: models downloaded (start the app once beforehand), browser zoom ~110%,
terminal font large, close unrelated windows.

| time | show | say (key points) |
|---|---|---|
| 0:00–0:30 | Repo on GitHub, README top | Task: one patient note + one trial criterion → label + cited sentence IDs. Fine-tuned Qwen2.5-1.5B (LoRA) on TrialGPT expert annotations, fixed 37/8/8 patient split. |
| 0:30–1:00 | Double-click `start.command`; terminal loads models; browser opens | One-click start; runs locally on the Mac (MPS); disclaimer banner: educational prototype, not a clinical decision. |
| 1:00–2:30 | Screening tab → *Load example* (a held-out test patient) → *Screen patient* | Per-criterion label badges, expert label column for comparison, evidence sentences quoted verbatim and highlighted in the numbered note, template explanation (grounded, no free-text claims), missing-information list, rule-based preliminary outcome and the rule shown under it. |
| 2:30–3:15 | Paste a NEW free-text note + own criteria (e.g. diabetes trial) → screen | Unseen input: note is split into numbered sentences automatically. Point at one "not enough information" row: missing evidence is never treated as eligibility. |
| 3:15–3:45 | Download HTML report, open it; open JSON, scroll to provenance | Export with provenance: model, adapter, decoding settings, device, timestamp, input hash, code version. |
| 3:45–4:15 | Clear → *Screen* with empty note; then criteria without a header | Invalid input → clear inline message, app keeps running. |
| 4:15–5:30 | Inference Lab → generation comparison on a val prompt (all three methods), then synthetic 512 tokens / 128 out with "ignore EOS" | Live: no-cache vs KV-cache vs speculative; all ✅ token-identical; KV cache faster as length grows; speculative shows acceptance rate and tokens per target pass; explain why speedup is limited at batch 1 (per-pass overhead, short JSON answers). |
| 5:30–6:00 | Inference Lab → attention, N=2048, causal | Reference vs tiled online softmax vs SDPA: errors ~1e-3 (fp16) vs float64 truth; tiled intermediate is one tile instead of N×N; skipped causal tiles; fused SDPA fastest. |
| 6:00–6:20 | Report PDF: results table + KV / attention figures | Before/after numbers, test result, where logs and the split manifest are in the repo. |
