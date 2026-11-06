"""TrialMatch browser app (Gradio, runs locally).

    python -m app.app                    # or double-click start.command (macOS) / ./run.sh
    python -m app.app --no-draft --port 7861

Tab 1 Screening: patient note + inclusion/exclusion criteria -> per-criterion label, verbatim
evidence, explanation, missing information, rule-based preliminary outcome, HTML/JSON export.
Tab 2 Inference Lab: live comparison of the three decoders (no cache, KV cache, speculative) on the
loaded models, and of the three attention implementations (reference, tiled online softmax, SDPA).
"""
from __future__ import annotations

import argparse
import html
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr
import torch

from src.data import load_split
from src.pipeline import (DISCLAIMER, InputError, criteria_to_text, load_screener, save_report)
from src.schema import number_sentences

DEFAULTS = {
    "model": "Qwen/Qwen2.5-1.5B-Instruct", "adapter": "outputs/ft_1p5b_r16/best",
    "draft": "Qwen/Qwen2.5-0.5B-Instruct", "draft_adapter": "outputs/ft_0p5b_r16/best",
}
LABEL_COLOR = {"included": "#1a7f37", "not excluded": "#1a7f37", "not included": "#cf222e", "excluded": "#cf222e",
               "not enough information": "#9a6700", "not applicable": "#57606a"}
OUTCOME_COLOR = {"Potentially eligible": "#1a7f37", "Needs more information": "#9a6700", "Likely ineligible": "#cf222e"}
CSS = """
.banner {background:#fff8c5;border:1px solid #d4a72c;border-radius:8px;padding:8px 12px;color:#3b2300;font-weight:600}
.badge {color:#fff;padding:2px 8px;border-radius:10px;font-size:12px;white-space:nowrap}
.res td, .res th {border:1px solid #d0d7de;padding:6px;vertical-align:top;font-size:14px}
.res {border-collapse:collapse;width:100%}
.note li {margin:2px 0;font-size:14px} .hl {background:#fff3b0;border-radius:3px}
.outcome {font-size:20px;font-weight:700;padding:10px 14px;border-radius:8px;color:#fff;display:inline-block}
"""


# ------------------------------------------------------------------ examples (held-out test patients)
def load_examples() -> dict[str, dict]:
    ex = {}
    for r in load_split("test"):
        key = f"{r['patient_id']} × {r['trial_id']}"
        e = ex.setdefault(key, {"sentences": r["sentences"], "criteria": [], "gold": {}})
        e["criteria"].append((r["criterion_type"], r["criterion_text"].strip()))
        e["gold"][(r["criterion_type"], r["criterion_text"].strip())] = r["gold_label"]
    return dict(sorted(ex.items(), key=lambda kv: len(kv[1]["criteria"])))


# ------------------------------------------------------------------ rendering
def badge(label: str | None, status: str, raw: str | None) -> str:
    if status == "unparseable":
        return "<span class='badge' style='background:#8250df'>unparseable</span>"
    if status == "invalid_label":
        return f"<span class='badge' style='background:#8250df'>invalid: {html.escape(str(raw))}</span>"
    return f"<span class='badge' style='background:{LABEL_COLOR[label]}'>{html.escape(label)}</span>"


def render_results(rep: dict, gold: dict | None) -> tuple[str, str, str]:
    o = rep["overall"]
    outcome = (f"<div class='outcome' style='background:{OUTCOME_COLOR[o['outcome']]}'>Preliminary outcome: {o['outcome']}</div>"
               f"<p>{html.escape(o['reason'])}.<br><small><b>Rule-based aggregation, not a clinical decision:</b> "
               f"{html.escape(o['rule'])}</small></p>")
    show_gold = bool(gold) and any((r["criterion_type"], r["criterion"]) in gold for r in rep["criteria"])
    head = "<tr><th>#</th><th>Type</th><th>Criterion</th><th>Label</th>" + ("<th>Expert label</th>" if show_gold else "") + \
           "<th>Evidence (verbatim)</th><th>Explanation</th></tr>"
    rows = []
    for r in rep["criteria"]:
        ev = "<br>".join(f"<b>[{i}]</b> {html.escape(t)}" for i, t in zip(r["evidence_ids"], r["evidence_text"])) or "<i>none cited</i>"
        g = gold.get((r["criterion_type"], r["criterion"])) if show_gold else None
        gcell = (f"<td>{html.escape(g) if g else '—'}{' ✅' if g and g == r['label'] else (' ❌' if g else '')}</td>") if show_gold else ""
        miss = "<br><b>⚠ missing information</b>" if r["missing_information"] else ""
        rows.append(f"<tr><td>{r['index']}</td><td>{r['criterion_type']}</td><td>{html.escape(r['criterion'])}</td>"
                    f"<td>{badge(r['label'], r['status'], r['raw_label'])}{miss}</td>{gcell}<td>{ev}</td>"
                    f"<td>{html.escape(r['explanation'])}</td></tr>")
    table = f"<table class='res'>{head}{''.join(rows)}</table>"
    cited = {i for r in rep["criteria"] for i in r["evidence_ids"]}
    note = "<ol class='note' start='0'>" + "".join(
        f"<li value='{i}'><span class='{'hl' if i in cited else ''}'>{html.escape(s)}</span></li>"
        for i, s in enumerate(rep["sentences"])) + "</ol><small>Highlighted: sentences cited as evidence.</small>"
    missing = [r for r in rep["criteria"] if r["missing_information"]]
    extra = ""
    if missing:
        extra += "<h4>Missing information</h4><ul>" + "".join(
            f"<li>Criterion {r['index']}: {html.escape(r['criterion'])}</li>" for r in missing) + "</ul>"
    if rep["warnings"]:
        extra += "<h4>Warnings</h4><ul>" + "".join(f"<li>{html.escape(w)}</li>" for w in rep["warnings"]) + "</ul>"
    return outcome + table + extra, note, rep["provenance"]


# ------------------------------------------------------------------ app
def build(screener, examples: dict, load_msg: str):
    has_draft = screener.draft is not None

    def on_example(key):
        if not key:
            return gr.update(), gr.update(), None
        e = examples[key]
        return number_sentences(e["sentences"]), criteria_to_text(e["criteria"]), e["gold"]

    def on_screen(note, criteria, mode, k, gold, progress=gr.Progress()):
        try:
            m = "speculative" if mode.startswith("Speculative") else "kv"
            if m == "speculative" and not has_draft:
                raise InputError("Speculative mode is unavailable: the app was started without a draft model.")
            rep = screener.screen(note, criteria, mode=m, k=int(k),
                                  progress=lambda i, n, c: progress(i / n, desc=f"criterion {i + 1}/{n}"))
            res, note_html, prov = render_results(rep, gold)
            jp, hp = save_report(rep)
            status = f"✅ Screened {len(rep['criteria'])} criteria in {rep['provenance']['runtime_s']} s. Report saved to `{hp.relative_to(ROOT)}`."
            return status, res, note_html, prov, [str(hp), str(jp)]
        except InputError as e:
            return f"❌ **Input problem:** {e}", "", "", None, None
        except Exception as e:  # keep the app alive and tell the user what happened
            traceback.print_exc()
            return f"❌ **Unexpected error:** {type(e).__name__}: {e}. The app is still running; see the terminal for details.", "", "", None, None

    def on_clear():
        return "", "", None, "", "", "", None, None

    from app.lab import attention_lab, generation_lab, lab_prompt_choices

    with gr.Blocks(title="TrialMatch") as demo:
        gr.HTML(f"<div class='banner'>⚠️ {DISCLAIMER}</div>")
        gr.Markdown(f"# TrialMatch — patient-to-criterion screening\n{load_msg}")
        with gr.Tab("Screening"):
            gold_state = gr.State(None)
            with gr.Row():
                with gr.Column(scale=1):
                    ex = gr.Dropdown(list(examples), label="Load example (held-out test patient × trial)", value=None)
                    note = gr.Textbox(label="Patient note (free text or one numbered sentence per line)", lines=12,
                                      placeholder="A 58-year-old woman presents with chest pain...")
                    crit = gr.Textbox(label="Trial criteria", lines=10,
                                      placeholder="Inclusion:\n- Age 18 years or older\n- ...\n\nExclusion:\n- Pregnancy\n- ...")
                    with gr.Row():
                        mode = gr.Radio(["KV-cache greedy", "Speculative (0.5B draft)"], value="KV-cache greedy",
                                        label="Decoding", interactive=has_draft)
                        k = gr.Slider(2, 6, value=4, step=1, label="Draft length k (speculative)")
                    with gr.Row():
                        run = gr.Button("Screen patient", variant="primary")
                        clear = gr.Button("Clear")
                    status = gr.Markdown()
                with gr.Column(scale=2):
                    results = gr.HTML()
                    with gr.Accordion("Numbered patient note with cited evidence", open=True):
                        note_view = gr.HTML()
                    files = gr.File(label="Download report (HTML / JSON)", file_count="multiple", interactive=False)
                    with gr.Accordion("Provenance", open=False):
                        prov = gr.JSON()
            ex.change(on_example, ex, [note, crit, gold_state])
            run.click(on_screen, [note, crit, mode, k, gold_state], [status, results, note_view, prov, files])
            clear.click(on_clear, None, [note, crit, gold_state, status, results, note_view, prov, files])

        with gr.Tab("Inference Lab"):
            gr.Markdown("### Generation: no cache vs KV cache vs speculative decoding\n"
                        "All three run live on the loaded fine-tuned model (greedy). The table checks the outputs are token-identical.")
            with gr.Row():
                src = gr.Dropdown(lab_prompt_choices(), value=lab_prompt_choices()[0], label="Prompt")
                n_in = gr.Slider(32, 1024, value=256, step=32, label="Synthetic prompt length (tokens)")
                n_out = gr.Slider(8, 256, value=64, step=8, label="Max new tokens")
            with gr.Row():
                ignore_eos = gr.Checkbox(False, label="Ignore EOS (fixed output length)")
                lab_k = gr.Slider(2, 6, value=4, step=1, label="Draft length k")
                methods = gr.CheckboxGroup(["no cache", "KV cache", "speculative"],
                                           value=["no cache", "KV cache"] + (["speculative"] if has_draft else []), label="Methods")
                gen_btn = gr.Button("Run generation comparison", variant="primary")
            gen_table = gr.Dataframe(label="Results", wrap=True)
            gen_text = gr.Textbox(label="Decoded output (KV cache)", lines=3)
            gen_btn.click(lambda *a: generation_lab(screener, *a), [src, n_in, n_out, ignore_eos, lab_k, methods],
                          [gen_table, gen_text])

            gr.Markdown("### Attention: reference vs tiled online softmax vs PyTorch SDPA\n"
                        "Random Q/K/V with the Qwen2.5-1.5B head layout (12 query heads, 2 KV heads). Errors are measured "
                        "against a float64 ground truth.")
            with gr.Row():
                a_n = gr.Slider(64, 4096, value=1024, step=64, label="Sequence length N")
                a_d = gr.Dropdown([64, 128], value=128, label="Head dim d")
                a_block = gr.Dropdown([32, 64, 128, 256], value=128, label="Tile size")
                a_causal = gr.Checkbox(True, label="Causal")
                a_dtype = gr.Radio(["fp32", "fp16"], value="fp16" if screener.device.type != "cpu" else "fp32", label="dtype")
                att_btn = gr.Button("Run attention comparison", variant="primary")
            att_table = gr.Dataframe(label="Results", wrap=True)
            att_btn.click(lambda *a: attention_lab(screener.device, *a), [a_n, a_d, a_block, a_causal, a_dtype], att_table)
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("TM_MODEL", DEFAULTS["model"]))
    ap.add_argument("--adapter", default=os.environ.get("TM_ADAPTER", DEFAULTS["adapter"]))
    ap.add_argument("--draft", default=os.environ.get("TM_DRAFT", DEFAULTS["draft"]))
    ap.add_argument("--draft-adapter", default=os.environ.get("TM_DRAFT_ADAPTER", DEFAULTS["draft_adapter"]))
    ap.add_argument("--no-draft", action="store_true", help="skip loading the draft model (no speculative mode)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--port", type=int, default=int(os.environ.get("TM_PORT", 7860)))
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    def resolve(adapter):
        if not adapter:
            return None
        p = ROOT / adapter
        return str(p) if p.exists() else adapter   # otherwise treat as a Hugging Face repo id

    t0 = time.time()
    print("Loading models (first start downloads ~4 GB from Hugging Face)...", flush=True)
    try:
        screener = load_screener(a.model, resolve(a.adapter), None if a.no_draft else a.draft,
                                 None if a.no_draft else resolve(a.draft_adapter), a.device)
    except Exception as e:
        print(f"\nERROR: could not load the models: {type(e).__name__}: {e}\n"
              "Fixes: check the internet connection (first start downloads the base models); make sure the adapter "
              f"folder exists ({a.adapter}) or pass a Hugging Face repo id with --adapter; on low memory start with "
              "--no-draft.", file=sys.stderr)
        sys.exit(1)
    msg = (f"Model **{a.model}** + LoRA `{a.adapter}` on **{screener.device}**"
           + (f"; speculative draft **{a.draft}**" + (f" + `{a.draft_adapter}`" if a.draft_adapter else " (base)")
              if screener.draft is not None else "; no draft model")
           + f" (loaded in {time.time() - t0:.0f} s).")
    print(msg.replace("*", "").replace("`", ""), flush=True)
    demo = build(screener, load_examples(), msg)
    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=a.port,
                                                   inbrowser=not a.no_browser, css=CSS,
                                                   theme=gr.themes.Soft(), allowed_paths=[str(ROOT / "reports")])


if __name__ == "__main__":
    main()
