"""B2: prompts contain only note + criterion type/text; output parsing never repairs invalid labels."""
from src import data
from src.schema import build_messages, format_target, parse_output


def test_prompt_contains_no_gpt4_content():
    df = data.load_raw().set_index("annotation_id")
    for split in ("train", "val", "test"):
        for r in data.load_split(split):
            msgs = build_messages(r["sentences"], r["criterion_type"], r["criterion_text"])
            text = "\n".join(m["content"] for m in msgs)
            row = df.loc[r["annotation_id"]]
            expl = row["gpt4_explanation"]
            # A few GPT-4 "explanations" are verbatim copies of a note sentence (e.g. row 604);
            # that text legitimately comes from the note, so only flag text from another source.
            legit = "\n".join(r["sentences"]) + "\n" + r["criterion_text"]
            if isinstance(expl, str) and len(expl) > 30 and expl not in legit:
                assert expl not in text
            assert "gpt4" not in text.lower()


def test_parse_valid():
    p = parse_output('{"label": "not excluded", "evidence": [0, 2]}', "exclusion", 5)
    assert p["parse_ok"] and p["label_valid"] and p["label"] == "not excluded" and p["evidence"] == [0, 2]


def test_parse_invalid_label_for_type_is_not_remapped():
    p = parse_output('{"label": "excluded", "evidence": []}', "inclusion", 5)
    assert p["parse_ok"] and not p["label_valid"] and p["label"] == "excluded"


def test_parse_out_of_range_ids_flagged():
    p = parse_output('{"label": "included", "evidence": [1, 9, -1]}', "inclusion", 5)
    assert p["evidence"] == [1] and p["unsupported_ids"] == [-1, 9]


def test_parse_garbage():
    p = parse_output("I think the patient is eligible", "inclusion", 5)
    assert not p["parse_ok"] and p["label"] is None and p["evidence"] == []


def test_parse_truncated_json_fallback():
    p = parse_output('{"label": "included", "evidence": [1, 2', "inclusion", 5)
    assert not p["parse_ok"] and p["label"] == "included" and p["label_valid"]


def test_target_roundtrip():
    t = format_target("not enough information", [3, 1, 3])
    p = parse_output(t, "exclusion", 5)
    assert p["label"] == "not enough information" and p["evidence"] == [1, 3]
