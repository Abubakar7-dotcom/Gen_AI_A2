"""C4/G4: input parsing, explanations and the rule-based aggregation (no model needed)."""
import pytest

from src.pipeline import InputError, aggregate, criteria_to_text, explain, parse_criteria, report_html, segment_note


def test_segment_numbered_note_from_zero_and_one():
    assert segment_note("0. A man.\n1. He smokes.") == ["A man.", "He smokes."]
    assert segment_note("1) A man.\n2) He smokes.") == ["A man.", "He smokes."]


def test_segment_free_text():
    s = segment_note("A 58-year-old woman presents with chest pain. She has diabetes!\nBP 140/90 mmHg.")
    assert s == ["A 58-year-old woman presents with chest pain.", "She has diabetes!", "BP 140/90 mmHg."]
    assert segment_note("Temp was 38.5 C today. Fine.") == ["Temp was 38.5 C today.", "Fine."]  # decimals kept


@pytest.mark.parametrize("bad", ["", "   \n  "])
def test_empty_note_rejected(bad):
    with pytest.raises(InputError):
        segment_note(bad)


def test_parse_criteria_sections_and_bullets():
    text = "Inclusion criteria:\n- Age >= 18\n2. Chest pain\n\nExclusion:\n* Pregnancy\nExclusion criteria apply to renal failure"
    assert parse_criteria(text) == [("inclusion", "Age >= 18"), ("inclusion", "Chest pain"),
                                    ("exclusion", "Pregnancy"), ("exclusion", "Exclusion criteria apply to renal failure")]
    assert parse_criteria("Inclusion: Age > 18") == [("inclusion", "Age > 18")]
    assert parse_criteria(criteria_to_text(parse_criteria(text))) == parse_criteria(text)


@pytest.mark.parametrize("bad", ["", "Age > 18\nInclusion:\n- x", "Inclusion:\n\n"])
def test_bad_criteria_rejected(bad):
    with pytest.raises(InputError):
        parse_criteria(bad)


def _row(i, label, status="ok"):
    return {"index": i, "label": label, "status": status}


def test_aggregate_rules():
    assert aggregate([_row(1, "included"), _row(2, "not excluded")])["outcome"] == "Potentially eligible"
    assert aggregate([_row(1, "included"), _row(2, "not applicable")])["outcome"] == "Potentially eligible"
    assert aggregate([_row(1, "included"), _row(2, "not enough information")])["outcome"] == "Needs more information"
    assert aggregate([_row(1, "included"), _row(2, None, "unparseable")])["outcome"] == "Needs more information"
    r = aggregate([_row(1, "not enough information"), _row(2, "excluded")])
    assert r["outcome"] == "Likely ineligible" and r["failing"] == [2]
    assert aggregate([_row(1, "not included")])["outcome"] == "Likely ineligible"


def test_explanations_quote_only_cited_sentences():
    sents = ["A woman, 58.", "She has type 2 diabetes.", "No allergies."]
    e = explain("excluded", "exclusion", "Diabetes", [1], sents, "ok")
    assert "She has type 2 diabetes." in e and "A woman" not in e
    assert "verify" in explain("not excluded", "exclusion", "Pregnancy", [], sents, "ok")
    assert "not contain enough information" in explain("not enough information", "inclusion", "HbA1c", [], sents, "ok")
    assert "unresolved" in explain(None, "inclusion", "x", [], sents, "unparseable")


def test_report_html_escapes_input():
    rep = {"disclaimer": "d", "sentences": ["<script>x</script>"], "warnings": [],
           "overall": {"outcome": "Potentially eligible", "reason": "r", "rule": "rule"},
           "criteria": [{"index": 1, "criterion_type": "inclusion", "criterion": "<b>", "label": "included", "status": "ok",
                         "raw_label": "included", "evidence_ids": [0], "evidence_text": ["<script>x</script>"], "explanation": "e"}],
           "provenance": {"model": "m"}}
    h = report_html(rep)
    assert "<script>" not in h and "&lt;script&gt;" in h
