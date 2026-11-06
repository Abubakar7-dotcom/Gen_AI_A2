"""C1-C3: metric definitions, including empty evidence sets and invalid predictions."""
import pytest

from src.metrics import INVALID, classification_metrics, evidence_metrics

L = ["included", "not included", "not enough information", "not applicable"]


def test_perfect_classification():
    m = classification_metrics(["included", "not included"], ["included", "not included"], L)
    assert m["accuracy"] == 1 and m["macro_f1"] == 1
    assert m["per_class"]["not applicable"]["f1"] is None  # absent class -> n/a, not 0


def test_invalid_counts_as_wrong_but_not_a_class():
    m = classification_metrics(["included", "included"], ["included", "excluded"], L)  # excluded invalid for inclusion
    assert m["accuracy"] == 0.5 and m["invalid_predictions"] == 1
    assert m["per_class"]["included"]["recall"] == 0.5
    assert INVALID not in m["macro_labels"]
    assert m["confusion_matrix"]["matrix"][0] == [1, 0, 0, 0, 1]


def test_predicted_absent_class_enters_macro():
    m = classification_metrics(["included", "included"], ["included", "not applicable"], L)
    assert "not applicable" in m["macro_labels"]
    assert m["per_class"]["not applicable"]["f1"] == 0.0


def test_evidence_empty_rules():
    m = evidence_metrics([[]], [[]])
    assert m["macro"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    m = evidence_metrics([[]], [[1]])            # cites something when nothing relevant
    assert m["macro"]["precision"] == 0 and m["macro"]["f1"] == 0 and m["micro"]["fp"] == 1
    m = evidence_metrics([[1]], [[]])            # misses evidence
    assert m["macro"]["recall"] == 0 and m["macro"]["f1"] == 0 and m["micro"]["fn"] == 1


def test_evidence_partial_overlap():
    m = evidence_metrics([[0, 1]], [[1, 2]])
    assert m["micro"]["precision"] == pytest.approx(0.5)
    assert m["micro"]["recall"] == pytest.approx(0.5)
    assert m["macro"]["f1"] == pytest.approx(0.5)


def test_micro_vs_macro_differ():
    m = evidence_metrics([[0], [0, 1, 2, 3]], [[0], []])
    assert m["micro"]["recall"] == pytest.approx(1 / 5)
    assert m["macro"]["recall"] == pytest.approx(0.5)
