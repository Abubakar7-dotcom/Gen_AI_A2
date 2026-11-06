"""B1: fixed 37/8/8 patient-grouped split, determinism, and no patient leakage."""
import json

import numpy as np
import pytest

from src import data


@pytest.fixture(scope="module")
def built():
    df = data.load_raw()
    split = data.make_split(df.patient_id.unique())
    records, dropped = data.build_records(df, split)
    return df, split, records, dropped


def test_split_sizes_and_disjoint(built):
    df, split, records, dropped = built
    assert {k: len(v) for k, v in split.items()} == {"train": 37, "val": 8, "test": 8}
    s = {k: set(v) for k, v in split.items()}
    assert not (s["train"] & s["val"]) and not (s["train"] & s["test"]) and not (s["val"] & s["test"])
    assert set().union(*s.values()) == set(df.patient_id.unique())


def test_split_follows_protocol(built):
    df, split, _, _ = built
    ids = np.array(sorted(df.patient_id.unique()))
    np.random.default_rng(42).shuffle(ids)
    ids = ids.tolist()
    assert split["train"] == ids[:37] and split["val"] == ids[37:45] and split["test"] == ids[45:]


def test_split_deterministic(built):
    df, split, _, _ = built
    assert data.make_split(df.patient_id.unique()) == split


def test_every_row_in_exactly_one_split(built):
    df, split, records, dropped = built
    ids = [r["annotation_id"] for rs in records.values() for r in rs]
    assert len(ids) == len(set(ids)) == len(df) - len(dropped)
    for s, rs in records.items():
        assert all(r["patient_id"] in split[s] for r in rs)


def test_manifest_matches_split(built):
    _, split, _, _ = built
    man = json.loads(data.MANIFEST_PATH.read_text())
    for s in split:
        assert man["splits"][s]["patient_ids"] == split[s]


def test_records_have_no_gpt4_fields(built):
    _, _, records, _ = built
    for rs in records.values():
        for r in rs:
            assert not any(k.startswith("gpt4") or k == "explanation_correctness" for k in r)
