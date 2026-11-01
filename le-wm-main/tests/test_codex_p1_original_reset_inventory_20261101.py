import json

import pytest

from experiments.spectral_reachability import codex_p1_original_reset_inventory_20261101 as m


def saved():
    return m.SOURCE.read_bytes(), m.JOURNAL.read_bytes()


def test_saved_journal():
    source, journal = saved()
    assert m.sha(source) == m.SOURCE_SHA
    result = m.extract(source, journal)
    assert (result["attempt_records"], result["recorded_reset_starts"], result["recorded_reset_completions"]) == (8, 16, 16)
    assert {row["seed"] for row in result["events"] if row["operation"] == "random_reset"} == set(range(202781010, 202781018))
    assert not result["freshness_certified"] and not result["training_exclusion_certified"]


def test_changed_journal_and_truncation_rejected():
    source, journal = saved()
    with pytest.raises(ValueError):
        m.extract(source, journal[:-1])
    with pytest.raises(ValueError):
        m.extract(source, journal.replace(b"random_reset", b"random_reseT", 1))


def test_mismatch_source_counts_rejected():
    source, journal = saved()
    value = json.loads(source)
    value["resources"]["reset"]["completed"] -= 1
    with pytest.raises(ValueError):
        m.extract(json.dumps(value).encode(), journal)


def test_duplicate_json_rejected():
    with pytest.raises(ValueError):
        m.parse(b'{"a":1,"a":2}')
