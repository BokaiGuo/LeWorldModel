import copy

import numpy as np
import pytest

from experiments.spectral_reachability import p1_coherent_suffix_family_20261101_leibniz as family


def roster():
    prefixes = {p: np.full((5, 2), i / 10, np.float32) for i, p in enumerate(family.PREFIXES)}
    tails = {"zero15": np.zeros((15, 2), np.float32)}
    tails.update({"own15_" + p: np.full((15, 2), (i + 1) / 10, np.float32)
                  for i, p in enumerate(family.PREFIXES)})
    cp5 = {p: {"pixel_sha256": "a" * 64, "state_sha256": "b" * 64} for p in family.PREFIXES}
    return prefixes, tails, cp5


def test_fixed_42_and_no_favorable_selection():
    prefixes, tails, cp5 = roster()
    schedule = family.make_schedule(prefixes, tails, cp5)
    family.validate_schedule(schedule)
    assert len(schedule) == 42
    assert {(s["prefix"], s["tail"]) for s in schedule} == {
        (p, t) for p in family.PREFIXES for t in family.TAILS}
    for slot in schedule:
        raw = np.asarray(slot["commands"])
        np.testing.assert_array_equal(raw[:5], prefixes[slot["prefix"]])
        np.testing.assert_array_equal(raw[5:], tails[slot["tail"]])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1.01])
def test_reject_invalid_action(bad):
    prefixes, tails, cp5 = roster()
    tails["own15_native"][0, 0] = bad
    with pytest.raises(ValueError, match="invalid action"):
        family.make_schedule(prefixes, tails, cp5)


def test_reject_roster_and_branch_mismatch():
    prefixes, tails, cp5 = roster()
    del tails["own15_native"]
    with pytest.raises(ValueError, match="tail roster"):
        family.make_schedule(prefixes, tails, cp5)
    prefixes, tails, cp5 = roster()
    schedule = family.make_schedule(prefixes, tails, cp5)
    with pytest.raises(ValueError, match="roster"):
        family.validate_schedule(schedule[:-1])
    modified = copy.deepcopy(schedule)
    modified[8]["commands"][7][0] += 0.1
    with pytest.raises(ValueError, match="tail mismatch"):
        family.validate_schedule(modified)


def test_reject_source_drift_and_conflicts(tmp_path):
    path = tmp_path / "source"
    path.write_text("original")
    sources = {str(path): family.sha(path)}
    family.verify_sources(sources)
    with pytest.raises(ValueError, match="conflicting"):
        family.merge_sources(sources, {str(path): "0" * 64})
    path.write_text("changed")
    with pytest.raises(ValueError, match="source mismatch"):
        family.verify_sources(sources)
