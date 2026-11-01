import numpy as np
import pytest

from experiments.spectral_reachability import p1_cp10_pair_common_tail_20261101_codex as subject


def test_schedule_is_fixed_two_by_two():
    prefixes = {"structured": np.full((5, 2), .25, np.float32).tolist(),
                "cem": np.full((5, 2), -.5, np.float32).tolist()}
    suffix = np.full((10, 2), .125, np.float32).tolist()
    rows = subject.schedule(prefixes, suffix)
    assert [(r["prefix_role"], r["suffix_role"]) for r in rows] == [
        ("structured", "common"), ("structured", "zero"), ("cem", "common"), ("cem", "zero")]
    assert all(r["commands"][:5] == prefixes[r["prefix_role"]] for r in rows)
    assert all(r["commands"][5:] == (suffix if r["suffix_role"] == "common"
                                     else np.zeros((10, 2)).tolist()) for r in rows)
    with pytest.raises(ValueError):
        subject.schedule({"structured": prefixes["structured"]}, suffix)


def test_actual_source_preflight_and_no_physical_calls():
    value = subject.build()
    assert len(value["schedule"]) == 4
    assert value["physical_outcomes_known_before_design"]
    assert not value["outcome_selected_for_this_run"] and not value["P1_admission"]
    assert value["neural_calls"] == 0
