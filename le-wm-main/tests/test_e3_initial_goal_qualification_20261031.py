import copy
import json

import pytest

from experiments.spectral_reachability import e3_initial_goal_qualification_20261031 as audit


@pytest.fixture(scope="module")
def data():
    return json.loads(audit.INPUT.read_bytes())


def test_g1_initial_state_is_success_and_cp5_candidates_are_recorded(data):
    result = audit.analyze(data)
    assert len(result["rows"]) == 16
    assert all(row["initial_success"] for row in result["rows"] if row["goal_index"] == 1)
    assert len(result["exploratory_cp5_candidates"]) == 2
    assert not result["production_authorized"]


def test_rejects_changed_root_identity(data):
    changed = copy.deepcopy(data)
    changed["cells"][4]["evaluator"]["native_goal"][0] += 1
    with pytest.raises(ValueError, match="G1 actual root identity"):
        audit.analyze(changed)
