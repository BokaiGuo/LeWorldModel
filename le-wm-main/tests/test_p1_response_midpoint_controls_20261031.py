import copy
import json

import pytest

from experiments.spectral_reachability import p1_response_midpoint_controls_20261031 as subject


@pytest.fixture(scope="module")
def sources():
    assert all(subject.sha(path) == digest for path, digest in subject.SOURCES)
    return [json.loads(path.read_text()) for path, _ in subject.SOURCES]


def test_join_preserves_censoring_and_control_identity(sources):
    result = subject.build(*sources)
    assert len(result["rows"]) == 13
    assert result["excluded"] == [1, 3, 5]
    assert result["equal_budget_certified"] is False
    row = next(row for row in result["rows"] if row["index"] == 4)
    assert row["controls"]["zero"]["outcome"]["status"] == "native_stopped"
    assert row["controls"]["zero"]["new_minus_control_fixed_endpoint"] is None


def test_wrong_control_identity_is_rejected(sources):
    changed = copy.deepcopy(sources)
    changed[6]["rows"][1]["role"] = "greedy"
    with pytest.raises((ValueError, AssertionError)):
        subject.build(*changed)


def test_mismatched_model_cost_is_rejected(sources):
    changed = copy.deepcopy(sources)
    changed[8]["per_row_calls_code_path"]["backward"] += 1
    with pytest.raises((ValueError, AssertionError)):
        subject.build(*changed)
