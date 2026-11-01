"""Stored-vector/packet completeness contracts, not neural experiments."""
import copy

import pytest

from experiments.spectral_reachability.audit_reacher_d2p_packet_rollout_fidelity import validate_packet_rows


def fixture():
    return [{"packet": i, "vectors": {"free": [1.] * 192, "teacher_forced": [1.] * 192,
        "observed": [0.] * 192, "goal": [2.] * 192},
        "costs": {"free": 1., "teacher_forced": 1., "observed": 4.}} for i in range(1, 5)]


def test_complete_four_packet_receipt():
    validate_packet_rows(fixture())


@pytest.mark.parametrize("kind", ["partial", "reordered", "nan", "shape", "cost", "first_history"])
def test_invalid_packet_receipt_fails_closed(kind):
    rows = copy.deepcopy(fixture())
    if kind == "partial":
        rows.pop()
    elif kind == "reordered":
        rows.reverse()
    elif kind == "nan":
        rows[1]["vectors"]["free"][0] = float("nan")
    elif kind == "shape":
        rows[1]["vectors"]["observed"].pop()
    elif kind == "cost":
        rows[1]["costs"]["observed"] = 0.
    else:
        rows[0]["vectors"]["teacher_forced"][0] += 1e-12
    with pytest.raises(ValueError):
        validate_packet_rows(rows)
