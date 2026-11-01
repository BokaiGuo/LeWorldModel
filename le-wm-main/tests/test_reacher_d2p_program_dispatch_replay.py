from copy import deepcopy

import pytest

from experiments.spectral_reachability.audit_reacher_d2p_program_dispatch_replay import summarize
from test_reacher_d2p_program_dispatch import _receipt


def _report():
    def branch(index, distance):
        return {"first_receipt": {"selected_proposal_index": index},
            "physical": {"final_qpos_distance": distance},
            "execution_identity": {"initial": {"reset_rgb_sha256": "root", "goal_rgb_sha256": "goal"}}}
    rows = [{"seed": 1, "source_episode": 10, "model_selection": _receipt(False),
        "zero_prefix": branch(0, 2.), "same_bank_greedy": branch(1, .5), "reconfiguration": None},
        {"seed": 2, "source_episode": 11, "model_selection": _receipt(True),
        "zero_prefix": branch(0, 2.), "same_bank_greedy": branch(1, .5), "reconfiguration": branch(2, .2)}]
    return {"status": "completed", "gates": {"integrity": True}, "sealed_protocol": {"root_seeds": [1, 2]}, "rows": rows}


def test_replay_preserves_roots_witnesses_and_actual_branches():
    result = summarize(_report())
    assert result["root_count"] == 2
    assert result["reconfiguration_count"] == result["progress_fallback_count"] == 1
    assert result["aggregate"]["mean_dispatch_minus_greedy"] == pytest.approx(-.15)
    assert result["aggregate"]["ties_against_greedy"] == 1


def test_replay_pairs_by_root_identity_not_array_order():
    report = _report()
    expected = summarize(report)
    report["rows"].reverse()
    assert summarize(report) == expected


@pytest.mark.parametrize("kind", ["missing", "duplicate", "greedy", "root", "witness", "nonfinite"])
def test_replay_rejects_broken_trace_bindings(kind):
    report = deepcopy(_report())
    if kind == "missing":
        report["rows"].pop()
    elif kind == "duplicate":
        report["rows"][1]["seed"] = 1
    elif kind == "greedy":
        report["rows"][0]["same_bank_greedy"]["first_receipt"]["selected_proposal_index"] = 0
    elif kind == "root":
        report["rows"][0]["zero_prefix"]["execution_identity"]["initial"]["reset_rgb_sha256"] = "different"
    elif kind == "witness":
        report["rows"][1]["reconfiguration"] = None
    else:
        report["rows"][0]["same_bank_greedy"]["physical"]["final_qpos_distance"] = float("nan")
    with pytest.raises(ValueError):
        summarize(report)
