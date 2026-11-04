import json

import pytest

from experiments.spectral_reachability import audit_fd_suffix_goal_q_saved_20261102 as audit
from experiments.spectral_reachability import reacher_pc_p1_common_tail_preflight_20261102_codexexclusive as preflight


def inputs():
    return [json.loads(p.read_text()) for p in
            (audit.FD, audit.GOAL, preflight.Q_PATH, audit.PLAN)]


def test_saved_scores_recomputed_and_tamper_rejected():
    fd, goal, q, plan = inputs()
    result = audit.audit(fd, goal, q, plan)
    assert [r["endpoints"] for r in result] == [45, 45]
    fd["rows"][0]["compiler_receipt"]["proposal_rows"][1]["final_costs"][0] += 1
    with pytest.raises(ValueError, match="recomputed Q mismatch"):
        audit.audit(fd, goal, q, plan)


def test_goal_identity_rejected():
    fd, goal, q, plan = inputs()
    goal["rows"][0]["goal_rgb_sha256"] = "different"
    with pytest.raises(ValueError, match="goal identity"):
        audit.audit(fd, goal, q, plan)
