import copy
import json

import pytest

from experiments.spectral_reachability import p1_adam_checkpoint_physical_20261031 as physical


def test_actual_fixed_plan_converts_both_budgets_and_rejects_tampering():
    assert physical.sha(physical.PLAN) == physical.PLAN_SHA
    plan = json.loads(physical.PLAN.read_bytes())
    for step in (5, 10):
        converted = physical.convert(plan, step)
        assert len(converted["rows"]) == 24
        assert converted["rows"][0]["selected_action_sha256"] == (
            plan["rows"][0]["checkpoints"][str(step)]["candidate_sha256"][
                plan["rows"][0]["checkpoints"][str(step)]["winner"]])
    tampered = copy.deepcopy(plan)
    tampered["rows"][0]["checkpoints"]["5"]["candidate_actions"][0][0][0][0] += 0.01
    with pytest.raises(ValueError, match="candidate action"):
        physical.convert(tampered, 5)
    tampered = copy.deepcopy(plan)
    tampered["rows"][0]["checkpoints"]["10"]["winner"] = (
        tampered["rows"][0]["checkpoints"]["10"]["winner"] + 1) % 4
    with pytest.raises(ValueError, match="winner"):
        physical.convert(tampered, 10)
