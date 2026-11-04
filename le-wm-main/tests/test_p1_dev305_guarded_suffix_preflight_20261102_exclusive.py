import json

import numpy as np
import pytest

from experiments.spectral_reachability import p1_dev305_guarded_suffix_preflight_20261102_exclusive as p


def goal(valid=True):
    image = np.full((224, 224, 3), 255, np.uint8)
    image[40:60 if valid else 60, 30:50] = [120, 140, 160]
    if not valid:
        image[40:60, 30:50] = 255
        image[210:224, 30:50] = [120, 140, 160]
    image[100:110, 100:110] = [50, 100, 230]
    return image


def bank():
    return [{"name": f"a{i}", "action": [0.5, 0.0]} for i in range(16)] + [
        {"name": "zero", "action": [0., 0.]}]


def run(g, actions=None):
    return p.preflight(g, ["candidate_02", "candidate_00", "zero"], actions or bank(),
                       environment_factory=lambda: pytest.fail("must not open environment"))


def test_invalid_goal_stops_all_before_environment():
    result = run(goal(False))
    assert result["status"] == "blocked_before_environment"
    assert len(result["rows"]) == 51
    assert all(r["executed_steps"] == 0 and r["terminal_reward"] is None for r in result["rows"])
    assert result["physical_steps"] == result["environment_factory_calls"] == 0


def test_valid_goal_is_not_rollout_or_admission():
    result = run(goal())
    assert result["status"] == "further_physical_preflight_required"
    assert not result["p1_admission"] and all(r["executed_steps"] == 0 for r in result["rows"])


def test_invalid_action_and_guard_fail_closed(monkeypatch):
    invalid = bank()
    invalid[0]["action"] = [2, 0]
    with pytest.raises(ValueError, match="invalid actions"):
        run(goal(), invalid)
    monkeypatch.setattr(p.guard, "_guard_intact", lambda: False)
    with pytest.raises(ValueError, match="guard source"):
        run(goal())


def test_actual_sealed_sources():
    report = p.analyze()
    assert report["status"] == "blocked_before_environment"
    assert len(report["rows"]) == 51 and report["physical_steps"] == 0
    assert len({x["plan_sha256"] for x in report["rows"]}) == 17
    assert json.dumps(report, allow_nan=False)
