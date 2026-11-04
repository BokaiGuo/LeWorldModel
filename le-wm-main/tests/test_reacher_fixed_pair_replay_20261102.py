"""Saved replay corruption and attribution rank edge cases."""
import copy
import json

import pytest

from experiments.spectral_reachability import audit_reacher_fixed_pair_rgb_replay_20261102 as replay_audit
from experiments.spectral_reachability import audit_reacher_fixed_pair_rgb_attribution_20261102 as ranks
from experiments.spectral_reachability import reacher_fixed_pair_rgb_attribution_20261102 as data


def test_replay_accepts_sealed_events_and_rejects_action_tamper():
    plan = json.loads(data.PLAN.read_text())
    events = data.replay.pair.audit.saved.verify_journal(data.JOURNAL)
    assert replay_audit.validate(plan, events, data.PLAN_SHA)["branches"] == 52
    altered = copy.deepcopy(events)
    intent = next(e for e in altered if e["kind"] == "step_intent")
    intent["action"][0] += 0.01
    with pytest.raises(ValueError, match="action/stop"):
        replay_audit.validate(plan, altered, data.PLAN_SHA)


def test_rank_sign_keeps_ties_and_rejects_nonfinite():
    assert [ranks.sign(x) for x in (-1.0, 0.0, 1.0)] == [-1, 0, 1]
    with pytest.raises(ValueError, match="nonfinite"):
        ranks.sign(float("nan"))
