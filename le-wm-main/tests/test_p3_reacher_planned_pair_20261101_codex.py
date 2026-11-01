"""Pure saved-identity comparison fixtures."""
from copy import deepcopy

from experiments.spectral_reachability import p3_reacher_planned_pair_20261101_codex as pair


def test_pair_requires_matching_roots_and_complete_raw_candidates():
    receipt = dict(failure=None, root_identity=dict(root_rgb_sha256="a", goal_rgb_sha256="b",
                                                    warmup_actions_sha256="c"))
    action = dict(physical_actions=[[.1, .2]] * 15, normalized_tokens=[[.3] * 10] * 3)
    assert pair.compare(receipt, receipt, action, action)["matched_descriptive_pair"]
    changed = deepcopy(action)
    changed["physical_actions"][7][0] = .4
    assert not pair.compare(receipt, receipt, action, changed)["identical_candidate"]
    other = deepcopy(receipt)
    other["root_identity"]["root_rgb_sha256"] = "different"
    assert not pair.compare(receipt, other, action, action)["matched_descriptive_pair"]
    other["failure"] = {"message": "failed"}
    assert not pair.compare(receipt, other, action, action)["complete"]
