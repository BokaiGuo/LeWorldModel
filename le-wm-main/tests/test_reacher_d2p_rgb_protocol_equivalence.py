import copy

import pytest

from experiments.spectral_reachability.audit_reacher_d2p_rgb_protocol_equivalence import compare_roots


def row(seed):
    return {"seed": seed, "source_episode": seed + 1, "source_start_row": seed + 2,
            "root_warmup_policy": "source", "root_rgb_sha256": "root", "goal_rgb_sha256": "goal",
            "proposal_support": {"hash": "bank"}, "candidate_raw_packets": [[0, 1], [1, 0]],
            "successor_rgb_sha256": ["a", "b"], "policies": [{"path": "original",
            "predicted_rgb_goal_costs": [0.1, 0.2], "observed_rgb_goal_costs": [0.2, 0.3]}]}


def test_pairs_by_root_identity_not_order_or_equal_panel_size():
    result = compare_roots([row(2), row(1)], [row(3), row(1), row(2)])
    assert result["status"] == "pass"
    assert [item["seed"] for item in result["rows"]] == [1, 2]


@pytest.mark.parametrize("field", ["root_rgb_sha256", "successor_rgb_sha256", "candidate_raw_packets"])
def test_changed_physical_evidence_is_not_equivalent(field):
    candidate = row(1)
    candidate[field] = "different"
    assert compare_roots([row(1)], [candidate])["status"] == "fail"


def test_even_tiny_cost_difference_is_not_exact_equivalence():
    candidate = copy.deepcopy(row(1))
    candidate["policies"][0]["observed_rgb_goal_costs"][0] += 1e-12
    assert compare_roots([row(1)], [candidate])["status"] == "fail"


def test_missing_and_duplicate_roots_are_rejected():
    with pytest.raises(ValueError, match="lacks"):
        compare_roots([row(1), row(2)], [row(1)])
    with pytest.raises(ValueError, match="duplicate"):
        compare_roots([row(1), row(1)], [row(1)])
