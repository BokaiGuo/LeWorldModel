import numpy as np

from experiments.spectral_reachability import pc_exact_orientation_mechanism_20261103_codex as pilot
from experiments.spectral_reachability import pc_exact_orientation_selected_audit_20261103_codex as audit


def test_restricted_prefix_and_response():
    goal = np.array([0.3, 0.7])
    selected, ledger = pilot.restricted(goal)
    actions = np.asarray(selected["chain"])
    assert actions[0, 0] == 0
    assert ledger["svd_calls"] == 9
    np.testing.assert_allclose(selected["trace"][0][:2], selected["trace"][1][:2])
    np.testing.assert_allclose(np.linalg.norm(selected["response"]), 1., atol=1e-9)
    assert selected["error"] < np.linalg.norm(goal)


def test_panel_and_combined_selection_are_bound():
    rows = pilot.run()
    assert len(rows) == 16
    for row in rows:
        if row["combined_selected"] == "structured":
            assert row["combined_actions"] == row["structured"]["chain"]
        else:
            assert row["combined_actions"] == row["gradient"]["32"]["actions"]
            assert not row["combined_witness"]
        assert len(row["cem_seeds"]) == 4
        assert row["gradient"]["32"]["ledger"]["objective_evaluations"] <= 32


def test_audit_checks_actual_gradient_chain():
    row = {"index": 0, "goal": [0., 0.9], "combined_selected": "gradient",
           "combined_actions": [[0., np.pi / 2], [0.9, 0.]]}
    result = audit.audit(row)
    assert result["source"] == "gradient"
    assert result["nonprogress"] and result["witness"]
    assert result["selected_prefix_reoptimized_distance"] < 1e-12
