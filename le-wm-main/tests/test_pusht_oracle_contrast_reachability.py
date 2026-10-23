import numpy as np

from experiments import audit_pusht_oracle_contrast_reachability as audit


def test_probe_anchored_bank_has_fixed_zero_plus_minus_order(monkeypatch):
    monkeypatch.setattr(audit, "edge_chart", lambda _env, _state: {"edge_id": "0:1", "contact_gap": 12.0})
    monkeypatch.setattr(
        audit,
        "matched_actions",
        lambda _chart, _inward, _tangent: {
            "zero": np.asarray([0.0, 0.0]),
            "plus": np.asarray([0.18, 0.55]),
            "minus": np.asarray([0.18, -0.55]),
            "pair_valid": True,
        },
    )
    _actions, bank = audit.probe_action_bank(object(), np.zeros(7), inward_magnitude=0.18)
    np.testing.assert_allclose(bank, [[0.0, 0.0], [0.18, 0.55], [0.18, -0.55]])


def test_disjoint_root_contract_rejects_repeated_source_state():
    root = {"state": np.zeros(7), "goal": np.zeros(7)}
    other = {"state": np.ones(7), "goal": np.ones(7)}
    audit.assert_root_disjoint([root], [other])
    try:
        audit.assert_root_disjoint([root], [root])
    except ValueError as error:
        assert "overlap" in str(error)
    else:  # pragma: no cover - protocol must fail closed
        raise AssertionError("overlapping root splits were accepted")


def test_disjoint_root_contract_rejects_same_state_with_different_goals():
    state = np.zeros(7)
    first = {"state": state, "goal": np.zeros(7)}
    second = {"state": state.copy(), "goal": np.ones(7)}
    # Complete identities differ, but a shared physical root would leak the
    # same rendered/history context across cohorts and must be rejected.
    assert audit.root_identity(first) != audit.root_identity(second)
    try:
        audit.assert_root_disjoint([first], [second])
    except ValueError as error:
        assert "source state" in str(error)
    else:  # pragma: no cover - protocol must fail closed
        raise AssertionError("same physical state was reused across cohorts")


def test_candidate_manifest_has_frozen_identifiers_and_hashes_actions():
    first = np.asarray([[0.0, 0.0], [0.18, 0.55], [0.18, -0.55]])
    second = np.asarray([[0.0, 0.0], [-0.18, 0.55], [-0.18, -0.55]])
    manifest = audit.candidate_manifest(first)
    assert [row["candidate_id"] for row in manifest] == list(audit.CANDIDATE_IDS)
    assert audit.candidate_manifest_fingerprint(first) != audit.candidate_manifest_fingerprint(second)


def test_nonterminal_collector_filters_terminal_roots(monkeypatch):
    terminal = {"state": np.zeros(7), "goal": np.zeros(7)}
    active = {"state": np.ones(7), "goal": np.zeros(7)}
    monkeypatch.setattr(audit, "make_boundary_contact_roots", lambda _env, _count, _seed: [terminal, active, active, active, active])
    monkeypatch.setattr(audit, "is_pose_success", lambda state, goal: bool(np.allclose(state, goal)))
    roots = audit.collect_nonterminal_contact_roots(object(), count=2, seed=4)
    assert len(roots) == 2
    assert all(not np.allclose(root["state"], root["goal"]) for root in roots)


def test_reachability_without_eligibility_falsifies_necessity():
    result = audit.empirical_necessity_sufficiency([
        {"contrast": {"eligible_for_later_visual_ranker": False}, "two_pulse_oracle": {"pose_success": True}},
        {"contrast": {"eligible_for_later_visual_ranker": True}, "two_pulse_oracle": {"pose_success": True}},
    ])
    assert result["empirical_necessity"]["status"] == "falsified"
    assert result["empirical_sufficiency"]["status"] == "not_falsified_not_proven"


def test_eligible_oracle_failure_falsifies_sufficiency():
    result = audit.empirical_necessity_sufficiency([
        {"contrast": {"eligible_for_later_visual_ranker": True}, "two_pulse_oracle": {"pose_success": False}},
        {"contrast": {"eligible_for_later_visual_ranker": False}, "two_pulse_oracle": {"pose_success": False}},
    ])
    assert result["empirical_sufficiency"]["status"] == "falsified"
    assert result["empirical_necessity"]["status"] == "not_falsified_not_proven"


def test_no_eligible_roots_makes_sufficiency_unevaluable_not_true():
    result = audit.empirical_necessity_sufficiency([
        {"contrast": {"eligible_for_later_visual_ranker": False}, "two_pulse_oracle": {"pose_success": False}},
    ])
    assert result["empirical_sufficiency"]["status"] == "unevaluable_no_eligible_roots"
    assert audit.RUNTIME_CONTRACT["uses_visual_or_learned_ranker"] is False
    assert audit.RUNTIME_CONTRACT["oracle_receives_eligibility_label"] is False
