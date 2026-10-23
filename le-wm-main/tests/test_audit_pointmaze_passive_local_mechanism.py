from experiments.audit_pointmaze_passive_local_mechanism import audit


def _row(root: int, *, local_index: int, visual_index: int, action: list[float]) -> dict:
    return {
        "episode_id": root,
        "trace": [{
            "selected_index": local_index,
            "selected_action": action,
            "action_transfer_gate": {
                "local_verified_release": True,
                "local_affine_selected_index": local_index,
                "local_affine_visual_index": visual_index,
                "local_affine_supported_candidate_fraction": 0.5,
            },
        }],
    }


def test_audit_distinguishes_nominal_release_from_actual_action_difference() -> None:
    passive = {"rows": [
        _row(1, local_index=2, visual_index=2, action=[0.2, 0.0]),
        _row(2, local_index=3, visual_index=1, action=[0.0, 0.2]),
    ]}
    htsc = {"rows": [
        _row(1, local_index=2, visual_index=2, action=[0.2, 0.0]),
        _row(2, local_index=1, visual_index=1, action=[0.2, 0.0]),
    ]}
    result = audit(passive, htsc)
    assert result["nominal_local_release_count"] == 2
    assert result["same_candidate_release_count"] == 1
    assert result["divergent_candidate_release_count"] == 1
    assert result["roots_with_any_executed_action_difference_vs_htsc"] == 1
    assert result["executed_action_difference_block_count_vs_htsc"] == 1
