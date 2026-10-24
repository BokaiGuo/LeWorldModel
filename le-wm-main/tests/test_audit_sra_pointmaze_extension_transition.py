from experiments.spectral_reachability.audit_sra_pointmaze_extension_transition import _contract_checks


def test_extension_audit_requires_the_high_level_float64_contract() -> None:
    manifest = {
        "state_action_dtype": "float64",
        "rollout_transition_contract": {
            "name": "mazeenv_high_level_action_v1",
            "state_clock": "states[0] is reset; states[t+1] follows actions[t]",
            "velocity_clip_before_action": 5.0,
            "raw_action_clip": [-1.0, 1.0],
            "mujoco_steps_per_action": 5,
            "physical_evaluator_sha256": "wrong",
        },
    }
    checks = _contract_checks(manifest)
    assert checks["declared_high_level_contract"]
    assert not checks["evaluator_hash_matches"]


def test_extension_audit_rejects_the_old_one_tick_or_float32_contract() -> None:
    checks = _contract_checks({"state_action_dtype": "float32", "rollout_transition_contract": {}})
    assert not checks["declared_high_level_contract"]
