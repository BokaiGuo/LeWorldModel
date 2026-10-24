import json

from experiments.spectral_reachability.bind_reacher_d2p_calibration_admission import bind


def _confirmation(seeds):
    return {"status": "calibration_only", "protocol": {"calibration_only": True, "action_parameterization": "box", "relinearization_steps": 4, "d2p_initialization": "screened", "d2p_proposal_samples": 128, "d2p_proposal_elites": 4, "warmup_policy": "warm", "execution_root_reconstruction": "snapshot"}, "inputs": {"runner_sha256": "r", "weights_sha256": "w", "adapter_sha256": "a"}, "gates": {"cuda": True}, "rows": [{"seed": seed} for seed in seeds]}


def test_binding_unions_disjoint_calibration_roots_and_preserves_selection(tmp_path):
    confirmation_path = tmp_path / "confirmation.json"; confirmation_path.write_text(json.dumps(_confirmation([3, 4])))
    import hashlib
    digest = hashlib.sha256(confirmation_path.read_bytes()).hexdigest()
    selection = {"schema": "d2p_reacher_damping_selection.v1", "status": "pass", "selected": {"damping": 10.0}, "calibration_root_seeds": [1, 2], "calibration_contract": {"runner": "r", "checkpoint": "w", "adapter": "a", "action_parameterization": "box", "relinearization_steps": 4, "d2p_initialization": "screened", "d2p_proposal_samples": 128, "d2p_proposal_elites": 4, "warmup_policy": "warm", "execution_root_reconstruction": "snapshot"}}
    admission = {"schema": "d2p_reacher_p0_model_physical_calibration_admission.v1", "status": "pass", "source_report": str(confirmation_path), "source_report_sha256": digest, "calibration_root_seeds": [3, 4], "precommitted_rule": {}, "strict_order_agreement_count": 2, "one_sided_exact_sign_p": 0.25}
    selection_path, admission_path = tmp_path / "selection.json", tmp_path / "admission.json"
    selection_path.write_text(json.dumps(selection)); admission_path.write_text(json.dumps(admission))

    result = bind(selection, admission, _confirmation([3, 4]), selection_path=selection_path, admission_path=admission_path)

    assert result["selected"] == {"damping": 10.0}
    assert result["calibration_root_seeds"] == [1, 2, 3, 4]
    assert result["model_physical_admission"]["root_seeds"] == [3, 4]
