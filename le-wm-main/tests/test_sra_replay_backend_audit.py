from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_versioned_replay_parent_is_cuda_gated_and_records_backend():
    source = (ROOT / "experiments/spectral_reachability/audit_sra_pointmaze_replay_legacy_backend_cuda.py").read_text()
    assert "require_cuda()" in source
    assert '"backend": backend' in source
    assert "no learned-model CPU fallback" in source


def test_backend_replays_the_high_level_step_contract():
    source = (ROOT / "experiments/spectral_reachability/audit_sra_pointmaze_replay_backend.py").read_text()
    assert "FRAME_SKIP, VELOCITY_LIMIT = 5, 5.0" in source
    assert "for _ in range(FRAME_SKIP)" in source
    assert 'choices=("mujoco", "mujoco_py")' in source


def test_parent_binds_mujoco_py_runtime_explicitly_when_requested():
    source = (ROOT / "experiments/spectral_reachability/audit_sra_pointmaze_replay_legacy_backend_cuda.py").read_text()
    assert 'environment["MUJOCO_PY_MUJOCO_PATH"]' in source
    assert 'environment["LD_LIBRARY_PATH"]' in source
    assert '"libmujoco_sha256": sha256_file(runtime_library)' in source


def test_legacy_launcher_selects_versioned_runtime_and_position_contract_gate():
    source = (ROOT / "experiments/spectral_reachability/run_sra_pointmaze_replay_legacy_backend_cuda.sh").read_text()
    assert "--engine mujoco_py" in source
    assert "--allow-position-label-contract" in source


def test_versioned_contract_binds_the_scripts_used_for_candidate_endpoints():
    source = (ROOT / "experiments/spectral_reachability/audit_sra_pointmaze_replay_legacy_backend_cuda.py").read_text()
    assert '"replay_backend_script_sha256"' in source
    assert '"endpoint_backend_script_sha256"' in source


def test_receipt_contract_launcher_uses_module_execution_and_frozen_root_panel():
    source = (ROOT / "experiments/spectral_reachability/run_sra_pointmaze_receipt_contract_cuda.sh").read_text()
    assert 'experiments.spectral_reachability.audit_sra_pointmaze_replay_legacy_backend_cuda' in source
    assert "audit_sra_pointmaze_replay_legacy_backend_cuda" in source
    assert "--engine mujoco" in source
    assert "1832,1833" in source
