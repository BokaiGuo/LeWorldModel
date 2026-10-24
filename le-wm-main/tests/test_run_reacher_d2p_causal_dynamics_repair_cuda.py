from pathlib import Path


def test_repair_launcher_is_cuda_only_serial_and_seals_outcomes_on_failed_gates():
    script = (Path(__file__).resolve().parents[1] / "experiments" / "spectral_reachability" / "run_reacher_d2p_causal_dynamics_repair_cuda.sh").read_text(encoding="utf-8")
    assert "--device cuda" in script
    assert "build_reacher_d2p_repair_training_pool" in script
    assert "CPU" in script
    assert "p0_damping_selection.json" in script
    assert "P0 did not unlock P1" in script
    assert "repaired checkpoint still fails" in script
    assert "--outcome-only" in script
    assert "observed_encoded_goal_cost" in script
    assert "-mindepth 1 -maxdepth 1" in script
