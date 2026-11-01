from pathlib import Path

from experiments.spectral_reachability.train_cube_d2p_dynamics_repair import _sha256
from experiments.spectral_reachability.audit_cube_d2p_p1_full_bank import calibration_gates


def test_cube_repair_sha256_is_streaming_equivalent(tmp_path: Path) -> None:
    path = tmp_path / "source.bin"
    path.write_bytes(b"source-locked-cube-repair")
    assert _sha256(path) == "61a44391c317d5d38b68df84ccdfb99cb557e873c3dbbe74edba76b68ae5112e"


def test_cube_repair_chain_requires_complete_manifest_and_new_output() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "spectral_reachability"
        / "run_cube_d2p_repair_p1_chain_cuda.sh"
    ).read_text(encoding="utf-8")
    assert "repair_manifest.json" in script
    assert "test ! -e" in script
    assert "--device cuda" in script
    assert "MUJOCO_GL=egl" in script


def test_cube_p1_calibration_rejects_the_historical_positive_control_failure() -> None:
    rows = [
        {
            "positive_control_selected": index < 3,
            "positive_control_physics_cost": 0.0,
            "physics_best_cost": 0.0,
            "spearman_model_vs_physics": rho,
        }
        for index, rho in enumerate((0.7176, -0.6, -0.1294, 0.6705, 0.6058, 0.5264, 0.3647, 0.7176))
    ]
    gates, aggregate = calibration_gates(rows)
    assert aggregate["positive_control_selected_count"] == 3
    assert not gates["positive_control_top1_at_least_75pct"]
    assert not gates["mean_spearman_at_least_0_60"]
    assert not gates["no_per_root_order_reversal"]
