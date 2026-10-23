import json
from pathlib import Path

import pytest

from experiments.gdd_wm.preflight_pointmaze_opportunity_router_v9 import (
    EXTENSION_SEED,
    ROOT_RANGE,
    SPLIT_COUNTS,
    _extension_contract,
    _interaction_headroom,
)


def test_v9_extension_contract_requires_a_new_independent_root_range(tmp_path: Path) -> None:
    (tmp_path / "extension_manifest.json").write_text(json.dumps({
        "schema": "pointmaze_independent_state_action_extension.v1",
        "seed": EXTENSION_SEED,
        "new_episode_count": ROOT_RANGE[1] - ROOT_RANGE[0],
        "new_root_id_range": [ROOT_RANGE[0], ROOT_RANGE[1] - 1],
        "base_root_count": ROOT_RANGE[0], "extended_root_count": ROOT_RANGE[1],
    }), encoding="utf-8")
    assert all(_extension_contract(tmp_path)["checks"].values())


def test_v9_extension_contract_rejects_reusing_the_v8_root_range(tmp_path: Path) -> None:
    (tmp_path / "extension_manifest.json").write_text(json.dumps({
        "schema": "pointmaze_independent_state_action_extension.v1",
        "seed": EXTENSION_SEED,
        "new_episode_count": 512,
        "new_root_id_range": [10808, 11319],
        "base_root_count": 10808, "extended_root_count": 11320,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="violates frozen contract"):
        _extension_contract(tmp_path)


def test_v9_launcher_is_fresh_cuda_only_and_uses_opportunity_gating() -> None:
    launcher = Path(__file__).resolve().parents[1] / "experiments" / "gdd_wm" / "run_pointmaze_opportunity_router_v9_cuda.sh"
    text = launcher.read_text(encoding="utf-8")
    assert "point_maze_v37_opportunity_router_r2_extension_20261022" in text
    assert "point_maze_v38_opportunity_router_r3_extension_20261022" in text
    assert "--root-id-min 14264 --root-id-max-exclusive 14776" in text
    assert "--train-root-count 160 --calibration-root-count 160 --outcome-root-count 64" in text
    assert SPLIT_COUNTS == {"train": 160, "calibration": 160, "outcome": 64}
    assert "--opportunity-gated" in text
    assert "--require-shared-candidate-support" in text
    assert "--recurrent-smoke" in text
    assert "refusing CPU fallback" in text
    assert "--seed 202610930" in text
    assert "512-root range" in text
    assert text.index("--recurrent-smoke") < text.index('--input-root "${BASE_POOL}"')


def test_s1_launcher_declares_its_own_disjoint_extension_contract() -> None:
    launcher = Path(__file__).resolve().parents[1] / "experiments" / "gdd_wm" / "run_pointmaze_opportunity_stability_router_s1_cuda.sh"
    text = launcher.read_text(encoding="utf-8")
    assert "point_maze_v39_opportunity_stability_s1_extension_20261022" in text
    assert "--extension-seed 202611000" in text
    assert "--root-id-min 14776 --root-id-max-exclusive 15288" in text
    assert "--router-kind advantage_bootstrap_lower" in text
    assert "--opportunity-gated" in text
    assert "--recurrent-smoke" in text
    assert "refusing CPU fallback" in text


def test_s2_launcher_uses_a_new_tail_robust_bank_without_lowering_headroom() -> None:
    launcher = Path(__file__).resolve().parents[1] / "experiments" / "gdd_wm" / "run_pointmaze_opportunity_stability_router_s2_cuda.sh"
    text = launcher.read_text(encoding="utf-8")
    assert "point_maze_v40_opportunity_stability_s2_extension_20261022" in text
    assert "--extension-seed 202611100" in text
    assert "--root-id-min 15288 --root-id-max-exclusive 15800" in text
    assert "expanded_tail_robust_goal_balanced_interaction_pairs" in text
    assert "--router-kind advantage_bootstrap_lower" in text
    assert "--recurrent-smoke" in text
    assert "refusing CPU fallback" in text
    assert "--reuse-identical-output" in text


def test_s3_launcher_increases_only_independent_calibration_power() -> None:
    launcher = Path(__file__).resolve().parents[1] / "experiments" / "gdd_wm" / "run_pointmaze_opportunity_stability_router_s3_cuda.sh"
    text = launcher.read_text(encoding="utf-8")
    assert "point_maze_v42_opportunity_stability_s3_power_extension_20261022" in text
    assert "--extension-seed 202611200" in text
    assert "--episode-count 640" in text
    assert "--root-id-min 15800 --root-id-max-exclusive 16440" in text
    assert "--train-root-count 160 --calibration-root-count 288 --outcome-root-count 64" in text
    assert "expanded_tail_robust_goal_balanced_interaction_pairs" in text
    assert "--router-kind advantage_bootstrap_lower" in text
    assert "--recurrent-smoke" in text
    assert "refusing CPU fallback" in text


def test_s4_launcher_uses_denser_preoutcome_tail_bank_and_fresh_roots() -> None:
    launcher = Path(__file__).resolve().parents[1] / "experiments" / "gdd_wm" / "run_pointmaze_opportunity_stability_router_s4_cuda.sh"
    text = launcher.read_text(encoding="utf-8")
    assert "point_maze_v43_opportunity_stability_s4_extension_20261022" in text
    assert "--episode-count 640" in text
    assert "--extension-seed 202611300" in text
    assert "--root-id-min 16440 --root-id-max-exclusive 17080" in text
    assert "ultra_tail_robust_goal_balanced_interaction_pairs" in text
    assert "--train-root-count 160 --calibration-root-count 288 --outcome-root-count 64" in text
    assert "--router-kind advantage_bootstrap_lower" in text
    assert "--recurrent-smoke" in text


def test_s5_launcher_changes_action_family_but_preserves_powered_router_protocol() -> None:
    launcher = Path(__file__).resolve().parents[1] / "experiments" / "gdd_wm" / "run_pointmaze_opportunity_stability_router_s5_cuda.sh"
    text = launcher.read_text(encoding="utf-8")
    assert "point_maze_v44_opportunity_stability_s5_extension_20261022" in text
    assert "--episode-count 640" in text
    assert "--extension-seed 202611400" in text
    assert "--root-id-min 17080 --root-id-max-exclusive 17720" in text
    assert "directional_tail_robust_goal_balanced_interaction_pairs" in text
    assert "--train-root-count 160 --calibration-root-count 288 --outcome-root-count 64" in text
    assert "--router-kind advantage_bootstrap_lower" in text
    assert "--recurrent-smoke" in text


def test_extension_contract_accepts_an_explicit_future_disjoint_range(tmp_path: Path) -> None:
    future_seed, future_range = 202611000, (14776, 15288)
    (tmp_path / "extension_manifest.json").write_text(json.dumps({
        "schema": "pointmaze_independent_state_action_extension.v1",
        "seed": future_seed,
        "new_episode_count": future_range[1] - future_range[0],
        "new_root_id_range": [future_range[0], future_range[1] - 1],
        "base_root_count": future_range[0], "extended_root_count": future_range[1],
    }), encoding="utf-8")
    contract = _extension_contract(tmp_path, extension_seed=future_seed, root_range=future_range)
    assert all(contract["checks"].values())


def test_tail_robust_headroom_schema_preserves_the_same_fixed_threshold_quantities() -> None:
    normalized = _interaction_headroom({
        "design_all_root_headroom": {
            "winner_entropy_normalized": 0.75,
            "best_second_gap_q10": 0.0015,
            "best_second_gap_ge_0p003_fraction": 0.6,
            "candidate_cost_spread_q10": 0.08,
        },
    })
    assert normalized == {
        "winner_entropy_normalized": 0.75,
        "best_second_gap_q10": 0.0015,
        "best_second_gap_ge_0p003_fraction": 0.6,
        "candidate_cost_spread_q10": 0.08,
    }


def test_v9_preflight_rejects_an_undeclared_router_before_state_or_render_access(tmp_path: Path) -> None:
    from experiments.gdd_wm.preflight_pointmaze_opportunity_router_v9 import preflight

    with pytest.raises(ValueError, match="declared advantage router"):
        preflight(
            dataset_root=tmp_path, contexts=[], root_seed=1, held_action_seed=2,
            train_action_seed=3, router_kind="not-a-router",
        )
