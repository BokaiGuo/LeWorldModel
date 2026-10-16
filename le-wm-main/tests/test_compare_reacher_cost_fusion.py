from __future__ import annotations

import json

from experiments.compare_reacher_cost_fusion import compare


def _write(path, *, blend, shuffled, success, final, minimum, elite=None):
    path.write_text(
        json.dumps(
            {
                "schema": "live_reacher_sequence_candidate_planning.v1",
                "protocol": {
                    "checkpoint": "checkpoint",
                    "horizon": 5,
                    "receding_horizon": 5,
                    "action_block": 5,
                    "candidate_bank": "capacity17",
                    "candidate_count": 17,
                    "final_action_selection": "best",
                    "target_delta": [0.65, -0.65],
                    "cost_metric": "capacity_safe_readout",
                    "capacity_readout_artifact": "readout.npz",
                    "capacity_base_cost_blend_weight": blend,
                    "capacity_shuffle_base_cost_control": shuffled,
                    "capacity_base_cost_elite_k": elite,
                },
                "frozen_lewm": {
                    "seeds": [1, 2, 3],
                    "episode_successes": success,
                    "final_qpos_distance": final,
                    "min_qpos_distance": minimum,
                    "runtime_diagnostics": {},
                },
            }
        ),
        encoding="utf-8",
    )


def test_compare_reacher_cost_fusion_keeps_placebo_explicit(tmp_path) -> None:
    no_dynamics = tmp_path / "no.json"
    fusion = tmp_path / "fusion.json"
    shuffled = tmp_path / "shuffled.json"
    _write(
        no_dynamics,
        blend=None,
        shuffled=False,
        success=[0, 1, 0],
        final=[0.4, 0.2, 0.3],
        minimum=[0.3, 0.1, 0.2],
    )
    _write(
        fusion,
        blend=0.5,
        shuffled=False,
        success=[1, 1, 0],
        final=[0.2, 0.2, 0.4],
        minimum=[0.1, 0.1, 0.3],
    )
    _write(
        shuffled,
        blend=0.5,
        shuffled=True,
        success=[0, 0, 0],
        final=[0.5, 0.3, 0.4],
        minimum=[0.4, 0.2, 0.3],
    )
    result = compare(no_dynamics, fusion, shuffled, draws=1000, seed=4)
    assert result["point_estimates"]["fusion"]["successes"] == 2
    assert result["paired"]["fusion_minus_no_dynamics"]["success"]["mean"] == 1 / 3
    assert result["paired"]["fusion_minus_shuffled"]["success"]["mean"] == 2 / 3
    assert result["paired"]["fusion_minus_no_dynamics"]["success_discordance"] == {
        "right_only_wins": 1,
        "left_only_wins": 0,
        "ties": 2,
    }


def test_compare_reacher_cost_elite_tiebreak(tmp_path) -> None:
    no_dynamics = tmp_path / "no.json"
    fusion = tmp_path / "fusion.json"
    shuffled = tmp_path / "shuffled.json"
    common = {
        "success": [0, 1, 0],
        "final": [0.4, 0.2, 0.3],
        "minimum": [0.3, 0.1, 0.2],
    }
    _write(no_dynamics, blend=None, shuffled=False, elite=None, **common)
    _write(fusion, blend=None, shuffled=False, elite=2, **common)
    _write(shuffled, blend=None, shuffled=True, elite=2, **common)
    result = compare(no_dynamics, fusion, shuffled, draws=100, seed=5)
    assert result["protocol"]["intervention_type"] == "source_action_elite_tiebreak"
    assert result["protocol"]["intervention_value"] == 2
