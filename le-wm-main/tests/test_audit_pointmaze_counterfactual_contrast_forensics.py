from __future__ import annotations

import json
from pathlib import Path

from experiments.decision_benchmark import audit_pointmaze_counterfactual_contrast_forensics as forensic


def _arm(winners: list[int], regrets: list[float]) -> dict[str, object]:
    return {
        "roots": {
            f"root-{index}": {"winner_candidate_id": winner, "regret": regret}
            for index, (winner, regret) in enumerate(zip(winners, regrets, strict=True))
        }
    }


def test_forensic_audit_separates_action_prior_from_unvalidated_visual_gain(tmp_path: Path) -> None:
    build = tmp_path / "build.json"
    frozen = tmp_path / "frozen.json"
    score = tmp_path / "score.json"
    build.write_text(json.dumps({"interaction_pair_design": {
        "candidate_design": "state_interaction_pairs", "held_pair_hashes": [f"candidate-{index}" for index in range(16)],
    }}))
    frozen.write_text(json.dumps({
        "authorization": {"permitted": False},
        "trained_interaction_diagnostics": {"interaction_active_at_1e-6": True},
    }))
    score.write_text(json.dumps({
        "action_only": _arm([8, 8, 8, 9], [0.0, 0.1, 0.0, 0.0]),
        "recurrent": _arm([8, 8, 9, 9], [0.0, 0.0, 0.0, 0.0]),
        "selected": _arm([8, 8, 9, 9], [0.0, 0.0, 0.0, 0.0]),
        "observability_suite": {
            "paired_gain_visual_over_action_only": {"lower": -0.1, "mean_gain": 0.02, "upper": 0.2},
            "paired_gain_state_over_action_only": {"lower": -0.2, "mean_gain": 0.01, "upper": 0.3},
            "paired_gain_visual_over_cyclic_shuffle": {"lower": -0.1, "mean_gain": 0.04, "upper": 0.2},
        },
    }))
    output = tmp_path / "audit.json"
    result = forensic.audit(build, frozen, score, output)
    assert result["mechanism_flags"]["authorization_rejected"] is True
    assert result["mechanism_flags"]["action_prior_concentrated"] is True
    assert result["mechanism_flags"]["history_interaction_active"] is True
    assert result["mechanism_flags"]["visual_beats_action_with_positive_lower_ci"] is False
    assert result["paired_gains"]["recurrent_over_action_only"]["mean_gain"] == 0.025
    assert result["ranking_intervention"]["changed_winner_root_count"] == 1
    assert result["mechanism_flags"]["recurrent_override_has_positive_lower_ci"] is False
    assert result["action_only"]["dominant_candidate_id"] == 8
    assert result["action_only"]["winner_support_fraction"] == 0.125
    assert output.is_file()
