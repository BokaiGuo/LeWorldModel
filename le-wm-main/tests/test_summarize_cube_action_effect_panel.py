import json

from experiments.summarize_cube_action_effect_panel import summarize


def _payload(rows):
    return {"rows": rows}


def test_summary_is_root_paired_and_reports_positive_distance_delta(tmp_path):
    base = tmp_path / "chart.json"
    effect = tmp_path / "effect.json"
    base.write_text(json.dumps(_payload([
        {"seed": 1, "success": False, "contact_observed": True,
         "final": {"block_goal_distance_m": 0.20}},
        {"seed": 2, "success": True, "contact_observed": True,
         "final": {"block_goal_distance_m": 0.04}},
    ])))
    effect.write_text(json.dumps(_payload([
        {"seed": 1, "success": True, "contact_observed": True,
         "effect_rank": 2, "final": {"block_goal_distance_m": 0.10}},
        {"seed": 3, "success": True, "contact_observed": True,
         "effect_rank": 2, "final": {"block_goal_distance_m": 0.02}},
    ])))
    report = summarize([base, effect])
    assert report["arms"]["effect"]["paired_roots"] == 1
    assert report["paired_deltas"]["effect"]["distance_improvement_m_positive_is_better"] == 0.1
