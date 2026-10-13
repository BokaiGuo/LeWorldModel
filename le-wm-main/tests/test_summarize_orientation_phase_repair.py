import json

from experiments.summarize_orientation_phase_repair import compare, load


def _payload(eval_seed: int, success: bool, xy: float, angle: float, source: str) -> dict:
    return {
        "protocol": {
            "checkpoint_immutable": True,
            "runtime_state_access": False,
            "runtime_successor_verification": False,
            "eval_seed": eval_seed,
        },
        "planning": {
            "episodes": [
                {"episode": 0, "strict_success": success, "block_xy_distance": xy, "angle_error": angle,
                 "trace": [{"action_source": source}]},
                {"episode": 1, "strict_success": False, "block_xy_distance": xy + 1, "angle_error": angle + 1,
                 "trace": [{"action_source": source}, {"action_source": "base"}]},
            ]
        },
    }


def test_compare_requires_shared_episode_set_and_reports_orientation_usage(tmp_path):
    baseline_path = tmp_path / "base.json"
    expert_path = tmp_path / "expert.json"
    baseline_path.write_text(json.dumps(_payload(9, False, 10.0, 1.0, "base")))
    expert_path.write_text(json.dumps(_payload(9, True, 8.0, 0.5, "orientation_expert")))
    report = compare(load(baseline_path), load(expert_path), "expert", 1)
    assert report["paired_deltas_expert_minus_base"]["strict_success"]["mean"] == 0.5
    assert report["orientation_phase_usage"]["episodes_using_orientation_expert"] == 2
