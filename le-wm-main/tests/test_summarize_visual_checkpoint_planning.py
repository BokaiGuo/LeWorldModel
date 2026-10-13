import json
from pathlib import Path

from experiments.summarize_visual_checkpoint_planning import summarize


def _write(path: Path, seed: int, success: int) -> None:
    episodes = [
        {"strict_success": int(index < success), "block_xy_distance": 10.0 + index, "angle_error": 0.1 + 0.01 * index}
        for index in range(4)
    ]
    payload = {
        "protocol": {
            "checkpoint_immutable": True,
            "runtime_state_access": False,
            "runtime_successor_verification": False,
            "interface": {"planning_horizon": 6, "cem_samples": 192},
        },
        "planning": {"episodes": episodes},
    }
    path.mkdir()
    (path / "checkpoint_planning.json").write_text(json.dumps(payload))


def test_multi_seed_summary_keeps_seed_as_bootstrap_unit(tmp_path: Path):
    first = tmp_path / "seed_1" / "checkpoint_planning.json"
    second = tmp_path / "seed_2" / "checkpoint_planning.json"
    _write(first.parent, 1, 1)
    _write(second.parent, 2, 2)
    report = summarize([first, second], expected_seeds=[1, 2])
    assert report["protocol"]["n_seed_clusters"] == 2
    assert report["aggregate"]["strict_success_rate"]["bootstrap_unit"] == "seed"
    assert report["acceptance"]["status"] == "negative_or_unstable_boundary"
