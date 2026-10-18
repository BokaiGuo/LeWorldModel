from experiments.run_tworoom_effect_map_panel import _command, _paired
from pathlib import Path


def test_effect_map_panel_uses_same_protocol_and_explicit_flag() -> None:
    class Args:
        device = "cuda"
        episodes = 32
        seed = 20269200
        max_episode_steps = 50
        horizon = 5
        receding_horizon = 5
        action_block = 5
        num_samples = 64
        n_steps = 8
        topk = 8
        hybrid_conflict_cos = 0.0
        hybrid_model_min_norm = 0.08

    direction = _command(Args(), Path("direction.json"), False)
    effect = _command(Args(), Path("effect.json"), True)
    assert "--hybrid-use-effect-map" not in direction
    assert effect[:-1] != direction[:-1]
    assert _paired([1, 0, 1], [0, 0, 1]) == {
        "left_wins": 1, "right_wins": 0, "ties": 2,
    }
