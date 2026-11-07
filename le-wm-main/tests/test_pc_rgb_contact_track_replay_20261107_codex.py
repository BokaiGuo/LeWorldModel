import pytest

from experiments.pc_rgb_contact_track_replay_20261107_codex import compare


def panel():
    new, old = [], []
    for root in range(16):
        conditions = []
        for color in ("white", "gray"):
            for side in (-1, 1):
                actions = {}
                for index, name in enumerate(("zero", "plus", "minus")):
                    value = float(index + root)
                    actions[name] = {"predicted_cost": value}
                    new.append({"root": root, "appearance": color, "side": side, "action": name, "score": value})
                conditions.append({"appearance": color, "side": side, "actions": actions})
        old.append({"root": root, "conditions": conditions})
    return new, old


def test_compare_recomputes_scores_and_choices():
    new, old = panel()
    result = compare(new, old)
    assert result["close_count"] == 192
    assert result["choice_matches"] == 64
    new[0]["score"] = 10
    result = compare(new, old)
    assert result["close_count"] == 191
    assert result["choice_matches"] == 63


def test_missing_or_duplicate_candidates_fail():
    new, old = panel()
    with pytest.raises(ValueError, match="denominator"):
        compare(new[:-1], old)
    new[-1] = new[0]
    with pytest.raises(ValueError, match="denominator"):
        compare(new, old)
