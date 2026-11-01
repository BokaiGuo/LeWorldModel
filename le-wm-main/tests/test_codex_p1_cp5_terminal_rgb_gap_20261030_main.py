import copy

import pytest

from experiments.spectral_reachability import codex_p1_cp5_terminal_rgb_gap_20261030_main as gap


def test_saved_panel_has_48_action_bound_terminal_inputs():
    pairs, _ = gap.read(gap.PAIRS, gap.PAIR_SHA)
    rows = gap.terminal_inputs(pairs)
    assert len(rows) == 48
    assert [(r["index"], r["label"]) for r in rows] == [
        (i, label) for i in range(24) for label in ("two_step", "twenty_step")
    ]


def test_tampered_pair_identity_is_rejected():
    pairs, _ = gap.read(gap.PAIRS, gap.PAIR_SHA)
    changed = copy.deepcopy(pairs)
    changed["rows"][0]["twenty_step"]["raw10_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="planner action/score binding"):
        gap.terminal_inputs(changed)
