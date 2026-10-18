from __future__ import annotations

import argparse

from experiments.run_cube_diagnostic_bundle import build_commands


def test_bundle_locks_shared_seed_roots_and_output_paths(tmp_path) -> None:
    args = argparse.Namespace(
        output=tmp_path,
        device="cuda",
        seed=17,
        calibration_roots=8,
        test_roots=3,
        tasks=[1, 2],
        macro_length=20,
        expanded_macro_length=40,
        goal_tolerance_m=0.05,
        macro_cycles=5,
        action_block=1,
        approach_steps=8,
        descent_steps=15,
        probe_steps=4,
    )
    commands = build_commands(args)
    assert commands["ceiling"][commands["ceiling"].index("--seed") + 1] == "17"
    assert commands["panel"][commands["panel"].index("--seed") + 1] == "17"
    assert commands["ceiling"][commands["ceiling"].index("--test-roots") + 1] == "3"
    assert commands["panel"][commands["panel"].index("--test-roots") + 1] == "3"
    assert commands["panel"][commands["panel"].index("--tasks") + 1:commands["panel"].index("--tasks") + 3] == ["1", "2"]
    assert str(tmp_path / "ceiling.json") in commands["failure_cause"]
    assert str(tmp_path / "panel" / "panel.json") in commands["failure_cause"]


def test_bundle_refuses_to_silently_change_device() -> None:
    args = argparse.Namespace(
        output="/tmp/cube-bundle", device="cuda", seed=1,
        calibration_roots=8, test_roots=1, tasks=[1], macro_length=20,
        expanded_macro_length=40,
        goal_tolerance_m=0.05, macro_cycles=1, action_block=1,
        approach_steps=8, descent_steps=15, probe_steps=4,
    )
    commands = build_commands(args)
    assert commands["panel"][commands["panel"].index("--device") + 1] == "cuda"
