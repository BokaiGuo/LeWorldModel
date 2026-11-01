"""Verify reporting metrics against the installed PushT native success function."""
import json
from pathlib import Path

import numpy as np
from stable_worldmodel.envs.pusht.env import PushT

from experiments.spectral_reachability.codex_p1_h10_paired_table_20261030_main import physical_metrics


def test_saved_endpoints_match_native_success_without_an_environment_instance():
    root = Path(__file__).resolve().parents[1] / "results"
    prepared = json.loads((root / "p1_response_prefix_h10_20261031_prepare_RUN01.json").read_text())
    planned = json.loads((root / "p1_response_prefix_cp5_physical_20261031_consume_RUN01.json").read_text())
    plans = {row["index"]: row for row in planned["rows"]}
    checked = 0
    for anchor in prepared["anchors"]:
        states = [anchor["root_state"], anchor["checkpoint"]["state"], plans[anchor["index"]]["completed_endpoint"]]
        for state in states:
            if state is None:
                continue
            goal_array, state_array = np.asarray(anchor["goal"]), np.asarray(state)
            native, _ = PushT.eval_state(None, goal_array, state_array)
            assert physical_metrics(state, anchor["goal"])["native_success"] == bool(native)
            checked += 1
    assert checked == 41


def test_native_success_includes_pusher_and_strict_boundaries():
    goal = np.zeros(7)
    state = np.zeros(7)
    assert PushT.eval_state(None, goal, state)[0]
    state[0] = 20
    assert not PushT.eval_state(None, goal, state)[0]
    assert not physical_metrics(state.tolist(), goal.tolist())["native_success"]
    state[0], state[4] = 0, np.pi / 9
    assert not PushT.eval_state(None, goal, state)[0]
    assert not physical_metrics(state.tolist(), goal.tolist())["native_success"]
