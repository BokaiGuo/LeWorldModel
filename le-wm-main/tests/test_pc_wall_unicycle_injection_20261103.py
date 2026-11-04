import json

import numpy as np

from experiments.spectral_reachability import pc_wall_unicycle_injection_20261103 as exp


def test_injected_witness_is_bound_and_fully_feasible():
    source = json.loads(exp.SOURCE.read_text())
    for prior in source["rows"]:
        actions = np.asarray(prior["selected"]["actions"])
        outcome = exp.wall.replay(actions)
        assert outcome["terminal_error"] < 1e-8
        assert not any(outcome["blocked"])


def test_recorded_injection_reproduces_one_search():
    source = json.loads(exp.SOURCE.read_text())
    saved = json.loads((exp.ROOT / "results/pc_wall_unicycle_injection_20261103_DEV01.json").read_text())
    row = exp.run(source["rows"][0]["seed"], source["rows"][0]["selected"]["actions"])
    expected = saved["rows"][0]
    for key in ("offered", "matching_context_branches", "matching_action_branches",
                "selected_actions", "selected_error", "compiler_counts", "backend"):
        assert row[key] == expected[key]
    assert row["offered"][1] > 0 and row["offered"][2] == 0
