import hashlib
import json

import numpy as np

from experiments.spectral_reachability import codex_p1_multistart20_cp5_20261030_main as planner
from experiments.spectral_reachability import codex_p1_multistart20_physical_20261030_main as physical


def test_roster_winner_actions_and_counts():
    result = json.loads(physical.PLAN.read_bytes())
    assert planner.STEPS == 20 and len(result["rows"]) == 24
    for row in result["rows"]:
        actions = np.asarray(row["selected_actions"], dtype="<f4")
        assert actions.shape == (2, 5, 2)
        assert hashlib.sha256(actions.tobytes()).hexdigest() == row["selected_action_sha256"]
        assert row["selected_index"] == planner.prior.pick(row["candidate_scores"])
        assert row["calls"]["backward"] == 80 and row["calls"]["predict"] == 168


def test_physical_receipt_preserves_stop():
    path = physical.run.old.ROOT/"results/codex_p1_multistart20_physical_20261030_main_SAVED_RUN01.json"
    rows = json.loads(path.read_bytes())["rows"]
    assert len(rows) == 24
    stopped = [row for row in rows if row["status"] == "native_stopped"]
    assert len(stopped) == 2
    assert all(row["completed_endpoint"] is None and row["endpoint_metrics"] is None for row in stopped)
