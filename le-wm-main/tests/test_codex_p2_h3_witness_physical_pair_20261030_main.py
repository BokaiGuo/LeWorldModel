import copy

import numpy as np
import pytest

from experiments.spectral_reachability import codex_p2_h3_witness_physical_pair_20261030_main as pairing


def fixtures():
    winners, cases, plans, rows, events = [], [], [], [], []
    for root in range(4):
        root_hash = f"root{root}"
        cases.append({"root_index": root, "root_sha256": root_hash,
                      "context": {"history": [root]}, "goal": [root]})
        for arm in ("joint", "suffix"):
            actions = np.full((15, 2), root / 10, dtype="<f4")
            commands = actions.tolist()
            raw = actions.tobytes().hex()
            winners.append({"root": root, "arm": arm, "winner_tag": arm, "raw_float32_hex": raw})
            plans.append({"root_index": root, "root_sha256": root_hash, "method": arm,
                          "predictor_context": {"history": [root]}, "goal": [root],
                          "commands": commands, "raw_float32_hex": raw})
            rows.append({"root_index": root, "root_sha256": root_hash, "method": arm,
                         "completed_commands": 15, "official": {"official_success": False}})
            events.extend({"kind": "env_step", "phase": "selected", "completed": False,
                           "root_sha256": root_hash, "method": arm, "command": cmd}
                          for cmd in commands)
    return winners, cases, {"plans": plans}, {"rows": rows}, events


def test_eight_slots_bound():
    rows = pairing.pair(*fixtures())
    assert len(rows) == 8
    assert all(row["physical_steps"] == 15 and not row["native_success"] for row in rows)


def test_rejects_action_and_context_drift_and_missing_step():
    for change in ("bits", "context", "missing"):
        args = list(copy.deepcopy(fixtures()))
        if change == "bits":
            args[0][0]["raw_float32_hex"] = "00" * 120
            # Root 0 contains zeros; alter one byte instead.
            args[0][0]["raw_float32_hex"] = "01" + "00" * 119
        elif change == "context":
            args[1][0]["context"] = {"history": [-1]}
        else:
            args[4].pop()
        with pytest.raises(ValueError):
            pairing.pair(*args)
