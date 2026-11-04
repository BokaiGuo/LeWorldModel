"""Independent offline checks and exact replays of the depth-3 ablation."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .d2p_composition_witness_v2 import digest
from .pc_atlas_matched_exact_dev_20261103_exclusive import ROOT
from .pc_atlas_unicycle_smoke_20261103 import UnicycleBackend

BASE = Path(__file__).resolve().parents[2]
RESULT = BASE / "results/pc_atlas_depth3_matched_root_source_20261103_DEV01.json"
EXPECTED = "cce9442bd7120bfcf71e8c443a3d0a3482244e46e9060dcab0d4c828fbd918c2"
ARMS = ("random", "spectral", "ordinary")


def verify(report):
    if report["schema"] != "pc_atlas_depth3_matched_root_source.v1" or len(report["rows"]) != 72:
        raise ValueError("unexpected report")
    backend = UnicycleBackend()
    replayed = 0
    for offset in range(0, 72, 3):
        group = report["rows"][offset:offset + 3]
        if [r["arm"] for r in group] != list(ARMS):
            raise ValueError("arm order")
        if any((r["index"], r["seed"]) != (group[0]["index"], group[0]["seed"]) for r in group):
            raise ValueError("paired identity")
        if any(r["root_bank"][:2] != group[0]["root_bank"][:2] for r in group):
            raise ValueError("root controls")
        for row in group:
            expected_counts = {"rollout_attempted": 108, "rollout_returned": 108,
                               "jacobian_attempted": 13, "jacobian_returned": 13}
            if row["counts"] != expected_counts or row["backend_action_steps"] != {"rollout": 116, "jacobian": 13}:
                raise ValueError("counts")
            angle = 2 * np.pi * row["index"] / 64 + .031
            goal = np.array([np.cos(angle), np.sin(angle), 0.])
            if not np.array_equal(goal, row["goal"]):
                raise ValueError("goal")
            if len(row["programs"]) != 4:
                raise ValueError("program count")
            costs = []
            for i, program in enumerate(row["programs"]):
                actions = np.asarray(program["actions"])
                if actions.shape != (3, 2) or not np.array_equal(actions[:1], row["root_bank"][i]):
                    raise ValueError("root action binding")
                replay = backend.rollout(ROOT, actions)
                replayed += 1
                if digest(replay) != digest(program["output_context"]):
                    raise ValueError("replay context")
                residual = goal - np.asarray(replay["latent_history"][-1])
                cost = float((residual[0] ** 2 + residual[1] ** 2) / 3)
                if not np.isclose(cost, program["terminal_cost"], rtol=1e-12, atol=1e-15):
                    raise ValueError("terminal cost")
                costs.append(cost)
            selected = min(range(4), key=lambda i: (costs[i], i))
            if selected != row["selected_index"] or row["selected_actions"] != row["programs"][selected]["actions"]:
                raise ValueError("selected program")
            selected_cost = costs[selected]
            if not np.isclose(row["selected_cost"], selected_cost, rtol=1e-12, atol=1e-15):
                raise ValueError("selected cost")
            if not np.isclose(row["terminal_error"], np.sqrt(3 * selected_cost), rtol=1e-12, atol=1e-15):
                raise ValueError("terminal error")
    return {"rows": 72, "replayed_programs": replayed, "status": "passed"}


def main():
    if hashlib.sha256(RESULT.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError("receipt sha256")
    print(json.dumps(verify(json.loads(RESULT.read_text())), sort_keys=True))


if __name__ == "__main__":
    main()
