"""Reconstruct the scored and dispatched fallback actions from sealed logs."""
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.pc_input_permissions import file_hash
from experiments import pc_rgb_timed_prefix_search_20261107_codex as timed

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/pc_tworoom_rgb_timed_prefix_feedback_20261107_codex.json"
RUNNER = ROOT / "experiments/pc_tworoom_rgb_timed_prefix_feedback_20261107_codex.py"
OUTPUT = ROOT / "results/pc_tworoom_timed_fallback_clock_audit_20261107_codex.json"


def action_hash(actions):
    return hashlib.sha256(np.ascontiguousarray(actions, dtype="<f4").tobytes()).hexdigest()


def audit(record):
    if record["adapter_sha256"] != file_hash(timed.__file__) or record["runner_sha256"] != file_hash(RUNNER):
        raise ValueError("historical source drift")
    checked = []
    for row in record["rows"]:
        for arm, outcome in row["arms"].items():
            previous = None
            for decision in outcome["decisions"]:
                actions = np.asarray(decision["actions"], dtype=np.float32)
                dispatched = np.asarray(decision["dispatched"], dtype=np.float32)
                if action_hash(actions) != decision["actions_sha256"]:
                    raise ValueError("recorded action hash mismatch")
                if (None if previous is None else action_hash(previous)) != decision["old_tail_sha256"]:
                    raise ValueError("old-tail lineage mismatch")
                if not np.array_equal(actions[:len(dispatched)], dispatched):
                    raise ValueError("dispatched actions differ from recorded chain")
                if decision["remaining_steps"] == 5:
                    parents, _, _ = timed.build(decision["candidate_seed"], previous, 5, count=13)
                    if previous is None:
                        parents[0] = 0
                    costs = np.asarray(decision["parent_cost"])
                    choice = decision["choice"]
                    if (actions.shape != (25, 2) or costs.shape != (13,)
                            or choice["source"] != "fallback"
                            or choice["index"] != int(np.argmin(costs))
                            or choice["cost"] != float(costs[choice["index"]])
                            or not np.array_equal(actions, parents[choice["index"]])):
                        raise ValueError("fallback reconstruction mismatch")
                    checked.append({"root": row["root"], "arm": arm, "returned_steps": len(actions),
                                    "scored_steps": 5, "dispatched_steps": len(dispatched),
                                    "choice_index": choice["index"]})
                previous = actions[len(dispatched):].copy()
    if not checked:
        raise ValueError("no fallback decisions found")
    return checked


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    record = json.loads(SOURCE.read_text())
    checked = audit(record)
    result = {"source_sha256": file_hash(SOURCE), "runner_sha256": file_hash(__file__),
              "fallback_decisions": len(checked), "rows": checked,
              "scope": "historical RGB development log audit; no model rescoring or independent confirmation"}
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(f"Reconstructed {len(checked)} fallback choices; dispatched actions match scored prefixes.")


if __name__ == "__main__":
    main()
