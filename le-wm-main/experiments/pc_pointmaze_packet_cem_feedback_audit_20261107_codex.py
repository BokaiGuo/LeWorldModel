"""Audit sealed feedback choices and action continuity without rerunning the planner."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_pointmaze_packet_cem_feedback_20261107_codex as run
from experiments.pc_input_permissions import file_hash

OUTPUT = run.ROOT / "results/pc_pointmaze_packet_cem_feedback_audit_20261107_codex.json"


def audit_record(row, initial):
    executed = np.asarray(row["executed_actions"], dtype=np.float32)
    initial = np.asarray(initial, dtype=np.float32)
    if executed.shape != (25, 2) or initial.shape != (25, 2) or not np.isfinite(executed).all() or np.abs(executed).max() > 1:
        raise ValueError("invalid executed or initial actions")
    np.testing.assert_array_equal(executed[:5], initial[:5])
    old = initial[5:].reshape(4, 5, 2)[:, 0]
    if len(row["decisions"]) != 4:
        raise ValueError("expected four decisions")
    retained = 0
    for k, decision in enumerate(row["decisions"]):
        n = 4-k
        if decision["step"] != 5*(k+1) or decision["horizon_packets"] != n or decision["query_count"] != 48:
            raise ValueError("clock or budget mismatch")
        np.testing.assert_array_equal(np.asarray(decision["old_tail"], np.float32), old)
        pools = np.asarray([r["candidates"] for r in decision["rounds"]], dtype=np.float32).reshape(48, n, 2)
        costs = np.asarray([r["costs"] for r in decision["rounds"]]).reshape(48)
        if not np.isfinite(pools).all() or not np.isfinite(costs).all() or np.abs(pools).max() > 1:
            raise ValueError("invalid candidate scores or actions")
        np.testing.assert_array_equal(pools[0], old)
        winner = int(np.argmin(costs))
        if winner != decision["chosen"] or not np.isclose(costs[winner], decision["selected_cost"], atol=1e-12):
            raise ValueError("selected score mismatch")
        selected = np.asarray(decision["selected"], dtype=np.float32)
        np.testing.assert_array_equal(selected, pools[winner])
        np.testing.assert_array_equal(executed[5*(k+1):5*(k+2)], np.repeat(selected[:1], 5, axis=0))
        retained += int(winner == 0)
        old = selected[1:]
    if row["feedback_queries"] != 192 or row["feedback_model_packet_steps"] != 480:
        raise ValueError("total budget mismatch")
    return retained


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    data = json.loads(run.OUTPUT.read_text())
    plan = run.source.check_plan()
    for key, path in (("protocol", run.PROTOCOL), ("runner", run.__file__),
                      ("bridge", Path(run.__file__).with_name("pc_pointmaze_state_feedback_bridge_20261107_codex.py")),
                      ("source_plan", run.source.PLAN), ("source_physical", run.source.PHYSICAL)):
        if data[key] != file_hash(path):
            raise ValueError(f"source drift: {key}")
    if len(data["rows"]) != 8:
        raise ValueError("root count mismatch")
    rows = []
    for i, (row, source) in enumerate(zip(data["rows"], plan["rows"], strict=True)):
        if row["root"] != 1884+i or source["root"] != row["root"]:
            raise ValueError("root mismatch")
        rows.append({"root": row["root"], "old_tail_selections": audit_record(row, source["selected"]["160"]["actions"])})
    with OUTPUT.open("x") as stream:
        json.dump({"feedback": file_hash(run.OUTPUT), "runner": file_hash(__file__), "rows": rows,
                   "scope": "stored decision and action consistency only; no independent physics replay"}, stream, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
