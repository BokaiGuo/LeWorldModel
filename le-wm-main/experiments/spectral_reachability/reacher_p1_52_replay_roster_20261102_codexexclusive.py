"""Freeze the exhaustive saved Q/L2 replay inputs; does not execute a model or simulator."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import audit_reacher_response_svd_cp5_cem_20261102 as l2_audit
from . import audit_reacher_response_svd_qcp5_cem_20261102 as q_audit
from . import reacher_pc_p1_fd_cp5_saved_20261102_codexexclusive as saved
from . import reacher_pc_p1_fd_cp5_host_20261102_codexexclusive as host

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "L2": ("reacher_response_svd_cp5_PLAN_20261102_v1.json",
           "f1087157d9cb4f6d4efdfc9900c74fed66a28f33169835e61093033c79077014",
           "reacher_response_svd_cp5_RUN_20261102_v1/events.jsonl",
           "e9c933f8d78abc9787198e324240afdb1186475fa08189a2787e39a3b9480f43"),
    "Q": ("reacher_response_svd_qcp5_PLAN_20261102_v1.json",
          "093874de230fb7e95c745817a94dbbadc168fc6af4d27edc5adafe300fa64f66",
          "reacher_response_svd_cp5_Q_RUN_20261102_v1/events.jsonl",
          "8c4733bceeca7b7891a713b276f557cad50575d160f7721c3611805eb4535afc"),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def make_roster():
    rows = []
    for group, (plan_name, plan_sha, journal_name, journal_sha) in SOURCES.items():
        plan_path, journal_path = ROOT / "results" / plan_name, ROOT / "results" / journal_name
        consumer = l2_audit.consume if group == "L2" else q_audit.consume
        receipt = consumer(plan_path, plan_sha, journal_path, journal_sha)
        host.require(receipt["status"] == "complete_saved_consistency", "saved audit")
        plan = json.loads(plan_path.read_text())
        events = saved.verify_journal(journal_path)
        for root in plan["roots"]:
            for arm in root["arms"]:
                events_arm = [e for e in events if e.get("seed") == root["seed"]
                              and e.get("arm") == arm["arm"]]
                contexts = [e for e in events_arm if e["kind"] == "context"]
                result = next(e["result"] for e in events_arm if e["kind"] == "arm_returned")
                actions = np.asarray(result["action_witness_raw"], dtype=np.float32)
                host.require(actions.shape == (15, 2) and np.isfinite(actions).all()
                             and np.array_equal(actions[:5], np.asarray(arm["raw_prefix"], dtype=np.float32)),
                             "15-step original action witness")
                successor = next(e["value"] for e in contexts if e["role"] == "successor")
                rows.append({"group": group, "seed": root["seed"], "arm": arm["arm"],
                             "prefix_index": arm["proposal_index"], "actions": actions.tolist(),
                             "action_sha256": host.bridge.sha_array(actions),
                             "root_rgb_sha256": root["root_rgb_sha256"],
                             "goal_rgb_sha256": root["goal_rgb_sha256"],
                             "cp5_context_sha256": digest(successor),
                             "history": successor["history"],
                             "cp5_rgb_sha256": result["observed_successor_context"]["root_rgb_sha256"],
                             "original_physical": result["physical"],
                             "original_result_sha256": digest(result),
                             "plan_sha256": plan_sha, "journal_sha256": journal_sha})
    validate_roster(rows)
    return {"schema": "p1_52_fixed_replay_roster_v1", "rows": rows,
            "source_files": SOURCES, "replay_is_independent_outcome": False,
            "execution_authorized": False, "neural_inference": "host CUDA only"}


def validate_roster(rows):
    expected = [(g, s, f"prefix_{i:02d}") for g in ("L2", "Q")
                for s in (20263160, 20263161) for i in range(13)]
    host.require([(r["group"], r["seed"], r["arm"]) for r in rows] == expected,
                 "full ordered 52-arm roster required")
    for i in range(26):
        a, b = rows[i], rows[i + 26]
        host.require(a["root_rgb_sha256"] == b["root_rgb_sha256"]
                     and a["goal_rgb_sha256"] == b["goal_rgb_sha256"]
                     and a["cp5_rgb_sha256"] == b["cp5_rgb_sha256"]
                     and a["history"] == b["history"]
                     and a["actions"][:5] == b["actions"][:5], "paired cp5 binding")


def rank_complete_panel(rows):
    """Rank within each group/root with average ranks for exact ties, retaining censored rows."""
    host.require(len(rows) == 13, "13 fixed prefixes required")
    host.require([r["prefix_index"] for r in rows] == list(range(13)), "prefix order")
    metrics = ("predicted_goal_distance", "actual_rgb_goal_distance", "physical_distance",
               "prediction_error")
    result = [dict(r) for r in rows]
    for metric in metrics:
        valid = []
        for i, row in enumerate(rows):
            if row["status"] != "complete":
                continue
            value = row[metric]
            host.require(isinstance(value, (int, float)) and not isinstance(value, bool)
                         and math.isfinite(value) and value >= 0, "invalid metric")
            valid.append((value, i))
        valid.sort()
        for row in result:
            row[metric + "_rank"] = None
        start = 0
        while start < len(valid):
            end = start + 1
            while end < len(valid) and valid[end][0] == valid[start][0]:
                end += 1
            rank = (start + 1 + end) / 2
            for _, index in valid[start:end]:
                result[index][metric + "_rank"] = rank
            start = end
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    host.require(output.parent == ROOT / "results" and "p1_52_replay_roster" in output.name,
                 "exclusive results path required")
    host.write_new(output, make_roster())
    print(json.dumps({"path": str(output), "sha256": host.bridge.sha_file(output), "arms": 52}))


if __name__ == "__main__":
    main()
