"""Model-only same-bank greedy binding to the already executed zero/CEM arm.

No physical execution or neural inference. This is a post-exposure alias
analysis, not a prospective freeze or another independent physical sample.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import reacher_pc_p1_fd_cp5_f64_host_20261102_codexexclusive as f64_host
from . import reacher_pc_p1_fd_cp5_saved_20261102_codexexclusive as saved

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
MODEL = RESULTS / "reacher_pc_fd_suffix_cuda_20261102_v1/report.json"
AUDIT = RESULTS / "reacher_pc_fd_suffix_saved_audit_20261102_v1.json"
PHYSICAL_PLAN = RESULTS / "reacher_pc_fd_common_physical_20261102_PLAN_v1.json"
PREFLIGHT = RESULTS / "reacher_pc_p1_common_tail_preflight_20261102_codexexclusive_02.json"
HOST_PLAN = RESULTS / "reacher_fd_cp5_host_f64_PREP_20261102_codexexclusive_01.json"
HOST_PLAN_SHA = "c4b33e3b67242173617d73bc0f577d8f2f9b2ac3f95a170970180d51dca4444e"
RUN = RESULTS / "reacher_fd_cp5_host_f64_RUN_20261102_codexexclusive_01"
SAVED = RESULTS / "reacher_fd_cp5_host_f64_SAVED_20261102_codexexclusive_01.json"
SAVED_SHA = "b8198c0067f350f028ff64cc79a53c3fec6eb4db8bb2f351ef71bdc25fb9add9"
JOURNAL_SHA = "587c88657831babbe270eeb250179688cff57d6cbe17801228a6cab6e66be3bf"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return f64_host.bridge.sha_file(path)


def greedy_index(proposals):
    """Model-only argument: the ordered nine proposal rows, no outcome input."""
    require(isinstance(proposals, list) and len(proposals) == 9, "nine proposals required")
    scores = []
    for i, row in enumerate(proposals):
        require(type(row.get("proposal_index")) is int and row["proposal_index"] == i,
                "proposal order")
        cost = row.get("immediate_cost")
        require(type(cost) in (int, float) and math.isfinite(cost), "nonfinite or invalid immediate Q")
        scores.append(float(cost))
    return min(range(9), key=lambda i: (scores[i], i))


def require_alias(greedy, zero):
    require(greedy["greedy_index"] == zero["greedy_index"] == 0, "greedy is not zero index")
    for key, shape, dtype in (("params", (1, 10), np.float64),
                              ("raw_prefix", (5, 2), np.float32)):
        left, right = np.asarray(greedy[key], dtype=dtype), np.asarray(zero[key], dtype=dtype)
        require(left.shape == right.shape == shape and np.isfinite(left).all()
                and np.array_equal(left, right), "greedy/zero " + key + " mismatch")


def prepare():
    """Bind only fixed model report and pre-outcome planning inputs."""
    f64_host.configure()
    require(sha(HOST_PLAN) == HOST_PLAN_SHA, "host plan hash mismatch")
    base = json.loads(HOST_PLAN.read_text())
    require(base == f64_host.plan(MODEL, AUDIT, PHYSICAL_PLAN, PREFLIGHT), "host/source drift")
    report = json.loads(MODEL.read_text())
    roots = []
    for row, root in zip(report["rows"], base["bridge"]["roots"], strict=True):
        require(row["seed"] == root["seed"], "root mismatch")
        receipt = row["compiler_receipt"]
        proposals = receipt["proposal_rows"]
        index = greedy_index(proposals)
        bank = np.asarray(row["bank_params"], dtype=np.float64)
        raw = np.asarray(row["prefix_raw"], dtype=np.float32)
        require(bank.shape == (9, 1, 10) and np.isfinite(bank).all()
                and raw.shape == (9, 5, 2) and np.isfinite(raw).all(), "bank/action shape")
        item = {"seed": root["seed"], "greedy_index": index,
                "selected_index": receipt["development_selected_index"],
                "strict_selected_index": receipt["strict_selected_index"],
                "immediate_q": [float(x["immediate_cost"]) for x in proposals],
                "params": bank[index].tolist(), "raw_prefix": raw[index].tolist(),
                "raw_prefix_sha256": f64_host.bridge.sha_array(raw[index]),
                "greedy_aliases_parameter_zero": index == 0}
        zero = root["arms"][1]
        require(zero["arm"] == "parameter_zero" and zero["proposal_index"] == 0, "zero identity")
        require_alias(item, {"greedy_index": 0, "params": zero["params"], "raw_prefix": zero["raw_prefix"]})
        require(item["raw_prefix_sha256"] == zero["raw_prefix_sha256"], "zero hash mismatch")
        roots.append(item)
    require([r["seed"] for r in roots] == [20263160, 20263161], "root roster")
    return {"schema": "fd_samebank_greedy_alias_v1", "host_plan_sha256": HOST_PLAN_SHA,
            "binding_code_sha256": sha(__file__), "saved_consumer_sha256": sha(saved.__file__),
            "source_model_sha256": sha(MODEL), "selection_rule": "min immediate model Q over indices 0..8; ties lowest index",
            "parameter_dtype": "float64", "roots": roots,
            "selection_frozen_before_old_outcomes": False, "physical_outcome_used_for_selection": False,
            "same_bank_as_fd_development_selector": True, "source_closure_complete": False,
            "P1_admission": False}


def consume(plan_path, expected_sha, run=RUN, archive=SAVED):
    """Recompute existing saved verifier, then attach zero arm as a greedy alias."""
    require(sha(plan_path) == expected_sha, "greedy plan hash mismatch")
    plan = json.loads(Path(plan_path).read_text())
    require(plan == prepare(), "greedy plan or source drift")
    require(sha(archive) == SAVED_SHA and sha(Path(run) / "events.jsonl") == JOURNAL_SHA
            and sha(Path(run) / "plan.json") == HOST_PLAN_SHA, "archived source mismatch")
    checked = saved.consume(Path(run) / "plan.json", HOST_PLAN_SHA,
                            Path(run) / "events.jsonl", JOURNAL_SHA)
    require(checked == json.loads(Path(archive).read_text())
            and checked["status"] == "complete_saved_consistency", "archived saved verification")
    events = saved.verify_journal(Path(run) / "events.jsonl")
    results = []
    for root in plan["roots"]:
        items = [e for e in events if e["kind"] == "arm_returned"
                 and e["seed"] == root["seed"] and e["arm"] == "parameter_zero"]
        require(len(items) == 1, "zero arm result cardinality")
        value = items[0]["result"]
        require(value["first_receipt"]["proposal_index"] == 0
                and value["first_receipt"]["raw_prefix_sha256"] == root["raw_prefix_sha256"]
                and np.array_equal(np.asarray(value["first_action_witness_raw"], np.float32),
                                   np.asarray(root["raw_prefix"], np.float32)), "greedy/zero witness mismatch")
        require(value["suffix_receipt"]["forward_rollout_evaluations"] == 32, "CEM budget mismatch")
        results.append({"seed": root["seed"], "greedy_index": 0, "alias_of": "parameter_zero",
                        "immediate_q": root["immediate_q"][0],
                        "raw_prefix_sha256": root["raw_prefix_sha256"],
                        "action_witness_raw": value["action_witness_raw"],
                        "suffix_receipt": value["suffix_receipt"],
                        "physical": value["physical"],
                        "stop": {"prefix": value["first_physical_execution"],
                                 "zero_control": value["zero_prefix_physical_execution"],
                                 "suffix": value["suffix_physical_execution"]}})
    return {"schema": "fd_samebank_greedy_alias_saved_v1", "status": "verified_existing_arm_alias",
            "greedy_plan_sha256": expected_sha, "archived_plan_sha256": HOST_PLAN_SHA,
            "archived_journal_sha256": JOURNAL_SHA, "archived_saved_sha256": SAVED_SHA,
            "roots": results, "new_physical_executions": 0, "new_cem_executions": 0,
            "selection_frozen_before_old_outcomes": False, "physical_outcome_used_for_selection": False,
            "independent_confirmation": False, "equal_total_compute": False, "P1_admission": False,
            "limitation": "post-exposure model-only alias of existing zero arm; immediate Q not independently recomputed"}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", required=True)
    c = sub.add_parser("consume")
    for key in ("plan", "sha256", "output"):
        c.add_argument("--" + key, required=True)
    args = parser.parse_args()
    result = prepare() if args.action == "prepare" else consume(args.plan, args.sha256)
    f64_host.host.write_new(args.output, result)
    print(json.dumps({"path": args.output, "sha256": sha(args.output), "status": result["schema"]}))


if __name__ == "__main__":
    main()
