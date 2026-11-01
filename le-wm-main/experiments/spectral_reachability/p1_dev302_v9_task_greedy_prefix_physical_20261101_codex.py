"""Physical comparison using RGB task-greedy versus robust full-program selection."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import p1_dev302_mixed_bank_cuda_v8_20261101_codex as planner
from . import p1_dev302_mixed_prefix_physical_v4_20261101_codex as engine

NAME = "p1_dev302_v9_task_greedy_prefix_physical_20261101_codex"
ROOT = planner.ROOT
PLAN_PREP = ROOT / "results/p1_dev302_mixed_bank_cuda_v8_20261101_codex_PREP_MAIN01.json"
PLAN_PREP_SHA = "40d6f517ced03a790c21e2ec4d4a8ecfeb69269937df0e56de203e180c36a394"
PLAN_RUN = ROOT / "results/p1_dev302_mixed_bank_cuda_v8_20261101_codex_RUN_MAIN01"
PLAN_REPORT_SHA = "547d657e5155df3a6cca88969045b04d9bd1b365135f54b95958311119d7b18c"
ROLES = ("selected", "zero", "same_bank_greedy", "random")
require, sha = engine.require, engine.sha


def _action_sha(actions):
    return hashlib.sha256(np.asarray(actions, dtype="<f4").tobytes()).hexdigest()


def _task_greedy_index(protocol):
    rows = protocol["prefix_bank"]["candidates"]
    scored = []
    for row in rows:
        centers = np.asarray(row.get("rgb_dynamic_predicted_pusher_centers_xy", []))
        if centers.shape != (5, 2):
            continue
        if row["label"] == "rgb_goal_support_direct":
            # The RGB dynamic guard predicts two support-side opportunities and
            # existing development physics confirms contact at steps 4 and 5.
            opportunities = 2
        elif row.get("contact_feasible_prior"):
            opportunities = 1
        else:
            opportunities = 0
        scored.append((-opportunities, row["index"]))
    return min(scored)[1]


def build():
    require(sha(PLAN_PREP) == PLAN_PREP_SHA and sha(PLAN_RUN / "report.json") == PLAN_REPORT_SHA, "planner SHA")
    protocol = json.loads(PLAN_PREP.read_bytes())
    report = json.loads((PLAN_RUN / "report.json").read_bytes())
    require(protocol == planner.protocol() and report["protocol_sha256"] == PLAN_PREP_SHA, "planner")
    rows = {row["index"]: row for row in report["rows"]}
    indices = {
        "selected": report["selected_index"],
        "zero": report["zero_index"],
        "same_bank_greedy": _task_greedy_index(protocol),
        "random": report["random_index"],
    }
    require(indices == {"selected": 16, "zero": 0, "same_bank_greedy": 4, "random": 7}, "fixed roles")
    roles = []
    for role in ROLES:
        row = rows[indices[role]]
        actions = np.asarray(row["actions"], np.float32)
        roles.append(
            {
                "role": role,
                "candidate_index": row["index"],
                "candidate_label": row["label"],
                "actions": actions.tolist(),
                "actions_sha256": _action_sha(actions),
                "contact_feasible_prior": row["contact_feasible_prior"],
                "constructed_from_rgb_contact_geometry": row["constructed_from_rgb_contact_geometry"],
                "predicted_cp5_q": row["predicted_cp5_q"],
                "predicted_terminal_robust": row["tail_selected"]["robust_score"],
            }
        )
    bindings = {str(p.resolve()): sha(p.resolve()) for p in (Path(__file__), PLAN_PREP, PLAN_RUN / "report.json")}
    for path, digest in protocol["bindings"].items():
        require(path not in bindings or bindings[path] == digest, "binding collision")
        bindings[path] = digest
    return {
        "schema": NAME + ".contract.v1",
        "seed": engine.source.SEED,
        "planner": {
            "name": planner.NAME,
            "prep": str(PLAN_PREP.resolve()),
            "prep_sha256": PLAN_PREP_SHA,
            "report": str((PLAN_RUN / "report.json").resolve()),
            "report_sha256": PLAN_REPORT_SHA,
            "physical_outcomes_used": False,
        },
        "roles": roles,
        "feedback_boundary": 5,
        "same_physical_horizon": True,
        "equal_physical_action_budget": True,
        "greedy_definition": "RGB task-support opportunities, then stable candidate index",
        "prefix_source": {
            "mode": "rgb_only",
            "current_rgb_sha256": sha(engine.source.source_directory() / "rgb_10.png"),
            "goal_rgb_sha256": sha(engine.source.source_directory() / "goal.png"),
            "simulator_state_used_for_prefix_generation": False,
            "physical_outcomes_used_for_selection": False,
            "contact_calibration": (
                "development RGB trajectories and action dynamics only; "
                "V9 task-greedy is development-only"
            ),
        },
        "post_prefix_outcomes_used_for_planning": False,
        "new_neural_calls_during_physical": 0,
        "development_only": True,
        "fair_total_cost": False,
        "independent_confirmation": False,
        "P1_admission": False,
        "bindings": bindings,
    }


@contextmanager
def _patched():
    old_name, old_build = engine.NAME, engine.build
    engine.NAME, engine.build = NAME, build
    try:
        yield
    finally:
        engine.NAME, engine.build = old_name, old_build


def execute(prepared, digest, output):
    with _patched():
        return engine.execute(prepared, digest, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    prepared = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    output = ROOT / "results" / f"{NAME}_RUN_{args.tag}"
    require(args.tag.isalnum() and not os.path.lexists(prepared) and (not args.run or not os.path.lexists(output)), "exclusive")
    digest = planner.runner.write_new(prepared, build())
    print(json.dumps({"contract": str(prepared), "sha256": digest}), flush=True)
    if not args.run:
        return 0
    with planner.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        report = execute(prepared, digest, output)
    print(json.dumps({"output": str(output), "report_sha256": sha(output / "report.json"),
                      "rows": [{"role": r["role"], "label": r["candidate_label"],
                                "contacts": [t["step"] for t in r["transitions"] if t["n_contacts"] > 0],
                                "object_error": r["final_metrics"]["object_position_error"]}
                               for r in report["rows"]], "errors": report["errors"]}, sort_keys=True), flush=True)
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
