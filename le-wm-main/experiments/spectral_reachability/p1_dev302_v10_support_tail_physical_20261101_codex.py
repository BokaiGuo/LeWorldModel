"""Physical execution of V10 equal-budget support-preserving tails."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import p1_dev302_v4_four_prefix_tail_physical_20261101_codex as engine
from . import p1_dev302_v10_support_tail_replan_20261101_codex as planner


NAME = "p1_dev302_v10_support_tail_physical_20261101_codex"
ROOT = planner.ROOT
PLAN_PREP = ROOT / "results/p1_dev302_v10_support_tail_replan_20261101_codex_PREP_MAIN01.json"
PLAN_PREP_SHA = "a3e767a8d5a0c158be946346589b3d67148f5fd42b763682205851578ce6f1bf"
PLAN_RUN = ROOT / "results/p1_dev302_v10_support_tail_replan_20261101_codex_RUN_MAIN01"
PLAN_REPORT_SHA = "18bbad2491b4072fd5e80fc74f98b777aabdcfe46918522f3a173d46258b4c2a"
ROLES = planner.ROLES
require, sha = engine.require, engine.sha


def _action_sha(actions):
    return hashlib.sha256(np.asarray(actions, dtype="<f4").tobytes()).hexdigest()


def build():
    require(
        sha(PLAN_PREP) == PLAN_PREP_SHA
        and sha(PLAN_RUN / "report.json") == PLAN_REPORT_SHA,
        "V10 input SHA",
    )
    protocol = json.loads(PLAN_PREP.read_bytes())
    report = json.loads((PLAN_RUN / "report.json").read_bytes())
    require(
        protocol == planner.protocol()
        and report["schema"] == planner.NAME + ".report.v1"
        and report["status"] == "model_only_complete_not_admitted"
        and report["protocol_sha256"] == PLAN_PREP_SHA
        and report["post_cp5_physical_outcomes_used"] is False
        and report["P1_admission"] is False,
        "V10 planner",
    )
    source_rows = {row["role"]: row for row in protocol["roles"]}
    report_rows = {row["role"]: row for row in report["rows"]}
    require(
        set(source_rows) == set(report_rows) == set(ROLES),
        "V10 role roster",
    )
    rows = []
    for role in ROLES:
        source = source_rows[role]
        row = report_rows[role]
        prefix = np.asarray(source["actual_prefix"], np.float32)
        tail = np.asarray(row["selected"]["actions"], np.float32).reshape(10, 2)
        actions = np.concatenate((prefix, tail), axis=0).astype(np.float32)
        require(
            prefix.shape == (5, 2)
            and tail.shape == (10, 2)
            and row["actual_prefix_sha256"] == _action_sha(prefix)
            and row["cp5_pixel_sha256"] == source["cp5_pixel_sha256"],
            f"{role} V10 plan identity",
        )
        rows.append(
            {
                "role": role,
                "candidate_index": row["candidate_index"],
                "candidate_label": row["candidate_label"],
                "actions": actions.tolist(),
                "actions_sha256": _action_sha(actions),
                "prefix_sha256": _action_sha(prefix),
                "expected_cp5_pixel_sha256": row["cp5_pixel_sha256"],
                "selected_tail": row["selected"],
            }
        )
    bindings = {
        str(path.resolve()): sha(path.resolve())
        for path in (
            Path(__file__),
            Path(planner.__file__),
            PLAN_PREP,
            PLAN_RUN / "report.json",
        )
    }
    for path, digest in protocol["bindings"].items():
        require(path not in bindings or bindings[path] == digest, "binding collision")
        bindings[path] = digest
    return {
        "schema": NAME + ".contract.v1",
        "seed": engine.source.SEED,
        "roles": rows,
        "feedback_boundary": 5,
        "physical_horizon": 15,
        "same_replanning_budget": True,
        "same_feedback_count": True,
        "post_cp5_physical_outcomes_used_for_selection": False,
        "new_neural_calls_during_physical": 0,
        "development_only": True,
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


def execute(prepared, prepared_sha, output):
    with _patched():
        return engine.execute(prepared, prepared_sha, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require(args.tag.isalnum(), "tag")
    prepared = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    output = ROOT / "results" / f"{NAME}_RUN_{args.tag}"
    require(
        not os.path.lexists(prepared) and (not args.run or not os.path.lexists(output)),
        "exclusive outputs",
    )
    digest = planner.write_new(prepared, build())
    print(json.dumps({"contract": str(prepared), "sha256": digest}), flush=True)
    if not args.run:
        return 0
    with planner.engine.base.locking.gpu_lock(
        ROOT / "results/e3_p2_host_cuda_cooperative.lock"
    ):
        report = execute(prepared, digest, output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(output),
                "report_sha256": sha(output / "report.json"),
                "rows": [
                    {
                        "role": row["role"],
                        "status": row["status"],
                        "native_success": row.get("final_metrics", {}).get(
                            "native_success"
                        ),
                        "object_position_error": row.get("final_metrics", {}).get(
                            "object_position_error"
                        ),
                        "contact_steps": [
                            item["step"]
                            for item in row["transitions"]
                            if item["n_contacts"] > 0
                        ],
                    }
                    for row in report["rows"]
                ],
                "errors": report["errors"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return int(report["status"] != "recorded_development_equal_budget_tails")


if __name__ == "__main__":
    raise SystemExit(main())
