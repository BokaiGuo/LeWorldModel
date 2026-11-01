"""Physical execution of V5 RGB-contact-feasible four-prefix tails."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import p1_dev302_v4_four_prefix_tail_physical_20261101_codex as engine
from . import p1_dev302_v5_contact_tail_replan_20261101_codex as planner

NAME = "p1_dev302_v5_contact_tail_physical_20261101_codex"
ROOT = planner.ROOT
PLAN_PREP = ROOT / "results/p1_dev302_v5_contact_tail_replan_20261101_codex_PREP_MAIN01.json"
PLAN_PREP_SHA = "1f78e038a7a68a9f15b2d1bd8975816a9577869a2e73da7068e464e0a3a6db65"
PLAN_RUN = ROOT / "results/p1_dev302_v5_contact_tail_replan_20261101_codex_RUN_MAIN01"
PLAN_REPORT_SHA = "1db874fb25a142b5f8c819e2a7e274858c863f01389343d829978e4ec48619e5"
ROLES = planner.ROLES
require, sha = engine.require, engine.sha
ENGINE_BUILD = engine.build


def _action_sha(actions):
    return hashlib.sha256(np.asarray(actions, dtype="<f4").tobytes()).hexdigest()


def build():
    require(
        sha(PLAN_PREP) == PLAN_PREP_SHA
        and sha(PLAN_RUN / "report.json") == PLAN_REPORT_SHA,
        "V5 input SHA",
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
        "V5 planner",
    )
    sources = {row["role"]: row for row in protocol["roles"]}
    rows = []
    for row in report["rows"]:
        source = sources[row["role"]]
        prefix = np.asarray(source["actual_prefix"], np.float32)
        tail = np.asarray(row["selected"]["actions"], np.float32).reshape(10, 2)
        actions = np.concatenate((prefix, tail)).astype(np.float32)
        require(actions.shape == (15, 2), "V5 plan shape")
        rows.append(
            {
                "role": row["role"],
                "candidate_index": row["candidate_index"],
                "candidate_label": row["candidate_label"],
                "actions": actions.tolist(),
                "actions_sha256": _action_sha(actions),
                "prefix_sha256": _action_sha(prefix),
                "expected_cp5_pixel_sha256": row["cp5_pixel_sha256"],
                "selected_tail": row["selected"],
            }
        )
    require([row["role"] for row in rows] == list(ROLES), "role order")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require(args.tag.isalnum(), "tag")
    prepared = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    output = ROOT / "results" / f"{NAME}_RUN_{args.tag}"
    require(not os.path.lexists(prepared) and (not args.run or not os.path.lexists(output)), "exclusive")
    digest = planner.write_new(prepared, build())
    print(json.dumps({"contract": str(prepared), "sha256": digest}), flush=True)
    if not args.run:
        return 0
    with planner.base.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
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
                        "native_success": row.get("final_metrics", {}).get("native_success"),
                        "object_position_error": row.get("final_metrics", {}).get(
                            "object_position_error"
                        ),
                        "contact_steps": [
                            item["step"] for item in row["transitions"] if item["n_contacts"] > 0
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
