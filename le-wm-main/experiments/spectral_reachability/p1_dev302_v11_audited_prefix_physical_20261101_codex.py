"""Audited V8 four-prefix physical runner with outcome-free RGB greedy."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path

import numpy as np

from . import p1_dev302_mixed_bank_cuda_v12_20261101_codex as planner
from . import p1_dev302_mixed_prefix_physical_v3_20261101_codex as engine
from . import p1_task_greedy_rgb_geometry_v11_20261101_codex as greedy


NAME = "p1_dev302_v11_audited_prefix_physical_20261101_codex"
ROOT = planner.ROOT
PLAN_PREP = ROOT / "results/p1_dev302_mixed_bank_cuda_v12_20261101_codex_PREP_MAIN01.json"
PLAN_PREP_SHA = "d2e5376480af4b75857bd6e9eb29ff24b514da4f9bb7827e90e4bba59a5a6f28"
PLAN_RUN = ROOT / "results/p1_dev302_mixed_bank_cuda_v12_20261101_codex_RUN_MAIN01"
PLAN_REPORT_SHA = "f7e66f1d045ff34f1468c02c480a296baf3c9b7f41dd4844f7d759f445f737ca"
ROLES = ("selected", "zero", "same_bank_greedy", "random")
source, base = engine.source, engine.base
require, sha = engine.require, engine.sha


def _action_sha(actions):
    return hashlib.sha256(np.asarray(actions, dtype="<f4").tobytes()).hexdigest()


def build():
    require(
        sha(PLAN_PREP) == PLAN_PREP_SHA
        and sha(PLAN_RUN / "report.json") == PLAN_REPORT_SHA,
        "sealed V12 planner SHA",
    )
    protocol = json.loads(PLAN_PREP.read_bytes())
    report = json.loads((PLAN_RUN / "report.json").read_bytes())
    _audit, _source_report, current, goal = planner.runner._images()
    require(
        protocol == planner.protocol()
        and report["schema"] == planner.NAME + ".report.v1"
        and report["status"] == "model_only_complete"
        and report["protocol_sha256"] == PLAN_PREP_SHA
        and report["physical_outcomes_used"] is False
        and report["P1_admission"] is False,
        "sealed V12 planner",
    )
    greedy_trace = greedy.score_candidates(protocol, current, goal)
    indices = {
        "selected": report["selected_index"],
        "zero": report["zero_index"],
        "same_bank_greedy": greedy.task_greedy_index(protocol, current, goal),
        "random": report["random_index"],
    }
    require(
        indices == {
            "selected": 16,
            "zero": 0,
            "same_bank_greedy": 4,
            "random": 7,
        },
        "fixed audited roles",
    )
    rows = {row["index"]: row for row in report["rows"]}
    roles = []
    for role in ROLES:
        row = rows[indices[role]]
        actions = np.asarray(row["actions"], np.float32)
        require(
            actions.shape == (5, 2)
            and np.isfinite(actions).all()
            and _action_sha(actions) == row["actions_sha256"],
            f"{role} action identity",
        )
        roles.append(
            {
                "role": role,
                "candidate_index": row["index"],
                "candidate_label": row["label"],
                "actions": actions.tolist(),
                "actions_sha256": row["actions_sha256"],
                "contact_feasible_prior": row["contact_feasible_prior"],
                "constructed_from_rgb_contact_geometry": row[
                    "constructed_from_rgb_contact_geometry"
                ],
                "predicted_cp5_q": row["predicted_cp5_q"],
                "predicted_terminal_robust": row["tail_selected"]["robust_score"],
            }
        )
    audit, receipt, _saved, _variants = source.material()
    env_path = Path(
        inspect.getsourcefile(
            importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
        )
    ).resolve()
    directory = source.source_directory()
    require(
        audit["seed"] == source.SEED
        and audit["P1_admission"] is False
        and receipt["history"]["reset_receipt_sha256"]
        == sha(directory / "reset_receipt.json"),
        "dev302 source authority",
    )
    paths = (
        Path(__file__).resolve(),
        Path(greedy.__file__).resolve(),
        Path(planner.__file__).resolve(),
        Path(engine.__file__).resolve(),
        env_path,
        PLAN_PREP,
        PLAN_RUN / "report.json",
        source.QUEUE,
        directory / "report.json",
        directory / "source_receipt.json",
        directory / "reset_receipt.json",
        directory / "goal.png",
        *(directory / f"rgb_{step:02d}.png" for step in (0, 5, 10)),
    )
    bindings = {str(path): sha(path) for path in paths}
    for path, digest in protocol["bindings"].items():
        require(path not in bindings or bindings[path] == digest, "binding collision")
        bindings[path] = digest
    return {
        "schema": NAME + ".contract.v1",
        "seed": source.SEED,
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
        "greedy_definition": (
            "stepwise RGB object-boundary support opportunity count, first "
            "opportunity, continuous deficit, stable candidate index"
        ),
        "greedy_trace": greedy_trace,
        "prefix_source": {
            "mode": "rgb_only",
            "current_rgb_sha256": sha(directory / "rgb_10.png"),
            "goal_rgb_sha256": sha(directory / "goal.png"),
            "simulator_state_used_for_prefix_generation": False,
            "physical_outcomes_used_for_selection": False,
            "contact_calibration": "frozen development RGB action-motion constants",
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
    names = {
        "NAME": NAME,
        "build": build,
        "PLAN_PREP_SHA": PLAN_PREP_SHA,
        "PLAN_REPORT_SHA": PLAN_REPORT_SHA,
    }
    previous = {key: getattr(engine, key) for key in names}
    for key, value in names.items():
        setattr(engine, key, value)
    try:
        yield
    finally:
        for key, value in previous.items():
            setattr(engine, key, value)


def execute(prepared, digest, output):
    with _patched():
        return engine.execute(prepared, digest, output)


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
    digest = planner.runner.write_new(prepared, build())
    print(json.dumps({"contract": str(prepared), "sha256": digest}), flush=True)
    if not args.run:
        return 0
    with planner.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        report = execute(prepared, digest, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "report_sha256": sha(output / "report.json"),
                "planner_prep_sha256": report["planner_prep_sha256"],
                "planner_report_sha256": report["planner_report_sha256"],
                "rows": [
                    {
                        "role": row["role"],
                        "contacts": [
                            item["step"]
                            for item in row["transitions"]
                            if item["n_contacts"] > 0
                        ],
                        "object_error": row["final_metrics"]["object_position_error"],
                    }
                    for row in report["rows"]
                ],
                "errors": report["errors"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
