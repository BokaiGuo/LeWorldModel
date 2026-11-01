"""Exploratory full-cross fixed prefixes and sealed-source common ten-step tails."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path

import numpy as np

from . import p1_dev303_v3_four_arm_physical_20261102_exclusive as prefix
from . import p1_dev303_v3_four_arm_physical_saved_20261102_exclusive as prefix_saved
from . import p1_dev303_cp5_tail_physical_20261102_exclusive as replay
from . import p1_dev303_cp5_tail_handoff_20261102_exclusive as gate

base, prior, require = prefix.base, prefix.prior, prefix.require
ROOT = prefix.ROOT
NAME = "p1_dev303_v3_cross_common_tail_20261102_exclusive"
PREFIX_PREP = ROOT / "results/p1_dev303_v3_four_arm_physical_20261102_exclusive_PREP_02.json"
PREFIX_PREP_SHA = "8e2692bf54cff2ddf1a7abcb4d447861ef25cc31836f64770f230d1358bed3d7"
PREFIX_DIR = ROOT / "results/p1_dev303_v3_four_arm_physical_20261102_exclusive_RUN_DEV01"
PREFIX_REPORT_SHA = "c74363f2c5a0f34866e9becdc0b9891f83c48c67872f88e69d14fff94c13ed93"
TAIL_PREP = prefix.audit.MODEL_PREP
TAIL_PREP_SHA = prefix.audit.PINS["model_prepare"]
AUTHORITY = "EXPLICIT_DEV303_CROSS_TAIL_PHYSICS"


def tail_family(prep):
    """All four frozen geometric candidates doubled, plus frozen zero doubled."""
    candidates = prep["candidates"]
    require(len(candidates) == 7, "tail source roster")
    tails = []
    for i in range(5):
        source = candidates[i]
        expected = f"rgb_boundary_approach_{i}" if i < 4 else "zero"
        require(source["index"] == i and source["label"] == expected
                and source["eligible_for_selection"] is (i < 4), "tail source identity")
        action = gate.actions(source["actions"], (5, 2))
        require(hashlib.sha256(action.tobytes()).hexdigest() == source["actions_sha256"],
                "tail source action hash")
        if i == 4:
            require(not action.any(), "zero source")
        tail = np.concatenate((action, action))
        tails.append({"tail": f"geometry_{i}" if i < 4 else "zero",
                      "source_index": i, "source_actions_sha256": source["actions_sha256"],
                      "actions": tail.tolist(),
                      "sha256": hashlib.sha256(tail.tobytes()).hexdigest()})
    return tails


def build():
    require(base.sha(TAIL_PREP) == TAIL_PREP_SHA, "sealed tail source SHA")
    tails = tail_family(base.read(TAIL_PREP))
    audit = prefix_saved.verify(PREFIX_PREP, PREFIX_PREP_SHA, PREFIX_DIR, PREFIX_REPORT_SHA)
    require(audit["run_status"] == "recorded_development"
            and all(r["status"] == "fixed_horizon_completed" and r["prefix_returned"] == 5
                    for r in audit["branches"]), "four complete verified prefixes required")
    old, prep = base.read(PREFIX_DIR / "report.json"), base.read(PREFIX_PREP)
    branches = []
    for spec, row in zip(prep["arms"], old["rows"], strict=True):
        require(row["arm"] == spec["role"], "prefix role identity")
        checkpoints = [p for p in row["observations"] if p["phase"] == "prefix"]
        require([p["step"] for p in checkpoints] == list(range(1, 6)), "prefix checkpoints")
        for tail in tails:
            branches.append({"arm": spec["role"] + "__" + tail["tail"],
                             "prefix_role": spec["role"], "tail_role": tail["tail"],
                             "prefix": spec["actions"], "prefix_sha256": spec["actions_sha256"],
                             "prefix_checkpoints": checkpoints,
                             "cp5_pixel_sha256": checkpoints[-1]["pixel_sha256"],
                             "tail": tail["actions"], "tail_sha256": tail["sha256"]})
    paths = [Path(__file__), Path(__file__).with_name(
        "p1_dev303_v3_cross_common_tail_saved_20261102_exclusive.py"),
        Path(prefix.__file__), Path(prefix_saved.__file__), Path(replay.__file__),
        Path(replay.verifier.__file__), Path(gate.__file__),
        Path(gate.planner.__file__), Path(base.__file__), Path(base.native.__file__),
        PREFIX_PREP, PREFIX_DIR / "report.json", TAIL_PREP]
    return {
        "schema": NAME + ".contract.v1",
        "classification": "post_prefix_development_nonindependent",
        "tail_definition": "all four frozen RGB geometry candidates repeated twice, plus frozen zero repeated twice",
        "tail_source_sha256": TAIL_PREP_SHA, "tail_family": tails,
        "prefix_report_sha256": PREFIX_REPORT_SHA, "prefix_prepare_sha256": PREFIX_PREP_SHA,
        "branches": branches, "ordering": "prefix-major then tail source index 0..4",
        "replay": "fresh environment; exact reset, zero warmup and every prefix frame/state/snapshot",
        "outcome_use": "prior prefix observations used for replay guard only; no tail or branch filtering",
        "censoring": "terminate branch on failure or native stop; preserve partial records; no padding or retry",
        "pose": "state vector recorded on each return and observation; object pose is state[2:5]",
        "ledger_verified": False, "equal_budget_claim": False,
        "P1_admission": False, "independent_confirmation": False,
        "physical_execution_authorized_by_contract": False,
        "bindings": {str(p.resolve()): base.sha(p) for p in paths},
    }


def execute(prepared, prepared_sha, output, authority):
    require(authority == AUTHORITY, "explicit development physics authority")
    prepared = Path(prepared).resolve(strict=True)
    require(base.sha(prepared) == prepared_sha and base.read(prepared) == build(),
            "contract or source drift")
    contract = base.read(prepared)
    arrays, receipt, reset = prior.source_data()
    store = base.Store(output)
    store.json("execution_binding.json", {"prepared": str(prepared), "prepare_sha256": prepared_sha})
    store.emit("run_started", prepare_sha256=prepared_sha)
    rows, errors = [], []
    for role in contract["branches"]:
        arm = role["arm"]
        row = {"arm": arm, "status": "running", "transitions": [], "observations": [], "error": None}
        env = None
        store.emit("branch_started", arm=arm)
        try:
            cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
            require(base.sha(inspect.getsourcefile(cls)) == base.native.ENV_SHA, "environment source SHA")
            env = cls(render_mode="rgb_array", resolution=224, with_target=False)
            replay.branch(env, store, role, receipt["history"], reset, arrays["goal_rgb"], row)
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
            errors.append({"arm": arm, **row["error"]})
            store.emit("branch_error", arm=arm, error=row["error"])
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception as exc:
                    error = {"arm": arm, "stage": "close", "type": type(exc).__name__,
                             "message": str(exc)[:1000]}
                    errors.append(error)
                    store.emit("close_error", **error)
        rows.append(row)
        store.json(arm + ".row.json", row)
        store.emit("branch_closed", arm=arm, status=row["status"])
    try:
        require(build() == contract, "postrun source closure")
    except Exception as exc:
        errors.append({"stage": "source_closure", "type": type(exc).__name__, "message": str(exc)[:1000]})
    store.emit("run_finished", errors=errors)
    report = {"schema": NAME + ".report.v1", "prepare_sha256": prepared_sha,
              "status": "partial_or_failed" if errors else "recorded_development",
              "rows": rows, "errors": errors, "journal_sha256": base.sha(store.path / "journal.jsonl"),
              "journal_tail": store.events[-1]["sha256"], "P1_admission": False,
              "independent_confirmation": False, "ledger_verified": False, "equal_budget_claim": False}
    store.json("report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--contract-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--authority")
    args = parser.parse_args()
    if args.command == "prepare":
        require(args.contract_sha256 is None and args.output is None and args.authority is None, "prepare only")
        print(json.dumps({"sha256": prior.write_new(args.contract, build())}))
    else:
        require(args.contract_sha256 is not None and args.output is not None, "run arguments")
        report = execute(args.contract, args.contract_sha256, args.output, args.authority)
        print(json.dumps({"status": report["status"], "report_sha256": base.sha(args.output / "report.json")}))


if __name__ == "__main__":
    main()
