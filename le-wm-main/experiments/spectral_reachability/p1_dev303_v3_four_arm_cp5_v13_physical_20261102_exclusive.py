"""Prepare and execute four fixed V13 tail witnesses after exact cp5 replay."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path

import numpy as np

from . import p1_dev303_v3_four_arm_cp5_v13_model_20261102_exclusive as model
from . import p1_dev303_cp5_tail_physical_20261102_exclusive as replay

inputs, base, require = model.inputs, model.base, model.require
ROOT = model.ROOT
NAME = "p1_dev303_v3_four_arm_cp5_v13_physical_20261102_exclusive"
MODEL_PREP = ROOT / "results/p1_dev303_v3_four_arm_cp5_v13_model_20261102_exclusive_PREP_01.json"
MODEL_PREP_SHA = "07828b3e7901839389490089b0fac84e583fc259e3a10476262ab0f01e47ef7b"
AUTHORITY = "EXPLICIT_DEV303_FOUR_ARM_V13_PHYSICS"


def build(report_path, report_sha):
    report_path = Path(report_path).resolve(strict=True)
    require(base.sha(MODEL_PREP) == MODEL_PREP_SHA and base.read(MODEL_PREP) == model.protocol(),
            "model preparation/source drift")
    require(base.sha(report_path) == report_sha, "external model report SHA")
    report, source = base.read(report_path), base.read(model.PREP)
    require(report["schema"] == model.NAME + ".report.v1" and report["status"] == "model_only_complete"
            and report["prepared_sha256"] == MODEL_PREP_SHA and report["input_sha256"] == model.PREP_SHA
            and report["model_before"] == report["model_after"]
            and report["model_before"]["original_state_dict"] == source["model_sha256"]
            and all(report[k] is False for k in ("ledger_verified", "equal_full_budget_certified",
                    "physical_actions_authorized", "P1_admission", "independent_confirmation")),
            "model report identity")
    rows = report["rows"]
    require([r["role"] for r in rows] == list(inputs.ROLES)
            and len({r["initial_bank_sha256"] for r in rows}) == 1, "four roles/common initialization")
    specs = []
    for role, row in zip(source["roles"], rows, strict=True):
        require(row["cp5_pixel_sha256"] == role["cp5_pixel_sha256"]
                and row["actual_prefix_sha256"] == role["actual_prefix_sha256"]
                and row["counts"] == base.read(MODEL_PREP)["per_arm_target"]
                and row["hooks"] == {"image_encoder": 1, "image_projector": 1,
                                     "action_encoder": 10, "predictor": 10}
                and [iteration["iteration"] for iteration in row["iterations"]] == list(range(5)),
                "role/model counts")
        tail = np.asarray(row["selected"]["actions"], dtype=np.float64)
        require(tail.shape == (2, 5, 2) and np.isfinite(tail).all() and (np.abs(tail) <= 1).all()
                and np.array_equal(tail.astype("<f4").astype(np.float64), tail), "bounded exact tail")
        tail = tail.astype("<f4").reshape(10, 2)
        require(hashlib.sha256(tail.tobytes()).hexdigest() == row["tail_actions_sha256"], "tail SHA")
        specs.append({"arm": role["role"], "prefix": role["actual_prefix"],
                      "prefix_checkpoints": role["prefix_checkpoints"],
                      "cp5_pixel_sha256": role["cp5_pixel_sha256"],
                      "tail": tail.tolist(), "tail_sha256": row["tail_actions_sha256"]})
    return {"schema": NAME + ".contract.v1", "model_report": str(report_path),
            "model_report_sha256": report_sha, "model_prepare_sha256": MODEL_PREP_SHA,
            "roles": specs, "classification": "posthoc_development_not_independent",
            "score_source": "local model report; calls/hooks reported, not independently authenticated",
            "ledger_verified": False, "equal_full_budget_certified": False,
            "P1_admission": False, "independent_confirmation": False,
            "physical_execution_authorized_by_contract": False,
            "sources": {str(Path(p).resolve()): base.sha(p) for p in
                        [__file__, Path(__file__).with_name(
                            "p1_dev303_v3_four_arm_cp5_v13_physical_saved_20261102_exclusive.py"),
                         replay.__file__, replay.verifier.__file__, model.__file__]}}


def execute(prepared, prepared_sha, output, authority):
    require(authority == AUTHORITY, "explicit physics authority")
    prepared = Path(prepared).resolve(strict=True)
    contract = base.read(prepared)
    require(base.sha(prepared) == prepared_sha
            and contract == build(contract["model_report"], contract["model_report_sha256"]),
            "physical preparation/source drift")
    arrays, receipt, reset = inputs.physical.prior.source_data()
    store = base.Store(output)
    store.json("execution_binding.json", {"prepared": str(prepared), "prepare_sha256": prepared_sha})
    store.emit("run_started", prepare_sha256=prepared_sha)
    rows, errors = [], []
    for role in contract["roles"]:
        arm = role["arm"]
        row = {"arm": arm, "status": "running", "transitions": [], "observations": [], "error": None}
        env = None
        store.emit("branch_started", arm=arm)
        try:
            cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
            require(base.sha(inspect.getsourcefile(cls)) == base.native.ENV_SHA, "native source")
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
        require(contract == build(contract["model_report"], contract["model_report_sha256"]),
                "postrun source closure")
    except Exception as exc:
        errors.append({"stage": "source_closure", "type": type(exc).__name__, "message": str(exc)[:1000]})
    store.emit("run_finished", errors=errors)
    report = {"schema": NAME + ".report.v1", "prepare_sha256": prepared_sha,
              "status": "partial_or_failed" if errors else "recorded_development",
              "rows": rows, "errors": errors, "journal_sha256": base.sha(store.path / "journal.jsonl"),
              "journal_tail": store.events[-1]["sha256"], "P1_admission": False,
              "independent_confirmation": False, "ledger_verified": False,
              "equal_full_budget_certified": False}
    store.json("report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--model-report", type=Path)
    parser.add_argument("--model-report-sha256")
    parser.add_argument("--contract-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--authority")
    args = parser.parse_args()
    if args.command == "prepare":
        require(args.model_report and args.model_report_sha256 and not args.authority
                and not args.output and not args.contract_sha256, "prepare arguments")
        print(json.dumps({"sha256": inputs.physical.prior.write_new(
            args.contract, build(args.model_report, args.model_report_sha256))}))
    else:
        require(args.output and args.contract_sha256 and not args.model_report
                and not args.model_report_sha256, "run arguments")
        report = execute(args.contract, args.contract_sha256, args.output, args.authority)
        print(json.dumps({"status": report["status"], "report_sha256": base.sha(args.output / "report.json")}))


if __name__ == "__main__":
    main()
