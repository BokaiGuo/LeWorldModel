"""Presealed, exploratory dev303 four-arm physical runner; no P1 admission."""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
from pathlib import Path

import numpy as np

from . import p1_dev303_rgb_chunk_diagnostic_20261102_exclusive as prior
from . import p1_dev303_v3_cuda_executor_20261102_exclusive as cuda
from . import p1_dev303_v3_four_arm_blocked_preflight_20261102_exclusive as audit
from . import p1_dev303_v3_model_evaluation_contract_20261102_exclusive as selector

base = prior.base
require = prior.require
NAME = "p1_dev303_v3_four_arm_physical_20261102_exclusive"
ROOT = prior.ROOT


def build():
    # Reconstruct the actual fixed selector and verify the pre-existing executor's
    # static source/model bindings. This is not runtime or budget attestation.
    identity = audit.audit_local()
    # The earlier preflight's blocked status was for its own stricter proposal;
    # retain the underlying facts, not that proposal's admission decision.
    identity.pop("status")
    identity["local_identity_status"] = "matched_not_independently_attested"
    cuda.verify_contract(audit.CUDA_PREP, audit.PINS["cuda_prepare"])
    prep = base.read(audit.MODEL_PREP)
    report = base.read(audit.RUN / "report.json")
    require(selector.select(prep, [row["score"] for row in report["rows"]])
            == report["selection"], "selector reconstruction")
    require(report["selection"]["score_provenance_verified"] is False
            and report["selection"]["ledger_verified"] is False, "report qualification")
    prior_contract = prior.build()
    verifier = Path(__file__).with_name("p1_dev303_v3_four_arm_physical_saved_20261102_exclusive.py")
    own = [Path(__file__), verifier, Path(audit.__file__), Path(prior.__file__),
           Path(selector.__file__), Path(cuda.__file__)]
    return {
        "schema": NAME + ".contract.v1",
        "classification": "exploratory_development_nonfair_not_independent",
        "model_identity": identity,
        "selection_audit": "pinned local report/events; fixed source selector reconstructed; "
                           "hooks reported and source instrumentation inspected, not independently attested",
        "ledger_verified": False, "equal_budget_claim": False,
        "encoding_and_total_compute": "historical encoding and full budget not measured",
        "source": prior_contract,
        "source_closure_sha256": {str(p.resolve()): base.sha(p) for p in own},
        "arms": identity["arms_fixed_before_physics"],
        "warmup_actions": 10, "prefix_actions": 5,
        "failure_policy": "stop arm on failure or native stop; preserve partial; no padding, retry or replacement",
        "outcome_selection": False, "new_model_calls": 0,
        "physical_execution_authorized_by_contract": False,
        "P1_admission": False, "independent_confirmation": False,
    }


def branch(env, store, spec, history, reset, goal_rgb, row):
    arm = spec["role"]
    goal = np.asarray(history["goal"], np.float64)
    checkpoints = {p["step"]: p for p in history["frames"]}
    obs, info = env.reset(seed=prior.SEED)
    require(np.array_equal(np.asarray(obs["state"], np.float64), np.asarray(reset["state"], np.float64))
            and np.array_equal(np.asarray(info["goal_state"], np.float64), goal)
            and np.array_equal(np.asarray(info["goal"]), goal_rgb), "reset/goal mismatch")
    store.emit("reset_verified", arm=arm)
    row["observations"].append(base.capture(store, env, arm, "warmup", 0, checkpoints[0]))
    for phase, actions in (("warmup", [[0.0, 0.0]] * 10), ("prefix", spec["actions"])):
        for i, value in enumerate(actions, 1):
            store.check()
            action = np.asarray(value, dtype="<f4")
            store.emit("action_dispatch", arm=arm, phase=phase, step=i, action=action.tolist())
            obs, reward, terminated, truncated, info = env.step(action.copy())
            observed = np.asarray(env.latest_action)
            store.emit("action_observed", arm=arm, phase=phase, step=i, action=observed.tolist(),
                       terminated=bool(terminated), truncated=bool(truncated))
            actual = prior._check_return(env, obs, info, goal, goal_rgb, action)
            require(np.isfinite(float(reward)), "nonfinite reward")
            transition = {"phase": phase, "step": i, "action": actual.tolist(),
                          "state": np.asarray(obs["state"]).tolist(), "reward": float(reward),
                          "terminated": bool(terminated), "truncated": bool(truncated)}
            row["transitions"].append(transition)
            store.emit("action_return", arm=arm, **transition)
            if phase == "prefix" or i in checkpoints:
                row["observations"].append(base.capture(store, env, arm, phase, i,
                                                        checkpoints.get(i) if phase == "warmup" else None))
            if terminated or truncated:
                if phase == "warmup":
                    raise ValueError("warmup stopped")
                row["status"] = "native_early_stop"
                return
        if phase == "warmup":
            require(np.array_equal(np.asarray(obs["state"], np.float64),
                                   np.asarray(history["final_state"], np.float64)), "warmup root mismatch")
    row["status"] = "fixed_horizon_completed"


def execute(prepared, prepared_sha, output, *, authority):
    require(authority == "EXPLICIT_DEV303_EXPLORATORY_PHYSICS", "explicit development physics authority")
    prepared = Path(prepared).resolve(strict=True)
    require(base.sha(prepared) == prepared_sha and base.read(prepared) == build(),
            "contract or source drift")
    contract = base.read(prepared)
    arrays, receipt, reset = prior.source_data()
    store = base.Store(output)
    store.json("execution_binding.json", {"prepared": str(prepared), "prepare_sha256": prepared_sha})
    store.emit("run_started", prepare_sha256=prepared_sha)
    rows, errors = [], []
    for spec in contract["arms"]:
        arm = spec["role"]
        row = {"arm": arm, "status": "running", "transitions": [], "observations": [], "error": None}
        env = None
        store.emit("branch_started", arm=arm)
        try:
            cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
            require(base.sha(inspect.getsourcefile(cls)) == base.native.ENV_SHA, "native environment SHA")
            env = cls(render_mode="rgb_array", resolution=224, with_target=False)
            branch(env, store, spec, receipt["history"], reset, arrays["goal_rgb"], row)
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
        errors.append({"stage": "source_closure", "type": type(exc).__name__,
                       "message": str(exc)[:1000]})
    store.emit("run_finished", errors=errors)
    report = {"schema": NAME + ".report.v1", "prepare_sha256": prepared_sha,
              "status": "partial_or_failed" if errors else "recorded_development",
              "rows": rows, "errors": errors, "journal_sha256": base.sha(store.path / "journal.jsonl"),
              "journal_tail": store.events[-1]["sha256"], "ledger_verified": False,
              "equal_budget_claim": False, "P1_admission": False, "independent_confirmation": False}
    store.json("report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--authority")
    args = parser.parse_args()
    if args.command == "prepare":
        require(args.contract_sha256 is None and args.output is None and args.authority is None,
                "prepare only")
        print(json.dumps({"sha256": prior.write_new(args.contract, build())}))
    else:
        require(args.contract_sha256 is not None and args.output is not None, "run arguments")
        result = execute(args.contract, args.contract_sha256, args.output, authority=args.authority)
        print(json.dumps({"status": result["status"], "report_sha256": base.sha(args.output / "report.json")}))


if __name__ == "__main__":
    main()
