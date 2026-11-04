"""Static contract audit for the Reacher P0/P1 runner chain.

This is read-only source/evidence auditing. It does not import the neural
runner, allocate CUDA, create MuJoCo environments, or execute a long run.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
SPEC = EXP / "spectral_reachability"

FILES = {
    "p0_cli": SPEC / "run_reacher_d2p_p0_cuda.sh",
    "p0_preflight": SPEC / "preflight_reacher_d2p_p0.py",
    "p0_runner": SPEC / "run_reacher_d2p_p0_goal_progress.py",
    "p1_preflight": SPEC / "preflight_reacher_d2p_p1_reconfiguration.py",
    "p1_runner": SPEC / "run_reacher_d2p_p1_reconfiguration.py",
    "context": SPEC / "reacher_d2p_context.py",
    "predict_instrumentation": SPEC / "d2p_model_call_instrumentation.py",
    "compute_ledger": SPEC / "d2p_planning_compute_ledger.py",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text(name: str) -> str:
    return FILES[name].read_text(encoding="utf-8")


def functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit() -> dict:
    for path in FILES.values():
        require(path.is_file(), f"missing source: {path}")

    p0_cli = text("p0_cli")
    p0 = text("p0_preflight")
    p0_runner = text("p0_runner")
    p1_preflight = text("p1_preflight")
    p1 = text("p1_runner")
    context = text("context")
    instrumentation = text("predict_instrumentation")
    ledger = text("compute_ledger")

    # CLI and CUDA boundary.
    require("preflight_reacher_d2p_p0" in p0_cli, "P0 CLI does not call the P0 preflight")
    require("--device cuda" in p0_cli, "P0 CLI does not force CUDA")
    require("CPU fallback" in p0 and 'args.device != "cuda"' in p0,
            "P0 preflight lacks an exact CUDA/no-fallback guard")
    require("require_new_output(args.output" in p0,
            "P0 preflight does not reserve a new output before work")
    require("_resolve_local_cli_inputs" in p1 and "Path(value).expanduser().resolve()" in p1,
            "P1 CLI path resolution is not source-bound")
    require("args.output.exists()" in p1 and "FileExistsError" in p1,
            "P1 runner lacks immutable output refusal")

    # P0 evidence and temporal feedback.
    require("d2p_reacher_p0_goal_progress.v1" in p0_runner,
            "P0 runner schema is not present")
    require("executed_context" in p0_runner and "root_rgb_sha256" in p0_runner
            and "warmup_actions_sha256" in p0_runner,
            "P0 evidence does not persist executed RGB/action identity")
    require("runtime_input" in p0_runner and "RGB/action history" in p0_runner,
            "P0 report does not declare RGB/action-history model input")
    require("_prepare_context" in p0 and "history_actions" in p0,
            "P0 preflight source does not bind RGB to action history")
    require("context_support" in p1_preflight and "warmup_policy" in p1_preflight
            and "qpos_readout_binding" in p1_preflight,
            "P1 preflight lacks P0 context/warmup/readout binding")
    require("_validate_upstream_p0" in p1 and "strict_admission_revalidated" in p1_preflight,
            "P1 runner does not revalidate P0 admission")

    # Physical execution is present only in the P1 physical runner; P0
    # preflight's warmup is intentionally not treated as candidate efficacy.
    require("collect_executed_context" in context and "_execute_raw_steps" in context,
            "executed warmup context helper is missing")
    require("advance_executed_context" in p1 and "p0._execute(" in p1,
            "P1 runner does not dispatch observed successor and candidate physical actions")
    require("execution_snapshot" in p1 and "restore_execution_snapshot" in p1,
            "P1 physical replay lacks evaluator-root restoration binding")
    require("qpos is evaluator-only" in p1,
            "P1 report does not state qpos evaluator-only boundary")
    require("p0._execute(" not in p0,
            "P0 preflight unexpectedly contains candidate physical dispatch")

    # Budget boundary: the existing instrumentation is explicitly predict-only
    # and the P1 runner does not instantiate the persistent/vector ledger.
    require("predict_calls" in instrumentation and "failed_predict_calls" in instrumentation,
            "predict attempt/failure accounting is missing")
    require("model.predict" in instrumentation and "action_encoder" not in instrumentation,
            "predict instrumentation scope changed unexpectedly")
    require("PredictBudget" not in p1 and "PredictionBudget" not in p1,
            "P1 runner unexpectedly claims an adaptive budget guard")
    require("env.step" not in p1 and "env.step" in context,
            "physical step boundary is not isolated in the context helper")
    require("record_resource_attempt" not in p1 and "record_resource_return" not in p1,
            "P1 runner unexpectedly contains durable resource journal calls")
    require("Resources" in ledger and "physical_env_steps" in ledger,
            "resource-vector ledger definition is missing")

    findings = [
        {
            "id": "cli",
            "status": "verified_static",
            "scope": "P0 shell entrypoint and P0/P1 Python CLI",
            "evidence": [
                "P0 shell invokes preflight_reacher_d2p_p0 with --device cuda",
                "P0 rejects non-cuda and reserves a new output",
                "P1 resolves local evidence paths and refuses existing output",
            ],
        },
        {
            "id": "p0_evidence",
            "status": "verified_static",
            "scope": "P0 report and P1 upstream admission",
            "evidence": [
                "P0 persists executed_context RGB/action digests",
                "P1 revalidates P0 schema, context support, warmup policy, qpos chart binding and stage admission",
            ],
        },
        {
            "id": "rgb_action_history_feedback",
            "status": "verified_static",
            "scope": "model-side context",
            "evidence": [
                "root-anchored RGB frames and causal action history are prepared together",
                "planner runtime input declares RGB/action history and rendered goal only",
                "qpos/qvel remain evaluator-only",
            ],
        },
        {
            "id": "physical_execution",
            "status": "verified_static",
            "scope": "P1 only",
            "evidence": [
                "P1 restores the evaluator root, advances an observed action block, recompiles from observed RGB, and dispatches p0._execute",
                "P0 preflight does not dispatch candidate physical actions",
            ],
        },
        {
            "id": "budget_ledger",
            "status": "gap",
            "scope": "P1 runner",
            "evidence": [
                "predict-only in-memory instrumentation counts predict calls/samples and failed predict calls",
                "the vector resource ledger is a standalone comparison utility, not wired into P1",
                "encoder, backward, env.step, attempt/return/failure persistence and fsync are not covered by P1",
            ],
            "restriction": "Do not call this an equal-compute or complete budget receipt.",
        },
    ]
    return {
        "schema": "audit_reacher_p1_runner_contract_20261102_exclusive.v1",
        "status": "static_contract_audit_with_budget_gap",
        "cuda_started": False,
        "long_run_started": False,
        "gate_modified": False,
        "files": {key: {"path": str(path.resolve()), "sha256": sha256(path)}
                  for key, path in FILES.items()},
        "functions": {key: sorted(functions(path)) for key, path in FILES.items()
                      if path.suffix == ".py"},
        "findings": findings,
        "claims": {
            "P0_evidence_is_physical_candidate_outcome": False,
            "P1_physical_execution_is_present_in_source": True,
            "complete_persistent_budget_ledger": False,
            "equal_total_compute": False,
            "P1_admission": False,
            "P2_admission": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    report = audit()
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
