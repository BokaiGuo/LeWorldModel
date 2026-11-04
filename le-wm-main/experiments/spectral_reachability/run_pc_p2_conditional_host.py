"""Source-sealed known-root conditional initialization development runner."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from experiments.spectral_reachability import p2_same_bank_host_development_20261102_exclusive as host
from experiments.spectral_reachability import p2_same_bank_native_objective_20261102_exclusive as native
from experiments.spectral_reachability import p2_same_bank_conditional_suffix_ablation_20261102_exclusive as bridge
from experiments.spectral_reachability import p2_common_objective_ledger_20261102_exclusive as ledger_core

NAME = "pc_p2_conditional_host_20261102"
ROOT = native.bank.ROOT
RESULTS = ROOT / "results"
METER_MODULE = "p2_conditional_hook_meter_20261102_exclusive.py"


def output_path(path, prefix):
    path = Path(path)
    if (not path.is_absolute() or path.parent != RESULTS or path.parent.resolve() != RESULTS
            or not path.name.startswith(NAME + prefix) or os.path.lexists(path)):
        raise ValueError("exclusive output under results required")
    return path


def source_bindings():
    result = host.source_bindings()
    directory = Path(__file__).parent
    for path in (Path(__file__), Path(bridge.__file__), directory / METER_MODULE,
                 ROOT.parent / "docs/PC_P2_CONDITIONAL_HOST_PROTOCOL_20261102.md"):
        result[str(path)] = host.file_sha(path)
    return result


def build_plan():
    fixed = native.verified_input()
    from experiments.spectral_reachability import p1_constructive_blind_encode_cuda_20261027_v1 as checkpoint
    return {
        "schema": NAME + ".plan_v1", "scope": "one known root, model-only conditional initialization",
        "source_sha256": source_bindings(),
        "input": {key: fixed[key] for key in ("context_sha256", "goal_sha256", "handoff_sha256", "pair_sha256")},
        "bank_sha256": [host.sha(raw) for raw in fixed["bank"]],
        "checkpoint_sha256": checkpoint.MODELS, "parameter_sha256": checkpoint.MODEL_TENSOR_SHA,
        "normalizer": checkpoint.native_metadata(), "device": "cuda:0",
        "optimizer": {"candidates": 16, "updates": 8, "step": 0.05,
                      "per_arm_score_calls": 144, "per_arm_gradient_calls": 128},
        "conditional": {"prefix_transitions": 16, "suffix_transition_forwards": 32,
                        "jacobians": 16, "requested_vjp_directions": 3072,
                        "bounded_solves": 16, "ridge": bridge.DAMPING, "radius": bridge.DELTA_CAP},
        "limitations": ["conditional compute is extra", "model load and inherited bank unmetered",
                        "FLOPs, energy and autograd engine internals unknown",
                        "source list is not full transitive closure", "known root", "no physical execution"],
        "equal_total_compute": False, "P1_admission": False, "P2_admission": False,
    }


def verify(plan_path, expected_sha):
    plan_path = Path(plan_path)
    if plan_path.parent != RESULTS or not plan_path.name.startswith(NAME + "_PLAN_"):
        raise ValueError("invalid plan path")
    if host.file_sha(plan_path) != expected_sha:
        raise ValueError("plan SHA mismatch")
    plan = json.loads(plan_path.read_text())
    if plan != build_plan() or host.file_sha(plan_path) != expected_sha:
        raise ValueError("plan or bound source drift")
    return plan


def execute(plan_path, expected_sha, output):
    plan = verify(plan_path, expected_sha)
    output = output_path(output, "_RUN_")
    if shutil.disk_usage(RESULTS).free < host.MIN_FREE_BYTES:
        raise OSError("insufficient disk")
    output.mkdir(mode=0o700)
    host.write_new(output / "binding.json", {"plan": str(plan_path), "sha256": expected_sha})
    ledger = ledger_core.Ledger(output / "journal.jsonl")
    started = time.monotonic()
    result = None
    failure = None
    counts = None
    parameter_unchanged = False
    try:
        from experiments.spectral_reachability import p2_conditional_hook_meter_20261102_exclusive as routed
        from experiments.spectral_reachability import p1_constructive_blind_encode_cuda_20261027_v1 as checkpoint
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        from experiments.spectral_reachability import pusht_full_packet_compiler_20261027 as compiler
        from experiments.checkpoint_compat import load_compatible_pretrained
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with native.locking.gpu_lock(native.LOCK):
            if not torch.cuda.is_available():
                raise RuntimeError("host CUDA required")
            native.check_normalizer(checkpoint.native_metadata(), p0.PUBLIC_ACTION_MEAN, p0.PUBLIC_ACTION_STD)
            for name, expected in checkpoint.MODELS.items():
                if checkpoint.sha(name) != expected:
                    raise ValueError("checkpoint drift")
            model = load_compatible_pretrained(str(checkpoint.POLICY), device="cuda:0").eval().requires_grad_(False)
            checkpoint.frozen(model)
            if checkpoint.tensor_hash(model) != checkpoint.MODEL_TENSOR_SHA:
                raise RuntimeError("loaded parameter hash mismatch")
            meter = routed.RoutedModelMeter(ledger)
            transition = compiler.make_transition(model, torch.device("cuda:0"))
            old_matmul = torch.backends.cuda.matmul.allow_tf32
            old_cudnn = torch.backends.cudnn.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                with sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", enabled=False):
                    with ledger_core.native_model_scope(model, meter):
                        result = bridge.run_bridge(native.verified_input(), transition, meter.ledger, "cuda:0")
                counts = meter.finish()
                torch.cuda.synchronize()
                checkpoint.frozen(model)
                parameter_unchanged = checkpoint.tensor_hash(model) == checkpoint.MODEL_TENSOR_SHA
                if not parameter_unchanged:
                    raise RuntimeError("parameter drift")
            finally:
                torch.backends.cuda.matmul.allow_tf32 = old_matmul
                torch.backends.cudnn.allow_tf32 = old_cudnn
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)[:1500]}
    finally:
        ledger.close()
    try:
        verify(plan_path, expected_sha)
    except Exception as exc:
        failure = failure or {"type": type(exc).__name__, "message": "postcheck: " + str(exc)[:1400]}
    audit = ledger_core.verify_ledger(output / "journal.jsonl")
    audit["counts"] = {category + "/" + axis: count for (category, axis), count in audit["counts"].items()}
    if audit["outstanding"]:
        failure = failure or {"type": "OutstandingEvents", "message": str(audit["outstanding"])}
    report = {"status": "completed_development" if failure is None else "failed_partial",
              "plan_sha256": expected_sha, "result": result, "failure": failure, "hook_attempt_counts": counts,
              "journal_sha256": host.file_sha(output / "journal.jsonl"), "ledger_audit": audit,
              "wall_seconds": time.monotonic() - started, "frozen_parameters_unchanged": parameter_unchanged,
              "limitations": plan["limitations"], "equal_total_compute": False,
              "P1_admission": False, "P2_admission": False}
    return report, host.write_new(output / "report.json", report)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--plan-sha256", required=True)
    run.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.command == "prepare":
        path = output_path(args.output, "_PLAN_")
        print(json.dumps({"plan": str(path), "sha256": host.write_new(path, build_plan())}))
    else:
        report, digest = execute(args.plan, args.plan_sha256, args.output)
        print(json.dumps({"status": report["status"], "report_sha256": digest}))
        if report["status"] != "completed_development":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
