"""Sealed known-root P2 joint/conditional CUDA development host.

No execution is authorized by preparing a plan. GPU use requires coordination.
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

from . import p2_joint_vs_conditional_matched_20261102_codexexclusive as matched
from . import p2_same_bank_host_development_20261102_exclusive as base
from . import p2_same_bank_native_objective_20261102_exclusive as native
from . import p2_common_objective_ledger_20261102_exclusive as core

NAME = "p2_joint_host_20261102_codexexclusive"
UUID = "GPU-4e8973ff-0278-53f3-2bae-2615d461e441"
AUTH = "P2_JOINT_CONDITIONAL_KNOWN_ROOT_DEVELOPMENT_ONLY"
RESULTS = base.RESULTS


def require_authorization(auth, visible):
    if auth != AUTH or visible != UUID:
        raise ValueError("explicit authorization and exclusive UUID required")


class TimedMeter(matched.MatchedMeter):
    """Synchronized inclusive wall time; nested durations must not be added."""
    def __init__(self, raw, synchronize, deadline=None):
        self.synchronize = synchronize
        self.deadline = deadline
        super().__init__(raw)

    def dispatch(self, axis, operation, *, hook):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self.failed = True
            self.raw.record("host_deadline_reject", {"axis": axis})
            raise TimeoutError("dispatch deadline; partial schedule retained")
        self.synchronize()
        start = time.perf_counter_ns()
        status = "failed"
        try:
            result = super().dispatch(axis, operation, hook=hook)
            self.synchronize()
            status = "returned"
            return result
        finally:
            self.raw.record("timed_call", {"axis": axis, "status": status,
                                           "elapsed_ns": time.perf_counter_ns() - start,
                                           "inclusive": True})


def measured_transition(transition, meter):
    def call(state, history, action):
        shapes = [list(x.shape) for x in (state, history, action)]
        if shapes != [[1, 3, 192], [1, 2, 10], [5, 2]]:
            raise ValueError("unexpected transition input shape")
        meter.raw.record("transition_shape", {"state_shape": shapes[0], "history_shape": shapes[1],
            "action_shape": shapes[2], "context_tokens": 3, "successor_tokens": 1})
        return transition(state, history, action)
    return call


def build_plan():
    from . import p1_constructive_blind_encode_cuda_20261027_v1 as checkpoint
    fixed = native.verified_input()
    files = base.source_bindings()
    directory = Path(__file__).parent
    for name in ("p2_joint_host_20261102_codexexclusive.py",
                 "p2_joint_saved_20261102_codexexclusive.py",
                 "p2_joint_vs_conditional_matched_20261102_codexexclusive.py",
                 "p2_same_bank_conditional_suffix_ablation_20261102_exclusive.py",
                 "p2_conditional_hook_meter_20261102_exclusive.py"):
        path = directory / name
        files[str(path)] = base.file_sha(path)
    protocol = native.bank.ROOT.parent / "docs/PC_P2_JOINT_HOST_SAVED_20261102_CODEXEXCLUSIVE.md"
    files[str(protocol)] = base.file_sha(protocol)
    return {"schema": NAME + ".plan_v1", "sources": files,
            "inputs": {k: fixed[k] for k in ("context_sha256", "goal_sha256", "handoff_sha256", "pair_sha256")},
            "bank_sha256": [base.sha(x) for x in fixed["bank"]],
            "checkpoint_files": checkpoint.MODELS, "model_tensor_sha256": checkpoint.MODEL_TENSOR_SHA,
            "normalizer": checkpoint.native_metadata(), "cuda_uuid": UUID, "lock": str(native.LOCK),
            "caps": matched.CAPS, "jacobian_shapes": [[192, 10], [192, 20]],
            "wall_dispatch_deadline_seconds": base.WALL_SECONDS,
            "control": "joint_gauss_newton_not_historical_sra",
            "limitations": ["known exposed B0 root", "not equal FLOPs or total compute",
              "historical SRA binding failure remains unresolved", "source manifest is not transitive closure",
              "inherited bank and model loading cost not matched", "no physical execution",
              "Jacobian internal backward calls and FLOPs unknown", "inclusive synchronized timing is not kernel time"],
            "equal_total_compute": False, "independent_confirmation": False, "P2_admission": False}


def path_new(path, kind):
    path = Path(path)
    if (not path.is_absolute() or path.parent != RESULTS or path.parent.resolve() != RESULTS
            or not path.name.startswith(NAME + kind) or os.path.lexists(path)):
        raise ValueError("exclusive results path required")
    return path


def verify_plan(path, digest):
    path = Path(path)
    if (not path.is_absolute() or path.parent != RESULTS or path.is_symlink()
            or not path.name.startswith(NAME + "_PLAN_") or base.file_sha(path) != digest):
        raise ValueError("plan identity")
    plan = json.loads(path.read_bytes())
    if plan != build_plan() or base.file_sha(path) != digest:
        raise ValueError("plan or source drift")
    return plan


def execute(plan_path, digest, output, authorization):
    require_authorization(authorization, os.environ.get("CUDA_VISIBLE_DEVICES"))
    plan = verify_plan(plan_path, digest)
    output = path_new(output, "_RUN_")
    if shutil.disk_usage(RESULTS).free < base.MIN_FREE_BYTES:
        raise OSError("insufficient disk")
    output.mkdir(mode=0o700)
    base.write_new(output / "binding.json", {"plan": str(plan_path), "plan_sha256": digest})
    ledger = core.Ledger(output / "journal.jsonl")
    started = time.monotonic()
    result = counts = device_info = None
    frozen = False
    failure = None
    try:
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from . import p1_constructive_blind_encode_cuda_20261027_v1 as checkpoint
        from . import preflight_pusht_d2p_p0 as p0
        from . import pusht_full_packet_compiler_20261027 as compiler
        with native.locking.gpu_lock(native.LOCK):
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError("one visible CUDA device required")
            props = torch.cuda.get_device_properties(0)
            actual = str(getattr(props, "uuid", ""))
            if not actual.lower().removeprefix("gpu-") == UUID.lower().removeprefix("gpu-"):
                raise RuntimeError("CUDA UUID unavailable or mismatch")
            device_info = {"uuid": UUID, "name": props.name, "torch": str(torch.__version__),
                           "cuda_runtime": torch.version.cuda}
            native.check_normalizer(checkpoint.native_metadata(), p0.PUBLIC_ACTION_MEAN, p0.PUBLIC_ACTION_STD)
            for name, expected in checkpoint.MODELS.items():
                if checkpoint.sha(name) != expected:
                    raise ValueError("checkpoint drift")
            model = load_compatible_pretrained(str(checkpoint.POLICY), device="cuda:0").eval().requires_grad_(False)
            checkpoint.frozen(model)
            if checkpoint.tensor_hash(model) != checkpoint.MODEL_TENSOR_SHA:
                raise ValueError("model tensor mismatch")
            meter = TimedMeter(ledger, lambda: torch.cuda.synchronize(0), started + base.WALL_SECONDS)
            transition = measured_transition(compiler.make_transition(model, torch.device("cuda:0")), meter)
            old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
                with sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", enabled=False):
                    with core.native_model_scope(model, meter):
                        result = matched.run_native(native.verified_input(), transition, meter, "cuda:0")
                counts = meter.finish()
                torch.cuda.synchronize(0)
                checkpoint.frozen(model)
                frozen = checkpoint.tensor_hash(model) == checkpoint.MODEL_TENSOR_SHA
                if not frozen:
                    raise ValueError("model tensor drift")
            finally:
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old
        verify_plan(plan_path, digest)
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)[:1500]}
    finally:
        ledger.close()
    report = {"schema": NAME + ".report_v1", "status": "completed_development" if failure is None else "failed_partial",
              "plan_sha256": digest, "journal_sha256": base.file_sha(output / "journal.jsonl"),
              "result": result, "counts": counts, "failure": failure, "device": device_info,
              "frozen_parameters_unchanged": frozen, "wall_seconds": time.monotonic() - started,
              "equal_total_compute": False, "P2_admission": False}
    return report, base.write_new(output / "report.json", report)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", required=True)
    run = sub.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--plan-sha256", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--authorization", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        path = path_new(args.output, "_PLAN_")
        print(json.dumps({"plan": str(path), "sha256": base.write_new(path, build_plan())}))
    else:
        report, sha = execute(args.plan, args.plan_sha256, args.output, args.authorization)
        print(json.dumps({"status": report["status"], "report_sha256": sha}))
        if report["failure"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
