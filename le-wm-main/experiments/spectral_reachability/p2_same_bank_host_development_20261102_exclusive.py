"""Pinned known-root same-objective CUDA development runner.

This is a repeatability control, not a composition ablation or scientific
admission. Source manifest is explicit, not the full transitive runtime closure.
Never invoke run without coordinated host GPU access and review of the plan.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

from . import p2_same_bank_native_objective_20261102_exclusive as native
from . import p2_same_bank_ledger_consumer_20261102_exclusive as consumer
from . import p2_common_objective_ledger_20261102_exclusive as ledger_core

NAME = "p2_same_bank_host_development_20261102_exclusive"
ROOT = native.bank.ROOT
RESULTS = ROOT / "results"
AUTH = "P2_KNOWN_ROOT_SAME_OBJECTIVE_HOST_DEVELOPMENT_ONLY"
WALL_SECONDS = 900
MIN_FREE_BYTES = 200 * 1024 * 1024
SOURCE_NAMES = (
    "p2_same_bank_host_development_20261102_exclusive.py",
    "p2_same_bank_native_objective_20261102_exclusive.py",
    "p2_same_bank_common_optimizer_20261102_exclusive.py",
    "p2_same_bank_ledger_consumer_20261102_exclusive.py",
    "p2_common_objective_ledger_20261102_exclusive.py",
    "p2_sra_common_optimizer_bridge_20261102_exclusive.py",
    "pusht_full_packet_compiler_20261027.py",
    "e3_p2_cooperative_gpu_lock_codex_20261030_v1.py",
    "codex_p2_program_model_scope_20261030_v1.py",
    "p1_constructive_blind_encode_cuda_20261027_v1.py",
    "preflight_pusht_d2p_p0.py",
    "pusht_d2p_packet_contract.py",
    "p2_cp15_capture_builder_20261102_codexexclusive.py",
    "p2_shared_capture_launch_20261102_exclusive.py",
    "p2_shared_rgb_pair_wiring_20261102_codexexclusive.py",
    "codex_sra_online_run_bank_20261030.py",
)
OTHER_SOURCES = ("experiments/checkpoint_compat.py", "experiments/run_pusht_counterfactual.py")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def raw_json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def file_sha(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing or symlink input: " + str(path))
    h = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
        after = os.fstat(stream.fileno())
    now = path.stat()
    identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    if identity(before) != identity(after) or identity(after) != identity(now) or path.is_symlink():
        raise ValueError("source changed while hashing: " + str(path))
    return h.hexdigest()


def exclusive_path(path, prefix):
    path = Path(path)
    if (not path.is_absolute() or path.parent != RESULTS or path.parent.resolve() != RESULTS or
            not path.name.startswith(NAME + prefix) or os.path.lexists(path)):
        raise ValueError("exclusive absolute output path required")
    return path


def source_bindings():
    directory = ROOT / "experiments/spectral_reachability"
    files = [directory / n for n in SOURCE_NAMES] + [ROOT / n for n in OTHER_SOURCES]
    return {str(p): file_sha(p) for p in files}


def build_plan():
    fixed = native.verified_input()
    from . import p1_constructive_blind_encode_cuda_20261027_v1 as checkpoint
    metadata = checkpoint.native_metadata()
    return {
        "schema": NAME + ".plan_v1",
        "scope": "known_root_same_bank_same_objective_repeatability_control",
        "source_sha256": source_bindings(),
        "input": {"handoff_sha256": fixed["handoff_sha256"], "pair_sha256": fixed["pair_sha256"],
                  "context_sha256": fixed["context_sha256"], "goal_sha256": fixed["goal_sha256"],
                  "capture_consumer_sha256": native.PROOF_SHA,
                  "selected_bank": "joint_multistart_all_16_in_index_order",
                  "initial_actions_sha256": [sha(x) for x in fixed["bank"]]},
        "checkpoint_sha256": checkpoint.MODELS, "model_tensor_sha256": checkpoint.MODEL_TENSOR_SHA,
        "normalizer": metadata, "device": "cuda:0",
        "lock_path": str(native.LOCK), "lock_scope": "native_session_model_load_through_postcheck",
        "objective": "two native 5x2 packets; mean squared terminal 192-vector error",
        "schedule": {"arms": list(native.optimizer.ARMS), "candidates_per_arm": 16,
                     "optimizer": "existing projected gradient, eight updates, best of nine",
                     "global_selection": "minimum returned model score across all 16; ties earliest index",
                     "learning_rate": native.optimizer.LEARNING_RATE,
                     "scores_per_arm": 144, "gradients_per_arm": 128,
                     "attempt_caps_per_arm": native.CAPS,
                     "attempt_counts_include_failed_calls": True},
        "failure_policy": {"durable_attempt_before_dispatch": True, "retain_partial": True,
                           "no_retry": True, "no_replace": True, "no_imputation": True},
        "limits": {"wall_seconds_dispatch_boundary": WALL_SECONDS, "wall_preemptive": False,
                   "minimum_free_bytes": MIN_FREE_BYTES, "flops": None, "energy": None,
                   "model_load_work_metered": False, "upstream_setup_cost_known": False,
                   "loader_and_initializer_costs_matched": False,
                   "source_manifest_full_transitive_closure": False,
                   "model_runtime_hardware_independently_authenticated": False},
        "authorization_required": AUTH, "P1_admission": False, "P2_admission": False,
        "composition_mechanism_authenticated": False, "equal_total_compute": False,
        "independent_confirmation": False,
    }


def write_new(path, value):
    raw = raw_json(value)
    with Path(path).open("xb") as stream:
        if stream.write(raw) != len(raw):
            raise OSError("short write")
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return sha(raw)


def prepare(path):
    path = exclusive_path(path, "_PLAN_")
    plan = build_plan()
    return write_new(path, plan)


def verify(path, expected):
    path = Path(path)
    if (not path.is_absolute() or path.parent != RESULTS or path.is_symlink() or
            not path.name.startswith(NAME + "_PLAN_")):
        raise ValueError("invalid plan path")
    raw = path.read_bytes()
    if sha(raw) != expected:
        raise ValueError("plan SHA mismatch")
    plan = json.loads(raw)
    if plan != build_plan() or sha(path.read_bytes()) != expected:
        raise ValueError("plan or bound source drift")
    return plan


def winner_handoff(result, plan_sha, audit):
    """Derive one winner per arm only from a fully audited ordered schedule."""
    if audit is None or audit["candidate_count_per_arm"] != 16:
        raise ValueError("complete ledger audit required")
    rows = result["rows"]
    if [row["index"] for row in rows] != list(range(16)):
        raise ValueError("winner input census")
    winners = []
    for arm in native.optimizer.ARMS:
        options = []
        for row in rows:
            matches = [x for x in row["arms"] if x["arm"] == arm]
            if len(matches) != 1:
                raise ValueError("winner arm census")
            entry = matches[0]
            score = entry["model_score"]
            action = bytes.fromhex(entry["raw_float32_hex"])
            if type(score) not in (int, float) or not math.isfinite(score) or len(action) != 80:
                raise ValueError("winner score or action")
            native.optimizer.raw_actions(native.np.frombuffer(action, dtype="<f4").reshape(10, 2))
            options.append((score, row["index"], entry, action))
        score, index, entry, action = min(options, key=lambda x: (x[0], x[1]))
        winners.append({"arm": arm, "candidate_index": index,
                        "iteration": entry["winner_iteration"], "model_score": score,
                        "raw_float32_hex": action.hex(), "raw_sha256": sha(action)})
    return {"schema": NAME + ".winners_v1", "plan_sha256": plan_sha,
            "journal_sha256": audit["ledger_sha256"], "winners": winners,
            "physical_execution_authorized": False, "known_root": True,
            "P1_admission": False, "P2_admission": False,
            "composition_mechanism_authenticated": False, "equal_total_compute": False}


def _execute(plan_path, plan_sha, output, session, profile):
    """Separate injectable session for CPU tests; CLI always passes native_session."""
    plan = verify(plan_path, plan_sha)
    output = exclusive_path(output, "_RUN_")
    if shutil.disk_usage(RESULTS).free < MIN_FREE_BYTES:
        raise ValueError("insufficient free disk")
    output.mkdir(mode=0o700)
    write_new(output / "binding.json", {"plan": str(plan_path), "plan_sha256": plan_sha,
                                      "profile": profile})
    journal = ledger_core.Ledger(output / "journal.jsonl")
    started = time.monotonic()
    result, failure, counts, callback, meter = None, None, None, None, None
    try:
        fixed = native.verified_input()
        def emit(event):
            journal.record("optimizer_" + event["kind"], event)
            if time.monotonic() - started > WALL_SECONDS:
                raise TimeoutError("nonpreemptive dispatch boundary wall cap")
            if shutil.disk_usage(RESULTS).free < MIN_FREE_BYTES:
                raise OSError("minimum free disk reached")
        with session(fixed, journal) as (callback, meter):
            result = native.run_schedule(fixed, callback, emit)
            counts = {"callback_attempts": callback.attempted, "instrumented_attempts": meter.counts}
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)[:1500]}
    finally:
        if callback is not None and meter is not None:
            counts = {"callback_attempts": callback.attempted, "instrumented_attempts": meter.counts}
        journal.close()
    post_error, audit, winners_sha = None, None, None
    try:
        verify(plan_path, plan_sha)
    except Exception as exc:
        post_error = {"type": type(exc).__name__, "message": str(exc)[:1500]}
    if failure is None and post_error is None:
        try:
            audit = consumer.audit(output / "journal.jsonl", profile)
            winners_sha = write_new(output / "winners.json", winner_handoff(result, plan_sha, audit))
        except Exception as exc:
            failure = {"type": type(exc).__name__, "message": str(exc)[:1500]}
    status = "completed_development_control" if failure is None and post_error is None else "failed_partial"
    report = {
        "schema": NAME + ".report_v1", "status": status, "profile": profile,
        "plan_sha256": plan_sha, "journal_sha256": file_sha(output / "journal.jsonl"),
        "wall_seconds": time.monotonic() - started, "failure": failure,
        "postcheck_failure": post_error, "attempted_counts": counts,
        "ledger_audit": audit, "winners_sha256": winners_sha, "result": result,
        "known_root": True, "new_physical_execution": False,
        "equal_total_compute": False, "composition_mechanism_authenticated": False,
        "independent_confirmation": False, "P1_admission": False, "P2_admission": False,
        "limits": plan["limits"],
    }
    report_sha = write_new(output / "report.json", report)
    return report, report_sha


def run(plan, plan_sha, output, authorization):
    if authorization != AUTH:
        raise ValueError("explicit known-root development authorization required")
    return _execute(plan, plan_sha, output, native.native_session, "native_hooks")


def consume(output, expected, profile="native_hooks"):
    output = Path(output)
    if (not output.is_absolute() or output.parent != RESULTS or output.is_symlink() or
            not output.name.startswith(NAME + "_RUN_")):
        raise ValueError("invalid run output")
    report_path = output / "report.json"
    raw = report_path.read_bytes()
    if report_path.is_symlink() or sha(raw) != expected:
        raise ValueError("report SHA")
    report = json.loads(raw)
    if report["status"] != "completed_development_control" or report["profile"] != profile:
        raise ValueError("incomplete or wrong-profile run")
    binding = json.loads((output / "binding.json").read_bytes())
    if binding["profile"] != profile or binding["plan_sha256"] != report["plan_sha256"]:
        raise ValueError("plan binding")
    verify(binding["plan"], binding["plan_sha256"])
    audit = consumer.audit(output / "journal.jsonl", profile)
    if audit != report["ledger_audit"] or audit["ledger_sha256"] != report["journal_sha256"]:
        raise ValueError("saved ledger mismatch")
    winners = winner_handoff(report["result"], report["plan_sha256"], audit)
    winner_bytes = (output / "winners.json").read_bytes()
    if (sha(winner_bytes) != report["winners_sha256"] or winner_bytes != raw_json(winners) or
            sha(report_path.read_bytes()) != expected):
        raise ValueError("saved winner or report mismatch")
    return {"status": "saved_development_ledger_verified_not_admitted",
            "report_sha256": expected, "profile": profile, "winners": winners["winners"],
            "native_execution_independently_authenticated": False,
            "equal_total_compute": False, "physical_execution_authorized": False,
            "P1_admission": False, "P2_admission": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", required=True)
    for name in ("verify", "run"):
        p = sub.add_parser(name)
        p.add_argument("--plan", required=True)
        p.add_argument("--plan-sha256", required=True)
        if name == "run":
            p.add_argument("--output", required=True)
            p.add_argument("--authorization", required=True)
    c = sub.add_parser("consume")
    c.add_argument("--output", required=True)
    c.add_argument("--report-sha256", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps({"plan": args.output, "sha256": prepare(args.output)}))
    elif args.command == "verify":
        verify(args.plan, args.plan_sha256)
        print(json.dumps({"status": "verified_no_cuda"}))
    elif args.command == "run":
        report, report_sha = run(args.plan, args.plan_sha256, args.output, args.authorization)
        print(json.dumps({"status": report["status"], "report_sha256": report_sha}))
        if report["status"] != "completed_development_control":
            raise SystemExit(1)
    else:
        print(json.dumps(consume(args.output, args.report_sha256), sort_keys=True))


if __name__ == "__main__":
    main()
