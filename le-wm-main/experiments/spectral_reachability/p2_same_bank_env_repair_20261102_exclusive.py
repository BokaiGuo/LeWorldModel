"""Separately authorized environment-repair development attempt; never auto-retry."""
import os

# Set before importing the numerical stack, while retaining a directly approved
# interpreter command. Record the inherited value in the environment witness.
INHERITED_BLAS = os.environ.get("OPENBLAS_NUM_THREADS")
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

from . import p2_same_bank_host_development_20261102_exclusive as core

ROOT = core.ROOT
RESULTS = core.RESULTS
NAME = "p2_same_bank_env_repair_20261102_exclusive"
PLAN = RESULTS / (NAME + "_PLAN_01.json")
OUTPUT = RESULTS / (core.NAME + "_RUN_ENV_REPAIR_01")
OLD_PLAN = RESULTS / (core.NAME + "_PLAN_01.json")
OLD_RUN = RESULTS / (core.NAME + "_RUN_01")
OLD_PLAN_SHA = "2cf0b49d0769d735a8ac82b8c6e19b6f942a56f7f893a3c87f356a91a5df645f"
OLD_FILES = {
    "report.json": "69454fe8fb16c9a210f131ac00f6ff041a1aa0be1db94ae781747ed8652d058a",
    "binding.json": "e664443b39c66dbe81b59ef7d40b1e24fdc2aa4f7de9f50ddd909f18309ef53e",
    "journal.jsonl": "4c299c40b1a4b5e79b5545f2ca9ed2f4ed8cb98dd4c55685c4dadabd9fc0e3d0",
}
AUTH = "P2_ENV_REPAIR_NEW_KNOWN_ROOT_DEVELOPMENT_ONLY"


def build_plan():
    technical = core.verify(OLD_PLAN, OLD_PLAN_SHA)
    for name, expected in OLD_FILES.items():
        if core.file_sha(OLD_RUN / name) != expected:
            raise ValueError("old failure artifact drift: " + name)
    old = json.loads((OLD_RUN / "report.json").read_bytes())
    if old["status"] != "failed_partial" or old["failure"] != {
        "type": "RuntimeError", "message": "No CUDA GPUs are available"
    }:
        raise ValueError("old failure mismatch")
    return {
        "schema": NAME + ".plan_v1",
        "new_attempt_not_old_retry": True,
        "scope": "environment_repair_known_root_development_control",
        "technical_plan_path": str(OLD_PLAN), "technical_plan_sha256": OLD_PLAN_SHA,
        "old_failure_directory": str(OLD_RUN), "old_failure_files_sha256": OLD_FILES,
        "repair_source_sha256": core.file_sha(Path(__file__)),
        "output": str(OUTPUT), "authorization_required": AUTH,
        "environment": {
            "entrypoint": "direct .venv/bin/python -B; host execution required",
            "openblas_threads": "1 set before numerical imports",
            "probe": "torch cuda availability and device count under cooperative lock",
            "lock": technical["lock_path"],
            "lock_gap": "probe releases lock; native session reacquires nonblocking; abort if busy",
            "gpu_probe_and_model_load_counted_separately": False,
            "host_cuda_visibility_user_reported_not_verified_by_prepare": True,
        },
        "failure_policy": "no automatic retry; retain new partial and old failure; no overwrite",
        "constraints": technical["limits"],
        "P1_admission": False, "P2_admission": False,
        "independent_confirmation": False, "equal_total_compute": False,
        "composition_mechanism_authenticated": False,
    }


def prepare():
    if os.path.lexists(PLAN) or os.path.lexists(OUTPUT):
        raise ValueError("repair plan or output already exists")
    return core.write_new(PLAN, build_plan())


def verify(expected):
    if PLAN.is_symlink() or core.file_sha(PLAN) != expected:
        raise ValueError("repair plan SHA")
    plan = json.loads(PLAN.read_bytes())
    if plan != build_plan() or core.file_sha(PLAN) != expected:
        raise ValueError("repair plan or source drift")
    return plan


@contextmanager
def repaired_session(fixed, journal, output, repair_sha, probe=None):
    """Preflight and native loading are both locked, with a non-atomic gap."""
    witness = {"repair_plan_sha256": repair_sha, "python": sys.executable,
               "inherited_openblas_threads": INHERITED_BLAS,
               "effective_openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        with core.native.locking.gpu_lock(core.native.LOCK):
            if probe is None:
                import torch
                witness["torch_version"] = str(torch.__version__)
                witness["torch_cuda_build"] = torch.version.cuda
                available, count = bool(torch.cuda.is_available()), int(torch.cuda.device_count())
            else:
                available, count = probe()
            witness.update(cuda_available=available, cuda_device_count=count)
            if available is not True or type(count) is not int or count < 1:
                raise RuntimeError("host CUDA preflight failed")
        witness["status"] = "probe_passed_lock_released_before_native_reacquisition"
    except Exception as exc:
        witness.update(status="preflight_failed", error_type=type(exc).__name__, error=str(exc)[:1000])
        core.write_new(output / "environment.json", witness)
        raise
    core.write_new(output / "environment.json", witness)
    with core.native.native_session(fixed, journal) as session:
        yield session


def run(expected, authorization):
    if authorization != AUTH:
        raise ValueError("new explicit environment repair authorization required")
    verify(expected)
    if os.path.lexists(OUTPUT):
        raise ValueError("new output already exists")
    def session(fixed, journal):
        return repaired_session(fixed, journal, OUTPUT, expected)
    report, report_sha = core._execute(OLD_PLAN, OLD_PLAN_SHA, OUTPUT, session, "native_hooks")
    verify(expected)
    return report, report_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    check = sub.add_parser("verify")
    check.add_argument("--plan-sha256", required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--plan-sha256", required=True)
    execute.add_argument("--authorization", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps({"plan": str(PLAN), "sha256": prepare()}))
    elif args.command == "verify":
        verify(args.plan_sha256)
        print(json.dumps({"status": "verified_no_cuda"}))
    else:
        report, digest = run(args.plan_sha256, args.authorization)
        print(json.dumps({"status": report["status"], "report_sha256": digest}))
        if report["status"] != "completed_development_control":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
