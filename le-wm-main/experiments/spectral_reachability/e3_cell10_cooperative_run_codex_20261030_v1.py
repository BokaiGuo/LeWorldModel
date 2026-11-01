"""One known-root cell10 feedback attempt; cooperative single-host use only.

No execution on import. A fresh child runs the unchanged native host. Old
entrypoints can bypass this marker; this entry is not a scientific admission.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.spectral_reachability import e3_cell10_durable_claim_codex_20261030_v1 as claim

ROOT = Path("/home/gbk/code/LeWorldModel/le-wm-main")
RESULTS = ROOT / "results"
SOURCE = RESULTS / "e3_cell10_feedback_prepare_codex_20261030_v1_LOCAL01.json"
AUTHORITY = RESULTS / "e3_cell10_startup_authority_codex_20261030_v1_CPU01.json"
CLOSURE = RESULTS / "e3_cell10_startup_closure_codex_20261030_v1_CPU01.json"
AUTH_SHA = "33064f0c5bae40aa06d662acd0673bf0127eb7faad1ff98e4247977ef84cf484"
CLOSURE_SHA = "3e50d68ca985cf6eeb63bad41f1df6319e7dea15d2eb1f04de888724086c2ec1"
OUTPUT = RESULTS / "e3_cell10_cooperative_native_codex_20261030_v1_MAIN01"
JOURNAL = RESULTS / "e3_cell10_cooperative_parent_codex_20261030_v1_MAIN01.jsonl"
CONFIRM = "CELL10_SINGLE_HOST_EXPLORATORY_ONCE"
WALL_SECONDS = 1800
CHILD = """import sys
from experiments.spectral_reachability import agent_e3_chunked_identity_host_20261029_v1 as r
x = r.run(sys.argv[1], sys.argv[2], sys.argv[3], r.launch.AUTH)
sys.exit(0 if x.get('complete') is True else 1)
"""


def _precheck_command():
    return [sys.executable, "-B", "-m",
            "experiments.spectral_reachability.agent_e3_startup_closure_20261029_v1",
            "precheck", "--source", str(SOURCE), "--source-sha256", AUTH_SHA,
            "--closure", str(CLOSURE), "--closure-sha256", CLOSURE_SHA,
            "--authority", str(AUTHORITY), "--authority-sha256", AUTH_SHA]


def _child_command():
    return [sys.executable, "-B", "-c", CHILD, str(AUTHORITY), AUTH_SHA, str(OUTPUT)]


def run(confirmation):
    if confirmation != CONFIRM:
        raise ValueError("explicit single-host coordination required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1"):
        raise ValueError("CUDA explicitly disabled")
    if Path.cwd().resolve() != ROOT:
        raise ValueError("fixed project working directory required")
    if os.path.lexists(OUTPUT) or os.path.lexists(JOURNAL):
        raise FileExistsError("fixed output or parent journal already exists")
    started = time.monotonic()
    claim.claim(CLOSURE_SHA)
    # Marker is intentionally never removed, including failures opening journal.
    fd = os.open(JOURNAL, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb", buffering=0) as journal:
        dfd = os.open(RESULTS, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
        def record(stage, **extra):
            data = json.dumps(dict(stage=stage, elapsed_seconds=time.monotonic()-started,
                                   **extra), sort_keys=True, allow_nan=False).encode() + b"\n"
            if journal.write(data) != len(data):
                raise OSError("short parent journal write")
            os.fsync(journal.fileno())
        def invoke(stage, command):
            record(stage + "_intent")
            remaining = WALL_SECONDS - (time.monotonic()-started)
            if remaining <= 0:
                raise TimeoutError("whole parent wall exhausted before " + stage)
            proc = subprocess.run(command, cwd=ROOT, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL, timeout=remaining, check=False)
            record(stage + "_return", returncode=proc.returncode)
            if proc.returncode != 0:
                raise RuntimeError(stage + " child failed")
            if time.monotonic() - started > WALL_SECONDS:
                raise TimeoutError("whole parent wall exceeded at " + stage)
        try:
            record("started", cell=10, authority_sha256=AUTH_SHA, closure_sha256=CLOSURE_SHA,
                   source="known_root_saved_control5", exploratory=True,
                   gpu_time="unknown", flops="unknown", loading_full_cost="unknown",
                   prior_control5_cost="outside_parent_scope", global_fence=False)
            invoke("cpu_precheck", _precheck_command())
            if os.path.lexists(OUTPUT):
                raise FileExistsError("fixed output appeared before dispatch")
            invoke("native_host", _child_command())
            record("finished", transport_complete=True, closed_loop_complete="unknown",
                   full_saved_strict=False, scientific_admission=False)
        except BaseException as exc:
            try:
                record("failure", error_type=type(exc).__name__, message=str(exc)[:1024])
            except BaseException as secondary:
                exc.add_note("parent journal failure: " + repr(secondary))
            raise
    return {"transport_complete": True, "closed_loop_complete": "unknown",
            "scientific_admission": False, "global_fence": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.confirm), sort_keys=True))


if __name__ == "__main__":
    main()
