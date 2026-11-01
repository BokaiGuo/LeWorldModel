"""One-attempt external deadline and physical GPU admission for the existing P3 four-slot plan."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from experiments.spectral_reachability import p3_feedback_online_binding_20261030_v1 as binding

NAME = "p3_feedback_supervised_launch_20261030_codex_v1"
PLAN = binding.RESULTS / "p3_feedback_online_binding_20261030_v1_plan_ONLINE_AUDIT02.json"
PLAN_SHA = "d43c5f20290114bb14c37e3c42a16f2b36e69eb5aceea764ded5750075d3b3d1"
BINDING_SHA = "7bfe19d3872a519207cb0f9639de5d4622d94ac94f07b16f8ef7fc47be65e5c6"
AUTH = "MAIN_THREAD_APPROVED_P3_MATCHED4_SUPERVISED"
UUID = re.compile(r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z")


def require(value, message):
    if not value:
        raise ValueError(message)


def preflight():
    require(binding.budget.sha(binding.__file__) == BINDING_SHA, "binding source drift")
    plan = binding.verify_plan(PLAN, PLAN_SHA)
    for key in ("run_path", "receipt_path", "analysis_path"):
        binding.absent(Path(plan[key]))
    return plan


def gpu_snapshot(uuid, run=subprocess.run):
    require(UUID.fullmatch(uuid) is not None, "physical GPU UUID")
    def query(fields):
        result = run(["nvidia-smi", "-i", uuid, "--query-gpu=" + fields,
                      "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=10)
        lines = result.stdout.strip().splitlines()
        require(len(lines) == 1, "single GPU required")
        return [x.strip() for x in lines[0].split(",")]
    info = query("uuid,memory.total,memory.free,compute_mode")
    mig = query("mig.mode.current")
    require(len(info) == 4 and info[0] == uuid and info[1].isdigit() and info[2].isdigit()
            and info[3] in ("Default", "Exclusive_Process") and mig in (["Disabled"], ["N/A"], ["[N/A]"]),
            "GPU identity or mode")
    return dict(uuid=uuid, total_mib=int(info[1]), free_mib=int(info[2]), compute_mode=info[3], mig_mode=mig[0])


def admit(gpu, minimum_free, minimum_total):
    require(type(minimum_free) is int and type(minimum_total) is int and 1 <= minimum_free <= minimum_total
            and 0 <= gpu["free_mib"] <= gpu["total_mib"] and gpu["free_mib"] >= minimum_free
            and gpu["total_mib"] >= minimum_total, "GPU memory admission")


def command(reference):
    binding.coordination(PLAN_SHA, reference)
    return [str(binding.host.ROOT / ".venv/bin/python"), "-m",
            "experiments.spectral_reachability." + binding.NAME, "run",
            "--plan", str(PLAN), "--plan-sha", PLAN_SHA, "--reference", reference,
            "--authorization", binding.host.AUTH, "--acknowledgment", binding.ACK]


def supervise(cmd, env, out, err, seconds):
    started = time.monotonic()
    child = subprocess.Popen(cmd, cwd=binding.host.ROOT, env=env, stdin=subprocess.DEVNULL,
                             stdout=out, stderr=err, start_new_session=True, close_fds=True)
    try:
        try:
            rc = child.wait(timeout=max(0, seconds - (time.monotonic() - started)))
            status = "exited"
        except subprocess.TimeoutExpired:
            status = "timeout"
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                rc = child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                rc = None
    except BaseException:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        raise
    return dict(status=status, returncode=rc, pid=child.pid, elapsed_seconds=time.monotonic() - started)


def emit(path, value):
    with path.open("ab") as stream:
        stream.write((json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def launch(args):
    require(args.authorization == AUTH and UUID.fullmatch(args.gpu_uuid) is not None
            and type(args.hard_seconds) is int and 1 <= args.hard_seconds <= 7200
            and 1 <= args.min_free_mib <= args.min_total_mib, "launch policy")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,55}", args.suffix) is not None, "suffix")
    cmd = command(args.reference)
    preflight()
    code_sha = binding.budget.sha(__file__)
    require(re.fullmatch(r"[0-9a-f]{64}", args.source_sha) is not None and
            code_sha == args.source_sha, "external supervision source SHA")
    outdir = binding.RESULTS / (NAME + "_" + args.suffix)
    require(not outdir.exists() and not outdir.is_symlink(), "supervision output exists")
    outdir.mkdir()
    journal = outdir / "supervision.jsonl"
    emit(journal, dict(event="created", source_sha256=code_sha, plan_sha256=PLAN_SHA,
                       gpu_uuid=args.gpu_uuid, hard_seconds=args.hard_seconds,
                       min_free_mib=args.min_free_mib, min_total_mib=args.min_total_mib,
                       reference=args.reference, retry=False))
    try:
        gpu = gpu_snapshot(args.gpu_uuid)
        admit(gpu, args.min_free_mib, args.min_total_mib)
        preflight()
        require(binding.budget.sha(__file__) == code_sha, "supervision source drift")
        emit(journal, dict(event="admitted", gpu=gpu))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1")
        emit(journal, dict(event="launch_attempt", command=cmd, cuda_visible_devices=args.gpu_uuid))
        with (outdir / "stdout.log").open("xb") as out, (outdir / "stderr.log").open("xb") as err:
            def interrupted(signum, frame):
                raise InterruptedError(f"signal {signum}")
            previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
            try:
                execution = supervise(cmd, env, out, err, args.hard_seconds)
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
            for stream in (out, err):
                stream.flush()
                os.fsync(stream.fileno())
        emit(journal, dict(event="finished", execution=execution, retry=False,
                           result="requires_independent_saved_consumption"))
        return 0 if execution["status"] == "exited" and execution["returncode"] == 0 else 2
    except Exception as exc:
        emit(journal, dict(event="failed", error_type=type(exc).__name__, error=str(exc)[:500], retry=False))
        return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("authorization", "gpu-uuid", "suffix", "reference", "source-sha"):
        parser.add_argument("--" + key, required=True)
    for key in ("min-free-mib", "min-total-mib", "hard-seconds"):
        parser.add_argument("--" + key, required=True, type=int)
    raise SystemExit(launch(parser.parse_args()))


if __name__ == "__main__":
    main()
