"""Execute the unmodified B0 saved consumer in a read-only bubblewrap view."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from . import p2_b0_sealed_source_recovery_20261102_exclusive as recovery

ROOT = recovery.ROOT
OUTPUT = ROOT / "results/p2_b0_isolated_consume_20261102_exclusive_run01.json"
REPORT_SHA = "d39f8e4f5f292034331b8a60b3fe0682b04d81206fd904200db2295a38e3ef2d"
RUN = ROOT / "results/p2_shared_rgb_pair_wiring_20261102_codexexclusive_RUN_20261102_p2continuation_host01"
CHILD = (
    "import json; from pathlib import Path; "
    "from experiments.spectral_reachability import p2_shared_capture_launch_20261102_exclusive as s; "
    f"print(json.dumps(s.audit(Path({str(RUN)!r}), {REPORT_SHA!r}), sort_keys=True))"
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def command(historical):
    """All host paths read-only, with a minimal /dev and no visible CUDA devices."""
    return ["bwrap", "--die-with-parent", "--ro-bind", "/", "/", "--dev", "/dev",
            "--proc", "/proc", "--ro-bind", str(historical), str(recovery.TARGET),
            "--chdir", str(ROOT), "--", "/usr/bin/env", "CUDA_VISIBLE_DEVICES=",
            "PYTHONDONTWRITEBYTECODE=1", "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1",
            str(ROOT / ".venv/bin/python"), "-B", "-c", CHILD]


def historical_bytes():
    raw = subprocess.run(["git", "--no-optional-locks", "show",
                          recovery.COMMIT + ":" + recovery.GIT_PATH],
                         cwd=recovery.REPO, check=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE).stdout
    if digest(raw) != recovery.HISTORICAL_SHA:
        raise ValueError("historical source SHA mismatch")
    return raw


def run():
    source = historical_bytes()
    before, error = recovery.checked_file(recovery.TARGET)
    if error:
        raise ValueError("current source unreadable: " + error)
    with tempfile.TemporaryDirectory(prefix="p2_b0_readonly_") as directory:
        historical = Path(directory) / "historical_p0.py"
        with historical.open("xb") as stream:
            stream.write(source)
            stream.flush()
            os.fsync(stream.fileno())
        historical.chmod(0o444)
        try:
            child = subprocess.run(command(historical), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   timeout=300, check=False)
            code, stdout, stderr, timed_out = (child.returncode, child.stdout.decode("utf-8", "replace"),
                                                child.stderr.decode("utf-8", "replace"), False)
        except subprocess.TimeoutExpired as exc:
            code, stdout, stderr, timed_out = (None, (exc.stdout or b"").decode("utf-8", "replace"),
                                                (exc.stderr or b"").decode("utf-8", "replace"), True)
    after, error = recovery.checked_file(recovery.TARGET)
    if error or before != after:
        raise ValueError("working tree target changed during verification")
    result = None
    if code == 0:
        result = json.loads(stdout)
        if result.get("status") != "verified_saved_pair_known_root_not_admitted":
            raise ValueError("unexpected consumer result")
    return {
        "schema": "p2_b0_isolated_consume_20261102_exclusive_v1",
        "status": "consumer_returned_success" if code == 0 else "consumer_failed_or_timed_out",
        "return_code": code, "timed_out": timed_out, "stdout": stdout, "stderr": stderr,
        "consumer_result": result, "historical_git_commit": recovery.COMMIT,
        "historical_source_sha256": digest(source), "working_tree_sha256_before_after": before,
        "report_sha256": REPORT_SHA, "readonly_root": True, "minimal_dev": True,
        "network_namespace_isolated": False, "cuda_visible_devices": "",
        "new_model_inference": False, "new_physical_execution": False,
        "independent_confirmation": False, "P1_admission": False, "P2_admission": False,
        "limitations": [
            "Saved-data consumer, not an independent rerun of historical model or physics.",
            "Historical source provided through a read-only mount; original shared paths and results unchanged.",
            "No claim of equal total budget, new root, or scientific admission.",
        ],
    }


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError(OUTPUT)
    value = run()
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(f"created {OUTPUT} sha256={digest(raw)} status={value['status']}")


if __name__ == "__main__":
    main()
