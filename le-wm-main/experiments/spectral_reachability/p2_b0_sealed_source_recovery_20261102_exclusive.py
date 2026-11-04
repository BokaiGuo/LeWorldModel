"""Read-only B0 binding census with one SHA-verified Git historical source.

This does not execute historical code, patch the worktree, or run the original
consumer; a matching binding census is not full B0 provenance verification.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent
RECEIPT = ROOT / "results/codex_p2_multistart_joint_physical_20261030_v1_B0_RECEIPT_01.json"
RECEIPT_SHA = "88a8d7bb3f04479dac0150a7c917b9b1840bfc67d962437c0cfaad7fcb98cddf"
TARGET = ROOT / "experiments/spectral_reachability/run_reacher_d2p_p0_goal_progress.py"
HISTORICAL_SHA = "5d3463aa69038d3dbe5e6cf6c6f5ad31ceba56211e872e72a2573de9a9f18dc0"
COMMIT = "3a87dbc"
GIT_PATH = "le-wm-main/experiments/spectral_reachability/run_reacher_d2p_p0_goal_progress.py"
OUTPUT = ROOT / "results/p2_b0_sealed_source_recovery_20261102_exclusive_run01.json"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def checked_file(path):
    """Hash one regular non-symlink file, rejecting a concurrent modification."""
    if path.is_symlink() or not path.is_file():
        return None, "missing_or_symlink"
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            h = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
            after = os.fstat(stream.fileno())
        now = path.stat()
    except OSError:
        return None, "read_error"
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if identity(before) != identity(after) or identity(after) != identity(now) or path.is_symlink():
        return None, "changed_during_read"
    return h.hexdigest(), None


def census(bindings, target, historical):
    if not isinstance(bindings, dict) or str(target) not in bindings:
        raise ValueError("target binding missing")
    if digest(historical) != bindings[str(target)]:
        raise ValueError("historical bytes do not match sealed target")
    mismatches = []
    target_current = None
    for name, expected in sorted(bindings.items()):
        path = Path(name)
        if not path.is_absolute() or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("invalid binding")
        actual, error = checked_file(path)
        if path == target:
            target_current = actual
        if error or actual != expected:
            mismatches.append({"path": name, "expected_sha256": expected,
                               "current_sha256": actual, "error": error,
                               "historical_target_available": path == target})
    remaining = [row for row in mismatches if not row["historical_target_available"]]
    return {"binding_count": len(bindings), "current_mismatches": mismatches,
            "remaining_mismatches_if_target_source_used": remaining,
            "historical_target_sha256": digest(historical),
            "current_target_sha256": target_current,
            "all_binding_bytes_match_with_target_substitution": not remaining}


def audit():
    raw = RECEIPT.read_bytes()
    if digest(raw) != RECEIPT_SHA:
        raise ValueError("pinned B0 receipt SHA mismatch")
    data = json.loads(raw)
    historical = subprocess.run(
        ["git", "--no-optional-locks", "show", COMMIT + ":" + GIT_PATH],
        cwd=REPO, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    if digest(historical) != HISTORICAL_SHA:
        raise ValueError("historical Git source SHA mismatch")
    result = census(data["bindings"], TARGET, historical)
    return {
        "schema": "p2_b0_sealed_source_recovery_20261102_exclusive_v1",
        "receipt_path": str(RECEIPT), "receipt_sha256": RECEIPT_SHA,
        "git_commit": COMMIT, "git_path": GIT_PATH, "census": result,
        "original_consumer_run": False, "historical_code_executed": False,
        "P1_admission": False, "P2_admission": False,
        "limitations": [
            "B0 receipt binding census only; not complete source closure, artifact, journal or runtime verification.",
            "Git bytes were checked in memory; consumer still reads original absolute paths and was not modified.",
            "No model inference, GPU use, physical execution or independent confirmation.",
        ],
    }


def main():
    value = audit()
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(f"created {OUTPUT} sha256={digest(raw)}")


if __name__ == "__main__":
    main()
