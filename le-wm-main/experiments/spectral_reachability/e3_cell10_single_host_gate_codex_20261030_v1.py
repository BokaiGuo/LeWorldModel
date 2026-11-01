"""Cell10 single-host cooperative gate prototype; production dispatch is blocked.

The test path launches only an inert fixed child in a private /tmp directory.
Neither legacy entrypoints nor other hosts are fenced by this module.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

from experiments.spectral_reachability import e3_cell10_durable_claim_codex_20261030_v1 as marker

ROUTES = ("chunked", "identity", "continuous", "integrated", "startup")
CHILD = "print('inert-cell10-fixture-complete')"
_FIXTURE_LOCK = threading.Lock()


def production_dispatch():
    raise RuntimeError("production blocked: legacy entrypoints are not fenced")


def fixture_dispatch(directory, route, closure_sha, *, child=subprocess.run):
    """Prove cooperative exclusion across route labels, not production fencing."""
    path = Path(directory)
    tmp = Path("/tmp").resolve()
    if (not path.is_absolute() or path != path.resolve() or path == tmp
            or tmp not in path.parents or not path.is_dir()):
        raise ValueError("canonical private /tmp directory required")
    info = path.stat()
    if info.st_uid != os.geteuid() or info.st_mode & 0o077:
        raise ValueError("fixture directory must be owner-only")
    if route not in ROUTES:
        raise ValueError("unknown fixture route")
    # Each fixture invocation is a separate process. The production marker
    # location is never used; do not accept an arbitrary child command.
    with _FIXTURE_LOCK:
        previous = marker.DIRECTORY
        try:
            marker.DIRECTORY = path
            marker.claim(closure_sha)
        finally:
            marker.DIRECTORY = previous
    result = child([sys.executable, "-B", "-I", "-c", CHILD],
                   cwd=path, capture_output=True, text=True, timeout=10, check=False)
    if result.returncode != 0 or result.stdout.strip() != "inert-cell10-fixture-complete":
        raise RuntimeError("inert child failed; marker retained")
    return {"fixture_complete": True, "route": route, "production_authorized": False,
            "legacy_fenced": False, "scientific_admission": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run")
    test = sub.add_parser("fixture")
    test.add_argument("--directory", required=True)
    test.add_argument("--route", required=True, choices=ROUTES)
    test.add_argument("--closure-sha256", required=True)
    args = parser.parse_args()
    try:
        result = production_dispatch() if args.command == "run" else fixture_dispatch(
            args.directory, args.route, args.closure_sha256)
        print(json.dumps(result, sort_keys=True))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"error": str(exc), "production_authorized": False}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
