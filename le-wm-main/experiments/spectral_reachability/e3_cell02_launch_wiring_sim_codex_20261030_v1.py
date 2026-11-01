"""Fail-closed cell02 launch wiring and isolated synthetic protocol.

The production path is deliberately unavailable. The only child executable here
is a fixed, inert Python snippet; no caller-supplied command can be dispatched.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.spectral_reachability import e3_cell02_unified_gate_codex_20261030_v1 as gate

TEST_WORKER = "import json; print(json.dumps({'fixture_complete': True}))"
FIXTURE_CANDIDATE = gate.ENTRYPOINTS[0]
BLOCKED = 'production unavailable: legacy entrypoints are not fenced'


def production_dispatch(*_args, **_kwargs):
    """No input or assertion can turn this into a production dispatch."""
    raise RuntimeError(BLOCKED)


def simulate(directory, entrypoint, manifest_sha, *, history_clear, fence_live,
             manifest_current, recheck, worker=subprocess.run):
    """Synthetic orchestration only; recheck and worker are test seams.

    Boolean claims and hooks here are fixtures, never a production attestation.
    The permanent test marker is retained on every failure. No production
    entrypoint, user-provided command, physical process or GPU is launched.
    """
    path = Path(directory)
    if entrypoint != FIXTURE_CANDIDATE:
        raise ValueError('only the candidate has a proposed handoff')
    if not (history_clear is True and fence_live is True and manifest_current is True):
        raise ValueError('synthetic prerequisites not established')
    gate.temporary_gate(path, entrypoint, manifest_sha)

    # Re-evaluate after the irreversible claim, before even the inert worker.
    status = recheck()
    if (type(status) is not dict or set(status) != {'history_clear', 'fence_live', 'manifest_current'}
            or any(status[key] is not True for key in status)):
        raise ValueError('synthetic boundary recheck failed')
    if set(os.listdir(path)) != {gate.TEST_MARKER}:
        raise ValueError('synthetic history changed before dispatch')
    proc = worker([sys.executable, '-B', '-s', '-c', TEST_WORKER],
                  cwd=str(path), capture_output=True, text=True, timeout=10, check=False)
    if proc.returncode != 0:
        raise RuntimeError('synthetic child failed')
    try:
        result = json.loads(proc.stdout)
    except (ValueError, TypeError) as exc:
        raise RuntimeError('synthetic child returned invalid report') from exc
    if (type(result) is not dict or set(result) != {'fixture_complete'}
            or result['fixture_complete'] is not True):
        raise RuntimeError('synthetic child incomplete')
    status = recheck()
    if (type(status) is not dict or set(status) != {'history_clear', 'fence_live', 'manifest_current'}
            or any(status[key] is not True for key in status)
            or set(os.listdir(path)) != {gate.TEST_MARKER}):
        raise ValueError('synthetic post-dispatch recheck failed')
    return {'fixture_complete': True, 'production_authorized': False,
            'production_ready': False, 'scientific_admission': False,
            'full_saved_strict': False}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        production_dispatch()
    except RuntimeError as exc:
        print(json.dumps({'error': str(exc), 'production_authorized': False,
                          'production_ready': False, 'scientific_admission': False}),
              file=sys.stderr)
        return 1
    return 1


if __name__ == '__main__':
    sys.exit(main())
