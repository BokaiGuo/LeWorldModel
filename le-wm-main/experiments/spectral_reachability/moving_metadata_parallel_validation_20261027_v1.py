"""Run fixed mock/saved-metadata tests in isolated processes, never experiments."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
STEM = 'moving_metadata_parallel_validation_20261027_v1'
SUITES = (
    'test_p1_moving_live_feedback_teacher_fidelity_cuda_20261027_v2.py',
    'test_p1_moving_baseline_budget_matrix_20261027_v1.py',
    'test_root_exclusion_authority_resolution_20261027_v1.py',
    'test_moving_budget_scheduler_comparison_rgb_20261027_v1.py',
    'test_moving_budget_scheduler_prospective_queue_20261027_v1.py',
    'test_moving_authority_inventory_bridge_20261027_v1.py',
    'test_p1_teacher_fidelity_error_decomposition_20261027_v1.py',
    'test_moving_equal_budget_prospective_queue_20261027_v1.py',
    'test_moving_p2_composition_prospective_queue_20261027_v1.py',
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def run_suite(name):
    if name not in SUITES:
        raise ValueError('only the fixed mock/metadata test list is allowed')
    relative = 'tests/' + name
    command = [sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', relative]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTEST_ADDOPTS='')
    test_sha = digest((ROOT / relative).read_bytes())
    start = time.monotonic()
    try:
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                timeout=180, check=False)
        stdout, stderr = result.stdout, result.stderr
        code, status = result.returncode, 'passed' if result.returncode == 0 else 'failed'
    except subprocess.TimeoutExpired as exc:
        stdout, stderr = exc.stdout or b'', exc.stderr or b''
        code, status = None, 'timeout'
    unchanged = digest((ROOT / relative).read_bytes()) == test_sha
    if not unchanged:
        status = 'test_source_changed'
    return dict(test=relative, test_sha256=test_sha, test_source_unchanged=unchanged,
                command=command, status=status, returncode=code,
                seconds=time.monotonic() - start,
                stdout_sha256=digest(stdout), stderr_sha256=digest(stderr),
                stdout_tail=stdout[-8192:].decode(errors='replace'),
                stderr_tail=stderr[-8192:].decode(errors='replace'),
                output_truncated=len(stdout) > 8192 or len(stderr) > 8192)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError('preserve previous validation output')
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_suite, SUITES))
    passed = all(row['status'] == 'passed' for row in rows)
    report = dict(schema=STEM + '.report', status='passed' if passed else 'failed',
                  workers=args.workers, elapsed_seconds=time.monotonic() - start,
                  source_sha256=digest(Path(__file__).read_bytes()), rows=rows,
                  scope='fixed mock and saved-metadata tests, isolated interpreter per suite',
                  scientific_admission=False, CUDA_experiments_launched=False)
    raw = (json.dumps(report, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()
    with args.output.open('xb') as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(dict(output=str(args.output.absolute()), sha256=digest(raw),
                          status=report['status'], suites=len(rows),
                          workers=args.workers, seconds=report['elapsed_seconds'])))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
