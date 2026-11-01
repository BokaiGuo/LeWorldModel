"""Run frozen test modules in independent processes; never edit sealed tests."""
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
MODULES = (
    'main_raw_hscale_20261028_v1',
    'agent_raw_raster_factor_20261028_v1',
    'agent_archive_action_discrimination_20261028_v1',
    'agent_hilewm_rgb_high_low_20261028_v1',
    'agent_p2_three_program_20261028_v1',
    'agent_p2_three_program_physical_20261029_v1',
    'agent_p2_three_program_model_mechanism_20261029_v1',
    'agent_p3_fullcommit_live_20261028_v1',
)


def run(output):
    output = Path(output).resolve()
    if output.parent != ROOT / 'results' or not output.name.startswith('main_isolated_regression_20261029_v1_'):
        raise ValueError('exclusive result directory required')
    tests = {name: ROOT / 'tests' / (f'{name}_tests.py' if name == 'agent_p3_fullcommit_live_20261028_v1' else f'test_{name}.py') for name in MODULES}
    if not all(path.is_file() for path in tests.values()):
        raise ValueError('all eight explicit test paths required before launch')
    output.mkdir(exist_ok=False)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')

    def one(name):
        test = tests[name]
        before = hashlib.sha256(test.read_bytes()).hexdigest()
        command = [sys.executable, '-B', '-m', 'pytest', '-q', str(test)]
        start = time.monotonic()
        with (output / f'{name}.log').open('xb') as stream:
            try:
                code = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                      stderr=subprocess.STDOUT, timeout=900, check=False).returncode
            except subprocess.TimeoutExpired:
                code = 'timeout'
        after = hashlib.sha256(test.read_bytes()).hexdigest()
        return dict(module=name, command=command, returncode=code,
                    seconds=time.monotonic()-start, test_sha256=before,
                    test_unchanged=before == after,
                    log_sha256=hashlib.sha256((output / f'{name}.log').read_bytes()).hexdigest())

    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(one, MODULES))
    report = dict(schema='main_isolated_regression_20261029_v1', rows=rows,
                  passed=all(r['returncode'] == 0 and r['test_unchanged'] for r in rows),
                  scope='unit tests in fresh processes; no production neural or physics experiment',
                  scientific_admission=False)
    with (output / 'report.json').open('x') as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write('\n')
    print(json.dumps(report, sort_keys=True))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    raise SystemExit(run(parser.parse_args().output))
