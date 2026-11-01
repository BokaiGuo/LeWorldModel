"""Exclusive bounded writer around the original full P3 saved consumer."""
import argparse
import os
from pathlib import Path

from experiments.spectral_reachability import agent_p3_same_feedback_physical20_20261028_v1 as p3

NAME = 'main_p3_same_feedback_receipt_20261028_v1'


def main():
    q = argparse.ArgumentParser(description=__doc__)
    for k in ('output', 'report-sha', 'prepared', 'prepare-sha', 'receipt'):
        q.add_argument('--'+k, required=True)
    a = q.parse_args()
    path = Path(a.receipt).absolute()
    p3.require(path.resolve() == path and path.parent == p3.ROOT/'results' and
        path.name.startswith(NAME+'_') and path.suffix == '.json' and not path.exists(), 'new own receipt only')
    result = p3.consume(a.output, a.report_sha, a.prepared, a.prepare_sha)
    payload = p3.raw(dict(schema=NAME, original_consume=result,
        report_sha256=a.report_sha, prepare_sha256=a.prepare_sha,
        new_neural_calls=0, new_physics_calls=0, P1_admission=False, P3_admission=False))
    p3.require(len(payload) <= p3.contract()['final_reserve_bytes'], 'bounded full receipt; never truncate')
    with path.open('xb') as f:
        p3.require(f.write(payload) == len(payload), 'complete saved receipt')
        f.flush(); os.fsync(f.fileno())
    p3.signal.fsync_dir(path.parent)
    print(p3.raw(dict(path=str(path), sha256=p3.sha(path), status=result['status'],
        completed20=result.get('completed20'), fixed_denominator=result.get('fixed_denominator'))).decode())


if __name__ == '__main__':
    main()
