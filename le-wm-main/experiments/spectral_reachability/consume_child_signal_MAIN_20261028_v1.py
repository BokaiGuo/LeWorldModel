"""Persist the full verified P3 result without the prepare-only 64MiB CLI writer.

The frozen validator and every row are unchanged. This saved-only receipt has
its own 256MiB engineering cap; no CUDA run budget or scientific gate changes.
"""
import argparse
import os
from pathlib import Path
import shutil

from experiments.spectral_reachability import agent_p3_child_signal_cuda_20261028_v2 as producer

RECEIPT_CAP = 256*1024**2
FLOOR = 200*1024**2


def persist(path, result):
    path = Path(path).absolute()
    producer.require(path.parent == producer.ROOT/'results' and path.name.startswith('consume_child_signal_MAIN_20261028_v1_')
                     and not path.exists() and not any(p.is_symlink() for p in (path, *path.parents)),
                     'new exclusive saved receipt')
    payload = producer.canonical(result)
    producer.require(len(payload) <= RECEIPT_CAP, 'full saved receipt cap')
    producer.require(shutil.disk_usage(path.parent).free >= len(payload)+FLOOR, 'receipt disk floor')
    with path.open('xb') as stream:
        producer.require(stream.write(payload) == len(payload), 'full strict receipt write')
        stream.flush()
        os.fsync(stream.fileno())
    producer.base.fsync_dir(path.parent)
    return dict(path=str(path), sha256=producer.sha(path), bytes=len(payload),
                status=result['status'], saved_receipt_cap_bytes=RECEIPT_CAP,
                fixed_denominator=result['fixed_denominator'],
                budget_refusals_by_mode=result['budget_refusals_by_mode'],
                original_validator_unchanged=True, scientific_admission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report-sha', required=True)
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    # No fallback, filtering, or alternative success path: full original consume.
    result = producer.consume(args.output, args.report_sha)
    print(producer.canonical(persist(args.receipt, result)).decode())


if __name__ == '__main__':
    main()
