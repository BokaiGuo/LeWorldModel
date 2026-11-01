"""Persist the full saved teacher-library consumer without dumping dense arrays."""
import hashlib
import json
import os
from pathlib import Path

from experiments.spectral_reachability import agent_p1_teacher_cross_library_20261028_v1 as teacher


def main():
    root = Path(__file__).resolve().parents[2]
    folder = root / 'results/agent_p1_teacher_cross_library_20261028_v1_cuda_MAIN01'
    report_sha = hashlib.sha256((folder / 'report.json').read_bytes()).hexdigest()
    value = teacher.consume(folder, report_sha)
    data = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    output = root / 'results/consume_teacher_cross_MAIN_20261028_v1_MAIN01.json'
    with output.open('xb') as stream:
        if stream.write(data) != len(data):
            raise OSError('short strict receipt write')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(output), sha256=hashlib.sha256(data).hexdigest(),
                          report_sha256=report_sha, status=value['status'], plans=len(value['plans']),
                          resources=value['resources'])))


if __name__ == '__main__':
    main()
