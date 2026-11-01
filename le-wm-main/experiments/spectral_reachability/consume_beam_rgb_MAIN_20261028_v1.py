"""Persist full saved beam RGB consumption; no inference or physics."""
import hashlib
import json
import os
from pathlib import Path

from experiments.spectral_reachability import agent_kinodynamic_matched_rgb_20261028_v1 as rgb


def main():
    root = Path(__file__).resolve().parents[2]
    result = rgb.consume(root/'results/agent_kinodynamic_matched_rgb_20261028_v1_cuda_MAIN01',
                         '52e6d7c025dcb836aed3ec28aefc9b249d449be2eb19618c8e6ebdae7ea4db49')
    result['report_sha256'] = '52e6d7c025dcb836aed3ec28aefc9b249d449be2eb19618c8e6ebdae7ea4db49'
    data = json.dumps(result, sort_keys=True, allow_nan=False).encode()
    path = root/'results/consume_beam_rgb_MAIN_20261028_v1_MAIN01.json'
    with path.open('xb') as stream:
        if stream.write(data) != len(data):
            raise OSError('short full saved receipt')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(path), sha256=hashlib.sha256(data).hexdigest(),
                          branches=len(result['rows']), counts=result['counts'])))


if __name__ == '__main__':
    main()
