"""Persist full saved P5 native-replay consumption, not a scientific gate."""
import hashlib
import json
import os
from pathlib import Path

from experiments.spectral_reachability import agent_p5_action_words_physical_20261028_v1 as physical

ROOT = Path(__file__).resolve().parents[2]


def main():
    result = physical.consume(
        ROOT/'results/agent_p5_action_words_physical_20261028_v1_prepare_MAIN01/manifest.json',
        'd33b539c77b35b4256a5e7099e39cdd1da619d8e8c3466d6f753ca3fded351d9',
        ROOT/'results/agent_p5_action_words_physical_20261028_v1_physics_MAIN01/report.json',
        '796d757c6ffcf623f4e307f094dba9e39eda2a4810a140e5ab3cf0897b3c73a5')
    path = ROOT/'results/consume_p5_physical_MAIN_20261028_v1_MAIN01.json'
    data = json.dumps(result, sort_keys=True, allow_nan=False).encode()
    with path.open('xb') as stream:
        if stream.write(data) != len(data):
            raise OSError('short saved receipt')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(path), sha256=hashlib.sha256(data).hexdigest(), status=result['status'])))


if __name__ == '__main__':
    main()
