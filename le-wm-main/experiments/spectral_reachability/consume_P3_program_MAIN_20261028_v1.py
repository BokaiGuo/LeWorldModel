"""Persist strict native program receipt; saved arithmetic, no neural calls."""
import json
import os
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import agent_p3_native_program_20261028_v1 as producer


def main():
    folder = producer.ROOT / 'results/agent_p3_native_program_20261028_v1_cuda_MAIN01'
    expected = '7f5f16ffe9d440a5cd2731e1e87c2325b1b39de9c18508ebdcb28409cbe858d3'
    output = producer.ROOT / 'results/consume_P3_program_MAIN_20261028_v1_MAIN01.json'
    if output.exists():
        raise FileExistsError(output)
    receipt = producer.consume(folder, expected)
    report = json.loads((folder / 'report.json').read_bytes())
    prepared = json.loads(Path(report['prepared']).read_bytes())
    numerical = []
    for row in report['rows']:
        task = prepared['tasks'][row['slot']]
        terminal = np.asarray(row['whole'][-1]['latent_history'][-1], dtype=np.float64)
        delta = np.asarray(task['projection'], dtype=np.float64) @ (terminal - np.asarray(task['goal']))
        cost = float(np.dot(delta, delta) / delta.size)
        if not np.isclose(cost, row['cost'], rtol=0, atol=1e-12):
            raise ValueError('independent terminal arithmetic mismatch')
        numerical.append(dict(slot=row['slot'], mode=row['mode'], index=row['index'],
                              horizon=row['horizon'], independent_cost=cost))
    value = dict(schema='MAIN.P3_native_program_receipt.v1', receipt=receipt,
                 independent_terminal_arithmetic=numerical,
                 new_neural_calls=0, new_physics_calls=0,
                 equal_compute=False, equal_horizon=False, P1_admission=False,
                 caveat='Two known roots, H1 versus H3 root extensions; no physical or full C3 evidence.')
    payload = producer.canonical(value)
    with output.open('xb') as stream:
        if stream.write(payload) != len(payload):
            raise OSError('short receipt write')
        stream.flush()
        os.fsync(stream.fileno())
    producer.signal.fsync_dir(output.parent)
    print(producer.canonical(dict(output=str(output), sha256=producer.sha(output),
        status=receipt['status'], denominator=receipt['denominator'],
        per_branch=receipt['per_branch'])).decode())


if __name__ == '__main__':
    main()
