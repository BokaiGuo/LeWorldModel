"""Strict saved-only receipt and independent terminal-Q arithmetic."""
import json
import os
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import agent_p3_common_budget_20261028_v1 as producer


def independent_cost(task, row):
    terminal = np.asarray(row['whole'][-1]['latent_history'][-1], dtype=np.float64)
    residual = terminal - np.asarray(task['goal'], dtype=np.float64)
    projected = np.asarray(task['projection'], dtype=np.float64) @ residual
    value = float(np.dot(projected, projected) / projected.size)
    if not np.isfinite(value) or not np.isclose(value, row['cost'], rtol=0, atol=1e-12):
        raise ValueError('independent terminal-Q mismatch')
    return value


def main():
    folder = producer.ROOT / 'results/agent_p3_common_budget_20261028_v1_cuda_MAIN01'
    expected = '3f7004aeb92d09a8f6010cd5a5c078a0660b19574203c38d5ae3fbe72dfab67e'
    output = producer.ROOT / 'results/consume_P3_common_MAIN_20261028_v1_MAIN01.json'
    if output.exists():
        raise FileExistsError(output)
    receipt = producer.consume(folder, expected)
    report = json.loads((folder / 'report.json').read_bytes())
    prepared = json.loads(Path(report['prepared']).read_bytes())
    numerical = [dict(slot=r['slot'], mode=r['mode'], index=r['index'],
                      cost=independent_cost(prepared['tasks'][r['slot']], r))
                 for r in report['rows']]
    value = dict(schema='MAIN.P3_common_budget_receipt.v1', receipt=receipt,
                 independent_terminal_arithmetic=numerical, new_neural_calls=0,
                 new_physics_calls=0, equal_predict_API_and_samples=True,
                 equal_total_compute=False, equal_execution_horizon=False,
                 scientific_admission=False,
                 caveat='Two known roots; H1 versus H3, no physical benefit or full C3 claim.')
    payload = producer.canonical(value)
    with output.open('xb') as stream:
        if stream.write(payload) != len(payload):
            raise OSError('short receipt write')
        stream.flush()
        os.fsync(stream.fileno())
    producer.signal.fsync_dir(output.parent)
    print(producer.canonical(dict(output=str(output), sha256=producer.sha(output),
                                 status=receipt['status'], candidates=len(numerical))).decode())


if __name__ == '__main__':
    main()
