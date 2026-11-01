"""MAIN strict saved acceptance plus independent endpoint/selection arithmetic."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import agent_p3_fixed_execution_cuda_20261028_v1 as producer


def independent(panel, report):
    rows = report['rows']
    if len(rows) != len(panel['jobs']):
        raise ValueError('complete candidate denominator')
    summaries = []
    for slot in (0, 1):
        task = panel['tasks'][slot]
        goal = np.asarray(task['goal'], dtype=np.float64)
        projection = np.asarray(task['projection'], dtype=np.float64)
        for mode in ('fixed', 'adaptive'):
            group = [r for r in rows if (r['slot'], r['mode']) == (slot, mode)]
            branch = next(b for b in panel['branches'] if (b['slot'], b['mode']) == (slot, mode))
            if len(group) != branch['candidates'] or not group:
                raise ValueError('whole prespecified branch')
            costs = []
            for row in group:
                residual = projection @ (np.asarray(row['whole'][-1]['latent_history'][-1], dtype=np.float64)-goal)
                cost = float(np.mean(residual**2))
                if not np.isfinite(cost) or cost != row['cost']:
                    raise ValueError('independent terminal Q mismatch')
                costs.append((cost, row['index']))
            cost, index = min(costs)
            winner = next(w for w in report['winners'] if (w['slot'], w['mode']) == (slot, mode))
            if (winner['cost'], winner['index']) != (cost, index):
                raise ValueError('independent stable winner mismatch')
            row = next(r for r in group if r['index'] == index)
            prefix = next(p for p in report['prefixes'] if (p['slot'], p['mode']) == (slot, mode))
            commands = np.asarray(prefix['request']['commands'], dtype='<f4')
            original = np.asarray(row['actions'][:5], dtype='<f4')
            if commands.shape != (5, 2) or commands.tobytes() != original.tobytes():
                raise ValueError('independent five-step raw buffer mismatch')
            # Same execution boundary, even though selection uses each planning endpoint.
            delta = projection @ (np.asarray(row['whole'][0]['latent_history'][-1], dtype=np.float64)-goal)
            summaries.append(dict(slot=slot, mode=mode, selected_index=index,
                planning_packets=row['horizon'], requested_execution_steps=5,
                planning_terminal_Q=cost, execution_boundary_Q=float(np.mean(delta**2))))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report-sha', required=True)
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    destination = Path(args.destination).resolve()
    if destination.parent != producer.ROOT/'results' or not destination.name.startswith('consume_p3_fixed_execution_MAIN_20261028_v1_') or destination.exists():
        raise ValueError('exclusive own results output')
    receipt = producer.consume(args.output, args.report_sha)
    report_path = Path(args.output)/'report.json'
    report = json.loads(report_path.read_bytes())
    prepared = Path(report['prepared'])
    if producer.sha(prepared) != report['prepare_sha256']:
        raise ValueError('prepare changed after strict consumer')
    package = json.loads(prepared.read_bytes())
    numerical = independent(package['panel']['base'], report)
    if producer.sha(report_path) != args.report_sha:
        raise ValueError('report changed after arithmetic')
    result = dict(schema='MAIN.p3_fixed_execution_acceptance.v1', receipt=receipt,
        independent_endpoint_arithmetic=numerical, new_neural_calls=0, new_physics_calls=0,
        P1_admission=False, P3_admission=False, independent_confirmation=False,
        caveat='Two known roots; requested five steps only, no physical outcome or complete C3 evidence.')
    payload = producer.canonical(result)
    with destination.open('xb') as stream:
        if stream.write(payload) != len(payload):
            raise OSError('short acceptance write')
        stream.flush()
        os.fsync(stream.fileno())
    producer.signal.fsync_dir(destination.parent)
    print(producer.canonical(dict(output=str(destination), sha256=producer.sha(destination),
        independent_endpoint_arithmetic=numerical)).decode())


if __name__ == '__main__':
    main()
