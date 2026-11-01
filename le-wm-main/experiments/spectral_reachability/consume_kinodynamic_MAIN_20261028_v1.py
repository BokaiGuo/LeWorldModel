"""Persist strict actual beam acceptance and independent node-cost/argmin audit."""
import os

import numpy as np

from experiments.spectral_reachability import agent_kinodynamic_matched_domain_20261028_v1 as producer


def main():
    folder = producer.ROOT/'results/agent_kinodynamic_matched_domain_20261028_v1_cuda_MAIN01'
    expected = 'd4976aeca357b5324c74be8b2b7e3bfc254e82b834b3acfe1b0ab90a1ef4614c'
    destination = producer.ROOT/'results/consume_kinodynamic_MAIN_20261028_v1_MAIN01.json'
    if destination.exists():
        raise FileExistsError(destination)
    receipt = producer.consume(folder, expected)
    report = producer.read(folder/'report.json')
    prepared = producer.read(report['prepared'])
    audits = []
    for slot in prepared['slots']:
        result = producer.read(folder/f'root_{slot["source_slot"]}.result.json.gz')
        task = producer.sampling.old.load_export(slot['task_export'], slot['task_sha256'])['gn_task']
        goal = np.asarray(task['goal'], dtype=np.float64)
        for node in result['first']+result['second']:
            endpoint = np.asarray(node['context']['latent_history'][-1], dtype=np.float64)
            cost = float(np.mean((endpoint-goal)**2))
            if cost != node['cost'] or not np.isfinite(cost):
                raise ValueError('independent full node Q')
        retained = sorted(result['first'], key=lambda node: (node['cost'], node['path']))[:6]
        selected = min(result['second'], key=lambda node: (node['cost'], node['path']))
        if ([node['path'] for node in retained] != result['retained_paths'] or
                selected['path'] != result['selected_path'] or selected['commands'] != result['commands']):
            raise ValueError('independent stable frontier/winner')
        audits.append(dict(slot=slot['source_slot'], checked_nodes=len(result['first'])+len(result['second']),
            selected_path=selected['path'], model_terminal_Q=selected['cost']))
    if producer.sha(folder/'report.json') != expected:
        raise ValueError('final actual report SHA')
    result = dict(schema='MAIN.kinodynamic_acceptance.v1', receipt=receipt,
        independent_node_arithmetic=audits, new_neural_calls=0, new_physics_calls=0,
        P1_admission=False, equal_total_compute=False, independent_confirmation=False)
    payload = producer.canonical(result)
    with destination.open('xb') as stream:
        if stream.write(payload) != len(payload):
            raise OSError('short acceptance write')
        stream.flush()
        os.fsync(stream.fileno())
    print(producer.canonical(dict(output=str(destination), sha256=producer.sha(destination),
        status=receipt['status'], independent_node_arithmetic=audits)).decode())


if __name__ == '__main__':
    main()
