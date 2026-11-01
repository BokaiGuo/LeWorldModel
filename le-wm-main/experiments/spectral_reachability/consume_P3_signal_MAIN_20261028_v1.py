"""Persist real strict consumption plus independent least-squares projection."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPORT_SHA = '5c776ea604f615081b9f50b92648e22e2612ef46d8443bf1949f69341ed4864d'
SOURCE_SHA = '6ce7722eca40cdd5f8958ba9590d8dc3d76d38bd10895d5ca494c1db3b6d100b'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source = Path(__file__).with_name('agent_p3_local_signal_cuda_20261028_v1.py')
    require(sha(source) == SOURCE_SHA, 'frozen reviewed actual consumer source')
    from experiments.spectral_reachability import agent_p3_local_signal_cuda_20261028_v1 as consumer
    directory = ROOT/'results/agent_p3_local_signal_cuda_20261028_v1_cuda_MAIN02'
    require(sha(directory/'report.json') == REPORT_SHA, 'actual report SHA')
    receipt = consumer.consume(directory, REPORT_SHA)
    report = json.loads((directory/'report.json').read_bytes())
    prepared = json.loads(Path(report['prepared']).read_bytes())
    independent = []
    for slot in (0, 1):
        task = prepared['tasks'][slot]
        q = np.asarray(task['projection'], np.float64)
        responses = [r['response'] for r in report['rows'][2*slot:2*slot+2]]
        target = q@(np.asarray(task['goal'])-np.asarray(responses[0]['nominal']))
        chis = []
        for response, spectrum in zip(responses, report['diagnostics'][slot]['spectra'], strict=True):
            j = q@np.asarray(response['jacobian'], np.float64)
            coefficients, _, rank, _ = np.linalg.lstsq(j, target, rcond=1e-6)
            residual = target-j@coefficients
            chi = float(residual@residual/(target@target)) if target@target else 0.
            require(rank == spectrum['rank'] and np.isclose(chi, spectrum['short_nominal_reference']['chi'],
                    rtol=0, atol=1e-12), 'independent lstsq residual agrees with saved spectral projector')
            chis.append(chi)
        policy = report['runtime_rule']['policies'][slot]
        gain = max(0., chis[0]-chis[1])
        for branch in report['decision_rows'][2*slot:2*slot+2]:
            fixed = branch['mode'] == 'fixed'
            high = chis[0] >= policy['chi_threshold']
            expected_type = 'progress' if fixed or not high else 'reconfiguration'
            expected_horizon = 1 if fixed or gain < policy['gain_threshold'] else 3
            expected_depth = policy['max_depth'] if fixed or high or gain >= policy['gain_threshold'] else 1
            decision = branch['queue']['decision']
            require((decision['program_type'], decision['horizon'], decision['program_depth_limit']) ==
                    (expected_type, expected_horizon, expected_depth), 'independent fixed-rule decision')
            require(branch['model_plan'] is None and branch['queue']['program_plan'] is None,
                    'requests not actual compiled programs')
        independent.append(dict(slot=slot, lstsq_chi_H1=chis[0], lstsq_chi_H3=chis[1],
                                signed_gain=chis[0]-chis[1], no_physical_result=True))
    result = dict(schema='MAIN.P3_signal_saved_independent.v1', receipt=receipt,
                  independent_lstsq=independent, report_sha256=REPORT_SHA, consumer_source_sha256=SOURCE_SHA,
                  new_neural_calls=0, new_physics=0, P1_admission=False, P3_admission=False,
                  full_C3_demonstrated=False, equal_total_compute=False)
    require(sha(source) == SOURCE_SHA and sha(directory/'report.json') == REPORT_SHA, 'final source/input seal')
    output = ROOT/'results/consume_P3_signal_MAIN_20261028_v1_MAIN01.json'
    with output.open('xb') as stream:
        stream.write((json.dumps(result, sort_keys=True, allow_nan=False)+'\n').encode())
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(output), sha256=sha(output), independent_lstsq=independent)))


if __name__ == '__main__':
    main()
