"""Recompute saved exploratory nonlinear suffix trajectories without GPU."""
import argparse
from copy import deepcopy
import json
import numpy as np
from experiments.spectral_reachability import p1_uphill_nonlinear_suffix_host_20261029 as producer
from experiments.spectral_reachability import p1_joint_suffix_fd_host_20261029 as source
from experiments.spectral_reachability import agent_p1_joint_vreach_saved_diff_20261029_v1 as saved


def verify(result, report, prepared):
    if result['source_report_sha256'] != source.REPORT_SHA or result['source_receipt_sha256'] != source.RECEIPT_SHA or result['updates'] != producer.UPDATES or result['calls'] != {'predict':1440,'action_encoder':1440,'reverse':1280}:
        raise ValueError('source or accounting mismatch')
    rows = result['rows']
    if len(rows) != 160:
        raise ValueError('candidate denominator')
    for index, row in enumerate(rows):
        arm = report['rows'][index//20]
        candidate = arm['result']['candidates'][index%20]
        if (row['slot'],row['arm'],row['start'],row['iteration']) != (arm['slot'],arm['arm'],candidate['start'],candidate['iteration']) or row['stage_uphill'] != candidate['stage_uphill']:
            raise ValueError('candidate binding')
        task = prepared['requests'][(index//20)//2]['task']
        goal = np.asarray(task['goal'],np.float64)
        x = np.asarray(candidate['suffix_solve']['params'],np.float64)
        evaluations = row['evaluations']
        if len(evaluations) != producer.UPDATES+1:
            raise ValueError('iteration denominator')
        for iteration, e in enumerate(evaluations):
            if e['iteration'] != iteration or not np.array_equal(np.asarray(e['logits'],np.float64), x):
                raise ValueError('iterate binding')
            raw = np.tanh(x.astype(np.float32)).reshape(5,2)
            if not np.array_equal(raw,np.asarray(e['raw'],np.float32)):
                raise ValueError('raw chart')
            end = np.asarray(e['endpoint'],np.float64)
            if end.shape != (192,) or not np.isfinite(end).all():
                raise ValueError('endpoint')
            if iteration == 0 and not np.array_equal(end,np.asarray(candidate['end']['latent_history'][-1],np.float64)):
                raise ValueError('incumbent endpoint')
            q = float(np.mean((end-goal)**2))
            regularized = q+saved.DAMPING*float(x@x)/192
            if not np.isclose(q,e['terminal_q'],rtol=1e-12,atol=1e-12) or not np.isclose(regularized,e['regularized'],rtol=1e-12,atol=1e-12):
                raise ValueError('objective mismatch')
            if iteration < producer.UPDATES:
                grad = np.asarray(e['raw_gradient'],np.float64)
                if grad.shape != (5,2) or not np.isfinite(grad).all():
                    raise ValueError('gradient')
                x = producer.proposal(x,grad,raw)
            elif e['raw_gradient'] is not None:
                raise ValueError('last gradient')
        best = min(evaluations,key=lambda e:(e['regularized'],e['iteration']))
        if (row['best_iteration'],row['best_regularized'],row['best_terminal_q']) != (best['iteration'],best['regularized'],best['terminal_q']):
            raise ValueError('argmin mismatch')
    rebuilt = producer.summarize(report,deepcopy(rows))
    if rebuilt['summary'] != result['summary'] or rebuilt['rows'] != rows:
        raise ValueError('summary mismatch')
    return dict(verified_saved_arithmetic=True,candidates=len(rows),hardware_attested=False,gradient_recomputed=False,
                independent_roots=False,physical_release=False,P1_admission=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',required=True); parser.add_argument('--sha256',required=True)
    args=parser.parse_args()
    result=saved.read_verified(args.result,args.sha256)
    report=saved.read_verified(source.REPORT,source.REPORT_SHA)
    receipt=saved.read_verified(source.RECEIPT,source.RECEIPT_SHA)
    if receipt['report_sha256'] != source.REPORT_SHA: raise ValueError('receipt binding')
    prepared=saved.read_verified(report['prepared'],report['prepare_sha256'])
    saved.analyze(report,receipt,prepared)
    print(json.dumps(verify(result,report,prepared)))


if __name__ == '__main__': main()
