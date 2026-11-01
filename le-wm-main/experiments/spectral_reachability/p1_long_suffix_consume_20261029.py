"""Independent saved-arithmetic verification, not a model or GPU attestation."""
import argparse
from copy import deepcopy
import json
import numpy as np
from experiments.spectral_reachability import p1_long_suffix_host_20261029 as producer


def verify(result, report, prepared):
    source, saved = producer.one.source, producer.one.saved
    if (result['source_report_sha256'], result['source_receipt_sha256']) != (source.REPORT_SHA, source.RECEIPT_SHA):
        raise ValueError('source mismatch')
    if result['calls'] != dict(predict=2880, action_encoder=2880, reverse=1280):
        raise ValueError('call denominator')
    rows = result['rows']
    if len(rows) != 160:
        raise ValueError('candidate denominator')
    for index, row in enumerate(rows):
        arm = report['rows'][index//20]
        candidate = arm['result']['candidates'][index%20]
        if (row['slot'], row['arm'], row['start'], row['iteration'], row['stage_uphill']) != (arm['slot'], arm['arm'], candidate['start'], candidate['iteration'], candidate['stage_uphill']):
            raise ValueError('candidate binding')
        goal = np.asarray(prepared['requests'][(index//20)//2]['task']['goal'], np.float64)
        x = np.stack((np.asarray(candidate['suffix_solve']['params'], np.float64), np.zeros(10)))
        if len(row['evaluations']) != producer.UPDATES+1:
            raise ValueError('iteration denominator')
        for i, e in enumerate(row['evaluations']):
            if e['iteration'] != i or not np.array_equal(e['logits'], x):
                raise ValueError('iterate mismatch')
            raw = np.tanh(x.astype(np.float32)).reshape(2, 5, 2)
            if not np.array_equal(e['raw'], raw):
                raise ValueError('raw mismatch')
            end = np.asarray(e['endpoint'], np.float64)
            if end.shape != (192,) or not np.isfinite(end).all():
                raise ValueError('endpoint')
            q = float(np.mean((end-goal)**2))
            regularized = q + saved.DAMPING*float(np.sum(x*x))/192
            if not np.isclose(q, e['terminal_q'], rtol=1e-12, atol=1e-12) or not np.isclose(regularized, e['regularized'], rtol=1e-12, atol=1e-12):
                raise ValueError('objective mismatch')
            if i < producer.UPDATES:
                grad = np.asarray(e['raw_gradient'], np.float64)
                if grad.shape != (2, 5, 2) or not np.isfinite(grad).all():
                    raise ValueError('gradient')
                x = np.stack([producer.one.proposal(x[k], grad[k], raw[k]) for k in range(2)])
            elif e['raw_gradient'] is not None:
                raise ValueError('last gradient')
        best = min(row['evaluations'], key=lambda e: (e['regularized'], e['iteration']))
        if (row['best_iteration'], row['best_regularized']) != (best['iteration'], best['regularized']):
            raise ValueError('argmin')
    copied = deepcopy(rows)
    summary = producer.summarize(report, copied)
    if summary != result['summary'] or copied != rows:
        raise ValueError('summary mismatch')
    return dict(saved_arithmetic=True, candidates=len(rows), gradient_recomputed=False, hardware_attested=False, physical_release=False, P1_admission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    source, saved = producer.one.source, producer.one.saved
    result = saved.read_verified(args.result, args.sha256)
    report = saved.read_verified(source.REPORT, source.REPORT_SHA)
    receipt = saved.read_verified(source.RECEIPT, source.RECEIPT_SHA)
    if receipt['report_sha256'] != source.REPORT_SHA:
        raise ValueError('receipt mismatch')
    prepared = saved.read_verified(report['prepared'], report['prepare_sha256'])
    saved.analyze(report, receipt, prepared)
    print(json.dumps(verify(result, report, prepared)))


if __name__ == '__main__':
    main()
