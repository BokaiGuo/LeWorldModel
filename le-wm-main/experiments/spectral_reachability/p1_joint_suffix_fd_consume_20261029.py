"""Independently recompute a fixed-panel saved H1 FD result, without model calls."""
import argparse
import json

import numpy as np
from experiments.spectral_reachability import p1_joint_suffix_fd_host_20261029 as producer
from experiments.spectral_reachability import agent_p1_joint_vreach_saved_diff_20261029_v1 as saved


def verify(result, report, prepared):
    expected = producer.candidates(report, prepared)
    if result['source_report_sha256'] != producer.REPORT_SHA or result['source_receipt_sha256'] != producer.RECEIPT_SHA or result['model_calls'] != 160 or len(result['rows']) != 16:
        raise ValueError('source or denominator mismatch')
    for row, (slot, iteration, task, candidate) in zip(result['rows'], expected, strict=True):
        if (row['slot'], row['iteration']) != (slot, iteration):
            raise ValueError('candidate identity mismatch')
        direction = np.asarray(candidate['suffix'], np.float32).reshape(10)
        if not np.array_equal(direction, np.asarray(row['direction'], np.float32)):
            raise ValueError('direction mismatch')
        nominal = np.asarray(candidate['signal']['nominal_context']['latent_history'][-1], float)
        endpoint = np.asarray(candidate['end']['latent_history'][-1], float)
        if not np.isclose(row['nominal_max_abs'], np.max(np.abs(np.asarray(row['zero'])-nominal)), rtol=1e-12, atol=1e-12):
            raise ValueError('nominal mismatch')
        if not np.isclose(row['full_max_abs'], np.max(np.abs(np.asarray(row['full'])-endpoint)), rtol=1e-12, atol=1e-12):
            raise ValueError('full mismatch')
        if len(row['scales']) != len(producer.SCALES):
            raise ValueError('scale denominator')
        for record, scale in zip(row['scales'], producer.SCALES, strict=True):
            if record['scale'] != scale or record['epsilon_float32'] != float(np.float32(scale)):
                raise ValueError('scale identity')
            plus, minus = np.asarray(record['plus'], float), np.asarray(record['minus'], float)
            jacobian = np.asarray(candidate['signal']['jacobian'], float)
            if plus.shape != (192,) or minus.shape != (192,) or not np.isfinite(plus).all() or not np.isfinite(minus).all():
                raise ValueError('finite difference vector')
            fd = (plus-minus)/(2*record['epsilon_float32'])
            reference = jacobian@direction.astype(np.float64)
            denom, error = float(np.linalg.norm(reference)), float(np.linalg.norm(fd-reference))
            values = dict(fd_l2=float(np.linalg.norm(fd)), reference_l2=denom, error_l2=error,
                          relative_error=None if denom == 0 else error/denom)
            for key, value in values.items():
                if value is None:
                    if record[key] is not None:
                        raise ValueError('zero denominator mismatch')
                elif not np.isclose(record[key], value, rtol=1e-12, atol=1e-12):
                    raise ValueError('finite difference mismatch')
    return {'verified_saved_arithmetic': True, 'rows': len(expected), 'physical_release': False,
            'hardware_attested': False, 'P1_admission': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    result = saved.read_verified(args.result, args.sha256)
    report = saved.read_verified(producer.REPORT, producer.REPORT_SHA)
    receipt = saved.read_verified(producer.RECEIPT, producer.RECEIPT_SHA)
    if receipt['report_sha256'] != producer.REPORT_SHA:
        raise ValueError('receipt binding')
    prepared = saved.read_verified(report['prepared'], report['prepare_sha256'])
    saved.analyze(report, receipt, prepared)
    print(json.dumps(verify(result, report, prepared)))


if __name__ == '__main__':
    main()
