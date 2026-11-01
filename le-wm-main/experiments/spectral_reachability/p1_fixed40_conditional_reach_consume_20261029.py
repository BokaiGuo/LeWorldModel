"""Recompute bounded ridge from saved CUDA outputs; no neural inference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from experiments.spectral_reachability import p1_fixed40_conditional_reach_host_20261029 as host
from experiments.spectral_reachability import agent_p1_joint_vreach_saved_diff_20261029_v1 as independent


def verify(path, sha):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != sha:
        raise ValueError('hash mismatch')
    report = json.loads(data)
    host.prefix_consumer.verify(host.PREFIX, host.PREFIX_SHA)
    prep = host.prefix.history.rgb.read(host.prefix.history.TEACHER_PREP, host.prefix.history.TEACHER_PREP_SHA)
    if report['prefix_sha256'] != host.PREFIX_SHA or report['model_hash_before'] != prep['expected_model_hash'] or report['model_hash_after'] != prep['expected_model_hash'] or len(report['rows']) != 8:
        raise ValueError('source or roster mismatch')
    results = []
    for i, (row, anchor) in enumerate(zip(report['rows'], prep['anchors'], strict=True)):
        if row['anchor_index'] != i or row['anchor_id'] != anchor['anchor_id'] or row['task_id'] != anchor['task_id']:
            raise ValueError('anchor mismatch')
        values = {}
        for name in ('predicted', 'observed'):
            case = row['cases'][name]
            j = np.asarray(case['jacobian'], dtype=np.float64)
            residual = np.asarray(anchor['goal'], dtype=np.float64)-np.asarray(case['nominal'], dtype=np.float64)
            x, _ = independent.independent_solve(j, residual, host.compiler.DAMPING, host.compiler.CAP)
            score = independent.compare(j, residual, x, host.compiler.DAMPING)['score_raw']
            old = case['solve']['own_reach_value']
            if not np.isclose(score, old, rtol=1e-8, atol=1e-10):
                raise ValueError('independent ridge score mismatch')
            values[name] = score
        results.append(dict(anchor_index=i, **values))
    return dict(report_sha256=sha, values=results, model_rerun=False, finite_difference_recomputed=False,
                interpretation='independent ridge scores only; no physical capability or hardware attestation')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', required=True)
    p.add_argument('--sha256', required=True)
    a = p.parse_args()
    print(json.dumps(verify(a.report, a.sha256), sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
