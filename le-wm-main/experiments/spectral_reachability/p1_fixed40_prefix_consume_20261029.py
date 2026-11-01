"""Recompute saved prefix scoring and identity; does not rerun neural inference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from experiments.spectral_reachability import p1_fixed40_prefix_host_20261029 as host


def verify(path, expected):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('report hash mismatch')
    result = json.loads(data)
    old = host.observed.build()
    prep = host.history.rgb.read(host.history.TEACHER_PREP, host.history.TEACHER_PREP_SHA)
    if result['source_bindings'] != old['source_bindings'] or result['observed_sha256'] != host.OBSERVED_SHA:
        raise ValueError('source mismatch')
    if result['model_hash_before'] != prep['expected_model_hash'] or result['model_hash_after'] != prep['expected_model_hash']:
        raise ValueError('checkpoint mismatch')
    if result['model_calls'] != 8 or len(result['rows']) != 8 or result['P1_admission'] is not False:
        raise ValueError('roster or admission mismatch')
    for i, (row, anchor, observed) in enumerate(zip(result['rows'], prep['anchors'], old['rows'], strict=True)):
        raw = np.asarray(anchor['prefix_commands'], dtype='<f4')
        if (row['anchor_index'] != i or row['anchor_id'] != anchor['anchor_id'] or
            row['task_id'] != anchor['task_id'] or row['prefix_sha256'] != hashlib.sha256(raw.tobytes()).hexdigest() or
            row['observed_stage'] != observed['observed_stage'] or
            row['model_stage_Q'] != host.score(row['predicted_latent'], anchor['goal'])):
            raise ValueError('saved identity or scoring mismatch')
    pairs = []
    for i in range(0, 8, 2):
        a, b = result['rows'][i:i+2]
        if a['task_id'] != b['task_id'] or a['observed_stage']['start_Q'] != b['observed_stage']['start_Q']:
            raise ValueError('pair context mismatch')
        pairs.append(dict(task_id=a['task_id'], model_S_minus_Z=b['model_stage_Q']-a['model_stage_Q'],
                          observed_S_minus_Z=b['observed_stage']['end_Q']-a['observed_stage']['end_Q']))
    return dict(report_sha256=expected, pairs=pairs, model_rerun=False, physical_rerun=False,
                interpretation='saved scores and identities only; not hardware attestation or scientific admission')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', required=True)
    p.add_argument('--sha256', required=True)
    args = p.parse_args()
    print(json.dumps(verify(args.report, args.sha256), sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
