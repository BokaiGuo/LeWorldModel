"""Check stored directional finite differences without rerunning the network."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from experiments.spectral_reachability import p1_fixed40_conditional_reach_consume_20261029 as original


def verify(path, expected):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('result hash mismatch')
    result = json.loads(data)
    original.verify(original.host.ROOT/'results/p1_fixed40_conditional_reach_host_20261029_MAIN01.json',
                    '9281d3353c8a36bcf7d836614a3142afbc73321064efb68edd52ce2533e9439c')
    previous = json.loads((original.host.ROOT/'results/p1_fixed40_conditional_reach_host_20261029_MAIN01.json').read_bytes())
    if len(result['rows']) != 8 or result['model_hash_after'] != result['model_hash_before'] or result['prefix_sha256'] != original.host.PREFIX_SHA:
        raise ValueError('model/source/roster mismatch')
    errors = []
    for row, old in zip(result['rows'], previous['rows'], strict=True):
        if row['anchor_id'] != old['anchor_id']:
            raise ValueError('anchor mismatch')
        for key in ('predicted', 'observed'):
            case, before = row['cases'][key], old['cases'][key]
            for name in ('nominal', 'jacobian', 'solve'):
                if case[name] != before[name]:
                    raise ValueError('fixed panel repeat differs: '+name)
            j = np.asarray(case['jacobian'], dtype=np.float32)
            d = np.asarray(case['direction'], dtype=np.float32)
            plus, minus = [np.asarray(case[x], dtype=np.float32) for x in ('plus', 'minus')]
            if j.shape != (192,10) or d.shape != (10,) or plus.shape != (192,) or minus.shape != (192,) or case['epsilon'] != .001:
                raise ValueError('invalid difference inputs')
            fd = (plus-minus)/(2*case['epsilon'])
            analytic = j@d
            err = float(np.linalg.norm(fd-analytic)/max(np.linalg.norm(analytic),1e-12))
            if not np.isclose(err, case['fd_directional_relative_error'], rtol=1e-7, atol=1e-10):
                raise ValueError('finite difference arithmetic mismatch')
            errors.append(err)
    return dict(report_sha256=expected, cases=len(errors), min_error=min(errors), max_error=max(errors),
                model_rerun=False, independent_roots=0, hardware_attestation=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', required=True)
    p.add_argument('--sha256', required=True)
    a = p.parse_args()
    print(json.dumps(verify(a.report,a.sha256), sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
