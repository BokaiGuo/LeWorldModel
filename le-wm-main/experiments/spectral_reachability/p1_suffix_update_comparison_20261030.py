"""Compare fixed saved panels; no model execution or scientific admission."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import p1_long_suffix_consume_20261029 as old_consumer
from experiments.spectral_reachability import p1_regularized_suffix_consume_20261030 as new_consumer

OLD_SHA = 'b178a90b90b63f7a04d9eeafa7c55f71697a1bd121c5050b4de01de3a2bb334b'
NEW_SHA = 'aa162a9ff2b8fb66d13fbc8d3db625c753b9f090160d6101e38b1e300e291799'
NAME = 'p1_suffix_update_comparison_20261030'


def compare(old, new):
    if len(old['rows']) != 160 or len(new['rows']) != 160:
        raise ValueError('candidate denominator')
    decisions = []
    changed_actions = changed_best_iteration = changed_flags = 0
    max_score_delta = max_raw_delta = 0.
    for index, (a, b) in enumerate(zip(old['rows'], new['rows'], strict=True)):
        if any(a[k] != b[k] for k in ('slot', 'arm', 'start', 'iteration', 'stage_uphill')):
            raise ValueError('candidate identity mismatch')
        ae, be = a['evaluations'][a['best_iteration']], b['evaluations'][b['best_iteration']]
        ar, br = np.asarray(ae['raw'], np.float32), np.asarray(be['raw'], np.float32)
        if ar.shape != (2, 5, 2) or br.shape != ar.shape or not np.isfinite(ar).all() or not np.isfinite(br).all():
            raise ValueError('action shape or finite')
        changed_actions += ar.tobytes() != br.tobytes()
        changed_best_iteration += a['best_iteration'] != b['best_iteration']
        changed_flags += a['uphill_and_better'] != b['uphill_and_better']
        max_score_delta = max(max_score_delta, abs(a['best_regularized']-b['best_regularized']))
        max_raw_delta = max(max_raw_delta, float(np.max(np.abs(ar.astype(float)-br.astype(float)))))
    for arm in range(8):
        lo = arm*20
        ai = min(range(20), key=lambda i: (old['rows'][lo+i]['best_regularized'], i))
        bi = min(range(20), key=lambda i: (new['rows'][lo+i]['best_regularized'], i))
        a, b = old['rows'][lo+ai], new['rows'][lo+bi]
        decisions.append(dict(slot=a['slot'], arm=a['arm'], old_index=ai, new_index=bi,
            same_candidate=ai == bi, old_stage_uphill=a['stage_uphill'], new_stage_uphill=b['stage_uphill'],
            old_score=a['best_regularized'], new_score=b['best_regularized']))
    return dict(candidates=160, decisions=decisions, changed_candidate_actions=changed_actions,
        changed_best_iteration=changed_best_iteration, changed_uphill_better_flag=changed_flags,
        max_abs_score_change=max_score_delta, max_abs_selected_raw_change=max_raw_delta,
        scope='known-root post-hoc model panels; no new physical result',
        independent_roots=False, P1_admission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = old_consumer.producer.one.source
    saved = old_consumer.producer.one.saved
    output = Path(args.output)
    if not output.is_absolute() or output.parent != source.ROOT/'results' or output.resolve() != output or not output.name.startswith(NAME+'_'):
        raise ValueError('exclusive output path')
    old = saved.read_verified(source.ROOT/'results/p1_long_suffix_host_20261029_MAIN01.json', OLD_SHA)
    new = saved.read_verified(source.ROOT/'results/p1_regularized_suffix_host_20261030_MAIN01.json', NEW_SHA)
    report = saved.read_verified(source.REPORT, source.REPORT_SHA)
    receipt = saved.read_verified(source.RECEIPT, source.RECEIPT_SHA)
    if receipt['report_sha256'] != source.REPORT_SHA:
        raise ValueError('receipt binding')
    prepared = saved.read_verified(report['prepared'], report['prepare_sha256'])
    saved.analyze(report, receipt, prepared)
    old_consumer.verify(old, report, prepared)
    new_consumer.verify(new, report, prepared)
    value = compare(old, new)
    value.update(old_sha256=OLD_SHA, new_sha256=NEW_SHA)
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    with output.open('xb') as stream:
        if stream.write(data) != len(data):
            raise OSError('short write')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(path=str(output), sha256=hashlib.sha256(data).hexdigest(), result=value)))


if __name__ == '__main__':
    main()
