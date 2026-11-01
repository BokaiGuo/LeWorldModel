"""Saved fixed-panel argmin and within-slot action identity audit; no model calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import p1_long_suffix_consume_20261029 as consumer

RESULT_SHA = 'b178a90b90b63f7a04d9eeafa7c55f71697a1bd121c5050b4de01de3a2bb334b'
NAME = 'p1_long_suffix_selector_audit_20261029'


def analyze(result, report):
    rows = result['rows']
    if len(rows) != 160 or len(report['rows']) != 8:
        raise ValueError('fixed panel')
    winners, witnessed = [], []
    for arm_index in range(8):
        group = rows[20*arm_index:20*(arm_index+1)]
        winner_index = min(range(20), key=lambda i: (group[i]['best_regularized'], i))
        winner = group[winner_index]
        winners.append(dict(slot=winner['slot'], arm=winner['arm'], candidate_index=winner_index,
            start=winner['start'], iteration=winner['iteration'], score=winner['best_regularized'],
            stage_uphill=winner['stage_uphill']))
        for i, row in enumerate(group):
            if row['uphill_and_better']:
                candidate = report['rows'][arm_index]['result']['candidates'][i]
                selected = row['evaluations'][row['best_iteration']]
                raw = np.concatenate((np.asarray(candidate['prefix'], np.float32), np.asarray(selected['raw'], np.float32).reshape(10, 2)))
                witnessed.append(dict(slot=row['slot'], arm=row['arm'], start=row['start'], iteration=row['iteration'],
                    score=row['best_regularized'], winner_score=winner['best_regularized'],
                    disadvantage_to_full_argmin=row['best_regularized']-winner['best_regularized'],
                    original_reach_better=candidate['reach_better'], original_reach=candidate['own_reach_value'],
                    original_baseline_reach=report['rows'][arm_index]['result']['baseline_reach_value'],
                    raw15_sha256=hashlib.sha256(raw.tobytes()).hexdigest()))
    unique = len({(x['slot'], x['raw15_sha256']) for x in witnessed})
    return dict(arms=len(winners), argmin_uphill=sum(x['stage_uphill'] for x in winners),
        witnessed_occurrences=len(witnessed), distinct_slot_raw15=unique,
        witnessed_original_reach_better=sum(x['original_reach_better'] for x in witnessed), winners=winners, witnessed=witnessed,
        scope='known-root post-hoc model only; raw bytes do not prove physical identity', P1_admission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    producer = consumer.producer
    source, saved = producer.one.source, producer.one.saved
    root = source.ROOT
    output = Path(args.output)
    if not output.is_absolute() or output.parent != root/'results' or output.resolve() != output or not output.name.startswith(NAME+'_'):
        raise ValueError('exclusive output path')
    result = saved.read_verified(root/'results/p1_long_suffix_host_20261029_MAIN01.json', RESULT_SHA)
    report = saved.read_verified(source.REPORT, source.REPORT_SHA)
    receipt = saved.read_verified(source.RECEIPT, source.RECEIPT_SHA)
    if receipt['report_sha256'] != source.REPORT_SHA:
        raise ValueError('receipt binding')
    prepared = saved.read_verified(report['prepared'], report['prepare_sha256'])
    saved.analyze(report, receipt, prepared)
    consumer.verify(result, report, prepared)
    value = analyze(result, report)
    value['source_result_sha256'] = RESULT_SHA
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    with output.open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    print(json.dumps(dict(path=str(output), sha256=hashlib.sha256(data).hexdigest(), summary={k:value[k] for k in ('arms','argmin_uphill','witnessed_occurrences','distinct_slot_raw15','witnessed_original_reach_better')})))


if __name__ == '__main__':
    main()
