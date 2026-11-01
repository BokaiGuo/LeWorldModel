"""Saved-only exact action-witness alignment, not new NN/physics evidence."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

NAME = 'main_p3_action_witness_alignment_20261029_v1'
ROOT = Path(__file__).resolve().parents[2]
PINS = {
    'report': ('agent_p3_shared_panel_20261029_v1_MAIN01/report.json', '668db35d79eb650419e4efa42dfc21a15cdf91725700b2f202e2a2ec975bcd6a'),
    'strict': ('agent_p3_shared_panel_20261029_v1_consume_MAIN01.json', 'd23ffa48bc954306bc27a15f924da8daafffc9b4bc76a91e59aa949ab1abbafd'),
    'summary': ('agent_p3_shared_panel_summary_20261029_v1_MAIN01.json', '4e89812a384a352840f41b84ef9a0ac1ea07622288949c6d5ff6c5795cc28224'),
}


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def raw(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''):
            h.update(b)
    return h.hexdigest()


def unique(items):
    d = {}
    for k, v in items:
        need(k not in d, 'duplicate JSON key')
        d[k] = v
    return d


def read(p, expected):
    p = Path(p)
    need(p.is_file() and not p.is_symlink() and p.stat().st_size <= 256*1024**2, 'bounded original JSON')
    need(sha(p) == expected, 'actual external SHA')
    x = json.loads(p.read_bytes(), object_pairs_hook=unique)
    raw(x)
    return x


def actions(x):
    a = np.asarray(x, np.float64)
    need(a.shape == (len(x), 2) and np.isfinite(a).all(), 'complete raw action pairs')
    need(not any(type(v) is bool for v in np.asarray(x, object).flat), 'no bool actions')
    b = a.astype(np.float32)
    need((abs(b) <= 1).all() and np.array_equal(a, b.astype(np.float64)), 'literal float32 raw domain')
    return b


def align(planned, executed, boundary, horizon):
    need(type(boundary) is int and boundary in (5, 10, 15) and type(horizon) is int and horizon in (1, 3), 'original node identity')
    p, e = actions(planned), actions(executed)
    need(len(p) == 5*horizon and boundary <= len(e) <= 20, 'full forecast / actual returned horizon')
    available = min(len(p), len(e)-boundary)
    equal = [p[i].tobytes() == e[boundary+i].tobytes() for i in range(available)]
    full = available == len(p)
    return dict(predicted_endpoint=boundary+len(p), actual_returned=len(e), planned_steps=len(p),
                compared_steps=available, missing_steps=len(p)-available, full_action_witness_observed=full,
                all_compared_raw_bits_equal=all(equal),
                exact_full_action_match=all(equal) if full else None,
                first_differing_control_step=next((boundary+i+1 for i, same in enumerate(equal) if not same), None),
                equal_by_control_step=[dict(step=boundary+i+1, raw_bits_equal=same) for i, same in enumerate(equal)],
                future_actions_missing=not full)


def summarize(report, strict, summary):
    need(report['schema'] == 'agent_p3_shared_panel_20261029_v1.report' and
         strict['schema'] == 'agent_p3_shared_panel_20261029_v1.consume' and
         summary['schema'] == 'agent_p3_shared_panel_summary_20261029_v1.summary', 'exact own upstream ABI')
    need(strict['complete'] is True and strict['report_sha256'] == PINS['report'][1] and
         summary['authority']['external_actual_strict_sha256'] == PINS['strict'][1], 'external strict identity inherited, not rebuilt')
    need(report['errors'] == [] and len(report['rows']) == len(summary['rows']) == 8 and len(report['nodes']) == 24, 'fixed eight/24 denominator')
    rr = {r['episode_id']: r for r in report['rows']}
    ss = {r['episode_id']: r for r in summary['rows']}
    need(len(rr) == len(ss) == 8 and set(rr) == set(ss), 'unique same eight episode identities')
    records = []
    counts = Counter()
    for node in report['nodes']:
        eid, b = node['episode_id'], node['boundary']
        row, sr = rr[eid], ss[eid]
        sn = next(v for v in sr['nodes'] if v['boundary'] == b)
        need(node['status'] == sn['status'] and row['control_returned'] == sr['control_returned'], 'actual node/time binding')
        if node['receipt'] is None:
            need(node['status'] == 'not_reached', 'missing only unreached')
            records.extend(dict(episode_id=eid, boundary=b, horizon=h, status='not_reached', alignment=None) for h in (1, 3))
            continue
        receipt = node['receipt']
        need([p['horizon'] for p in receipt['panels']] == [1, 3], 'both full panels')
        for panel, sp in zip(receipt['panels'], sn['panels'], strict=True):
            h = panel['horizon']
            need(h == sp['horizon'] and panel['winner']['index'] == sp['winner_index'], 'same saved winner, no re-selection')
            a = align(panel['winner']['actions'], row['executed_raw'], b, h)
            chosen = h == receipt['selected_horizon']
            if chosen:
                need(all(v['raw_bits_equal'] for v in a['equal_by_control_step'][:5]), 'selected first5 actually executed')
            same = a['exact_full_action_match'] is True and sp['error_decomposition'] is not None
            if not a['full_action_witness_observed']:
                need(sp['error_decomposition'] is None, 'unobserved endpoint not scored')
            category = 'same_action_endpoint' if same else 'changed_action_trajectory' if a['first_differing_control_step'] is not None else 'incomplete_action_witness'
            counts[f'H{h}_{category}'] += 1
            records.append(dict(episode_id=eid, boundary=b, horizon=h, status=node['status'], selected=chosen,
                alignment=a, category=category, original_conservative_H1_flag=sp['causal_endpoint_pair'],
                same_action_error_decomposition=sp['error_decomposition'] if same else None,
                descriptive_error_decomposition=sp['error_decomposition'] if not same else None))
    need(len(records) == 48, 'all48 forecast slots retained')
    return dict(schema=NAME+'.summary', panel_denominator=48, node_denominator=24, episode_denominator=8,
                known_roots=2, counts=dict(counts), records=records, full_strict_rebuilt_here=False,
                new_neural_calls=0, new_physics_calls=0, P1_admission=False, P3_admission=False,
                interpretation='Exact same actions are required for endpoint fidelity; changed MPC actions are not model prediction errors.')


def analyze():
    refs = {k: (ROOT/'results'/p, h) for k, (p, h) in PINS.items()}
    result = summarize(**{k: read(p, h) for k, (p, h) in refs.items()})
    result['bindings'] = {str(p): h for p, h in refs.values()}
    for p in (Path(__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py')):
        result['bindings'][str(p)] = sha(p)
    for p, h in result['bindings'].items():
        need(sha(p) == h, 'all inputs unchanged after saved analysis')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('analyze', 'verify'))
    p.add_argument('--output')
    p.add_argument('--summary')
    p.add_argument('--summary-sha')
    a = p.parse_args()
    result = analyze()
    if a.command == 'verify':
        need(raw(read(a.summary, a.summary_sha)) == raw(result), 'whole deterministic saved reconstruction')
        print(raw(dict(verified=True, sha256=a.summary_sha)).decode())
    else:
        dest = Path(a.output).absolute()
        need(dest.parent == ROOT/'results' and dest.name.startswith(NAME+'_') and not dest.exists(), 'exclusive own result')
        import os
        with dest.open('xb') as f:
            f.write(raw(result)); f.flush(); os.fsync(f.fileno())
        print(raw(dict(path=str(dest), sha256=sha(dest))).decode())


if __name__ == '__main__':
    main()
