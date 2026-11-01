"""Saved independent-summary ABI only; no producer imports or runtime replay."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np

NAME = 'agent_p3_horizon_activation_summary_20261029_v1'
UPSTREAM = 'agent_p3_shared_panel_summary_20261029_v1'
ARMS = ('fixedH1_endpoints', 'fixedH1_midpoints', 'adaptive_endpoints', 'adaptive_midpoints')
ROOT = Path(__file__).resolve().parents[2]
CAP = 128 * 1024**2


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def encoded(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    p = Path(path)
    require(p.is_file() and not p.is_symlink(), 'regular non-symlink file')
    return hashlib.sha256(p.read_bytes()).hexdigest()


def pairs_unique(items):
    out = {}
    for k, v in items:
        require(k not in out, 'duplicate JSON key')
        out[k] = v
    return out


def read(path, expected):
    p = Path(path)
    require(p.is_absolute() and str(p.resolve()) == str(p), 'canonical absolute path')
    require(p.stat().st_size <= CAP and sha(p) == expected, 'bounded SHA-bound input')
    obj = json.loads(p.read_bytes(), object_pairs_hook=pairs_unique)
    encoded(obj)
    return obj


def bits(value, shape, dtype):
    a = np.asarray(value, dtype=np.float64)
    require(a.shape == shape and np.isfinite(a).all(), 'finite exact shape')
    require(not any(type(v) is bool for v in np.asarray(value, dtype=object).flat), 'no boolean coordinates')
    b = a.astype(dtype)
    require(np.array_equal(a, b.astype(np.float64)), 'exact representable coordinates')
    return b.tobytes()


def reasons(decision):
    if decision is None:
        return ['decision_missing']
    keys = ('program_type', 'extension_type', 'continue_expanding', 'horizon')
    missing = ['missing_' + k for k in keys if k not in decision]
    out = list(missing)
    for k in ('program_type', 'extension_type'):
        if k in decision and decision[k] != 'progress':
            out.append(k + '_not_progress')
    if 'continue_expanding' in decision and decision['continue_expanding'] is not True:
        out.append('continue_expanding_not_true')
    if 'horizon' in decision and (type(decision['horizon']) is not int or decision['horizon'] not in (1, 3)):
        out.append('horizon_not_H1_H3')
    return out


def node_summary(node, returned):
    b, status, chosen = node['boundary'], node['status'], node['executed_horizon']
    require(status in ('sealed', 'refused', 'not_reached'), 'complete-summary node status')
    rs = reasons(node['decision'])
    if status == 'not_reached':
        require(chosen is None and node['decision'] is None and node['panels'] == [], 'unreached null node')
    else:
        require([p['horizon'] for p in node['panels']] == [1, 3], 'full ordered two panels')
        require(node['decision'] is not None, 'reached decision')
        if status == 'sealed':
            require(type(chosen) is int and chosen in (1, 3) and chosen == node['decision']['horizon'], 'chosen decision horizon')
            require(not rs, 'sealed exact supported progress predicate')
        else:
            require(chosen is None and rs and not any(r.startswith('missing_') for r in rs), 'refusal exact unsupported predicate')
    panels = []
    if status == 'not_reached':
        panels = [dict(horizon=h, category='uncomputed_H'+str(h), endpoint=b+5*h,
            available=False, missing_actual_reason='node_not_reached', error_decomposition=None) for h in (1, 3)]
    for p in node['panels']:
        h = p['horizon']
        require(p['predicted_endpoint_step'] == b + 5*h, 'endpoint identity')
        require(type(p['selected_for_execution']) is bool and p['selected_for_execution'] == (h == chosen), 'chosen panel identity')
        require(p['causal_endpoint_pair'] is (chosen == h == 1), 'H1 causal designation')
        bits(p['first5_raw'], (5, 2), np.float32)
        available = p['error_decomposition'] is not None
        require(available == (p['actual_Q_at_predicted_endpoint'] is not None), 'error availability identity')
        if b + 5*h > 20:
            require(not available and p['missing_actual_reason'] == 'beyond_executed20_lookahead', 'beyond horizon remains missing')
        same_action = chosen == h == 1 and returned >= b+5 and available
        panels.append(dict(horizon=h, category='same_action_H1' if same_action else 'descriptive_H3' if h == 3 else 'unexecuted_or_incomplete_H1',
            endpoint=b+5*h, available=available, missing_actual_reason=p['missing_actual_reason'],
            error_decomposition=deepcopy(p['error_decomposition'])))
    n = min(5, max(0, returned-b))
    require(status == 'sealed' or n == 0, 'no actions after refused/unreached node')
    return dict(boundary=b, status=status, support=status == 'sealed' if status != 'not_reached' else None,
        rejection_reasons=rs if status == 'refused' else [], decision=deepcopy(node['decision']),
        chosen_horizon=chosen, actual_selected_steps=n, panels=panels)


def summarize(s):
    require(s['schema'] == UPSTREAM+'.summary' and s['status'] == 'known2_descriptive_NOT_admitted', 'explicit independent summary ABI')
    for k, v in dict(fixed_denominator=8, node_denominator=24, panel_denominator=48, pair_denominator=8, known_roots=2, fresh_roots=0).items():
        require(type(s[k]) is int and s[k] == v, 'fixed denominator '+k)
    require(all(s[k] is False for k in ('P1_admission', 'P3_admission', 'freshness', 'scientific_admission')), 'no admission')
    require([(r['source_slot'], r['arm']) for r in s['rows']] == [(i, a) for i in (0, 1) for a in ARMS], 'all eight arms order')
    lookup = {}; rows = []; raw_by_id = {}; tally = Counter(); rejection = Counter(); categories = Counter()
    for r in s['rows']:
        eid = f"s{r['source_slot']}_{r['arm']}"
        require(r['episode_id'] == eid and eid not in lookup, 'unique episode identity')
        require(type(r['control_returned']) is int and 0 <= r['control_returned'] <= 20, 'actual returned count')
        require(len(r['steps']) == 21 and [n['boundary'] for n in r['nodes']] == [5, 10, 15], 'all steps and nodes')
        for i, step in enumerate(r['steps']):
            require((step is not None) == (i not in r['missing_steps']), 'explicit missing steps')
            if step is not None:
                require(step['step'] == i and i <= r['control_returned'], 'actual observed time')
                bits(step['state'], (7,), np.float64)
        nodes = [node_summary(n, r['control_returned']) for n in r['nodes']]
        actions = {}
        for original, n in zip(r['nodes'], nodes):
            tally[n['status']] += 1
            if n['chosen_horizon'] is not None:
                tally['chosen_H'+str(n['chosen_horizon'])] += 1
                if n['actual_selected_steps']:
                    tally['dispatched_H'+str(n['chosen_horizon'])] += 1
                chosen = next(p for p in original['panels'] if p['selected_for_execution'])
                for j in range(n['actual_selected_steps']):
                    actions[n['boundary']+j+1] = chosen['first5_raw'][j]
            rejection.update(n['rejection_reasons'])
            for p in n['panels']:
                categories[p['category']+('_available' if p['available'] else '_missing')] += 1
        if r['arm'].startswith('fixedH1'):
            require(all(n['chosen_horizon'] != 3 for n in nodes), 'fixed H1 cannot choose H3')
        lookup[eid] = r; raw_by_id[eid] = actions
        rows.append(dict(episode_id=eid, source_slot=r['source_slot'], arm=r['arm'], control_returned=r['control_returned'], nodes=nodes))
    require(len(s['pairs']) == 8, 'full eight pair denominator')
    expected = [(i, f's{i}_{l}', f's{i}_{r}') for i in (0, 1) for l, r in (
        ('fixedH1_endpoints', 'adaptive_endpoints'), ('fixedH1_midpoints', 'adaptive_midpoints'),
        ('fixedH1_endpoints', 'fixedH1_midpoints'), ('adaptive_endpoints', 'adaptive_midpoints'))]
    require([(p['source_slot'], p['left'], p['right']) for p in s['pairs']] == expected, 'fixed contrasts order')
    pairs = []
    for p in s['pairs']:
        l, r = lookup[p['left']], lookup[p['right']]; steps = []
        require(len(p['steps']) == 21, 'all paired times')
        for i, (a, b, supplied) in enumerate(zip(l['steps'], r['steps'], p['steps'])):
            state_equal = None if a is None or b is None else bits(a['state'], (7,), np.float64) == bits(b['state'], (7,), np.float64)
            require(supplied['step'] == i and supplied['state_bits_equal'] is state_equal, 'independent state bit pairing')
            al, ar = raw_by_id[p['left']].get(i), raw_by_id[p['right']].get(i)
            raw_equal = None if al is None or ar is None else bits([al], (1, 2), np.float32) == bits([ar], (1, 2), np.float32)
            steps.append(dict(step=i, raw_bits_equal=raw_equal, state_bits_equal=state_equal,
                raw_missing_reason='not_an_action' if i == 0 else 'prefix5_not_in_summary_ABI' if i <= 5 else 'one_or_both_not_returned' if raw_equal is None else None))
        pairs.append(dict(source_slot=p['source_slot'], left=p['left'], right=p['right'], steps=steps,
            first_observed_raw_divergence=next((v['step'] for v in steps if v['raw_bits_equal'] is False), None),
            first_observed_state_divergence=next((v['step'] for v in steps if v['state_bits_equal'] is False), None),
            raw_counts={k:sum(('missing' if v['raw_bits_equal'] is None else 'same' if v['raw_bits_equal'] else 'different') == k for v in steps[1:]) for k in ('missing','same','different')},
            state_counts={k:sum(('missing' if v['state_bits_equal'] is None else 'same' if v['state_bits_equal'] else 'different') == k for v in steps) for k in ('missing','same','different')}))
    require(sum(categories.values()) == 48, 'all panel slots including uncomputed')
    return dict(schema=NAME+'.summary', fixed_denominator=8, node_denominator=24, panel_denominator=48, pair_denominator=8,
        rows=rows, pairs=pairs, node_counts={k:tally[k] for k in ('sealed','refused','not_reached','chosen_H1','chosen_H3','dispatched_H1','dispatched_H3')},
        refusal_reasons_nonexclusive=dict(rejection), error_categories={k:categories[k] for k in (
            'same_action_H1_available','same_action_H1_missing','descriptive_H3_available','descriptive_H3_missing',
            'unexecuted_or_incomplete_H1_available','unexecuted_or_incomplete_H1_missing','uncomputed_H1_missing','uncomputed_H3_missing')}, known_roots=2, fresh_roots=0,
        P1_admission=False, P3_admission=False, freshness=False, scientific_admission=False,
        uncertainty='descriptive only; no CI', full_strict_rebuilt_here=False, producer_imported=False,
        interpretation='chosen is not necessarily dispatched; no observed divergence is not a mechanism-negative conclusion; H3 is descriptive')


def analyze(summary, summary_sha):
    s = read(summary, summary_sha)
    out = summarize(s)
    files = [Path(__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')]
    out.update(arguments=dict(summary=summary, summary_sha=summary_sha), upstream_bindings=deepcopy(s['bindings']),
        upstream_authority=deepcopy(s['authority']), sources={str(p):sha(p) for p in files})
    require(sha(summary) == summary_sha, 'input unchanged')
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('analyze', 'verify'))
    for k in ('summary','summary-sha','output','analysis','analysis-sha'):
        ap.add_argument('--'+k)
    a = ap.parse_args(argv)
    if a.command == 'verify':
        saved = read(a.analysis, a.analysis_sha)
        require(encoded(saved) == encoded(analyze(**saved['arguments'])), 'deterministic complete reconstruction')
        result = dict(verified=True, sha256=a.analysis_sha)
    else:
        result = analyze(a.summary, a.summary_sha); body = encoded(result); p = Path(a.output)
        require(p.is_absolute() and str(p.resolve()) == str(p) and p.name.startswith(NAME+'_') and len(body) <= CAP, 'exclusive bounded own output')
        with p.open('xb') as f:
            require(f.write(body) == len(body), 'whole output')
        result = dict(path=str(p), sha256=sha(p))
    print(encoded(result).decode())


if __name__ == '__main__':
    main()
