"""Independent saved scalar analysis; no producer import or model execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import zlib

import numpy as np

NAME = 'agent_p2_full30_control15_feedback_summary_20261029_v1'
PRODUCER = 'agent_p2_full30_control15_feedback_20261029_v2'
ARMS = ('two_program_transfer', 'joint_two_round')
FLAGS = ('C3', 'P1_admission', 'P2_admission', 'P2_mechanism', 'equal_FLOPs',
         'fresh', 'physical_release', 'scientific_admission')
TOTAL = dict(predict=216, J=8, VJP=1536, J_elements=46080, AD_primal=24)
PINS = dict(prepare='498b561855d7859681a18ed47e0157908a146a796145b929b5436a1e73b79630',
            report='6e2c82633140e071c880aed92c3cde025635cf7dd92c6687d9e0334bc30db3a8',
            completion='8622879f9c674d6acffcbc6646fbaccf25023486b0534626c3b63a5bab1d7329',
            receipt='4e7d5e1f6b4b03b28b68eb8a31016211f7d1e0ff3115671923b6df2e97f3d893')


def require(ok, why):
    if not ok:
        raise ValueError(why)


def raw(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()+b'\n'


def digest(x):
    return hashlib.sha256(raw(x)).hexdigest()


def same(a, b, why):
    require(raw(a) == raw(b), why)


def array(x, shape, f32=False):
    def numeric(v):
        return all(numeric(t) for t in v) if isinstance(v, list) else type(v) in (float, int)
    require(numeric(x), 'numeric literals, no bool')
    a = np.asarray(x, dtype=np.float64)
    require(a.shape == shape and np.isfinite(a).all(), 'finite exact vector shape')
    if f32:
        require(np.array_equal(a, a.astype('<f4').astype(float)), 'literal float32, no repair')
    return a


def context(c):
    require(set(c) == {'layout', 'context_length', 'latent_dim', 'action_dim',
                       'latent_history', 'action_history'}, 'context exact fields')
    require(c['layout'] == 'states_with_between_state_actions_v1', 'context layout')
    for k, v in [('context_length', 3), ('latent_dim', 192), ('action_dim', 10)]:
        require(type(c[k]) is int and c[k] == v, 'context dimensions')
    array(c['latent_history'], (3, 192), True)
    array(c['action_history'], (2, 10), True)
    return c


def chart(u, size=30):
    return np.tanh(array(u, (size,))).astype('<f4').reshape(-1, 5, 2)


def scores(c, goal):
    z = array(context(c)['latent_history'][-1], (192,), True)
    g = array(goal, (192,))
    # Same float64 subtraction/square as released route; independent reduction.
    delta = z-g
    return math.fsum(float(t)*float(t) for t in delta)/192, float(np.mean(delta**2))


def sign(x):
    return 'positive' if x > 0 else 'negative' if x < 0 else 'tie'


def flags(x):
    for k in FLAGS:
        require(x[k] is False, 'scientific flag must be literal false: '+k)


def audit_job(p, j, b, row, rr, plan, slot):
    bi, policy = slot//2, ('I', 'F')[slot % 2]
    root, arm = bi//2, ARMS[bi % 2]
    for obj in (j, b, row, rr, plan):
        require(type(obj['base_index']) is int and obj['base_index'] == bi and obj['arm'] == arm, 'base/arm binding')
    for obj in (j, b, rr, plan):
        require(type(obj['root_slot']) is int and obj['root_slot'] == root, 'known root binding')
    for obj in (j, row, rr, plan):
        require(obj['policy'] == policy, 'I/F job order')
    require(b['policy'] == 'O' and b['model_context_scope'] == 'inherited only; no new O prediction', 'O scope')
    require(row['slot'] == rr['slot'] == slot and row['index'] == root and
            row['status'] == rr['status'] == 'sealed', 'sealed slot identity')
    require(rr['file'] == f'slot{slot}.json.gz', 'fixed slot file')
    require(j['task_id'] == b['task_id'] == plan['task_id'], 'task identity')
    require(j['parent_witness_sha256'] == b['parent_witness_sha256'] == plan['parent_witness_sha256'], 'parent witness')
    original = chart(j['parent_logits'], 60).reshape(30, 2)
    require(original.tobytes().hex() == j['parent_raw_bits'] == b['parent_raw_bits'] == plan['parent_raw_bits'], 'parent raw bits')
    require(array(b['parent_commands'], (30, 2), True).astype('<f4').tobytes() == original.tobytes(), 'O whole raw30')
    same(j['incumbent_logits'], j['parent_logits'][30:], 'original suffix iterate0')
    same(j['prefix_raw'], original[:15].tolist(), 'original prefix15')
    context(j['input_context'])
    result = row['result']
    flags(result)
    require(result['input_context_sha256'] == digest(j['input_context']), 'exact job input context')
    prop, es = result['proposals'], result['evaluations']
    require(len(es) == len(prop['logits']) == 5, 'all five candidates retained')
    u = array(j['incumbent_logits'], (30,))
    d = array(prop['direction'], (30,))
    g = array(prop['gradient'], (30,))  # inherited AD; no J load or new differentiation
    norm = float(np.linalg.norm(g))
    target = u.copy() if norm == 0 else -.5*j['suffix_cap']*g/norm
    expected_d = target-u
    slope = float(g@expected_d)
    if slope >= 0:
        expected_d = np.zeros_like(u)
    same(prop['direction'], expected_d.tolist(), 'direction from inherited gradient')
    require(prop['slope'] == slope, 'inherited slope arithmetic')
    same(prop['logits'], [u.tolist()]+[(u+a*d).tolist() for a in (1., .5, .25, .125)], 'literal fixed alpha proposals')
    require(math.isfinite(prop['slope']), 'finite inherited slope')
    candidates = []
    for i, e in enumerate(es):
        require(type(e['index']) is int and e['index'] == i and len(e['contexts']) == 3, 'complete candidate order')
        same(e['logits'], prop['logits'][i], 'candidate logits')
        for c in e['contexts']:
            context(c)
        fsum_q, numpy_q = scores(e['contexts'][-1], j['goal'])
        require(type(e['Q']) is float and numpy_q == e['Q'], 'exact original NumPy mean Q')
        candidates.append(dict(index=i, Q_fsum=fsum_q, Q_numpy_original=numpy_q,
                               fsum_minus_numpy=fsum_q-numpy_q))
    selected = min(range(5), key=lambda i: (candidates[i]['Q_numpy_original'], i))
    alt = min(range(5), key=lambda i: (candidates[i]['Q_fsum'], i))
    require(type(result['selected']) is int and result['selected'] == selected, 'stable original (Q,index) selector')
    winner = es[selected]
    same(result['suffix_logits'], winner['logits'], 'chosen suffix')
    for k in ('contexts', 'segmented_contexts'):
        same(result[k], winner['contexts'], 'chosen whole/segment full context')
    require(result['initial_Q'] == es[0]['Q'] and result['new_suffix_Q'] == winner['Q'] <= es[0]['Q'], 'same-input incumbent comparison')
    full = np.concatenate((original[:15], chart(winner['logits']).reshape(15, 2)))
    require(array(result['commands'], (30, 2), True).astype('<f4').tobytes() == full.tobytes() and
            result['raw_float32_bits'] == full.tobytes().hex(), 'prefix15 plus exact selected suffix raw30')
    same(rr['contexts'], result['contexts'], 'report endpoint contexts')
    require(rr['new_suffix_Q'] == result['new_suffix_Q'], 'report scalar')
    # Join every saved native forward path, including independently paid replays.
    paths = [('incumbent', j['incumbent_logits'], es[0]['contexts'], j['input_context']),
             ('raw_J/forward', j['incumbent_logits'], es[0]['contexts'], j['input_context'])]
    paths += [(f'candidate{i}', es[i]['logits'], es[i]['contexts'], j['input_context']) for i in range(1, 5)]
    paths += [(tag, winner['logits'], winner['contexts'], j['input_context']) for tag in ('selected_replay', 'whole15')]
    paths += [('segment5', winner['logits'][:10], winner['contexts'][:1], j['input_context']),
              ('segment10', winner['logits'][10:], winner['contexts'][1:], winner['contexts'][0])]
    fs = row['forwards']
    require(len(fs) == 27, 'all27 forwards, no dropped calls')
    n = 0
    for tag, logits, cs, initial in paths:
        packets = np.tanh(np.asarray(logits, dtype=np.float64)).astype('<f4').reshape(-1, 5, 2)
        prev = initial
        for step, (packet, c) in enumerate(zip(packets, cs)):
            f = fs[n]
            require(f['completed'] is True and f['index'] == n and f['tag'] == f'{tag}/{step}', 'forward ordered identity')
            same(f['input'], prev, 'forward exact incoming history')
            same(f['output'], c, 'forward full shifted context')
            require(f['request']['input_context_sha256'] == digest(prev), 'forward input digest')
            desc = f['raw']
            require(desc['dtype'] == 'float32' and desc['shape'] == [5, 2] and
                    desc['raw_hex'] == packet.tobytes().hex() and
                    desc['sha256'] == hashlib.sha256(packet.tobytes()).hexdigest(), 'forward literal raw packet')
            same(f['request'], dict(index=n, tag=f'{tag}/{step}', completed=False,
                 input_context_sha256=digest(prev), raw_sha256=digest(desc)), 'native forward request')
            require([m['resource'] for m in f['modules']] == ['action_encoder', 'predictor', 'pred_proj'] and
                    all(m['completed'] is True for m in f['modules']), 'three completed native modules')
            n += 1
            prev = c
    require(len(row['AD']) == 1 and row['AD'][0]['completed'] is True, 'one inherited completed AD')
    ad = row['AD'][0]
    same(ad['contexts'], es[0]['contexts'], 'inherited AD primal context')
    same(ad['request']['at_logits'], j['incumbent_logits'], 'inherited AD iterate0')
    require(ad['request']['input_context_sha256'] == digest(j['input_context']), 'inherited AD input')
    expected = dict(schema=PRODUCER+'.model_plan', **{k:j[k] for k in
                    ('base_index', 'root_slot', 'arm', 'policy', 'task_id', 'task_sha256',
                     'parent_witness_sha256', 'parent_raw_bits', 'input_context', 'goal', 'suffix_cap')},
                    feedback_control=15, model_hash=p['expected_model_hash'],
                    **{k:result[k] for k in ('commands', 'raw_float32_bits', 'suffix_logits', 'new_suffix_Q')},
                    new_contexts=result['contexts'], selection_uses_future_labels=False,
                    old_fields_scope='parent only; new suffix/context/witness are separate',
                    **dict.fromkeys(FLAGS, False))
    expected['new_witness_sha256'] = digest(expected)
    same(plan, expected, 'complete independent new witness reconstruction')
    same(row['plan'], plan, 'gzip/receipt plan exact')
    same(row['resources'], rr['resources'], 'resource lane')
    rc = rr['resources']
    for k, v in dict(predict_attempted=27, predict_completed=27, jacobian_attempted=1,
                     jacobian_completed=1, requested_AD_primal=3, requested_J_elements=5760,
                     requested_VJP=192, encoder=0, projector=0, physics=0, SVD=0, solve=0).items():
        require(type(rc[k]) is int and rc[k] == v, 'full resource vector '+k)
    same(rc['modules_completed'], dict(action_encoder=27, predictor=27, pred_proj=27), 'module resources')
    require(rc['FLOPs'] is None and rc['engine_backward'] is None, 'unknown cost stays unknown')
    gain = candidates[0]['Q_fsum']-candidates[selected]['Q_fsum']
    return dict(slot=slot, base_index=bi, root_slot=root, arm=arm, policy=policy,
                task_id=j['task_id'], input_context_sha256=digest(j['input_context']),
                parent_witness_sha256=j['parent_witness_sha256'], new_witness_sha256=expected['new_witness_sha256'],
                parent_raw_sha256=hashlib.sha256(original.tobytes()).hexdigest(),
                prefix15_sha256=hashlib.sha256(original[:15].tobytes()).hexdigest(),
                raw30_sha256=hashlib.sha256(full.tobytes()).hexdigest(), candidates=candidates,
                original_selected=selected, descriptive_fsum_selected=alt, selectors_agree=selected == alt,
                zero_direction=bool(np.all(d == 0)), zero_selected_step=es[selected]['logits'] == es[0]['logits'],
                incumbent_minus_selected_fsum=gain, gain_sign=sign(gain),
                incumbent_minus_selected_numpy=es[0]['Q']-winner['Q'], resources=rc,
                wall_seconds=rr['wall_seconds'])


def build(p, r, c, s, rows, prepare_sha, report_sha, completion_sha, prepared):
    for x, suffix in ((p, 'prepare'), (r, 'report'), (c, 'completion'), (s, 'consume')):
        require(x['schema'] == PRODUCER+'.'+suffix, 'actual authority schema')
    for x in (p, r, s):
        flags(x)
    require(r['prepared'] == prepared and r['prepare_sha256'] == s['prepare_sha256'] == prepare_sha, 'prepare authority')
    require(c['report_sha256'] == s['report_sha256'] == report_sha and s['completion_sha256'] == completion_sha, 'report/completion authority')
    require(c['boundary'] == 'after_report_fsync_and_hash' and
            0 <= r['wall_seconds'] <= c['elapsed_seconds'] <= 1800 and
            s['whole_through_report_hash_seconds'] == c['elapsed_seconds'], 'inclusive fsync/hash boundary')
    require(r['status'] == 'all8_feedback_sealed_NOT_admitted' and
            s['status'] == 'control15_feedback_verified_NOT_admitted' and
            r['errors'] == s['errors'] == [] and r['first_error'] is None, 'complete actual authorities')
    require(r['source_verified'] is True and r['model_before'] == r['model_after'] == p['expected_model_hash'], 'frozen source/model')
    require(p['full_upstream_rebuilt_here'] is s['full_upstream_rebuilt_here'] is False and
            s['own_native_saved_reconstructed_here'] is True and s['new_consume_model_calls'] == 0, 'inherited strict scope')
    require(p['generator_labels_max_control'] == 15, 'no future labels')
    same(r['contract'], p['contract'], 'frozen contract')
    same(r['inherited_resources'], p['inherited_resources'], 'inherited resources distinct')
    same(s['actual_new'], TOTAL, 'full new resource vector')
    same(p['contract']['total'], TOTAL, 'contract resource vector')
    require(p['contract']['new_jobs'] == 8 and p['contract']['policy_slots'] == 12 and
            p['contract']['alphas'] == [1., .5, .25, .125], 'fixed candidate schedule')
    mods = dict(action_encoder=216, predictor=216, pred_proj=216)
    same(r['actual_resources'], dict(forward_attempted=216, forward_returned=216,
         jacobian_attempted=8, jacobian_returned=8, modules_attempted=mods,
         modules_returned=mods, predict_API_attempted=216, predict_API_returned=216), 'durable actual census')
    require(len(p['bases']) == 4 and all(len(x) == 8 for x in
            (p['jobs'], r['rows'], s['plans'], rows)) and
            len(r['policy_roster']) == len(s['policy_slots']) == 12, 'fixed8 jobs /12 policies')
    jobs = [audit_job(p, j, p['bases'][i//2], rows[i], r['rows'][i], s['plans'][i], i)
            for i, j in enumerate(p['jobs'])]
    for bi in range(4):
        a, b = p['jobs'][2*bi:2*bi+2]
        for k in ('task_id', 'task_sha256', 'parent_logits', 'parent_raw_bits',
                  'parent_witness_sha256', 'incumbent_logits', 'prefix_raw', 'goal', 'suffix_cap'):
            same(a[k], b[k], 'I/F same base authority, not same input: '+k)
    policies = []
    for bi, b in enumerate(p['bases']):
        for pol in ('O', 'I', 'F'):
            idx = bi*3+('O', 'I', 'F').index(pol)
            if pol == 'O':
                entry = dict(base_index=bi, policy=pol, status='inherited_openloop', parent=b)
                same(s['policy_slots'][idx], entry, 'O inherited receipt roster')
                same(r['policy_roster'][idx], entry, 'O report roster')
                policies.append(dict(base_index=bi, policy=pol, new_model_Q=None,
                                     scope='inherited openloop only', parent_witness_sha256=b['parent_witness_sha256']))
            else:
                slot = 2*bi+('I', 'F').index(pol)
                same(s['policy_slots'][idx], dict(base_index=bi, policy=pol,
                     status='new_model_plan', plan=s['plans'][slot]), 'new policy receipt roster')
                same(r['policy_roster'][idx], dict(base_index=bi, policy=pol,
                     status='sealed', model_job_slot=slot), 'new policy report roster')
                policies.append(dict(base_index=bi, policy=pol, model_job_slot=slot))
    return dict(schema=NAME+'.summary', status='saved_scalar_verified_NOT_admitted',
                denominator=dict(known_roots=2, bases=4, jobs=8, policies=12, candidate_Q=40),
                jobs=jobs, policies=policies, actual_new=TOTAL, actual_census=r['actual_resources'],
                inherited_resources=p['inherited_resources'], inclusive_seconds=c['elapsed_seconds'],
                comparison_scope='same-job incumbent minus selected only; I/F different-input Q is not a real win',
                provenance_scope='strict receipt inherited; no upstream/J/event-ledger reconstruction',
                new_model_calls=0, **dict.fromkeys(FLAGS, False))


def canonical(path):
    p = Path(path)
    require(p.is_absolute() and str(p.resolve()) == str(p) and not p.is_symlink(), 'canonical absolute path required')
    return p


def decode(data):
    def pairs(items):
        d = {}
        for k, v in items:
            require(k not in d, 'duplicate JSON key')
            d[k] = v
        return d
    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def read(path, sha, compressed=False):
    p = canonical(path)
    require(p.stat().st_size <= 64*1024**2, 'bounded saved input')
    data = p.read_bytes()
    require(hashlib.sha256(data).hexdigest() == sha, 'saved byte SHA: '+p.name)
    if compressed:
        z = zlib.decompressobj(16+zlib.MAX_WBITS)
        data = z.decompress(data, 64*1024**2+1)
        require(len(data) <= 64*1024**2 and z.eof and not z.unused_data and not z.unconsumed_tail, 'one complete bounded gzip member')
    return decode(data)


def analyze(paths, hashes):
    require(set(paths) == set(hashes) == set(PINS), 'four explicit authority inputs')
    p, r, c, s = [read(paths[k], hashes[k]) for k in PINS]
    out = canonical(paths['report']).parent
    rows = [read(str(out/f'slot{i}.json.gz'), r['artifact_sha256'][f'slot{i}.json.gz'], True) for i in range(8)]
    result = build(p, r, c, s, rows, hashes['prepare'], hashes['report'], hashes['completion'], paths['prepare'])
    result['sources'] = {k:dict(path=paths[k], sha256=hashes[k]) for k in PINS}
    result['saved_slots'] = {f'slot{i}.json.gz':r['artifact_sha256'][f'slot{i}.json.gz'] for i in range(8)}
    return result


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('command', choices=('analyze', 'verify'))
    for k in PINS:
        a.add_argument('--'+k, required=True)
        a.add_argument('--'+k+'-sha', default=PINS[k])
    a.add_argument('--output', required=True)
    a.add_argument('--summary-sha')
    args = a.parse_args()
    result = analyze({k:getattr(args, k) for k in PINS}, {k:getattr(args, k+'_sha') for k in PINS})
    output = canonical(args.output)
    if args.command == 'analyze':
        with output.open('xb') as f:
            f.write(raw(result))
    else:
        require(args.summary_sha is not None, 'verify requires summary SHA')
        same(read(str(output), args.summary_sha), result, 'whole independently rebuilt summary')
    print(json.dumps(dict(status='ok', output=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
