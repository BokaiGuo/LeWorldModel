"""Model-free fixtures; never scientific or native execution evidence."""
from copy import deepcopy
import gzip
import hashlib
import json

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p2_full30_control15_feedback_summary_20261029_v1 as m


def ctx(value):
    return dict(layout='states_with_between_state_actions_v1', context_length=3,
                latent_dim=192, action_dim=10,
                latent_history=np.full((3, 192), value, dtype='<f4').tolist(),
                action_history=np.zeros((2, 10)).tolist())


def fixture(zero=False):
    flags = dict.fromkeys(m.FLAGS, False)
    p = dict(schema=m.PRODUCER+'.prepare', bases=[], jobs=[], expected_model_hash='model',
             inherited_resources=dict(model={'predict': 123}, native={'public': 100},
                                      encoding={'encoder': 12}, charged_again_as_new=False),
             full_upstream_rebuilt_here=False, generator_labels_max_control=15,
             contract=dict(total=m.TOTAL, new_jobs=8, policy_slots=12, alphas=[1., .5, .25, .125]), **flags)
    r = dict(schema=m.PRODUCER+'.report', prepared='/fixture/prepare.json', prepare_sha256='p',
             rows=[], policy_roster=[], artifact_sha256={}, model_before='model', model_after='model',
             source_verified=True, wall_seconds=1., status='all8_feedback_sealed_NOT_admitted',
             errors=[], first_error=None, **flags)
    c = dict(schema=m.PRODUCER+'.completion', report_sha256='r',
             boundary='after_report_fsync_and_hash', elapsed_seconds=2.)
    s = dict(schema=m.PRODUCER+'.consume', prepare_sha256='p', report_sha256='r', completion_sha256='c',
             status='control15_feedback_verified_NOT_admitted', errors=[], plans=[], policy_slots=[],
             full_upstream_rebuilt_here=False, own_native_saved_reconstructed_here=True,
             new_consume_model_calls=0, whole_through_report_hash_seconds=2., actual_new=m.TOTAL, **flags)
    mods = dict(action_encoder=216, predictor=216, pred_proj=216)
    r['actual_resources'] = dict(forward_attempted=216, forward_returned=216,
                                jacobian_attempted=8, jacobian_returned=8, modules_attempted=mods,
                                modules_returned=mods, predict_API_attempted=216, predict_API_returned=216)
    r['contract'], r['inherited_resources'] = deepcopy(p['contract']), deepcopy(p['inherited_resources'])
    rows = []
    for bi in range(4):
        full_logits = np.zeros(60).tolist()
        parent = m.chart(full_logits, 60).reshape(30, 2)
        b = dict(base_index=bi, root_slot=bi//2, arm=m.ARMS[bi % 2], policy='O', task_id=f'task{bi}',
                 parent_witness_sha256=f'parent{bi}', parent_raw_bits=parent.tobytes().hex(),
                 parent_commands=parent.tolist(), model_context_scope='inherited only; no new O prediction',
                 parent_model_report_sha256='inherited')
        p['bases'].append(b)
        o = dict(base_index=bi, policy='O', status='inherited_openloop', parent=b)
        r['policy_roster'].append(deepcopy(o)); s['policy_slots'].append(deepcopy(o))
        for policy in ('I', 'F'):
            slot = len(rows)
            j = dict(base_index=bi, root_slot=bi//2, arm=b['arm'], policy=policy, task_id=b['task_id'],
                     task_sha256=f'tasksha{bi}', parent_witness_sha256=b['parent_witness_sha256'],
                     parent_logits=full_logits, parent_raw_bits=b['parent_raw_bits'],
                     prefix_raw=parent[:15].tolist(), incumbent_logits=full_logits[30:],
                     suffix_cap=2., goal=np.zeros(192).tolist(), input_context=ctx(slot+1))
            p['jobs'].append(j)
            u = np.asarray(j['incumbent_logits'])
            g = np.zeros(30) if zero else np.ones(30)
            n = np.linalg.norm(g)
            d = np.zeros(30) if zero else -.5*j['suffix_cap']*g/n-u
            prop = dict(logits=[u.tolist()]+[(u+a*d).tolist() for a in (1., .5, .25, .125)],
                        gradient=g.tolist(), direction=d.tolist(), slope=float(g@d))
            es = []
            for i in range(5):
                cs = [ctx(2. if zero or i == 0 else 1.+i/8) for _ in range(3)]
                q = m.scores(cs[-1], j['goal'])[1]
                es.append(dict(index=i, logits=prop['logits'][i], contexts=cs, Q=q))
            selected = min(range(5), key=lambda i: (es[i]['Q'], i))
            w = es[selected]
            commands = np.concatenate((parent[:15], m.chart(w['logits']).reshape(15, 2)))
            result = dict(selected=selected, proposals=prop, evaluations=es, contexts=w['contexts'],
                          segmented_contexts=w['contexts'], initial_Q=es[0]['Q'], new_suffix_Q=w['Q'],
                          suffix_logits=w['logits'], commands=commands.tolist(),
                          raw_float32_bits=commands.tobytes().hex(), input_context_sha256=m.digest(j['input_context']), **flags)
            plan = dict(schema=m.PRODUCER+'.model_plan', **{k:j[k] for k in
                        ('base_index', 'root_slot', 'arm', 'policy', 'task_id', 'task_sha256',
                         'parent_witness_sha256', 'parent_raw_bits', 'input_context', 'goal', 'suffix_cap')},
                        feedback_control=15, model_hash='model',
                        **{k:result[k] for k in ('commands', 'raw_float32_bits', 'suffix_logits', 'new_suffix_Q')},
                        new_contexts=result['contexts'], selection_uses_future_labels=False,
                        old_fields_scope='parent only; new suffix/context/witness are separate', **flags)
            plan['new_witness_sha256'] = m.digest(plan)
            rc = dict(predict_attempted=27, predict_completed=27, jacobian_attempted=1, jacobian_completed=1,
                      requested_AD_primal=3, requested_J_elements=5760, requested_VJP=192,
                      encoder=0, projector=0, physics=0, SVD=0, solve=0, FLOPs=None, engine_backward=None,
                      modules_completed=dict(action_encoder=27, predictor=27, pred_proj=27))
            row = dict(slot=slot, index=bi//2, arm=b['arm'], base_index=bi, policy=policy,
                       status='sealed', result=result, plan=deepcopy(plan), resources=rc, forwards=[],
                       AD=[dict(completed=True, contexts=es[0]['contexts'],
                                request=dict(at_logits=j['incumbent_logits'], input_context_sha256=m.digest(j['input_context'])))])
            paths = [('incumbent', es[0]['logits'], es[0]['contexts'], j['input_context']),
                     ('raw_J/forward', es[0]['logits'], es[0]['contexts'], j['input_context'])]
            paths += [(f'candidate{i}', es[i]['logits'], es[i]['contexts'], j['input_context']) for i in range(1, 5)]
            paths += [(tag, w['logits'], w['contexts'], j['input_context']) for tag in ('selected_replay', 'whole15')]
            paths += [('segment5', w['logits'][:10], w['contexts'][:1], j['input_context']),
                      ('segment10', w['logits'][10:], w['contexts'][1:], w['contexts'][0])]
            for tag, logits, cs, initial in paths:
                prev = initial
                packets = np.tanh(np.asarray(logits)).astype('<f4').reshape(-1, 5, 2)
                for k, (a, output) in enumerate(zip(packets, cs)):
                    n = len(row['forwards'])
                    desc = dict(dtype='float32', shape=[5, 2], raw_hex=a.tobytes().hex(), sha256=hashlib.sha256(a.tobytes()).hexdigest())
                    row['forwards'].append(dict(index=n, tag=f'{tag}/{k}', completed=True, input=prev, output=output,
                        raw=desc, modules=[dict(resource=name, completed=True) for name in ('action_encoder', 'predictor', 'pred_proj')],
                        request=dict(index=n, tag=f'{tag}/{k}', completed=False,
                                     input_context_sha256=m.digest(prev), raw_sha256=m.digest(desc))))
                    prev = output
            rows.append(row)
            r['rows'].append(dict(slot=slot, root_slot=bi//2, arm=b['arm'], base_index=bi, policy=policy, status='sealed',
                                  file=f'slot{slot}.json.gz', wall_seconds=.1, contexts=result['contexts'], new_suffix_Q=w['Q'], resources=deepcopy(rc)))
            s['plans'].append(deepcopy(plan))
            s['policy_slots'].append(dict(base_index=bi, policy=policy, status='new_model_plan', plan=deepcopy(plan)))
            r['policy_roster'].append(dict(base_index=bi, policy=policy, status='sealed', model_job_slot=slot))
    # Persisted JSON cannot share Python object references across independent paths.
    # Round-trip the fixture so a single forward mutation cannot silently alter AD/evaluation too.
    return json.loads(m.raw([p, r, c, s, rows]))


def build(f):
    return m.build(*f, 'p', 'r', 'c', '/fixture/prepare.json')


@pytest.mark.parametrize('zero', [False, True])
def test_all8_all12_all40_complete(zero):
    a = build(fixture(zero))
    assert a['denominator'] == dict(known_roots=2, bases=4, jobs=8, policies=12, candidate_Q=40)
    assert len(a['jobs']) == 8 and len(a['policies']) == 12
    assert all(len(j['candidates']) == 5 for j in a['jobs'])
    assert all(j['zero_selected_step'] == zero for j in a['jobs'])
    assert all(j['original_selected'] == (0 if zero else 1) for j in a['jobs'])
    assert all(p['new_model_Q'] is None for p in a['policies'] if p['policy'] == 'O')
    assert not a['P2_admission'] and not a['fresh'] and a['new_model_calls'] == 0
    assert a['jobs'][0]['input_context_sha256'] != a['jobs'][1]['input_context_sha256']
    assert 'I/F different-input' in a['comparison_scope']


@pytest.mark.parametrize('mutation', [
    lambda f: f[0]['jobs'].pop(),
    lambda f: f[3]['policy_slots'].pop(),
    lambda f: f[0]['jobs'][0].update(policy='F'),
    lambda f: f[0]['jobs'][0].update(parent_raw_bits='00'),
    lambda f: f[0]['jobs'][0]['prefix_raw'][0].__setitem__(0, .1),
    lambda f: f[0]['jobs'][0]['input_context']['latent_history'][0].__setitem__(0, .25),
    lambda f: f[4][0]['result']['evaluations'][0].update(Q=99.),
    lambda f: f[4][0]['result'].update(selected=4),
    lambda f: f[4][0]['result']['proposals']['logits'].pop(),
    lambda f: f[4][0]['result']['proposals']['direction'].__setitem__(0, 1.),
    lambda f: f[4][0]['result']['commands'][0].__setitem__(0, 1.),
    lambda f: f[3]['plans'][0].update(new_witness_sha256='bad'),
    lambda f: f[3]['plans'][0].update(task_sha256='other'),
    lambda f: f[4][0]['forwards'][0]['request'].update(raw_sha256='bad'),
    lambda f: f[4][0]['forwards'][26].update(completed=False),
    lambda f: f[4][0]['forwards'][24].update(tag='whole15/0'),
    lambda f: f[4][0]['forwards'][5]['output']['action_history'][0].__setitem__(0, 2.),
    lambda f: f[4][0]['AD'][0]['request'].update(input_context_sha256='bad'),
    lambda f: f[4][0]['resources'].update(FLOPs=0),
    lambda f: f[1]['actual_resources'].update(predict_API_returned=215),
    lambda f: f[1].update(source_verified=False),
    lambda f: f[1].update(model_after='new'),
    lambda f: f[2].update(boundary='before_report'),
    lambda f: f[3].update(completion_sha256='wrong'),
    lambda f: f[3].update(P2_admission=True),
    lambda f: f[3].update(own_native_saved_reconstructed_here=False),
])
def test_fail_closed_mutations(mutation):
    f = deepcopy(fixture())
    mutation(f)
    with pytest.raises((ValueError, KeyError, IndexError)):
        build(f)


def test_fsum_and_numpy_separate_and_ties():
    c = ctx(0.)
    c['latent_history'][-1] = [float(np.float32(1.e8))]+[1.]*191
    q, n = m.scores(c, [0.]*192)
    assert q != n  # reduction rounding is deliberately not a forced equality
    assert q == __import__('math').fsum([1.e16]+[1.]*191)/192
    assert [m.sign(v) for v in (-1.e-300, 0., 1.e-300)] == ['negative', 'tie', 'positive']
    assert all(j['original_selected'] == 0 and j['gain_sign'] == 'tie' for j in build(fixture(True))['jobs'])


def test_json_codec_is_compact_newline_not_default():
    assert m.digest([1, 2]) != hashlib.sha256(json.dumps([1, 2]).encode()).hexdigest()
    assert m.raw([1, 2]) == b'[1,2]\n'


def persist(tmp_path):
    p, r, c, s, rows = fixture()
    paths = {k:str(tmp_path/(k+'.json')) for k in m.PINS}
    hashes = {}
    r['prepared'] = paths['prepare']
    for i, row in enumerate(rows):
        data = gzip.compress(m.raw(row), mtime=0)
        (tmp_path/f'slot{i}.json.gz').write_bytes(data)
        r['artifact_sha256'][f'slot{i}.json.gz'] = hashlib.sha256(data).hexdigest()
    for k, x in [('prepare', p), ('report', r), ('completion', c), ('receipt', s)]:
        if k == 'report':
            r['prepare_sha256'] = hashes['prepare']
        elif k == 'completion':
            c['report_sha256'] = hashes['report']
        elif k == 'receipt':
            s.update(prepare_sha256=hashes['prepare'], report_sha256=hashes['report'], completion_sha256=hashes['completion'])
        data = m.raw(x)
        __import__('pathlib').Path(paths[k]).write_bytes(data)
        hashes[k] = hashlib.sha256(data).hexdigest()
    return paths, hashes


def test_real_file_writer_reader_whole_rebuild(tmp_path, monkeypatch, capsys):
    paths, hashes = persist(tmp_path)
    # Only four authority files and exactly eight gzip slots may be read.
    got = m.analyze(paths, hashes)
    assert len(got['jobs']) == 8
    out = tmp_path/'summary.json'
    argv = ['summary', 'analyze']
    for k in m.PINS:
        argv += ['--'+k, paths[k], '--'+k+'-sha', hashes[k]]
    monkeypatch.setattr('sys.argv', argv+['--output', str(out)])
    m.main()
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    monkeypatch.setattr('sys.argv', [argv[0], 'verify']+argv[2:]+['--output', str(out), '--summary-sha', sha])
    m.main()
    a = m.decode(out.read_bytes()); a['jobs'][0]['original_selected'] = 99
    out.write_bytes(m.raw(a))
    monkeypatch.setattr('sys.argv', [argv[0], 'verify']+argv[2:]+['--output', str(out), '--summary-sha', hashlib.sha256(out.read_bytes()).hexdigest()])
    with pytest.raises(ValueError, match='whole independently'):
        m.main()


def test_gzip_and_sha_tampering(tmp_path):
    paths, hashes = persist(tmp_path)
    path = tmp_path/'slot0.json.gz'
    path.write_bytes(path.read_bytes()+gzip.compress(b'{}'))
    rpath = tmp_path/'report.json'
    r = m.decode(rpath.read_bytes())
    r['artifact_sha256']['slot0.json.gz'] = hashlib.sha256(path.read_bytes()).hexdigest()
    # Direct reader checks trailing members even if caller pins their byte hash.
    with pytest.raises(ValueError, match='one complete bounded gzip'):
        m.read(str(path), r['artifact_sha256']['slot0.json.gz'], True)
    with pytest.raises(ValueError, match='saved byte SHA'):
        m.analyze(paths, dict(hashes, receipt='0'*64))


@pytest.mark.parametrize('payload', [b'{"x":1,"x":2}', b'{"x":NaN}'])
def test_strict_json(payload):
    with pytest.raises(ValueError):
        m.decode(payload)


def test_absolute_paths_and_f32_not_repaired():
    with pytest.raises(ValueError, match='canonical absolute'):
        m.canonical('relative.json')
    with pytest.raises(ValueError, match='literal float32'):
        m.array([.1], (1,), True)
    with pytest.raises(ValueError, match='no bool'):
        m.array([True], (1,))
