"""Independent saved CUDA FD arithmetic/provenance checks; no runner imports.

Only JSON, hashes and NumPy; tolerances compare arithmetic receipts, never
define a scientific FD/J pass criterion. No model, AD or physics execution.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'results/p1_direct_fd_cuda_20261027_v1_cuda_main01'
PREPARED = ROOT/'results/p1_direct_fd_cuda_20261027_v1_prepared01'
PREPARE_SHA = 'fdb4c376c574ed61915e88b569980330edf6fa89dd974e8fa9033b9da9af149c'
REPORT_SHA = 'd77f7aadf3fd3b6bd7da8c32ce499e5dc486a6989b0c178622aba4ff4478eaa1'
ANALYSIS_SHA = '4091379f9f179a28338ca8e4b65f48b59e9ce2eeb5c76226eff662e271b064dd'
JOURNAL_SHA = '1c96ccc2a324abf36cbe399cd2511cd7940866c9bee8c039186f2935fda0a0c8'
EPS = (1e-3, 1e-2)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for raw in iter(lambda: stream.read(1024**2), b''):
            h.update(raw)
    return h.hexdigest()


def read(path, expected=None):
    path = Path(path)
    assert path.is_file() and not path.is_symlink() and path.stat().st_size < 16*1024**2
    if expected is not None:
        assert sha(path) == expected
    return json.loads(path.read_text())


def identity(task):
    return hashlib.sha256(json.dumps(task, sort_keys=True, allow_nan=False).encode()).hexdigest()


def equal(a, b):
    # Serialization/arithmetic consistency only, NOT a derivative accuracy gate.
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)


def finite(a, shape):
    a = np.asarray(a, dtype=float)
    assert a.shape == shape and np.isfinite(a).all()
    return a


def fixed_directions():
    e0 = np.eye(20)[0]; e10 = np.eye(20)[10]
    rng = np.random.Generator(np.random.PCG64(20261027))
    v = rng.standard_normal(20); v /= np.linalg.norm(v)
    return [(name, x.astype(np.float32).astype(float)) for name, x in (
        ('packet0_e0', e0), ('packet1_e0', e10),
        ('equal_packet_e0', (e0+e10)/np.sqrt(2)), ('seed20261027_unitrandom', v))]


def load_saved():
    prepare = read(PREPARED/'prepare.json', PREPARE_SHA)
    protocol = read(PREPARED/'protocol.json', prepare['protocol_sha256'])
    report = read(RUN/'report.json', REPORT_SHA)
    analysis = read(RUN/'analysis.json', ANALYSIS_SHA)
    assert sha(RUN/'fd.events.jsonl') == JOURNAL_SHA == report['journal_sha256']
    assert report['analysis_sha256'] == ANALYSIS_SHA
    binding = read(RUN/'execution_binding.json')
    assert binding == dict(prepared=str(PREPARED), prepare_sha256=PREPARE_SHA)
    assert report['prepare_sha256'] == PREPARE_SHA
    for path, expected in {**protocol['sources_sha256'], **protocol['input_bindings'],
            **protocol['original_derivative_source_bindings'], **protocol['model_bindings']}.items():
        assert sha(path) == expected, path
    evaluations = [read(RUN/f'eval_{i:02d}.json') for i in range(32)]
    assert {p.name for p in RUN.glob('eval_*.json')} == {f'eval_{i:02d}.json' for i in range(32)}
    events = [json.loads(line) for line in (RUN/'fd.events.jsonl').read_text().splitlines()]
    originals = []
    for root in protocol['roots']:
        c = read(root['original_component'], protocol['input_bindings'][root['original_component']])
        path = Path(root['original_journal'])
        assert sha(path) == protocol['input_bindings'][str(path)]
        originals.append(dict(component=c, events=[json.loads(x) for x in path.read_text().splitlines()]))
    return dict(protocol=protocol, report=report, analysis=analysis,
                evaluations=evaluations, events=events, originals=originals)


def check_original(root, original):
    task = root['task']; c = original['component']; events = original['events']
    assert identity(task) == root['task_id'] == c['task_id']
    assert events[0]['task'] == task
    run = c['direct_runs'][0]; zero = run[0]['incumbent']; derivative = run[1]['derivative']
    assert run[0]['iteration'] == 0 and run[0]['derivative'] is None
    assert np.array_equal(zero['logits'], np.zeros((2,10)))
    assert np.array_equal(zero['commands'], np.zeros((10,2)))
    assert derivative['at_logits'] == zero['logits']
    assert derivative['at_terminal'] == zero['contexts'][-1]['latent_history'][-1] == root['nominal_terminal']
    first_joint = next(i for i,e in enumerate(events) if e['kind']=='joint_logit_jacobian')
    raw_rows = [e for e in events[:first_joint] if e['kind']=='jacobian_row']
    assert [e['row'] for e in raw_rows] == list(range(192))
    raw_j = finite([e['raw_control_derivative'] for e in raw_rows], (192,20))
    j = finite(derivative['raw_terminal_jacobian'], (192,20))
    np.testing.assert_array_equal(j, raw_j)  # tanh chain AT ZERO is identity
    np.testing.assert_array_equal(j, root['logit_jacobian'])
    np.testing.assert_array_equal(j, events[first_joint]['jacobian'])
    packets = [e for e in events[:first_joint] if e['kind']=='jacobian_forward_packet']
    assert [e['packet'] for e in packets] == [0,1]
    assert [e['context'] for e in packets] == zero['contexts']
    return j


def check_contexts(task, raw, contexts):
    before = task['root_context']
    tokens = (raw.reshape(2,10)-np.tile(np.asarray(task['action_mean'],np.float32),5))/np.tile(np.asarray(task['action_std'],np.float32),5)
    assert len(contexts) == 2
    for k, c in enumerate(contexts):
        assert {key:c[key] for key in ('layout','context_length','latent_dim','action_dim')} == dict(
            layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10)
        z = finite(c['latent_history'], (3,192))
        np.testing.assert_array_equal(z[:2], np.asarray(before['latent_history'],np.float32)[1:])
        expected = np.concatenate((np.asarray(before['action_history'],np.float32),tokens[k:k+1]))[-2:]
        np.testing.assert_array_equal(c['action_history'], expected)
        before = c


def audit(saved):
    p,r,a = saved['protocol'],saved['report'],saved['analysis']
    assert r['status']=='cuda_fd_completed' and r['first_error'] is None and r['errors']==[]
    assert r['source_verified'] is r['model_unchanged'] is True
    assert r['model_hash_before']==r['model_hash_after']==p['expected_model_tensor_sha256']
    assert r['evaluations_completed']==32
    assert r['resources']=={k:dict(attempted=64,completed=64) for k in ('predict','action_encoder')}
    assert all(r[k]==0 for k in ('encoder_calls','projector_calls','AD_calls','physics_steps'))
    assert r['training'] is r['new_Q'] is r['P1_admission'] is False
    assert r['scientific_pass_threshold'] is a['scientific_pass_threshold'] is None
    assert r['scientific_verdict']==a['scientific_verdict']=='not_assigned'
    assert len(p['roots'])==2 and len({x['root_sha256'] for x in p['roots']})==2
    directions=fixed_directions()
    assert [(d['id'],d['vector']) for d in p['directions']] == [(n,v.tolist()) for n,v in directions]
    assert p['epsilons']==list(EPS) and p['signs']==[-1,1]
    order=[(root,name,v,eps,sign) for root in p['roots'] for name,v in directions for eps in EPS for sign in (-1,1)]
    assert len(saved['events'])==32*12
    terminals={}; ledger=[]
    for i,(root,name,v,eps,sign) in enumerate(order):
        e=saved['evaluations'][i]
        assert (e['order'],e['root_index'],e['root_sha256'],e['task_id'],e['direction'],e['epsilon_requested'],e['sign']) == (
            i,root['root_index'],root['root_sha256'],root['task_id'],name,eps,sign)
        logits=(v.astype(np.float32)*np.float32(eps)*np.float32(sign)).reshape(2,10)
        raw=np.tanh(logits).reshape(10,2)
        np.testing.assert_array_equal(e['logits'],logits)
        np.testing.assert_array_equal(e['commands'],raw)
        assert e['epsilon_float32']==float(np.float32(eps))
        check_contexts(root['task'],raw,e['contexts'])
        terminals[root['root_index'],name,eps,sign]=finite(e['contexts'][-1]['latent_history'][-1],(192,))
        events=saved['events'][12*i:12*(i+1)]
        assert events[0]==dict(kind='evaluation_attempt',evaluation={k:v for k,v in e.items() if k!='contexts'})
        pos=1
        for packet in (0,1):
            for resource in ('action_encoder','predict'):
                entry=dict(index=len(ledger),resource=resource,samples=1,completed=True,evaluation_order=i,packet=packet)
                assert events[pos]==dict(kind='module_attempt',entry=dict(entry,completed=False))
                assert events[pos+1]==dict(kind='module_return',entry=entry)
                ledger.append(entry);pos+=2
            assert events[pos]==dict(kind='native_packet',evaluation_order=i,packet=packet,mode='whole_replay',context=e['contexts'][packet]);pos+=1
        assert events[-1]==dict(kind='evaluation_saved',order=i,file=f'eval_{i:02d}.json',sha256=sha(RUN/f'eval_{i:02d}.json'))
    assert ledger==r['module_entries'] and len(ledger)==128
    recomputed=[]
    for root, old, row_group in zip(p['roots'],saved['originals'],(a['rows'][:4],a['rows'][4:])):
        j=check_original(root,old)
        for (name,v), claimed in zip(directions,row_group):
            assert (claimed['root_index'],claimed['root_sha256'],claimed['task_id'],claimed['direction']) == (root['root_index'],root['root_sha256'],root['task_id'],name)
            expected_jv=j@v;jnorm=float(np.linalg.norm(expected_jv));fds=[]
            for eps, c in zip(EPS,claimed['epsilons']):
                plus,minus=(terminals[root['root_index'],name,eps,s] for s in (1,-1))
                fd=(plus-minus)/(2*float(np.float32(eps)));difference=fd-expected_jv;fnorm=float(np.linalg.norm(fd))
                metrics=dict(absolute_l2=float(np.linalg.norm(difference)),absolute_linf=float(np.max(np.abs(difference))),
                    absolute_rmse=float(np.sqrt(np.mean(difference**2))),saved_jv_l2=jnorm,fd_l2=fnorm,
                    relative_l2=float(np.linalg.norm(difference)/jnorm) if jnorm else None,
                    symmetric_relative_l2=float(np.linalg.norm(difference)/max(jnorm,fnorm)) if max(jnorm,fnorm) else None,
                    cosine=float(np.dot(fd,expected_jv)/(fnorm*jnorm)) if fnorm and jnorm else None,
                    midpoint_drift_from_saved_nominal_l2=float(np.linalg.norm((plus+minus)/2-root['nominal_terminal'])),
                    direction_norm_float32=float(np.linalg.norm(v)))
                assert c['epsilon_requested']==eps and c['epsilon_float32']==float(np.float32(eps))
                equal(c['fd'],fd);equal(c['saved_jv'],expected_jv)
                for k,value in metrics.items():
                    if value is None:assert c[k] is None
                    else:equal(c[k],value)
                fds.append(fd)
                recomputed.append(dict(root=root['root_index'],direction=name,epsilon=eps,**metrics))
            stability=float(np.linalg.norm(fds[0]-fds[1]))
            equal(claimed['epsilon_stability_l2'],stability)
            equal(claimed['epsilon_stability_relative_to_saved_jv'],stability/jnorm)
    assert len(a['rows'])==8 and a['comparisons']==16 and a['epsilon_stability_pairs']==8
    return recomputed


@pytest.fixture(scope='module')
def saved():return load_saved()


def test_actual_all_raw_eval_journal_jv_errors_and_epsilon_stability(saved):
    rows=audit(saved)
    assert len(rows)==16


def test_recomputed_error_ranges_without_scientific_threshold(saved):
    rows=audit(saved)
    expected={.001:(.001354660614166085,.0025700692872082024,.0002210074917858992,.0008234180181454952),
              .01:(.0006828963865910734,.002727013574918472,.00022287025067457426,.00041974519042571167)}
    for eps,(lo,hi,rlo,rhi) in expected.items():
        subset=[r for r in rows if r['epsilon']==eps]
        equal([min(x['absolute_l2'] for x in subset),max(x['absolute_l2'] for x in subset)], [lo,hi])
        equal([min(x['relative_l2'] for x in subset),max(x['relative_l2'] for x in subset)], [rlo,rhi])
    # Descriptive difference in epsilon behavior, not choose-the-best-eps tuning.
    for root in (0,1):
        for direction,_ in fixed_directions():
            pair=[x['absolute_l2'] for x in rows if x['root']==root and x['direction']==direction]
            assert (pair[1]<pair[0]) is (root==0)


@pytest.mark.parametrize('corruption',['fd','jv','relative','stability','raw','history','sign','task','module_count','oldJ'])
def test_memory_only_counterexamples_are_rejected(saved,corruption):
    bad=deepcopy(saved)
    c=bad['analysis']['rows'][0]
    if corruption=='fd':c['epsilons'][0]['fd'][0]+=.1
    elif corruption=='jv':c['epsilons'][0]['saved_jv'][0]+=.1
    elif corruption=='relative':c['epsilons'][0]['relative_l2']+=.1
    elif corruption=='stability':c['epsilon_stability_l2']+=.1
    elif corruption=='raw':bad['evaluations'][0]['commands'][0][0]+=.1
    elif corruption=='history':bad['evaluations'][0]['contexts'][1]['action_history'][0][0]+=.1
    elif corruption=='sign':bad['evaluations'][0]['sign']=1
    elif corruption=='task':bad['evaluations'][0]['task_id']='other'
    elif corruption=='module_count':bad['report']['module_entries'][0]['completed']=False
    else:bad['originals'][0]['component']['direct_runs'][0][1]['derivative']['raw_terminal_jacobian'][0][0]+=.1
    with pytest.raises(AssertionError):audit(bad)


if __name__=='__main__':
    rows=audit(load_saved())
    summary={str(eps):{metric:[min(r[metric] for r in rows if r['epsilon']==eps),
        max(r[metric] for r in rows if r['epsilon']==eps)] for metric in ('absolute_l2','relative_l2','cosine')} for eps in EPS}
    print(json.dumps(dict(actual_saved_recomputed=True,ranges=summary,scientific_pass_threshold=None),sort_keys=True))
