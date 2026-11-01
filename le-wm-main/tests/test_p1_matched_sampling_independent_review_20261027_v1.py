"""Independent pure-array review. No real torch/model/simulator execution."""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import gzip
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import p1_matched_sampling_cuda_20261027_v1 as m


def task():
    raw=np.linspace(-.3,.3,20,dtype=np.float32).reshape(10,2)
    t=dict(environment='array_mock', task_contract_sha256='f'*64,
        root_raw_action_history=raw.tolist(), goal=[0.]*192,
        projection=np.eye(192).tolist(), model_sha256='a'*64,
        model_config_sha256='b'*64, metric_source_sha256='c'*64,
        goal_source_sha256='d'*64, representation='native_projected_cls',
        latent_dim=192, horizon=2, packet_steps=5, control_dim=2,
        action_mean=[.1, -.2], action_std=[.2, .25],
        packet_logit_cap=float(.75*np.sqrt(5)), feedback='open_loop_model_only',
        root_context=dict(layout='states_with_between_state_actions_v1', context_length=3,
            latent_dim=192, action_dim=10,
            latent_history=np.full((3, 192), .125).tolist(),
            action_history=np.zeros((2,10)).tolist()))
    t['root_context']['action_history']=m.core.normalized_tokens(t,raw).tolist()
    return t


class ArrayBackend:
    @contextmanager
    def guard(self, meter):
        yield

    def rollout(self, t, raw, meter, mode):
        c=deepcopy(t['root_context']); paths=[]
        for token in m.core.normalized_tokens(t, raw):
            meter.call('action_encoder', lambda: None)
            # Both old histories influence successor, detecting truncation/padding.
            def advance():
                z=np.asarray(c['latent_history']).sum(axis=0)*.125
                z[:10] += np.asarray(c['action_history']).sum(axis=0)*.01 + token*.1
                return z.tolist()
            z=meter.call('predict', advance)
            c=dict(c, latent_history=c['latent_history'][1:]+[z],
                   action_history=c['action_history'][1:]+[token.tolist()])
            paths.append(deepcopy(c))
        return paths


@pytest.mark.parametrize('method', m.METHODS)
def test_independent_full_history_and_exact_phase_work(method):
    events=[]; t=task()
    r=m.run_arm(t, method, m.seed_for(1, method), ArrayBackend(), events.append)
    assert r['status']=='sampling_completed_not_admitted', r['error']
    cs=[e['candidate'] for e in events if e['kind']=='candidate_complete']
    assert len(cs)==60
    for c in cs:
        before=t['root_context']
        for k, after in enumerate(c['contexts']):
            raw=np.asarray(c['commands'][k*5:k*5+5], np.float32)
            token=m.core.normalized_tokens(t, raw)[0]
            z=np.asarray(before['latent_history']).sum(axis=0)*.125
            z[:10]+=np.asarray(before['action_history']).sum(axis=0)*.01+token*.1
            assert after['latent_history']==before['latent_history'][1:]+[z.tolist()]
            assert after['action_history']==before['action_history'][1:]+[token.tolist()]
            before=after
    assert r['winner']==min(cs, key=lambda c:(c['cost'], c['index']))
    for phase, caps in m.budgets(method).items():
        assert r['resources'][phase]=={k:dict(attempted=v, completed=v) for k,v in caps.items()}
    assert sum(r['resources'][p]['predict']['completed'] for p in r['resources'])==126


@pytest.mark.parametrize('method', m.METHODS)
def test_population_draws_zero_and_seed_table(method):
    cap=float(.75*np.sqrt(5)); mean=np.zeros((2,10)); sigma=mean+.5
    seed=20261027+(100 if method=='iid_random' else 0)
    assert m.seed_for(0, method)==seed and m.seed_for(1, method)==seed+1000
    a=np.random.Generator(np.random.PCG64(seed)); b=np.random.Generator(np.random.PCG64(seed))
    all_values=[]
    for batch in range(6):
        n=9 if batch==0 else 10
        if method=='cem':
            expected=b.normal(mean, sigma, (n,2,10))
        else:
            d=b.normal(size=(n,2,10))
            expected=d/np.linalg.norm(d, axis=-1, keepdims=True)*(cap*b.random((n,2,1))**.1)
        expected=[m.screen.projected_logits(v, cap) for v in expected]
        if batch==0: expected.insert(0, np.zeros((2,10)))
        actual=m.population(method, a, mean, sigma, batch, cap)
        assert np.array_equal(actual, expected)
        all_values.extend(actual)
    values=np.asarray(all_values)
    assert len(values)==60 and not values[0].any()
    assert np.all(np.any(values[1:]!=0, axis=(1,2)))
    assert (np.linalg.norm(values, axis=-1)<=cap).all()
    assert a.bit_generator.state==b.bit_generator.state


def test_independent_cem_stable_elite_and_population_std():
    a=np.arange(200).reshape(10,2,10)/200
    costs=np.array([2,1,1,1,3,4,5,6,7,8])
    mean, sigma, ids=m.cem_update(a, costs)
    assert ids==[1,2,3]
    assert np.array_equal(mean, a[[1,2,3]].mean(axis=0))
    assert np.array_equal(sigma, np.maximum(a[[1,2,3]].std(axis=0,ddof=0), .05))


@pytest.mark.xfail(strict=True, reason='R1: arm report write failure omits durable journal identity/hash from final report')
def test_failed_arm_report_still_indexes_existing_journal(tmp_path, monkeypatch):
    """Fake runtime only; inject failure AFTER durable arm journal is created."""
    runtime=ModuleType('torch')
    runtime.device=lambda *a: 'mock'
    runtime.backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)))
    runtime.cuda=SimpleNamespace(synchronize=lambda: None)
    runtime.inference_mode=nullcontext
    attention=ModuleType('torch.nn.attention')
    attention.SDPBackend=SimpleNamespace(MATH='mock')
    attention.sdpa_kernel=lambda *a: nullcontext()
    checkpoint=ModuleType('experiments.checkpoint_compat')
    model=SimpleNamespace()
    model.eval=lambda: model
    model.requires_grad_=lambda flag: model
    checkpoint.load_compatible_pretrained=lambda *a, **kw: model
    preflight=ModuleType('experiments.gpu_preflight')
    preflight.require_cuda_allocation=lambda *a: None
    for name, mod in [('torch',runtime),('torch.nn.attention',attention),
            ('experiments.checkpoint_compat',checkpoint),('experiments.gpu_preflight',preflight)]:
        monkeypatch.setitem(sys.modules,name,mod)
    monkeypatch.setattr(m.io,'frozen',lambda model: None)
    monkeypatch.setattr(m.io,'tensor_hash',lambda model: 'a'*64)
    t=task()
    p=dict(inputs=dict(policy='mock',tasks=[dict(task=t,root_sha256='b'*64,task_id=m.core.task_id(t))],
        inherited={},source_purpose='calibration_exploration',history_semantics='mock'))
    monkeypatch.setattr(m,'verify',lambda *a: p)
    monkeypatch.setattr(m.screen,'NativeCuda',lambda *a: ArrayBackend())
    original=m.Store.put
    def put(store, name, value, **kwargs):
        if name=='root_0_cem.json': raise OSError('injected_arm_report_write')
        return original(store,name,value,**kwargs)
    monkeypatch.setattr(m.Store,'put',put)
    out=tmp_path/'output'
    r=m.run(tmp_path/'prepared','a'*64,out,authorize_host_cuda=True)
    assert r['status']=='partial_or_failed'
    assert 'injected_arm_report_write' in r['errors'][0]['message']
    journal=out/'root_0_cem.jsonl.gz'
    assert journal.exists() and b'candidate_complete' in gzip.decompress(journal.read_bytes())
    assert r['resources']['predict']==dict(attempted=126,completed=126)
    assert m.io.read(out/'report.json')==r
    assert r['arms'][0]['journal_file']==journal.name
    assert r['arms'][0]['journal_sha256']==m.io.sha(journal)
