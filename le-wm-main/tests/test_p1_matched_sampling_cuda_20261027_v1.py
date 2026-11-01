"""NumPy/mock only. No torch model, GPU, simulator, or efficacy assertions."""
from contextlib import contextmanager
from copy import deepcopy
import gzip
import inspect
from pathlib import Path
from types import SimpleNamespace
import zlib
import numpy as np
import pytest

from experiments.spectral_reachability import p1_matched_sampling_cuda_20261027_v1 as m


def task():
    t=dict(environment='array_mock',task_contract_sha256='f'*64,
        root_raw_action_history=np.zeros((10,2)).tolist(),goal=[.15]*10+[0.]*182,
        projection=np.eye(192).tolist(),model_sha256='a'*64,model_config_sha256='b'*64,
        metric_source_sha256='c'*64,goal_source_sha256='d'*64,representation='native_projected_cls',
        latent_dim=192,horizon=2,packet_steps=5,control_dim=2,action_mean=[0.,0.],action_std=[.2,.25],
        packet_logit_cap=float(.75*np.sqrt(5)),feedback='open_loop_model_only')
    t['root_context']=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,
        action_dim=10,latent_history=np.zeros((3,192)).tolist(),action_history=np.zeros((2,10)).tolist())
    return t


class Backend:
    def __init__(self,fault=None):self.calls=[];self.fault=fault;self.closed=False

    @contextmanager
    def guard(self,meter):
        try:yield
        finally:self.closed=True

    def rollout(self,t,raw,meter,mode):
        self.calls.append(mode);c=deepcopy(t['root_context']);out=[]
        for k,token in enumerate(m.core.normalized_tokens(t,raw)):
            meter.call('action_encoder',lambda:None)
            def predict():
                if self.fault=='predict':raise RuntimeError('primary_predict')
                z=.7*np.asarray(c['latent_history'][-1]);z[:10]+=token
                return z.tolist()
            z=meter.call('predict',predict)
            c=dict(c,latent_history=c['latent_history'][1:]+[z],action_history=c['action_history'][1:]+[token.tolist()])
            if self.fault=='chronology':c['action_history'][-1][0]+=1
            meter.emit('mock_packet',context=c)
            out.append(deepcopy(c))
        if self.fault=='replay' and mode=='whole_replay':out[-1]['latent_history'][-1][0]+=1
        return out


@pytest.fixture(scope='module',params=m.METHODS)
def complete(request):
    events=[];b=Backend();t=task();r=m.run_arm(t,request.param,m.seed_for(0,request.param),b,events.append)
    return t,r,events,b


def test_exact_full_budget_all_slots(complete):
    t,r,e,b=complete
    assert r['status']=='sampling_completed_not_admitted' and r['error'] is None and b.closed
    assert len(r['candidates'])==60
    assert b.calls==['candidate']*60+['nominal','whole_replay','segmented_replay']
    for phase,caps in m.budgets(r['method']).items():
        for resource,n in caps.items():assert r['resources'][phase][resource]==dict(attempted=n,completed=n)
    assert sum(v['predict']['completed'] for v in r['resources'].values())==126
    assert not r['P1_admission']


def test_exact_candidate_winner_original_cost_contexts(complete):
    t,r,events,_=complete
    cs=[e['candidate'] for e in events if e['kind']=='candidate_complete']
    assert len(cs)==60 and r['winner']==min(cs,key=lambda c:(c['cost'],c['index']))
    m.core.check_prediction(t,r['witness'])
    for c in cs:
        assert c['cost']==m.core.model_cost(t,c['contexts'][-1]['latent_history'][-1])
        assert c['commands']==np.tanh(np.asarray(c['logits'],np.float32)).reshape(10,2).tolist()
        assert c['commands']==m.bridge.edge(t,t['root_context'],c['logits'][0],c['contexts'][0])+m.bridge.edge(t,c['contexts'][0],c['logits'][1],c['contexts'][1])
    assert r['winner']['contexts']==r['witness']['contexts']


def test_zero_only_first_and_float32_packet_ball(complete):
    t,r,_,_=complete
    a=np.asarray([c['logits'] for c in r['candidates']])
    assert np.array_equal(a,a.astype(np.float32))
    assert (np.linalg.norm(a,axis=-1)<=t['packet_logit_cap']).all()
    assert not a[0].any() and np.all(np.any(a[1:]!=0,axis=(1,2)))
    assert len({m.core.digest(x.tolist()) for x in a})==60


def test_cem_updates_are_batch_elite_empirical_not_softmax(complete):
    _,r,events,_=complete
    updates=[e for e in events if e['kind']=='elite_update']
    assert len(updates)==(6 if r['method']=='cem' else 0)
    if updates:
        for batch,u in enumerate(updates):
            cs=r['candidates'][batch*10:batch*10+10]
            mean,std,ids=m.cem_update([c['logits'] for c in cs],[c['cost'] for c in cs])
            assert u['elite_slots']==ids and u['mean']==mean.tolist() and u['sigma']==std.tolist()
            if batch<5:
                p=next(e for e in events if e['kind']=='population_frozen' and e['batch']==batch+1)
                assert p['mean']==u['mean'] and p['sigma']==u['sigma']


def test_candidate_arrays_durable_before_selection(complete):
    _,_,events,_=complete
    for i,e in enumerate(events):
        if e['kind']=='candidate_complete':
            assert events[i+1]['kind']=='operation_attempted'
            assert events[i+1]['event']['resource']=='selection'


def test_cem_stable_ties_ddof0_floor():
    values=np.arange(200,dtype=float).reshape(10,2,10)/100
    mean,std,ids=m.cem_update(values,np.zeros(10))
    assert ids==[0,1,2]
    np.testing.assert_array_equal(mean,values[:3].mean(axis=0))
    np.testing.assert_array_equal(std,np.maximum(values[:3].std(axis=0,ddof=0),.05))
    _,std,_=m.cem_update(np.zeros_like(values),np.arange(10))
    assert np.all(std==.05)


@pytest.mark.parametrize('method',m.METHODS)
def test_reproducible_stream_seed_and_no_mean_injection(method):
    def draw(seed):
        rng=np.random.Generator(np.random.PCG64(seed));mean=np.zeros((2,10));std=np.full((2,10),.5)
        return np.asarray([m.population(method,rng,mean,std,b,float(.75*np.sqrt(5))) for b in range(6)])
    a=draw(m.seed_for(0,method));assert np.array_equal(a,draw(m.seed_for(0,method)))
    assert not np.array_equal(a,draw(m.seed_for(1,method)))
    assert np.any(a[1:,0]!=0,axis=(1,2)).all()


def test_uniform_ball_radius_moment():
    rng=np.random.default_rng(7);cap=task()['packet_logit_cap']
    samples=np.asarray([m.population('iid_random',rng,np.zeros((2,10)),np.ones((2,10)),1,cap) for _ in range(200)])
    assert abs(np.mean(np.linalg.norm(samples,axis=-1)**2)/cap**2-10/12)<.015


@pytest.mark.parametrize('method',m.METHODS)
def test_zero_is_retained_incumbent_on_all_positive_mock_costs(method):
    t=task();t['goal']=[0.]*192
    r=m.run_arm(t,method,m.seed_for(0,method),Backend(),lambda e:None)
    assert r['winner']['index']==0 and len(r['candidates'])==60


@pytest.mark.parametrize('resource',['jacobian','vjp_directions','svd','linear_solve','encoder','projector','unknown'])
def test_forbidden_resources_predispatch(resource):
    meter=m.Meter('cem',lambda e:None)
    with pytest.raises(ValueError,match='predispatch'):meter.begin(resource)
    assert not meter.events and not meter.extended


def test_phase_caps_and_failed_work_not_refunded():
    meter=m.Meter('iid_random',lambda e:None)
    for _ in range(120):meter.begin('predict')
    with pytest.raises(ValueError):meter.begin('predict')
    meter.phase='winner'
    for _ in range(6):meter.begin('predict')
    with pytest.raises(ValueError):meter.begin('predict')
    c=m.counts(meter.extended,'iid_random')
    assert c['sampling']['predict']==dict(attempted=120,completed=0)


@pytest.mark.parametrize('fault',['predict','chronology','replay'])
def test_faults_preserve_actual_partial_counts(fault):
    events=[];b=Backend(fault);r=m.run_arm(task(),'cem',123,b,events.append)
    assert r['status']=='failed' and r['error'] and b.closed
    if fault=='predict':
        assert r['resources']['sampling']['predict']==dict(attempted=1,completed=0)
        assert r['resources']['sampling']['action_encoder']==dict(attempted=1,completed=1)
        assert 'primary_predict' in r['error']['message']
    if fault=='replay':assert len(r['candidates'])==60


def test_primary_failure_survives_final_journal_failure():
    def sink(e):
        if e['kind']=='arm_finished':raise OSError('secondary_sink')
    r=m.run_arm(task(),'cem',123,Backend('predict'),sink)
    assert 'primary_predict' in r['error']['message']
    assert 'secondary_sink' in r['secondary_errors'][0]['message']


def test_lossless_durable_partial_and_finished_stream(tmp_path):
    store=m.Store(tmp_path/'new');sink=store.sink('test.jsonl.gz')
    value=dict(float=np.nextafter(1.,2.),negative=-0.,tiny=1e-300,array=np.random.default_rng(8).normal(size=192).tolist())
    sink(value)
    partial=(store.output/'test.jsonl.gz').read_bytes()
    decoded=zlib.decompressobj(31).decompress(partial).splitlines()
    assert m.io.io.decode(decoded[-1])==value
    assert store.close()==[]
    assert m.io.io.decode(gzip.decompress((store.output/'test.jsonl.gz').read_bytes()).splitlines()[-1])==value


def test_store_global_quota_final_reserve_and_no_overwrite(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'LIMIT',4096);monkeypatch.setattr(m,'FINAL',1024)
    s=m.Store(tmp_path/'output');s.write('data',b'x'*3072)
    with pytest.raises(ValueError,match='quota'):s.write('overflow',b'x')
    s.put('report.json',dict(status='failed'),final=True)
    with pytest.raises(FileExistsError):s.put('report.json',{},final=True)
    assert sum(p.stat().st_size for p in s.output.iterdir())<=4096
    with pytest.raises(ValueError):m.Store(s.output)
    with pytest.raises(ValueError):s.put('../escape',{})


def test_200mib_floor_each_write(tmp_path,monkeypatch):
    s=m.Store(tmp_path/'new')
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:SimpleNamespace(free=m.FLOOR))
    with pytest.raises(ValueError,match='floor'):s.put('data',{})
    with pytest.raises(ValueError,match='reserve'):m.Store(tmp_path/'other')


def test_protocol_total_504_not_total_compute_matched():
    assert sum(m.budgets(method)[p]['predict'] for _ in range(2) for method in m.METHODS for p in ('sampling','winner'))==504
    for method in m.METHODS:
        assert sum(m.budgets(method)[p]['action_encoder'] for p in ('sampling','winner'))==126
    assert 'NOT AD/FLOPs' in m.PROTOCOL['match']
    assert not m.PROTOCOL['P1_admission']


@pytest.mark.parametrize('field,value',[('projection',np.zeros((192,192)).tolist()),('packet_logit_cap',1.)])
def test_original_task_scale_Q_rejected(field,value):
    t=task();t[field]=value
    with pytest.raises(ValueError):m.run_arm(t,'cem',123,Backend(),lambda e:None)


def test_explicit_cuda_authorization_before_loading():
    with pytest.raises(ValueError,match='authorization'):m.run(None,None,None)


def test_prepare_hash_only_verify_source_drift(tmp_path,monkeypatch):
    f=tmp_path/'own.py';f.write_text('# fixture')
    inputs=dict(direct_prepared='fixture',direct_prepare_sha256='a'*64,task_directory='fixture',
                bindings={},model_bindings={},tasks=[])
    monkeypatch.setattr(m,'collect',lambda *a:deepcopy(inputs))
    monkeypatch.setattr(m,'source_files',lambda:[f])
    out=tmp_path/'prepared';m.prepare('fixture','a'*64,'fixture',out)
    sha=m.io.sha(out/'prepare.json');assert m.verify(out,sha)['inputs']==inputs
    assert {p.name for p in out.iterdir()}=={'protocol.json','prepare.json'}
    with pytest.raises(ValueError):m.verify(out,'b'*64)
    f.write_text('# drift')
    with pytest.raises(ValueError,match='drift'):m.verify(out,sha)


def test_collector_only_cached_input_no_witness_or_direct_losses():
    text=inspect.getsource(m.collect)
    assert 'bridge.collect' in text and 'bridge.run' not in text and 'bridge.verify' not in text
    text=inspect.getsource(m.bridge.collect)
    assert "directory/'report.json'" in text and "'actual_direct_resources'" in text
    assert 'direct_runs' not in text and 'witness.json' not in text


def test_four_dense_float32_arms_lossless_fit_complete_output_quota(tmp_path):
    class Dense(Backend):
        def __init__(self):
            super().__init__();self.matrix=np.random.default_rng(42).normal(size=(192,10)).astype(np.float32)*.03
        def rollout(self,t,raw,meter,mode):
            c=deepcopy(t['root_context']);out=[]
            for token in m.core.normalized_tokens(t,raw):
                meter.call('action_encoder',lambda:None)
                z=meter.call('predict',lambda:(.7*np.asarray(c['latent_history'][-1],np.float32)+self.matrix@token).tolist())
                c=dict(c,latent_history=c['latent_history'][1:]+[z],action_history=c['action_history'][1:]+[token.tolist()])
                meter.emit('native_packet',mode=mode,context=c);out.append(deepcopy(c))
            return out
    store=m.Store(tmp_path/'dense');totals=0
    for root in range(2):
        for method in m.METHODS:
            name=f'{root}_{method}'
            r=m.run_arm(task(),method,m.seed_for(root,method),Dense(),store.sink(name+'.jsonl.gz'))
            assert r['status']=='sampling_completed_not_admitted'
            store.put(name+'.json',r)
            totals+=sum(c['predict']['completed'] for c in r['resources'].values())
    assert store.close()==[] and totals==504
    store.put('report.json',dict(status='mock_only'),final=True)
    assert sum(p.stat().st_size for p in store.output.iterdir())<m.LIMIT
    for p in store.output.glob('*.gz'):
        events=[m.io.io.decode(line) for line in gzip.decompress(p.read_bytes()).splitlines()]
        assert len([e for e in events if e['kind']=='candidate_complete'])==60


def test_failed_compressed_write_preserves_decodable_prefix(tmp_path,monkeypatch):
    s=m.Store(tmp_path/'partial');emit=s.sink('test.jsonl.gz');emit(dict(saved=1))
    original=s.write
    def fail(name,raw,**kwargs):
        if not kwargs.get('final'):raise OSError('medium failure')
        return original(name,raw,**kwargs)
    monkeypatch.setattr(s,'write',fail)
    with pytest.raises(OSError):emit(dict(unsaved=2))
    assert s.close()[0]['stage']=='journal_finish'
    raw=zlib.decompressobj(31).decompress((s.output/'test.jsonl.gz').read_bytes())
    assert m.io.io.decode(raw.splitlines()[-1])==dict(saved=1)


def test_outer_model_load_and_synchronize_faults_keep_final_report(tmp_path,monkeypatch):
    import sys
    from types import ModuleType
    from contextlib import nullcontext
    runtime=ModuleType('torch');runtime.device=lambda name:name
    runtime.backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)))
    def sync():raise RuntimeError('secondary_sync')
    runtime.cuda=SimpleNamespace(synchronize=sync)
    att=ModuleType('torch.nn.attention');att.SDPBackend=SimpleNamespace(MATH='mock')
    att.sdpa_kernel=lambda *a:nullcontext()
    checkpoint=ModuleType('experiments.checkpoint_compat')
    def load(*a,**kw):raise RuntimeError('primary_load')
    checkpoint.load_compatible_pretrained=load
    preflight=ModuleType('experiments.gpu_preflight');preflight.require_cuda_allocation=lambda *a:None
    for name,module in [('torch',runtime),('torch.nn.attention',att),('experiments.checkpoint_compat',checkpoint),('experiments.gpu_preflight',preflight)]:
        monkeypatch.setitem(sys.modules,name,module)
    p=dict(inputs=dict(policy='fixture',tasks=[],inherited={},source_purpose='calibration_exploration',history_semantics='causal_padding'))
    monkeypatch.setattr(m,'verify',lambda *a:p)
    out=tmp_path/'failure';r=m.run(tmp_path/'prepared','a'*64,out,authorize_host_cuda=True)
    assert r['status']=='partial_or_failed' and r['source_verified']
    assert 'primary_load' in r['errors'][0]['message'] and 'secondary_sync' in r['errors'][1]['message']
    assert m.io.read(out/'report.json')==r
