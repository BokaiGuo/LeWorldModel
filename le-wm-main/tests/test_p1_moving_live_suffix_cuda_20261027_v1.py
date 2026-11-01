"""RAM-only NumPy/hook stand-ins: no torch model, CUDA, physics or temp files."""
from copy import deepcopy
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
from experiments.spectral_reachability import p1_moving_live_suffix_cuda_20261027_v1 as m

spec=importlib.util.spec_from_file_location('suffix_ram_helpers',Path(__file__).with_name('test_moving_history_observed_suffix_cuda_20261027_v1.py'))
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


class RAM:
    def __init__(self,fail=None):self.output=Path('/RAM');self.files={};self.events=[];self.fail=fail
    def _write(self,name,raw):
        if name==self.fail:raise OSError('first disk failure')
        assert name not in self.files
        self.files[name]=raw
    def event(self,e):
        if e['kind']==self.fail:raise OSError('first journal failure')
        self.events.append(deepcopy(e))


@pytest.fixture(autouse=True)
def disk(monkeypatch):
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:NS(free=10*m.CAP+m.FLOOR))


def packet(mode='R',slot=0,arm=None):
    b=h.branch(slot,arm);c=b['observed_context']
    mb=dict(tensor_sha256=m.MODEL_SHA,model_files=deepcopy(m.frozen.observed.enc.MODELS),
        action_mean=b['action_mean'],action_std=b['action_std'],training_provenance='unknown',
        source_closure={'/RAM/source.py':'1'*64})
    p=dict(schema=m.NAME+'.packet',mode=mode,context3=deepcopy(c),actualrawhistory10=b['observed_raw_history'],
        oldrawsuffix5=b['old_suffix_commands'],goal=b['goal'],identityQ=dict(kind='identity',dimension=192),modelbinding=mb)
    s=dict(episode_id=f'ep{slot}_{mode}',source_slot=slot,prefix_arm=arm or m.planner.ARMS[0],mode=mode,
        parent_task_id='2'*64,task_export_sha256=m.planner.EXPORT_SHAS[slot],parent_selection_sha256='3'*64,
        step=5,remaining_public_steps=5,context_kind='observed' if mode=='R' else 'imagined',
        rgb_file_sha256='4'*64,rgb_pixels_sha256='5'*64,encoded_npy_sha256='6'*64,
        encoded_latent_sha256=m.digest(c['latent_history'][-1]) if mode=='R' else '7'*64,
        encoder_receipt_sha256='8'*64,imagined_context_sha256=None if mode=='R' else m.digest(c),
        observed_event_index=15,encoded_event_index=16)
    for k,v in (('context',c),('rawhistory',p['actualrawhistory10']),('oldsuffix',p['oldrawsuffix5']),
                ('goal',p['goal']),('Q',p['identityQ']),('modelbinding',mb)):s[k+'_sha256']=m.digest(v)
    s['seal_sha256']=m.digest(s);p['episode_observation_seal']=s
    return p


def runner(fail=None):
    def run(b,sink,row):
        meter=m.frozen.Meter(sink.event);meter.require_native=True
        modules=m.frozen.Modules(NS(),sink,'branch_00',meter)
        ad=m.frozen.SuffixAD(NS(),m.frozen.blind.ARMS[1],sink.event,lambda:None);ad.stage=lambda:meter.stage
        model=NS(predict=lambda:None)
        api=m.PredictAPI(model,meter,sink,lambda:None)
        class Backend(h.Backend):
            def step(self,c,raw):
                model.predict()
                return super().step(c,raw)
            def jacobian(self,c,operation):
                def calculate():
                    def forward(**kw):
                        self.nominal(c)
                        return ad.dispatch('grad',lambda **kw:self.j.copy(),(),dict(is_grads_batched=True))
                    return ad.dispatch('jacobian',forward,(),dict(vectorize=True))
                return operation('jacobian',calculate)
        backend=Backend(b,modules,fail)
        try:
            with api:m.frozen.evaluate(b,backend,meter,sink,row)
        finally:
            row.update(resources={**modules.counts(),**meter.counts()},AD_API_resources=ad.counts(),
                module_ledger=deepcopy(modules.entries),math_ledger=deepcopy(meter.entries),
                AD_API_ledger=deepcopy(ad.events),predict_API_resources=api.counts(),predict_API_ledger=deepcopy(api.entries))
    return run


def execute(p=None,store=None,run=None,check_model=None,check_source=None,check_limits=None):
    store=store or RAM()
    r=m._execute(p or packet(),store=store,check_limits=check_limits or (lambda:None),runner=run or runner(),
        check_model=check_model or (lambda:m.MODEL_SHA),check_source=check_source or (lambda b:None))
    return r,store


@pytest.mark.parametrize('mode',['I','R'])
@pytest.mark.parametrize('slot',[0,1])
@pytest.mark.parametrize('arm',m.planner.ARMS)
def test_fixed_all_strata_five_forwards_fresh_j_and_nonzero_history(mode,slot,arm):
    p=packet(mode,slot,arm);r,s=execute(p)
    assert m.verify_suffix_seal(p,r)==r['dispatch_raw5']
    assert len(r['row']['module_ledger'])==15 and len(r['row']['predict_API_ledger'])==5
    assert r['row']['math_ledger'][0]['mathematical_VJP']==192
    assert r['row']['AD_API_resources']=={k:dict(attempted=1,completed=1) for k in ('grad','jacobian')}
    assert [e['kind'] for e in s.events].index('new_suffix_frozen_before_evaluation') < [e['kind'] for e in s.events].index('suffix_evaluation_attempt')
    assert not r['P1_admission'] and not r['P4_admission']
    assert 'receipt.json.gz' in s.files and 'packet.json' in s.files
    assert np.any(np.asarray(p['actualrawhistory10'])[:5])


@pytest.mark.parametrize('where',['packet','context3','modelbinding','seal'])
@pytest.mark.parametrize('field',['state','witness','outcome'])
def test_privileged_unknown_fields_rejected_before_runner(where,field):
    p=packet();target=p if where=='packet' else p['episode_observation_seal'] if where=='seal' else p[where]
    target[field]=0
    with pytest.raises(m.SuffixFailure) as e:execute(p,run=lambda *a:pytest.fail('must not invoke backend'))
    assert not e.value.receipt['commands_released']


@pytest.mark.parametrize('bad',['O','goal','Q','zero_history','old_suffix','encoded','seal','slot','mode','sequence','nan','precision','normalizer'])
def test_packet_mutation_failclosed(bad):
    p=packet();s=p['episode_observation_seal']
    if bad=='O':p['mode']='O'
    elif bad=='goal':p['goal'][0]+=.125
    elif bad=='Q':p['identityQ']['dimension']=191
    elif bad=='zero_history':p['actualrawhistory10'][:5]=np.zeros((5,2)).tolist()
    elif bad=='old_suffix':p['oldrawsuffix5'][0][0]=1.
    elif bad=='encoded':s['encoded_latent_sha256']='a'*64
    elif bad=='seal':s['seal_sha256']='a'*64
    elif bad=='slot':s['source_slot']=1
    elif bad=='mode':s['context_kind']='imagined'
    elif bad=='sequence':s['encoded_event_index']=s['observed_event_index']
    elif bad=='nan':p['goal'][0]=float('nan')
    elif bad=='precision':p['context3']['latent_history'][0][0]+=1e-12
    elif bad=='normalizer':p['modelbinding']['action_std'][0]=.2
    with pytest.raises(ValueError):m.validate_packet(p)


def test_I_does_not_contain_actual_latent_value():
    p=packet('I');assert p['episode_observation_seal']['encoded_latent_sha256']!=m.digest(p['context3']['latent_history'][-1])
    m.validate_packet(p)


def test_old_score_better_still_dispatches_new_ridge():
    p=packet();j=np.zeros((192,10));j[:10]=np.eye(10)*.5
    p['goal']=(np.asarray(p['context3']['latent_history'][-1])+j@np.asarray(p['oldrawsuffix5']).reshape(10)).astype(np.float32).astype(float).tolist()
    s=p['episode_observation_seal'];s['goal_sha256']=m.digest(p['goal'])
    s['seal_sha256']=m.digest({k:v for k,v in s.items() if k!='seal_sha256'})
    r,_=execute(p);old,new,_=r['row']['evaluations']
    assert old['cost']==0 and new['cost']>0
    assert r['dispatch_raw5']==new['commands'] and r['dispatch_raw5']!=p['oldrawsuffix5']


def test_final_source_changed_and_global_limit_on_return_failclosed():
    calls=[]
    def source(b):
        calls.append(1)
        if len(calls)>1:raise RuntimeError('source drift after compile')
    with pytest.raises(m.SuffixFailure) as e:execute(check_source=source)
    assert e.value.receipt['first_error']['stage']=='final_source'
    assert not e.value.receipt['commands_released']


def test_missing_AD_or_forward_receipts_no_release():
    real=runner()
    def omitted(b,s,r):
        real(b,s,r);r['AD_API_resources']['grad']['completed']=0
    with pytest.raises(m.SuffixFailure) as e:execute(run=omitted)
    assert not e.value.receipt['commands_released']


def test_predict_API_cap_predispatch_and_restore():
    entered=[];model=NS(predict=lambda:entered.append(1));meter=NS(stage='nominal')
    api=m.PredictAPI(model,meter,RAM(),lambda:None)
    with api:
        for stage in m.frozen.STAGES:meter.stage=stage;model.predict()
        with pytest.raises(ValueError):model.predict()
    assert len(entered)==5 and api.counts()['predict']==dict(attempted=5,completed=5)


@pytest.mark.parametrize('phase',m.frozen.STAGES)
def test_neural_partial_attempts_no_retry_no_dispatch(phase):
    s=RAM()
    with pytest.raises(m.SuffixFailure) as e:execute(store=s,run=runner(fail=phase))
    r=e.value.receipt
    assert 'original forward failed' in r['first_error']['message']
    assert not r['commands_released'] and 'dispatch_raw5' not in r and 'failure.json.gz' in s.files
    assert r['row']['resources']['predict']['attempted']==m.frozen.STAGES.index(phase)+1
    assert r['row']['resources']['predict']['completed']==m.frozen.STAGES.index(phase)


@pytest.mark.parametrize('failure',['packet.json','module_attempt','suffix_release_seal','receipt.json.gz'])
def test_durable_failure_never_releases(failure):
    s=RAM(fail=failure)
    with pytest.raises(m.SuffixFailure) as e:execute(store=s)
    assert not e.value.receipt['commands_released'] and 'dispatch_raw5' not in e.value.receipt
    assert 'first' in e.value.receipt['first_error']['message']


def test_compute_error_preserved_if_final_model_and_failure_write_fail():
    calls=[]
    def model():
        calls.append(1)
        if len(calls)>1:raise RuntimeError('secondary model error')
        return m.MODEL_SHA
    s=RAM(fail='failure.json.gz')
    with pytest.raises(m.SuffixFailure) as e:execute(store=s,run=runner('new_suffix'),check_model=model)
    r=e.value.receipt
    assert 'original forward failed' in r['first_error']['message']
    assert [x['stage'] for x in r['errors']]==['execution','final_model','failure_persist']


def test_floor_checked_before_model(monkeypatch):
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:NS(free=m.FLOOR-1))
    with pytest.raises(m.SuffixFailure) as e:execute(check_model=lambda:pytest.fail('no model'))
    assert 'disk floor' in e.value.receipt['first_error']['message']


def test_local_cap_and_global_limit_before_dispatch():
    s=m.Sink(RAM(),lambda:None)
    with pytest.raises(ValueError):s._write('large',bytes(m.CAP-m.FINAL+1))
    assert not s.files
    with pytest.raises(m.SuffixFailure):execute(check_limits=lambda:(_ for _ in ()).throw(RuntimeError('global wall cap')))


@pytest.mark.parametrize('mutate',['commands','packet','errors','budget','model'])
def test_release_tampering_rejected(mutate):
    p=packet();r,_=execute(p)
    if mutate=='commands':r['dispatch_raw5'][0][0]+=.25
    elif mutate=='packet':p['episode_observation_seal']['episode_id']='other_episode'
    elif mutate=='errors':r['errors']=[dict(error='not complete')]
    elif mutate=='budget':r['row']['resources']['predict']['completed']=4
    else:r['model_hash_after']='b'*64
    with pytest.raises(ValueError):m.verify_suffix_seal(p,r)


def test_predict_API_restores_inherited_method_after_first_error():
    class Model:
        def predict(self):raise RuntimeError('primary error')
    model=Model();meter=NS(stage='nominal');api=m.PredictAPI(model,meter,RAM(),lambda:None)
    with pytest.raises(RuntimeError,match='primary'):
        with api:model.predict()
    assert 'predict' not in model.__dict__ and api.counts()==dict(predict=dict(attempted=1,completed=0))


def test_explicit_cuda_only_entry_no_loader_and_fixed_signature():
    assert list(inspect.signature(m.compile_suffix).parameters)==['request','artifact_sink','model','device','check_limits']
    with pytest.raises(ValueError,match='CUDA'):m.compile_suffix(packet(),RAM(),model=NS(),device=NS(type='cpu'),check_limits=lambda:None)
    assert not m._LOCK.locked()
    source=Path(m.__file__).read_text()
    assert 'load_compatible_pretrained' not in source and 'load_task(' not in source and 'set_state(' not in source


def test_real_source_closure_is_saved_code_only():
    bindings=m.source_bindings()
    assert str(Path(m.__file__).resolve()) in bindings
    assert str(Path(m.frozen.__file__).resolve()) in bindings
    assert all('/results/' not in p and not p.endswith(('weights.pt','.png','.npy')) for p in bindings)
