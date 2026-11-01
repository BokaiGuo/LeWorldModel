"""RAM arrays and fake hook modules ONLY. No torch/model/environment execution."""
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_contextual_physical_rgb_encode_20261028_v1 as m


def h(value):return hashlib.sha256(str(value).encode()).hexdigest()
def z(value):return np.full(192,value,np.float32).tolist()


def fixture(missing=()):
    plans=[];actual=[];planner=[dict(results=[]) for _ in range(4)]
    for i in range(8):
        root,condition=m.physical.saved.producer.ORDER[i//2];arm=m.physical.ARMS[i%2]
        context=lambda v:dict(latent_history=[z(v),z(v+.125),z(v+.25)],action_history=[[0.]*10]*2)
        contexts=[context(.125),context(.5),context(.75)]
        commands=np.full((10,2),i/16,np.float32).tolist()
        plan=dict(root_slot=root,condition=condition,arm=arm,task_id=h(('task',i)),commands=commands,
            raw_float32_bits=m.physical.bits(commands),selection_digest=h(('physical',i)),compiler_selection_digest=h(('compiler',i)),
            root_context=contexts[0],witness=dict(receipt=dict(terminal_context=contexts[2])),goal=z(0.),
            predicted_Q=m.mse(contexts[-1]['latent_history'][-1],z(0.)),
            original_task_file='/RAM_ONLY/task.json.gz',original_task_sha256=h('task'),
            blind_path='/RAM_ONLY/blind.npz',blind_sha256=h('blind'),encoder_identity=dict(model_sha256=h('model')))
        frames=[]
        for step in (0,5,10):
            if (i,step) in missing:continue
            history=np.concatenate((m.physical.kernel.moving.warmup_commands(),np.asarray(commands,np.float32)[:step]))[-10:]
            frames.append(dict(step=step,path=f'/RAM_ONLY/r{i}_s{step}.png',png=f'r{i}_s{step}.png',
                file_sha256=h(('PNG',i,step)),pixel_sha256=h(('pixels',i,step)),condition=condition,
                task_id=plan['task_id'],compiler_selection_digest=plan['compiler_selection_digest'],
                original_task_file=plan['original_task_file'],original_task_sha256=plan['original_task_sha256'],
                original_blind_path=plan['blind_path'],original_blind_sha256=plan['blind_sha256'],
                encoder_identity=plan['encoder_identity'],encoding_performed=False,model_bound_by_prepare=True,
                raw_action_history10=history.tolist(),raw_action_history_bits=m.physical.bits(history.tolist()),
                state=[0.]*7,snapshot=dict(RAM_ONLY=True)))
        actual.append(dict(index=i,root_slot=root,condition=condition,arm=arm,selection_digest=plan['selection_digest'],
            compiler_selection_digest=plan['compiler_selection_digest'],rgb_encoding_inputs=frames,
            missing_checkpoints=[s for s in (0,5,10) if (i,s) in missing],status='completed',
            actual_metrics=[dict(step=s,native_success=False,native_coverage=None,joint_xy_error=float(s)) for s in range(1,11)],
            initial_metrics=dict(native_success=False,native_coverage=None),final_metrics=dict(native_success=False),full_horizon=True))
        plans.append(plan)
        planner[i//2]['results'].append(dict(arm=arm,winner=dict(segmented_contexts=contexts,witness=plan['witness'],cost=plan['predicted_Q'])))
    prep=dict(plans=plans)
    strict=dict(schema=m.physical.NAME+'.consume',status='engineering_verified_science_blocked',fixed_denominator=8,
        known_roots=2,P1_admission=False,P2_admission=False,physical_release=False,rows=actual)
    return prep,strict,planner


def prepared(missing=()):
    rows=m.fixed_rows(*fixture(missing));calls={k:sum(f['source'] is not None for row in rows for f in row['frames'])
        if k in ('encoder','projector') else sum(row['predict_eligible'] for row in rows) for k in m.KINDS}
    return dict(rows=rows,contract=m.contract(),expected_calls=calls)


class Memory(m.rgb.Artifacts):
    def __init__(self):self.files={};self.events=[]
    def write(self,name,raw,final=False):
        assert name not in self.files
        self.files[name]=raw;return hashlib.sha256(raw).hexdigest()
    def event(self,e):self.events.append(deepcopy(e))
    def load(self,r):return m.rgb.load_array(r,self.files.__getitem__)


class Tensor:
    """Explicit non-neural RAM double exercising production hook wiring."""
    def __init__(self,a):self.a=np.asarray(a,np.float32);self.device=SimpleNamespace(type='cuda');self.dtype=np.dtype('float32')
    @property
    def shape(self):return self.a.shape
    def detach(self):return self
    def cpu(self):return self
    def contiguous(self):return self
    def numpy(self):return self.a
    def __getitem__(self,k):return Tensor(self.a[k])


class Module:
    def __init__(self,fn):self.fn=fn;self.pre=[];self.post=[]
    def register_forward_pre_hook(self,fn):
        self.pre.append(fn);return SimpleNamespace(remove=lambda:self.pre.remove(fn))
    def register_forward_hook(self,fn):
        self.post.append(fn);return SimpleNamespace(remove=lambda:self.post.remove(fn))
    def __call__(self,*args):
        for fn in self.pre:fn(self,args)
        out=self.fn(*args)
        for fn in self.post:fn(self,args,out)
        return out


class Model:
    def __init__(self,tokens=257):
        self.encoder=Module(lambda x:Tensor(np.full((1,tokens,192),x.a.mean(),np.float32)))
        self.projector=Module(lambda x:Tensor(x.a+.125))
        self.action_encoder=Module(lambda x:Tensor(np.repeat(x.a.sum(-1,keepdims=True),192,axis=-1)))
        self.predictor=Module(lambda z,a:Tensor(z.a+a.a))
        self.pred_proj=Module(lambda x:Tensor(x.a+.125))
    def predict(self,z,a):return Tensor(self.pred_proj(self.predictor(z,a)[0]).a[None])


def fake_pixels(frame):return np.full((224,224,3),int(frame['step'])+64,np.uint8)


def execute(monkeypatch,missing=(),tokens=257):
    p=prepared(missing);store=Memory();rows=deepcopy(p['rows']);model=Model(tokens)
    torch=SimpleNamespace(float32=np.dtype('float32'),is_grad_enabled=lambda:False,cuda=SimpleNamespace(synchronize=lambda:None))
    monkeypatch.setattr(m,'load_pixels',fake_pixels)
    trace=m.Trace(model,torch,store,lambda:None,p['expected_calls']);original=model.predict
    def encode(pix,f,scope):
        with m.physical.saved.overlay.traced_scope(trace,scope,f):
            pre=m.pixels.preprocess_pixels(pix);x=Tensor(pre)
            f['preprocessed']=trace.save(x,f'r{scope["row"]}_s{scope["step"]}_preprocessed')
            return model.projector(model.encoder(Tensor(pre.reshape(1,3,224,224)))[:,0]).a[0].copy()
    def predict(inputs,f,scope):
        with m.physical.saved.overlay.traced_scope(trace,scope,f):
            return model.predict(Tensor(inputs['context']),model.action_encoder(Tensor(inputs['outgoing']))).a.copy()
    with trace:m.process(p,store,encode,predict,rows)
    assert model.predict==original and all(not module.pre and not module.post for module in
        (model.encoder,model.projector,model.action_encoder,model.predictor,model.pred_proj))
    r=dict(rows=rows,contract=m.contract(),errors=[],modules=trace.entries,counts=m.counts(trace.entries))
    return p,r,store


@pytest.fixture(scope='module')
def replay():
    with pytest.MonkeyPatch.context() as patch:return execute(patch)


def test_actual_hook_dispatch_all24_encode8_same_suffix_and_full_saved_chain(monkeypatch):
    p,r,s=execute(monkeypatch);out=m.validate(p,r,s.events,s.load)
    assert len(out)==8 and len(r['modules'])==80 and len(s.events)==224
    assert all(r['counts'][k]==dict(attempted=v,completed=v) for k,v in p['expected_calls'].items())
    assert all(not v['physical_benefit_claim'] and v['no_control_intervention'] for v in out)
    assert all(v['checkpoints'][0]['native']['native_coverage'] is None for v in out)


@pytest.mark.parametrize('missing',[(0,0),(0,5),(0,10)])
def test_missing_retained_no_padding_and_mid_dependency(monkeypatch,missing):
    p,r,s=execute(monkeypatch,(missing,));out=m.validate(p,r,s.events,s.load)
    assert len(out)==8 and p['expected_calls']['encoder']==23
    assert p['expected_calls']['predict_API']==(8 if missing[1]==10 else 7)
    assert out[0]['checkpoints'][(0,5,10).index(missing[1])]['latent_Q'] is None
    if missing[1]==10:assert out[0]['imagined_terminal_MSE'] is out[0]['real_mid_same_suffix_MSE'] is None
    else:assert out[0]['real_mid_same_suffix_predicted_Q'] is None


def test_197_not_257_rejected_at_production_hook(monkeypatch):
    with pytest.raises(ValueError,match='hidden257'):execute(monkeypatch,tokens=197)


def test_all8_feedback_denominator_exists_even_first_image_failure(monkeypatch):
    p=prepared();rows=deepcopy(p['rows']);s=Memory()
    monkeypatch.setattr(m,'load_pixels',lambda f:(_ for _ in ()).throw(ValueError('RAM image failure')))
    with pytest.raises(ValueError,match='RAM image failure'):
        m.process(p,s,lambda *a:pytest.fail('encode must not occur'),lambda *a:pytest.fail('predict'),rows)
    assert len(rows)==8 and all(row['feedback']==dict(status='not_started',prediction=None) for row in rows)
    assert rows[0]['frames'][0]['status']=='attempted' and len(s.events)==1


@pytest.mark.parametrize('field',['arm','condition','selection_digest','compiler_selection_digest','task_id','original_task_sha256',
    'original_blind_sha256','encoder_identity','model_bound_by_prepare','encoding_performed','raw_action_history_bits','raw_action_history10','path'])
def test_real_schema_identity_and_history_mutations_fail_closed(field):
    p,s,plan=fixture()
    if field in ('arm','selection_digest'):s['rows'][0][field]='wrong'
    else:
        f=s['rows'][0]['rgb_encoding_inputs'][1]
        if field=='raw_action_history10':f[field]=[[0.,0.]]*10
        elif field=='model_bound_by_prepare':f[field]=False
        elif field=='encoding_performed':f[field]=True
        elif field=='encoder_identity':f[field]={}
        else:f[field]='wrong'
    with pytest.raises((ValueError,KeyError)):m.fixed_rows(p,s,plan)


@pytest.mark.parametrize('field',['Q','commands','terminal','contexts','missing','count'])
def test_selected_plan_replay_and_denominator_mutations(field):
    p,s,pl=fixture()
    if field=='Q':p['plans'][0]['predicted_Q']+=1
    elif field=='commands':p['plans'][0]['commands'][0][0]=.5
    elif field=='terminal':pl[0]['results'][0]['winner']['segmented_contexts'][2]['latent_history'][2][0]=.5
    elif field=='contexts':pl[0]['results'][0]['winner']['segmented_contexts'].pop()
    elif field=='missing':s['rows'][0]['missing_checkpoints']=[5]
    else:s['rows'].pop()
    with pytest.raises((ValueError,KeyError)):m.fixed_rows(p,s,pl)


@pytest.mark.parametrize('fault',['missing_event','event_order','nonempty_bad_event','module_order','native_input','native_output',
    'history','suffix','goal','missing_to_zero','extra_count','partial','feedback_native3','preprocess','hidden197','row_dropped'])
def test_strict_saved_mutations_rejected(replay,monkeypatch,fault):
    p,original,store=replay;r=deepcopy(original);events=deepcopy(store.events);s=Memory();s.files=dict(store.files)
    monkeypatch.setattr(m,'load_pixels',fake_pixels)
    row=r['rows'][0];frame=row['frames'][0];feedback=row['feedback']
    if fault=='missing_event':events.pop()
    elif fault=='event_order':events[0],events[1]=events[1],events[0]
    elif fault=='nonempty_bad_event':events.insert(0,dict(kind='BAD'))
    elif fault=='module_order':feedback['modules'].reverse()
    elif fault=='native_input':feedback['inputs']['context']=s.array('bad_context',np.zeros((1,3,192),np.float32))
    elif fault=='native_output':feedback['modules'][1]['output']=s.array('bad_output',np.zeros((1,3,192),np.float32))
    elif fault=='history':row['frames'][1]['source']['raw_action_history10'][0][0]=.5
    elif fault=='suffix':row['full10'][9][0]=.5
    elif fault=='goal':row['goal192'][0]=.5
    elif fault=='missing_to_zero':frame['source']=None
    elif fault=='extra_count':r['counts']['encoder']['completed']+=1
    elif fault=='partial':r['errors']=[dict(stage='RAM failure')]
    elif fault=='feedback_native3':feedback['prediction']=s.array('bad_native3',np.zeros((1,1,192),np.float32))
    elif fault=='preprocess':frame['preprocessed']=s.array('bad_pre',np.zeros((1,1,3,224,224),np.float32))
    elif fault=='hidden197':frame['modules'][0]['output']=s.array('bad_197',np.zeros((1,197,192),np.float32))
    elif fault=='row_dropped':r['rows'].pop()
    with pytest.raises((ValueError,KeyError,IndexError)):m.validate(p,r,events,s.load)


def test_endpoint_not_native_first_two_or_shifted_history(replay):
    p,r,s=replay;row=r['rows'][0];out=m.summarize(p['rows'][0],row,s.load)
    native=s.load(row['feedback']['prediction']);actual=s.load(row['frames'][2]['latent'])
    assert out['real_mid_same_suffix_MSE']==m.mse(native[0,-1],actual)
    assert out['real_mid_same_suffix_predicted_Q']==m.mse(native[0,-1],p['rows'][0]['goal192'])
    assert out['real_mid_same_suffix_MSE']!=m.mse(native[0,0],actual)
    inputs=m.mid_inputs(p['rows'][0],s.load(row['frames'][0]['latent']),s.load(row['frames'][1]['latent']))
    assert np.array_equal(inputs['context'][0,0],p['rows'][0]['root_context']['latent_history'][1])
    token=m.physical.saved.frontier._normalized_token
    assert np.array_equal(inputs['outgoing'][0,-1],token(np.asarray(p['rows'][0]['full10'],np.float32)[5:]))


def test_AUTH_before_any_prepare_or_model_IO(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('no IO allowed'))
    with pytest.raises(ValueError,match='MAIN authority'):m.run('/RAM_ONLY','0'*64,'/RAM_ONLY','wrong')


def test_contract_unknown_costs_and_science_gates():
    c=m.contract()
    assert c['total_FLOPs'] is c['hidden_backward_engine_calls'] is None
    assert not any(c[k] for k in ('equal_FLOPs','P1_admission','P2_admission','physical_release','scientific_pass'))
    assert c['new_physics']==c['new_optimizer']==c['new_AD_requested']==0


def test_budget_wall_disk_and_final_reserve_RAM():
    now=[0.];b=m.Budget(clock=lambda:now[0],free=lambda:m.CAP+m.FLOOR)
    b.used=m.CAP-m.FINAL;b.check()
    with pytest.raises(ValueError):b.check(1)
    b.check(1,final=True);now[0]=601
    with pytest.raises(ValueError):b.check()
    b.check(final=True);b.free=lambda:m.FLOOR-1
    with pytest.raises(ValueError):b.check(final=True)


def test_array_hashes_no_zero_substitution(replay):
    _,r,s=replay;receipt=r['rows'][0]['frames'][0]['latent'];files=dict(s.files)
    files[receipt['file']]=b'not an array'
    with pytest.raises(ValueError):m.rgb.load_array(receipt,files.__getitem__)


def test_full_physical_authority_rebuild_not_status_or_SHA_only(monkeypatch):
    monkeypatch.setattr(m,'sha',lambda path:m.PHYSICAL_SHA if str(path)==m.physical.__file__ else '1'*64)
    monkeypatch.setattr(m.physical,'read',lambda path:dict(status='engineering_verified_science_blocked',rows=[]))
    calls=[]
    monkeypatch.setattr(m.physical,'consume',lambda *a:(calls.append(a) or dict(status='different')))
    with pytest.raises(ValueError,match='entire actual physical strict consumer reconstructed'):
        m.build('/RAM_ONLY/physical','2'*64,'/RAM_ONLY/receipt','1'*64)
    assert len(calls)==1
