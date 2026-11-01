"""RAM/mock only: no tmp, model weights, CPU neural, GPU, or simulator."""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import pytest
from experiments.spectral_reachability import p1_moving_live_feedback_teacher_fidelity_cuda_20261027_v1 as m


class RAM(m.Store):
    def __init__(self):self.output=Path('/RAM/teacher');self.files={};self.previous=None;self.index=0;self.events=[]
    def write(self,name,raw,append=False,final=False):
        assert append or name not in self.files;self.files[name]=self.files.get(name,b'')+raw
        assert sum(len(v) for v in self.files.values())<m.CAP
        return m.saved.sha_bytes(self.files[name])
    def event(self,e):
        super().event(e);self.events.append(deepcopy(e))
    def load(self,e):
        raw=self.files[e['file']];assert m.saved.sha_bytes(raw)==e['sha256']
        return np.load(BytesIO(raw),allow_pickle=False)


def fixture():
    inputs=[];evaluators=[];arrays={};mean=m.moving.normalization()['mean'];std=m.moving.normalization()['std']
    for i,(s,a,mode) in enumerate(m.saved.KEYS):
        identity=dict(index=i,episode_id=f'e{i//3:02}_{mode}',source_slot=s,prefix_arm=a,mode=mode,
            parent_task_id='a'*64,parent_selection_sha256='b'*64)
        prefix=np.ones(192,np.float32);suffix=np.full((5,2),np.float32(.02*(1+i)),np.float32)
        inputs.append(m.request(identity,np.zeros((3,192),np.float32),prefix,m.moving.fixed_raw(),
            np.full((5,2),np.float32(.05)),suffix,mean,std))
        receipt=lambda step:dict(path=f'/RAM/old/{identity["episode_id"]}_encode{step}/native192.npy',
            file_sha256='c'*64,array_sha256='d'*64,dtype='float32',shape=[192])
        ev=dict(identity=identity,label=receipt(10),prefix=receipt(5),goal=np.zeros(192,np.float32).tolist())
        arrays[ev['label']['path']]=np.full(192,3,np.float32);arrays[ev['prefix']['path']]=prefix;evaluators.append(ev)
    return dict(schema=m.NAME+'.prepare',inputs=inputs,evaluators=evaluators,contract=m.contract(),
        expected_model_hash=m.moving.low.MODEL_TENSOR_SHA),arrays


def fake_forward(ledger,store,*,fail_at=None,extra=False):
    def forward(r):
        m.validate_request(r);i=r['identity']['index']
        ae=ledger.begin('action_encoder');ae['inputs']=[store.array(f'm{i}_a_in.npy',m.validate_request(r)[None])]
        ae['output']=store.array(f'm{i}_a_out.npy',np.zeros((1,3,192),np.float32));ledger.end(ae)
        api=ledger.begin('predict_API')
        if i==fail_at:raise RuntimeError('artificial predictor failure')
        p=ledger.begin('predictor');p['inputs']=[store.array(f'm{i}_p_in.npy',m.literal(r['latents'],(3,192))[None]),ae['output']]
        p['output']=store.array(f'm{i}_p_out.npy',np.full((1,3,192),2,np.float32));ledger.end(p)
        q=ledger.begin('pred_proj');q['inputs']=[store.array(f'm{i}_q_in.npy',np.full((3,192),2,np.float32))]
        q['output']=store.array(f'm{i}_q_out.npy',np.full((3,192),2,np.float32));ledger.end(q);ledger.end(api)
        if extra:ledger.begin('predict_API')
        return np.full(192,2,np.float32)
    return forward


def execute(*,fail_at=None,label_error=False):
    p,arrays=fixture();store=RAM();ledger=m.Ledger(store);rows=m.initial_rows(p);calls=[]
    def labels(e):
        assert 'predictions_seal.json' in store.files and len([x for x in store.files if x.startswith('prediction_')])==12
        assert ledger.counts()=={k:dict(attempted=12,completed=12) for k in m.RESOURCES}
        calls.append(e['path'])
        if label_error:raise ValueError('artificial missing label')
        return arrays[e['path']].copy()
    seal=m.execute(p,store,ledger,fake_forward(ledger,store,fail_at=fail_at),rows,label_loader=labels)
    return p,arrays,store,ledger,rows,seal,calls


def test_fixed12_seal_before_any_terminal_decode_metrics_and_complete_trace():
    p,arrays,s,l,rows,seal,calls=execute()
    assert len(rows)==12 and all(r['status']=='scored' for r in rows) and len(calls)==24
    assert m.saved.sha_bytes(s.files['predictions_seal.json'])==seal
    assert all(r['metrics']==dict(factual_MSE=1.,predicted_Q=4.,actual_Q=9.,predicted_Q_minus_actual_Q=-5.,actual_prefix_terminal_mse=4.) for r in rows)
    order=[e['kind'] for e in s.events];assert order.index('all_predictions_sealed')<order.index('episode_scored')
    m.check_trace(l.entries,p['inputs'],s.load)
    assert l.counts()=={k:dict(attempted=12,completed=12) for k in m.RESOURCES}


@pytest.mark.parametrize('bad',['target','goal','state','path','terminal','mode','digest','history','std','slotbool'])
def test_no_future_or_privileged_forward_fields_or_wrong_context(bad):
    r=fixture()[0]['inputs'][0]
    if bad in ('target','goal','state','path','terminal'):r[bad]='forbidden'
    elif bad=='mode':r['identity']['mode']='R'
    elif bad=='digest':r['input_digest']='0'*64
    elif bad=='history':r['history']=[[0.]*10]
    elif bad=='std':r['std']=[1.,1.]
    else:r['identity']['source_slot']=False
    if bad!='digest':r['input_digest']=m.digest({k:v for k,v in r.items() if k!='input_digest'})
    with pytest.raises(ValueError):m.validate_request(r)


def test_actual_history_float32_time_major_no_zero_padding_or_double_scaling():
    r=fixture()[0]['inputs'][0];out=m.validate_request(r)
    raw=np.concatenate((m.moving.fixed_raw()[5:],np.full((5,2),np.float32(.05)),m.literal(r['raw5'],(5,2))))
    expected=((raw-np.asarray(r['mean'],np.float32))/np.asarray(r['std'],np.float32)).reshape(3,10)
    assert np.array_equal(out,expected) and not np.array_equal(out,np.zeros((3,10)))
    assert not np.array_equal(out,((raw/100-np.asarray(r['mean'],np.float32))/np.asarray(r['std'],np.float32)).reshape(3,10))


@pytest.mark.parametrize('i',[0,5,11])
def test_failed_predict_no_labels_no_retry_full_initial_denominator(i):
    p,arrays=fixture();s=RAM();l=m.Ledger(s);rows=m.initial_rows(p)
    with pytest.raises(RuntimeError):m.execute(p,s,l,fake_forward(l,s,fail_at=i),rows,label_loader=lambda e:pytest.fail('future label read'))
    assert len(m.saved.parse(s.files['initial_rows.json']))==12 and len(rows)==12
    assert rows[i]['status']=='attempted' and all(r['status']=='not_started' for r in rows[i+1:])
    assert 'predictions_seal.json' not in s.files
    assert l.counts()['predict_API']==dict(attempted=i+1,completed=i)
    with pytest.raises(ValueError):l.begin('predict_API')


def test_second_forward_rejected_before_more_work_and_no_label_fallback():
    p,_=fixture();s=RAM();l=m.Ledger(s);rows=m.initial_rows(p)
    with pytest.raises(ValueError):m.execute(p,s,l,fake_forward(l,s,extra=True),rows,label_loader=lambda _:pytest.fail('no labels'))
    assert l.counts()['predict_API']==dict(attempted=1,completed=1)


def test_label_failure_after_prediction_seal_retains_all_predictions():
    p,_=fixture();s=RAM();l=m.Ledger(s);rows=m.initial_rows(p)
    with pytest.raises(FileNotFoundError):m.execute(p,s,l,fake_forward(l,s),rows,label_loader=lambda _:(_ for _ in ()).throw(FileNotFoundError('label')))
    assert 'predictions_seal.json' in s.files and len(rows)==12 and all(r['status']=='predicted' for r in rows)
    assert l.counts()=={k:dict(attempted=12,completed=12) for k in m.RESOURCES}


def test_swapped_episode_label_identity_rejected_before_decoding():
    p,_=fixture();p['evaluators'][0],p['evaluators'][1]=p['evaluators'][1],p['evaluators'][0]
    s=RAM();l=m.Ledger(s)
    with pytest.raises(ValueError):m.execute(p,s,l,fake_forward(l,s),m.initial_rows(p),label_loader=lambda _:pytest.fail('wrong label decode'))
    assert 'predictions_seal.json' in s.files


def test_trace_tampering_and_native_metrics_shape_reject():
    p,_,s,l,_,_,_=execute();m.check_trace(l.entries,p['inputs'],s.load)
    l.entries[2]['inputs'][0]=l.entries[0]['output']
    with pytest.raises(ValueError):m.check_trace(l.entries,p['inputs'],s.load)
    for bad in (np.zeros(191,np.float32),np.zeros(192,np.float64),np.full(192,np.nan,np.float32)):
        with pytest.raises(ValueError):m.metrics(bad,np.zeros(192,np.float32),np.zeros(192),np.zeros(192,np.float32))


def test_ad_guards_restore_on_failure_and_cpu_native_rejected():
    f=lambda:None;torch=NS(autograd=NS(grad=f,functional=NS(jacobian=f)),Tensor=NS(backward=f),func=NS(vjp=f),linalg=NS(svd=f))
    s=RAM();l=m.Ledger(s);errors=[]
    with pytest.raises(ValueError):
        with m.no_ad(torch,l,errors):torch.autograd.grad()
    assert torch.autograd.grad is f and torch.autograd.functional.jacobian is f and not errors and l.blocked
    with pytest.raises(ValueError):m.Native(None,None,NS(type='cpu'),l,s)


def test_actual_ae_failure_does_not_invent_predict_api_attempt():
    l=m.Ledger(RAM());l.current=0;l.begin('action_encoder')
    assert l.counts()['predict_API']==dict(attempted=0,completed=0)


class Tensor:
    def __init__(self,a,device):self.a=np.asarray(a,np.float32);self.device=device;self.dtype='float32'
    def __getitem__(self,k):return Tensor(self.a[k],self.device)
    def reshape(self,*shape):return Tensor(self.a.reshape(*shape),self.device)
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.a


class Module:
    def __init__(self,fn):self.fn=fn;self.pre=[];self.post=[]
    def register_forward_pre_hook(self,h):self.pre.append(h);return NS(remove=lambda:self.pre.remove(h))
    def register_forward_hook(self,h,always_call=False):self.post.append(h);return NS(remove=lambda:self.post.remove(h))
    def __call__(self,*args):
        for h in self.pre:h(self,args)
        out=self.fn(*args)
        for h in self.post:h(self,args,out)
        return out


def native_fixture():
    dev=NS(type='cuda');torch=NS(float32='float32',tensor=lambda a,**kw:Tensor(a,kw['device']),
        is_grad_enabled=lambda:False,cuda=NS(synchronize=lambda d:None))
    class Model:
        def __init__(self):
            self.action_encoder=Module(lambda a:Tensor(np.zeros((1,3,192),np.float32),dev))
            self.predictor=Module(lambda z,a:Tensor(z.a+a.a,dev))
            self.pred_proj=Module(lambda z:Tensor(z.a+np.float32(1),dev))
            self.encoder=Module(lambda z:z);self.projector=Module(lambda z:z)
        def predict(self,z,a):return self.pred_proj(self.predictor(z,a).reshape(3,192)).reshape(1,3,192)
    return Model(),torch,dev


def test_real_native_wrapper_hooks_api_full3_and_restore_all_in_ram():
    model,torch,dev=native_fixture();p,arrays=fixture();s=RAM();l=m.Ledger(s);rows=m.initial_rows(p);old=model.predict
    with m.Native(model,torch,dev,l,s) as forward:
        def labels(e):
            assert 'predictions_seal.json' in s.files
            return arrays[e['path']].copy()
        m.execute(p,s,l,forward,rows,label_loader=labels)
    assert model.predict==old and 'predict' not in model.__dict__
    assert all(not module.pre and not module.post for module in (model.action_encoder,model.predictor,model.pred_proj,model.encoder,model.projector))
    assert all(r['metrics']['factual_MSE']==1 for r in rows)
    m.check_trace(l.entries,p['inputs'],s.load)


def test_native_image_encoder_forbidden_and_restored():
    model,torch,dev=native_fixture();s=RAM();l=m.Ledger(s)
    with pytest.raises(ValueError):
        with m.Native(model,torch,dev,l,s):model.encoder(Tensor(np.zeros((1,3,224,224)),dev))
    assert l.blocked==['image_encoder/projector'] and not model.encoder.pre


def test_caps_no_neural_imports_or_files():
    with patch.object(Path,'exists',return_value=False),patch.object(m.shutil,'disk_usage',return_value=NS(free=m.FLOOR+m.CAP-1)):
        with pytest.raises(ValueError):m.Store('/RAM/not_created')
    assert not any(k in sys.modules for k in ('torch','gymnasium','pymunk','stable_worldmodel'))


def test_saved_consume_full_artifacts_seal_labels_and_metrics(monkeypatch):
    p,arrays,s,l,rows,seal,_=execute();h='a'*64
    report=dict(schema=m.NAME+'.report',status='teacher_factual_recorded_not_admitted',errors=[],blocked=[],
        prepared='/RAM/p.json',prepare_sha256=h,source_verified=True,model_unchanged=True,
        model_hash_before=p['expected_model_hash'],model_hash_after=p['expected_model_hash'],contract=m.contract(),fixed_denominator=12,
        rows=rows,resources=l.counts(),ledger=l.entries,predictions_seal_sha256=seal,
        artifact_sha256={name:m.saved.sha_bytes(raw) for name,raw in s.files.items()})
    s.files['report.json']=m.canonical(report)
    monkeypatch.setattr(m,'verify',lambda *a:deepcopy(p))
    monkeypatch.setattr(m,'sha',lambda path:m.saved.sha_bytes(s.files[Path(path).name]))
    monkeypatch.setattr(m.saved,'read_bytes',lambda path,*a:s.files[Path(path).name])
    monkeypatch.setattr(Path,'read_bytes',lambda path:s.files[Path(path).name])
    monkeypatch.setattr(Path,'iterdir',lambda path:iter([s.output/n for n in s.files]))
    monkeypatch.setattr(Path,'is_symlink',lambda path:False)
    monkeypatch.setattr(m,'receipt_array',lambda receipt:arrays[receipt['path']].copy())
    out=m.consume(s.output,h);assert len(out['rows'])==12 and out['P1_admission'] is False
    s.files['extra.npy']=b'not authorized'
    with pytest.raises(ValueError,match='artifact set'):m.consume(s.output,h)


@pytest.fixture(scope='module')
def physical_saved():
    # Execute only frozen artificial environment/tensors, producing RAM bytes, not research outputs.
    from test_p1_moving_live_feedback_physical_20261027_v1 import fixture as driver_fixture,execute as driver_execute
    from experiments.spectral_reachability import p1_moving_live_feedback_physical_20261027_v1 as driver
    p,d=driver_fixture();p.update(schema=driver.NAME+'.prepare',modelbinding=d['tasks'][0]['modelbinding'])
    r,s,_,_=driver_execute(p=p,d=d)
    assert r['status']=='live_feedback_calibration_recorded'
    return p,d,r,s


def test_build_provider_join_reads_no_terminal_array_and_rejects_cross_episode(physical_saved,monkeypatch):
    p,d,r,s=deepcopy(physical_saved);base=s.path
    # RAM absolute authority paths; hashes are explicitly mocked, never prepare or actual results.
    rp=base/'report.json';pp=Path('/RAM/driver_prepare.json');ap=Path('/RAM/analysis.json')
    analysis=dict(schema=m.saved.NAME+'.report',observed_endpoint_count=24,
        inputs=dict(report_sha256=m.LIVE_SHA,prepare_sha256=m.LIVE_PREP_SHA),contrasts=[dict(prefix_pairing_exact=True)]*12)
    files={str(base/n):raw for n,raw in s.files.items()}
    files.update({str(rp):m.canonical(r),str(pp):m.canonical(p),str(ap):m.canonical(analysis),
        str(base/'execution_binding.json'):m.canonical(dict(prepared=str(pp),prepare_sha256=m.LIVE_PREP_SHA))})
    pins={str(rp):m.LIVE_SHA,str(pp):m.LIVE_PREP_SHA,str(ap):m.ANALYSIS_SHA,str(m.PREREG):m.PREREG_SHA,
        str(Path(m.saved.__file__).resolve()):m.SAVED_SHA,str(Path(m.moving.__file__).resolve()):m.MOVING_SHA,**m.moving.MODELS,
        **p['modelbinding']['source_closure']}
    pins.update({layer['task_export']:layer['task_sha256'] for layer in p['layers']})
    actualread=Path.read_bytes
    monkeypatch.setattr(Path,'read_bytes',lambda path:files[str(path)] if str(path) in files else actualread(path))
    actualstat=Path.stat
    monkeypatch.setattr(Path,'stat',lambda path,*a,**kw: NS(st_size=len(files[str(path)]),st_mode=33188) if str(path) in files else actualstat(path,*a,**kw))
    monkeypatch.setattr(m,'sha',lambda path:pins.get(str(path),m.saved.sha_bytes(files.get(str(path),b'RAMsource'))))
    monkeypatch.setattr(m.moving,'source_files',lambda:[])
    monkeypatch.setattr(m.moving,'load_task',lambda path,h:dict(task_id=next(x['task_id'] for x in p['layers'] if x['task_export']==path),
        gn_task=deepcopy(d['tasks'][next(x['source_slot'] for x in p['layers'] if x['task_export']==path)]['gn_task'])))
    from PIL import Image
    actualopen=Image.open
    monkeypatch.setattr(Image,'open',lambda path:actualopen(BytesIO(files[str(path)])))
    calls=[]
    def prefix_only(receipt):
        assert '_encode5/' in receipt['path'];calls.append(receipt['path'])
        return np.load(BytesIO(files[receipt['path']]),allow_pickle=False)
    monkeypatch.setattr(m,'receipt_array',prefix_only)
    out=m.build(str(rp),m.LIVE_SHA,str(pp),m.LIVE_PREP_SHA,str(ap),m.ANALYSIS_SHA)
    assert len(out['inputs'])==len(out['evaluators'])==12 and len(calls)==12
    assert all(set(x)==set('schema identity latents history raw5 mean std input_digest'.split()) for x in out['inputs'])
    # Source label arrays remain receipts, never decoded or packed in forward inputs.
    assert all('label' not in x for x in out['inputs'])
