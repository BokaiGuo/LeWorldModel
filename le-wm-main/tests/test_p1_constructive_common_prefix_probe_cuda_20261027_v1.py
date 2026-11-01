"""Pure RAM/mock tests: no torch, native simulator, GPU or disk tmp."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import p1_constructive_common_prefix_probe_cuda_20261027_v1 as m
import test_p1_constructive_common_prefix_probe_physical_20261027_v1 as fixture


def protocol():
    p=fixture.protocol(); tasks=[fixture.task(),fixture.task()]
    for t in tasks:
        t['action_mean']=m.base.d2p.ACTION_MEAN.tolist(); t['action_std']=m.base.d2p.ACTION_STD.tolist()
        t['root_context']['action_history']=m.core.normalized_tokens(t,np.zeros((10,2))).tolist()
    images=[]
    for root in (0,1):
        for role in ('warm5','warm10'): images.append(dict(index=len(images),root_slot=root,role=role,available=True))
    for pid in range(4): images.append(dict(index=len(images),root_slot=pid//2,role='prefix5',available=True))
    for i in range(84): images.append(dict(index=len(images),root_slot=i//42,role='terminal10',available=True))
    rows=[dict(index=i,position_id=i//21,probe_index=i%21,root_slot=i//42,terminal_image=8+i) for i in range(84)]
    p.update(tasks=tasks,images=images,rows=rows,contract=m.contract()); return p


class Store:
    def __init__(self): self.events=[]
    def event(self,e): self.events.append(deepcopy(e))


def callbacks(p,ledger,terminal=100.,fail=None):
    calls=[]
    def encode(item):
        def output():
            if fail==item['index']: raise RuntimeError('encoder primary')
            return None
        ledger.call('encoder',output); ledger.call('projector',lambda:None)
        index=item['index']
        return np.full(192,0. if index<4 else float(index-3) if index<8 else terminal+index,np.float32)
    def predict(context,commands):
        calls.append((deepcopy(context),commands.copy()))
        ledger.call('action_encoder',lambda:None); ledger.call('predict',lambda:None)
        z=np.asarray(context['latent_history'][-1],np.float32)+np.sum(commands,dtype=np.float32)
        return dict(context,latent_history=context['latent_history'][1:]+[z.astype(float).tolist()],
            action_history=[context['action_history'][-1],m.core.normalized_tokens(p['tasks'][0],commands)[0].tolist()])
    return encode,predict,calls


def execute(p,terminal=100.,fail=None):
    store=Store(); ledger=m.Ledger(store.event); encode,predict,calls=callbacks(p,ledger,terminal,fail)
    images=[dict(index=i,status='not_started') for i in range(92)]; rows=[dict(index=i,status='not_started') for i in range(84)]
    a=m.execute(p,encode,predict,ledger,store,images,rows)
    return a,ledger,store,images,rows,calls


def test_all92_native_encodings_all84_actual_H1_calls_and_chronology():
    p=protocol(); a,ledger,store,images,rows,calls=execute(p)
    assert ledger.counts()=={k:dict(attempted=v,completed=v) for k,v in m.contract()['caps'].items()}
    assert len(calls)==84 and a['planned']==84 and all(x['status']=='encoded' for x in images)
    assert all(x['status']=='predicted' for x in rows)
    for i,(context,commands) in enumerate(calls):
        pid=i//21; assert context['latent_history']==np.stack((np.zeros(192),np.zeros(192),np.full(192,pid+1))).tolist()
        raw=np.concatenate((np.zeros((5,2),np.float32),p['positions'][pid]['prefix_commands']))
        assert np.array_equal(context['action_history'],m.core.normalized_tokens(p['tasks'][pid//2],raw))
        assert np.array_equal(commands,p['contract']['probes'][i%21]['commands'])


def test_terminal_labels_never_change_any_prediction_input_or_probe():
    p=protocol(); x=execute(p,100.); y=execute(p,-900.)
    assert len(x[-1])==len(y[-1])==84
    for (c,u),(d,v) in zip(x[-1],y[-1],strict=True): assert c==d and np.array_equal(u,v)
    assert x[0]['positions'][0]['rows'][0]['actual_Q']!=y[0]['positions'][0]['rows'][0]['actual_Q']


@pytest.mark.parametrize('missing',(0,4,8,9))
def test_partial_fixed_slots_never_replace_or_zero_fill(missing):
    p=protocol(); p['images'][missing]['available']=False
    a,ledger,store,images,rows,calls=execute(p)
    assert len(images)==92 and len(rows)==a['planned']==84 and images[missing]['status']=='missing'
    assert ledger.counts()['encoder']==dict(attempted=91,completed=91)
    if missing in (0,4):
        count=42 if missing==0 else 21
        assert len(calls)==84-count and sum(x['status']=='missing_observed_context' for x in rows)==count
    else:
        assert len(calls)==84
        r=a['positions'][0]['rows'][missing-8]; assert r['actual_Q'] is None and r['predicted_vs_actual_mse'] is None
        if missing==8: assert all(r['actual_delta'] is None for r in a['positions'][0]['rows'])


def test_native_float32_and_raw_history_are_not_rescaled_or_recentered():
    p=protocol(); t=p['tasks'][0]; z=np.arange(192,dtype=np.float32); u=m.physical.raw5(p['positions'][1]['prefix_commands'])
    c=m.observed_context(t,z,z+1,z+2,u)
    assert c['latent_history']==np.stack((z,z+1,z+2)).astype(float).tolist()
    assert np.array_equal(c['action_history'][1],m.core.normalized_tokens(t,u)[0])
    with pytest.raises(ValueError): m.observed_context(t,z.astype(float),z,z,u)


def test_arithmetic_shared_direction_signed_secant_and_zero_missing():
    p=protocol(); pred=[]; actual=[]
    for pid in range(4):
        for probe in m.physical.probes():
            z=np.zeros(192); z[0]=probe['sign']*.5
            pred.append(z.tolist()); actual.append((2*z).tolist())
    a=m.arithmetic(p,pred,actual); row=a['positions'][0]; c=row['columns'][0]
    assert c['predicted']['B'][0]==2 and c['actual']['B'][0]==4
    assert c['predicted']['signed']['raw']==2 and c['actual']['signed']['unit']==pytest.approx(4/np.sqrt(192))
    assert np.asarray(row['predicted_B_192x10']).shape==(192,10)
    actual[1]=None; a=m.arithmetic(p,pred,actual); assert a['positions'][0]['columns'][0]['actual']['B'] is None
    assert a['positions'][0]['actual_B_192x10'] is None and len(a['positions'][0]['rows'])==21
    p['tasks'][0]['goal']=np.zeros(192).tolist(); a=m.arithmetic(p,pred,actual)
    assert not a['positions'][0]['direction_defined'] and a['positions'][0]['rows'][0]['predicted_signed'] is None


def test_durable_attempt_nonrefund_no_retry_and_scope_caps():
    sink=Store(); ledger=m.Ledger(sink.event); ledger.scope=('image',0)
    with pytest.raises(RuntimeError,match='primary'): ledger.call('encoder',lambda:(_ for _ in ()).throw(RuntimeError('primary')))
    assert ledger.counts()['encoder']==dict(attempted=1,completed=0)
    with pytest.raises(ValueError): ledger.call('encoder',lambda:None)
    assert sink.events[0]['kind']=='module_attempt'
    ledger=m.Ledger(sink.event)
    with pytest.raises(ValueError): ledger.call('encoder',lambda:None)
    ledger.scope=('predict',0)
    with pytest.raises(ValueError): ledger.call('predict',lambda:None)
    for i in range(84):
        ledger.scope=('predict',i); ledger.call('action_encoder',lambda:None); ledger.call('predict',lambda:None)
    ledger.scope=('predict',84)
    with pytest.raises(ValueError,match='cap'): ledger.call('action_encoder',lambda:None)


def test_synchronize_error_attempt_is_not_completed():
    ledger=m.Ledger(lambda e:None,sync=lambda:(_ for _ in ()).throw(RuntimeError('sync'))); ledger.scope=('image',0)
    with pytest.raises(RuntimeError): ledger.call('encoder',lambda:None)
    assert ledger.counts()['encoder']==dict(attempted=1,completed=0)


def test_actual_saved_material_RAM_only_no_prepare_no_native():
    p,_=m.material(str(m.ROOT/'results/p1_constructive_common_prefix_probe_physical_20261027_v1_main01/report.json'),m.REPORT_SHA)
    assert len(p['images'])==92 and len(p['rows'])==84 and all(x['available'] for x in p['images'])
    assert len(p['tasks'])==2 and p['contract']==m.contract() and len(m.canonical(p))<=2*1024**2


def test_external_wrong_report_SHA_rejected_before_reads():
    with pytest.raises(ValueError,match='pins'): m.material('/never/read','a'*64)


def test_zero_measurement_is_not_missing_or_direction_recentered():
    p=protocol(); zeros=[np.zeros(192).tolist() for _ in range(84)]
    a=m.arithmetic(p,zeros,zeros)
    assert a['positions'][0]['common_direction']==a['positions'][1]['common_direction']==p['tasks'][0]['goal']
    r=a['positions'][0]['rows'][0]
    assert r['predicted_vs_actual_mse']==0. and r['actual_signed']['raw']==0. and r['actual_delta']==[0.]*192


def test_guard_module_actual_API_shapes_inference_AD_denial_and_restore():
    fn=lambda *args,**kw:'output'
    model=SimpleNamespace(**{k:SimpleNamespace(forward=fn) for k in ('encoder','projector','action_encoder','predictor')})
    torch=SimpleNamespace(is_grad_enabled=lambda:False,autograd=SimpleNamespace(grad=fn,backward=fn,
        functional=SimpleNamespace(**{k:fn for k in ('jacobian','hessian','jvp','vjp','hvp','vhp')})),
        Tensor=SimpleNamespace(backward=fn),func=SimpleNamespace(**{k:fn for k in ('grad','grad_and_value','jacrev','jacfwd','hessian','jvp','vjp')}))
    ledger=m.Ledger(lambda e:None)
    tensor=lambda shape:SimpleNamespace(device=SimpleNamespace(type='cuda'),shape=shape)
    with m.guard(model,torch,ledger):
        ledger.scope=('image',0)
        assert model.encoder.forward(tensor((1,3,224,224)))=='output'
        assert model.projector.forward(tensor((1,192)))=='output'
        ledger.scope=('predict',0)
        model.action_encoder.forward(tensor((1,3,10))); model.predictor.forward(tensor((1,3,192)))
        with pytest.raises(ValueError,match='AD'): torch.autograd.grad()
    assert all(getattr(model,k).forward is fn for k in ('encoder','projector','action_encoder','predictor'))
    assert torch.autograd.grad is fn and ledger.failed
    assert ledger.counts()=={k:dict(attempted=1,completed=1) for k in m.contract()['caps']}
    ledger=m.Ledger(lambda e:None); ledger.scope=('image',0)
    with m.guard(model,torch,ledger):
        with pytest.raises(ValueError,match='native'): model.encoder.forward(tensor((2,3,224,224)))
    assert ledger.counts()['encoder']['attempted']==0


@pytest.fixture(scope='module')
def saved():
    path=m.ROOT/'results/p1_constructive_common_prefix_probe_physical_20261027_v1_main01/report.json'
    r=m.read(path); binding=m.read(path.parent/'execution_binding.json'); p=m.read(binding['prepared'])
    j=m.load_journal(path.parent/'journal.jsonl',r['journal_chain_tail'])
    return p,r,j


@pytest.mark.parametrize('damage',('root','probe','signature','guard','state','counts'))
def test_saved_join_tamper_failclosed_without_any_neural(saved,damage):
    p,r,j=deepcopy(saved)
    if damage=='root': r['rows'][0]['root_slot']=1
    elif damage=='probe': p['plans'][0]['commands'][5][0]=.125
    elif damage=='signature': r['rows'][1]['prefix_signature']['RGB_bytes_sha256']='a'*64
    elif damage=='counts': j['counts']['public_step']['completed']-=1
    else:
        e=next(x for x in j['events'] if x.get('branch')==0 and x['kind']==('fresh_observation' if damage=='state' else 'guard_receipt') and
            (x['receipt']['phase']=='plan' and x['receipt']['step']==5 if damage=='state' else x['phase']=='plan' and x['step']==5))
        if damage=='state': e['receipt']['state'][0]+=1
        else: e['file_sha256']='b'*64
    with pytest.raises(ValueError): m.assemble(p,r,j)


def test_saved_partial_terminal_preserves_all84_and_zero_missing(saved):
    p,r,j=deepcopy(saved); r['status']=j['last']['status']='partial_or_failed'
    row=r['rows'][0]; row.update(status='failed_partial',full_horizon=False,first_error=dict(message='mock terminal guard'))
    closed=next(e for e in j['events'] if e['kind']=='branch_closed' and e['receipt']['index']==0); closed['receipt']=deepcopy(row)
    images,rows=m.assemble(p,r,j)
    assert len(images)==92 and len(rows)==84 and not images[8]['available']
    assert all(x['available'] for x in images[:8]) and sum(x['available'] for x in images)==91
