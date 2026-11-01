"""RAM-only pure arrays and CUDA-interface stand-ins; no torch or simulator."""
from copy import deepcopy
from io import BytesIO
import gzip
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
from experiments.spectral_reachability import moving_history_observed_suffix_cuda_20261027_v1 as m


spec=importlib.util.spec_from_file_location('moving_test_helpers',Path(__file__).with_name('test_moving_history_planners_cuda_20261027_v1.py'))
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


def fixture(slot=0,arm=None):
    arm=arm or m.ARMS[0];i=h.inputs(17+slot)
    raw=np.linspace(-.2,.3,20,dtype=np.float32).reshape(10,2)
    plan=dict(schema=m.planner.NAME+'.model_plan',source_slot=slot,arm=arm,task_id=i['task_id'],
        task_export=m.planner.EXPORTS[slot],task_sha256=m.planner.EXPORT_SHAS[slot],blind_sha256='b'*64,
        commands=raw.astype(float).tolist(),with_target=False,selection_uses_physics=False,P1_admission=False)
    plan['selection_sha256']=m.core.digest(plan)
    item={k:plan[k] for k in ('source_slot','arm','task_id','task_export','task_sha256','blind_sha256')}
    item.update(root_slot=slot,role='prefix5',planner_selection_sha256=plan['selection_sha256'],
        selection_digest='c'*64,checkpoint=dict(step=5,phase='plan'))
    z=np.linspace(-.4,.7,192,dtype=np.float32);buf=BytesIO();np.save(buf,z,allow_pickle=False)
    entry=dict(path='/RAM/prefix.npy',file='prefix.npy',sha256=m.observed.enc.digest(buf.getvalue()),dtype='float32',shape=[192])
    return i,plan,item,z,entry,buf.getvalue()


def branch(slot=0,arm=None):
    i,p,item,z,entry,_=fixture(slot,arm)
    return m.branch_input(slot,p['arm'],i,p,item,z,entry)


def array(store,entry):
    raw=store.files[entry['file']];assert m.observed.enc.digest(raw)==entry['sha256']
    return np.load(BytesIO(raw),allow_pickle=False)


class Tensor:
    """Only a device-labelled NumPy interface stand-in, never a torch tensor."""
    def __init__(self,a,device='cuda'):self.a=np.asarray(a,np.float32);self.device=NS(type=device)
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.a


class Backend:
    def __init__(self,b,modules=None,fail=None):
        self.b,self.modules,self.fail=b,modules,fail;self.calls=[]
        self.j=np.zeros((192,10),np.float64);self.j[:10]=np.eye(10)*.5
    def step(self,c,raw):
        self.calls.append(self.modules.meter.stage if self.modules else 'step')
        incoming=m.planner.incoming(self.b,c,raw)
        z=(np.asarray(c['latent_history'][-1])+self.j@np.asarray(raw).reshape(10)).astype(np.float32)
        if self.modules:
            ae=np.asarray(incoming['action_history'],np.float32)[None]
            lat=np.asarray(incoming['latent_history'],np.float32)[None]
            out=lat.copy();out[0,-1]=z
            for kind,ins,output in (('action_encoder',[ae],ae),('predict',[lat,ae],out),('pred_proj',[out.reshape(3,192)],out.reshape(3,192))):
                obj=kind;args=tuple(Tensor(a) for a in ins)
                self.modules.pre(kind)(obj,args)
                if self.fail==self.modules.meter.stage and kind=='predict':
                    self.modules.post(obj,args,None);raise RuntimeError('original forward failed')
                self.modules.post(obj,args,Tensor(output))
        return dict(c,latent_history=deepcopy(c['latent_history'][1:])+[z.astype(float).tolist()],action_history=incoming['action_history'][-2:])
    def nominal(self,c):return np.asarray(self.step(c,np.zeros((5,2),np.float32))['latent_history'][-1])
    def jacobian(self,c,operation):
        def calculate():self.nominal(c);return self.j.copy()
        return operation('jacobian',calculate)


def execute(b=None,fail=None):
    b=b or branch();store=h.h.RAMStore();meter=m.Meter(store.event);meter.require_native=True
    modules=m.Modules(NS(),store,'branch_00',meter);backend=Backend(b,modules,fail)
    row=dict(index=0,branch_id=b['branch_id'])
    try:m.evaluate(b,backend,meter,store,row)
    finally:
        row.update(module_ledger=deepcopy(modules.entries),math_ledger=deepcopy(meter.entries),
            resources={**modules.counts(),**meter.counts()},AD_API_resources={k:dict(attempted=1,completed=1) for k in ('grad','jacobian')},
            AD_API_ledger=[dict(index=0,resource='jacobian',completed=True,is_grads_batched=False),dict(index=1,resource='grad',completed=True,is_grads_batched=True)])
    return b,store,row,backend,meter


def test_actual_pins_and_fixed_budget_no_outcome_rule():
    c=m.contract()
    assert c['total']==dict(predict=20,action_encoder=20,pred_proj=20,jacobian=4,svd=4)
    assert c['mathematical_VJP_total']==768 and not c['P1_admission'] and not c['P4_admission']
    assert m.RGB_REPORT_SHA=='7ec03c0a6a6e8411ff0dd0115044a872d099106edc95e4529b97c38c2b9b8f71'


@pytest.mark.parametrize('slot',[0,1])
@pytest.mark.parametrize('arm',m.ARMS)
def test_observed_context_exact_history_normalization(slot,arm):
    i,p,item,z,entry,_=fixture(slot,arm);b=m.branch_input(slot,arm,i,p,item,z,entry)
    assert b['observed_context']['latent_history']==i['gn_task']['root_context']['latent_history'][1:]+[z.astype(float).tolist()]
    raw=np.concatenate((m.planner.fixed_raw()[5:],np.asarray(p['commands'][:5],np.float32)))
    assert b['observed_context']['action_history']==m.core.normalized_tokens(i['gn_task'],raw).astype(float).tolist()
    assert b['observed_raw_history']==raw.astype(float).tolist()
    assert b['branch_id']==m.core.digest({k:v for k,v in b.items() if k!='branch_id'})


@pytest.mark.parametrize('bad',['role','step','phase','arm','root_slot','task_id','task_sha256','selection','raw','Q','warmup'])
def test_input_binding_tamper_rejected(bad):
    i,p,item,z,entry,_=fixture()
    if bad=='step':item['checkpoint']['step']=10
    elif bad=='phase':item['checkpoint']['phase']='warmup'
    elif bad=='role':item['role']='terminal10'
    elif bad=='selection':p['selection_sha256']='d'*64
    elif bad=='raw':p['commands'][0][0]+=.00000000001
    elif bad=='Q':i['q'][0][0]=2
    elif bad=='warmup':i['gn_task']['root_raw_action_history'][0][0]=0
    else:item[bad]=99 if bad=='root_slot' else 'changed'
    with pytest.raises(ValueError):m.branch_input(0,m.ARMS[0],i,p,item,z,entry)


def test_five_forwards_native_arrays_single_solve_and_unconditional_release():
    b,s,r,backend,meter=execute();m.check_result(b,r);m.saved_arrays(b,r,lambda e:array(s,e))
    m.blind.exact_counts(r['resources'],m.budget())
    assert backend.calls==list(m.STAGES) and len(r['module_ledger'])==15
    assert [e['resource'] for e in meter.entries]==['jacobian','svd']
    assert r['dispatch']=='new_suffix_unconditionally'
    assert sum(map(len,s.files.values()))<1024**2
    s.close();journal,eof=m.blind.inflate(s.files['events.jsonl.gz']);assert eof
    events=[m.blind.decode(x) for x in journal.splitlines()]
    kinds=[e['kind'] for e in events]
    assert kinds.index('new_suffix_frozen_before_evaluation')<kinds.index('suffix_evaluation_attempt')
    assert kinds.index('nominal_saved')<kinds.index('jacobian_saved')<kinds.index('new_suffix_frozen_before_evaluation')


def test_old_diagnostic_better_still_dispatches_new_suffix():
    b=branch();reference=Backend(b).step(b['observed_context'],b['old_suffix_commands'])
    b['goal']=reference['latent_history'][-1]
    b['branch_id']=m.core.digest({k:v for k,v in b.items() if k!='branch_id'})
    b,s,r,_,_=execute(b);m.check_result(b,r);m.saved_arrays(b,r,lambda e:array(s,e))
    assert r['evaluations'][0]['cost']==0 and r['evaluations'][1]['cost']>0
    assert r['dispatch']=='new_suffix_unconditionally'


def test_invalid_prepare_never_opens_output_or_imports_model(monkeypatch):
    def fail(*args):raise ValueError('external prepare mismatch')
    monkeypatch.setattr(m,'verify',fail)
    monkeypatch.setattr(m,'Store',lambda *args:pytest.fail('store opened before input verify'))
    with pytest.raises(ValueError,match='external prepare mismatch'):m.run('/RAM/prepare','a'*64,'/RAM/output')


@pytest.mark.parametrize('bad',['history','raw','cost','replay','dispatch','cap','new'])
def test_result_contract_rejects_tamper(bad):
    b,_,r,_,_=execute()
    if bad=='history':r['evaluations'][0]['context']['action_history'][0][0]+=1
    elif bad=='raw':r['evaluations'][0]['commands'][0][0]+=1
    elif bad=='cost':r['evaluations'][0]['cost']+=1
    elif bad=='replay':r['evaluations'][2]['context']['latent_history'][-1][0]+=1
    elif bad=='dispatch':r['dispatch']='best_old_or_new'
    elif bad=='cap':r['solve']['radius']*=2
    else:r['new_suffix_commands'][0][0]+=.001
    with pytest.raises(ValueError):m.check_result(b,r)


@pytest.mark.parametrize('bad',['input','output','order','math','AD','jacobian','array'])
def test_saved_native_trace_rejects_tamper(bad):
    b,s,r,_,_=execute()
    if bad in ('input','output'):
        e=r['module_ledger'][1];entry=e['inputs'][0] if bad=='input' else e['output'];a=array(s,entry).copy();a.flat[0]+=1
        new=m.save_array(s,'tampered.npy',a)
        if bad=='input':e['inputs'][0]=new
        else:e['output']=new
    elif bad=='order':r['module_ledger'][1]['stage']='nominal_wrong'
    elif bad=='math':r['math_ledger'].pop()
    elif bad=='AD':r['AD_API_ledger'][-1]['is_grads_batched']=False
    elif bad=='jacobian':r['arrays']['jacobian']=m.save_array(s,'bad_j.npy',np.zeros((192,9)))
    else:r['arrays']['observed_tokens']=m.save_array(s,'bad_tokens.npy',np.zeros((2,10),np.float32))
    with pytest.raises(ValueError):m.saved_arrays(b,r,lambda e:array(s,e))


@pytest.mark.parametrize('stage',m.STAGES)
def test_failed_forward_keeps_original_inputs_attempt_and_partial_arrays(stage):
    b=branch();s=h.h.RAMStore();meter=m.Meter(s.event);meter.require_native=True
    modules=m.Modules(NS(),s,'branch_00',meter);backend=Backend(b,modules,stage);r=dict(index=0,branch_id=b['branch_id'])
    with pytest.raises(RuntimeError,match='original forward failed'):m.evaluate(b,backend,meter,s,r)
    assert modules.entries[-1]['resource']=='predict' and not modules.entries[-1]['completed']
    assert len(modules.entries[-1]['inputs'])==2 and modules.failed
    assert backend.calls==list(m.STAGES[:m.STAGES.index(stage)+1])
    if stage!='nominal':assert 'nominal' in r['arrays']
    if stage in m.STAGES[2:]:assert 'jacobian' in r['arrays'] and 'solve' in r and 'new_suffix' in r['arrays']


def test_math_failure_no_retry_even_journal_failure():
    def fail(e):raise RuntimeError('journal unavailable')
    meter=m.Meter(fail);meter.stage='jacobian'
    with pytest.raises(RuntimeError):meter.call('jacobian',lambda:None)
    assert meter.counts()['jacobian']==dict(attempted=1,completed=0)
    with pytest.raises(ValueError):meter.call('jacobian',lambda:None)


def test_modules_refuse_cpu_and_wrong_nonzero_tokens():
    b=branch();meter=m.Meter(lambda e:None);meter.stage='nominal';meter.expected_input=m.planner.incoming(b,b['observed_context'],np.zeros((5,2),np.float32))
    modules=m.Modules(NS(),h.h.RAMStore(),'branch_00',meter)
    with pytest.raises(ValueError):modules.array(Tensor(np.zeros(3),'cpu'))
    with pytest.raises(ValueError):modules.pre('action_encoder')('ae',(Tensor(np.zeros((1,3,10))),))
    assert modules.counts()['action_encoder']==dict(attempted=1,completed=0)


def test_AD_vectorized_single_API_and_restoration():
    torch=h.h.fake_torch();events=[];grad=torch.autograd.grad
    torch.autograd.functional.jacobian=lambda *a,**kw:torch.autograd.grad(is_grads_batched=True)
    original=torch.autograd.functional.jacobian
    ad=m.SuffixAD(torch,m.blind.ARMS[1],events.append,lambda:None);ad.stage=lambda:'jacobian'
    with ad:
        torch.autograd.functional.jacobian()
        with pytest.raises(ValueError):torch.autograd.functional.jacobian()
    assert torch.autograd.grad is grad and torch.autograd.functional.jacobian is original
    assert ad.counts()=={k:dict(attempted=1,completed=1) for k in ('jacobian','grad')}


def test_AD_failed_journal_keeps_depth_and_attempt():
    torch=h.h.fake_torch()
    def fail(e):raise RuntimeError('journal failed')
    ad=m.SuffixAD(torch,m.blind.ARMS[1],fail,lambda:None);ad.stage=lambda:'jacobian'
    with ad:
        with pytest.raises(RuntimeError):torch.autograd.functional.jacobian()
    assert ad.depth==0 and ad.counts()['jacobian']==dict(attempted=1,completed=0)


def test_full_denominator_and_first_failure_no_replacement(monkeypatch):
    branches=[branch(s,a) for s in (0,1) for a in m.ARMS];rows=[dict(index=k,branch_id=b['branch_id'],status='not_started') for k,b in enumerate(branches)]
    monkeypatch.setattr(m,'load_npy',lambda *a:(np.zeros(192,np.float32),b'original bytes'))
    s=h.h.RAMStore();calls=[]
    def fail(b,store,row):calls.append(b['branch_id']);row['arrays']={'partial':m.save_array(store,'partial.npy',np.zeros(192,np.float32))};raise RuntimeError('first')
    with pytest.raises(RuntimeError,match='first'):m.compile_all(dict(branches=branches),s,fail,rows)
    assert len(calls)==1 and [r['status'] for r in rows]==['attempted']+['not_started']*3
    assert s.files['branch_00.original_prefix.native192.npy']==b'original bytes' and 'partial.npy' in s.files


def test_prepare_fixed_actual_report_gate_before_IO():
    with pytest.raises(ValueError):m.build('/RAM/report.json','a'*64,m.RGB_SOURCE_SHA)


@pytest.fixture
def saved_RAM(monkeypatch):
    branches=[branch(s,a) for s in (0,1) for a in m.ARMS]
    p=dict(branches=branches,contract=m.contract(),expected_model_hash='e'*64)
    rows=[dict(index=k,source_slot=b['source_slot'],arm=b['arm'],branch_id=b['branch_id'],status='not_started') for k,b in enumerate(branches)]
    store=h.h.RAMStore();ph='f'*64
    store.put('binding.json',dict(prepared='/RAM/prepare.json',prepare_sha256=ph))
    raw=fixture()[-1]
    monkeypatch.setattr(m,'load_npy',lambda *args:(np.load(BytesIO(raw),allow_pickle=False),raw))
    def runner(b,s,r):
        meter=m.Meter(s.event);meter.require_native=True
        modules=m.Modules(NS(),s,f'branch_{r["index"]:02d}',meter);backend=Backend(b,modules)
        original=backend.jacobian
        ad=[dict(index=0,resource='jacobian',completed=True,is_grads_batched=False),dict(index=1,resource='grad',completed=True,is_grads_batched=True)]
        def jac(c,operation):
            s.event(dict(kind='AD_API_attempt',entry=dict(ad[0],completed=False)))
            value=original(c,operation)
            s.event(dict(kind='AD_API_attempt',entry=dict(ad[1],completed=False)))
            s.event(dict(kind='AD_API_completed',entry=ad[1]));s.event(dict(kind='AD_API_completed',entry=ad[0]));return value
        backend.jacobian=jac
        m.evaluate(b,backend,meter,s,r)
        r.update(module_ledger=deepcopy(modules.entries),math_ledger=deepcopy(meter.entries),AD_API_ledger=ad,
            resources={**modules.counts(),**meter.counts()},AD_API_resources={k:dict(attempted=1,completed=1) for k in ('grad','jacobian')})
    m.compile_all(p,store,runner,rows);store.close()
    report=dict(schema=m.NAME+'.report',status='observed_suffix_calibration_sealed_not_admitted',commands_released=True,
        errors=[],first_error=None,source_verified=True,model_unchanged=True,model_hash_before='e'*64,model_hash_after='e'*64,
        prepared='/RAM/prepare.json',prepare_sha256=ph,rows=rows,contract=m.contract(),P1_admission=False,P4_admission=False,
        artifact_sha256={n:m.observed.enc.digest(raw) for n,raw in store.files.items()})
    files={'/RAM/run/'+n:raw for n,raw in store.files.items()}
    class RAMPath:
        def __init__(self,p):self.p=str(p)
        def __str__(self):return self.p
        def __truediv__(self,n):return RAMPath(self.p+'/'+n)
        def absolute(self):return self
        @property
        def name(self):return self.p.rsplit('/',1)[-1]
        def read_bytes(self):return files[self.p]
        def stat(self):return NS(st_size=len(files[self.p]))
        def iterdir(self):return [RAMPath(k) for k in files if k.startswith(self.p+'/')]
    def read(path,expected=None):
        data=files[str(path)]
        if expected is not None:m.require(m.observed.enc.digest(data)==expected,'RAM external hash')
        return m.blind.decode(gzip.decompress(data) if str(path).endswith('.gz') else data)
    def publish():
        files['/RAM/run/report.json']=m.canonical(report)
        return m.observed.enc.digest(files['/RAM/run/report.json'])
    monkeypatch.setattr(m,'Path',RAMPath);monkeypatch.setattr(m,'read',read);monkeypatch.setattr(m.blind,'read',read)
    monkeypatch.setattr(m,'sha',lambda p:m.observed.enc.digest(files[str(p)]))
    monkeypatch.setattr(m,'verify',lambda *a:deepcopy(p))
    monkeypatch.setattr(m.observed.rgb.old,'local',lambda d,n:d/n)
    return files,report,publish


def test_complete_four_branch_own_consume_RAM(saved_RAM):
    files,r,publish=saved_RAM;out=m.consume('/RAM/run',publish())
    assert out['status']=='observed_suffix_commands_verified_not_admitted' and out['errors']==[]
    assert len(out['plans'])==4 and all(p['blind_sha256']=='b'*64 and len(p['commands'])==10 for p in out['plans'])
    assert [(p['source_slot'],p['arm']) for p in out['plans']]==[(s,a) for s in (0,1) for a in m.ARMS]
    assert sum(map(len,files.values()))<m.CAP
    print('RAM full output bytes:',sum(map(len,files.values())))


@pytest.mark.parametrize('bad',['schema','partial','extra','artifact','roworder','blind','binding','journal'])
def test_consume_rejects_relabel_partial_and_artifact_tamper(saved_RAM,bad):
    files,r,publish=saved_RAM
    if bad=='schema':r['schema']='old.report'
    elif bad=='partial':r['commands_released']=False
    elif bad=='extra':files['/RAM/run/extra']=b'{}'
    elif bad=='artifact':files['/RAM/run/branch_00.nominal.npy']=b'bad'
    elif bad=='roworder':r['rows'].reverse()
    elif bad=='blind':r['rows'][0]['plan']['blind_sha256']='d'*64
    elif bad=='binding':files['/RAM/run/binding.json']=b'{}'
    else:files['/RAM/run/events.jsonl.gz']=files['/RAM/run/events.jsonl.gz'][:-8]
    with pytest.raises(ValueError):m.consume('/RAM/run',publish())

