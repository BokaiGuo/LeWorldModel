"""RAM-only native-interface doubles; no torch, real model or simulator."""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_contextual_midfeedback_recompile_20261028_v2 as m
import test_agent_p2_contextual_midfeedback_recompile_20261028_v1 as prior
import test_moving_history_observed_suffix_cuda_20261027_v1 as core
import test_agent_p2_contextual_composition_physical_handoff_20261028_v1 as phys


def branch():
    spec,plan,u=prior.inputs()
    b=m.old.branch(spec,plan,prior.rgbtest.z(.25),prior.rgbtest.z(.5),u)
    return m.with_incumbent(b,u)


class Memory:
    def __init__(self):self.files={};self.events=[]
    def _write(self,name,value):assert name not in self.files;self.files[name]=value
    def event(self,value):self.events.append(deepcopy(value))
    def load(self,e):
        b=self.files[e['file']];assert m.hashlib.sha256(b).hexdigest()==e['sha256']
        a=np.load(BytesIO(b),allow_pickle=False);assert str(a.dtype)==e['dtype'] and list(a.shape)==e['shape'];return a


class Backend:
    """Literal production module pre/post and event ABI, NumPy arithmetic only."""
    def __init__(self,b,modules,store,mode=None):
        self.b,self.modules,self.store,self.mode=b,modules,store,mode;self.api=[];self.ad=[]
        self.j=np.zeros((192,10),np.float64);self.j[:10]=np.eye(10)*.5
    def step(self,c,raw):
        meter=self.modules.meter;stage=meter.stage
        incoming=m.F.planner.incoming(self.b,c,raw)
        z=(np.asarray(c['latent_history'][-1])+self.j@np.asarray(raw).reshape(10)).astype(np.float32)
        if self.mode=='worse' and not np.array_equal(raw,self.b['old_suffix_commands']):z+=10
        ae=np.asarray(incoming['action_history'],np.float32)[None];lat=np.asarray(incoming['latent_history'],np.float32)[None]
        out=lat.copy();out[0,-1]=z
        entry=dict(index=len(self.api),stage=stage,completed=False);self.api.append(entry)
        for kind,ins,value in (('action_encoder',[ae],ae),('predict',[lat,ae],out),('pred_proj',[out.reshape(3,192)],out.reshape(3,192))):
            if kind=='predict':self.store.event(dict(kind='predict_API_attempt',entry=deepcopy(entry)))
            args=tuple(core.Tensor(a) for a in ins);self.modules.pre(kind)(kind,args);self.modules.post(kind,args,core.Tensor(value))
            if kind=='pred_proj':entry['completed']=True;self.store.event(dict(kind='predict_API_return',entry=deepcopy(entry)))
        return dict(c,latent_history=deepcopy(c['latent_history'][1:])+[z.astype(float).tolist()],action_history=incoming['action_history'][-2:])
    def jacobian_raw(self,c,incumbent,operation):
        def calculate():
            e=dict(index=0,resource='jacobian',completed=False,is_grads_batched=False);self.ad.append(e)
            self.store.event(dict(kind='AD_API_attempt',entry=deepcopy(e)))
            self.step(c,incumbent)
            g=dict(index=1,resource='grad',completed=False,is_grads_batched=True);self.ad.append(g)
            self.store.event(dict(kind='AD_API_attempt',entry=deepcopy(g)))
            g['completed']=True;self.store.event(dict(kind='AD_API_completed',entry=deepcopy(g)))
            e['completed']=True;self.store.event(dict(kind='AD_API_completed',entry=deepcopy(e)))
            return self.j.copy()
        return operation('jacobian',calculate)


def execute(b=None,mode=None):
    b=b or branch();s=Memory();meter=m.F.Meter(s.event);meter.require_native=True
    modules=m.Modules(NS(),s,'branch_00',meter);backend=Backend(b,modules,s,mode)
    r=dict(index=0,branch_id=b['branch_id']);m.evaluate(b,backend,meter,s,r)
    r.update(module_ledger=deepcopy(modules.entries),math_ledger=deepcopy(meter.entries),
        resources={**modules.counts(),**meter.counts()},predict_API_ledger=deepcopy(backend.api),
        predict_API_resources=dict(predict=dict(attempted=7,completed=7)),AD_API_ledger=deepcopy(backend.ad),
        AD_API_resources={k:dict(attempted=1,completed=1) for k in ('jacobian','grad')})
    return b,s,r


@pytest.fixture(scope='module')
def completed():return execute()


def test_full_native_schema_arrays_seven_calls_and_model_safeguard(completed):
    b,s,r=completed;m.check_compile(b,r,s.load);m.check_events(b,r,s.events)
    assert len(r['module_ledger'])==21 and len(r['evaluations'])==5
    assert r['replay']['cost']<=r['evaluations'][0]['cost'] and r['resources']['svd']==dict(attempted=0,completed=0)
    assert r['selected_index']>0 and r['proposal']['slope']<0


def test_nonlinear_worsening_retains_incumbent_not_new_or_physics():
    b,s,r=execute(mode='worse');m.check_compile(b,r,s.load);m.check_events(b,r,s.events)
    assert r['selected_index']==0 and r['new_suffix_commands']==b['old_suffix_commands']
    assert all(e['cost']>r['evaluations'][0]['cost'] for e in r['evaluations'][1:])


def test_zero_gradient_ties_keep_old_all_candidates_evaluated():
    b=branch();b['goal']=(np.asarray(b['observed_context']['latent_history'][-1])+
        Backend(b,None,None).j@np.asarray(b['old_suffix_commands']).reshape(10)).astype(np.float32).tolist()
    b['branch_id']=m.F.core.digest({k:v for k,v in b.items() if k!='branch_id'})
    b,s,r=execute(b);m.check_compile(b,r,s.load);m.check_events(b,r,s.events)
    assert r['selected_index']==0 and len(r['evaluations'])==5 and r['replay']['cost']==0
    assert not r['proposal']['descent_used'] and np.array_equal(r['proposal']['direction'],np.zeros(10))


def test_exact_chart_not_float32_tanh_and_no_inverse_rounding():
    b=branch();u=np.asarray(b['incumbent_logits'])
    assert np.array_equal(m.chart(u),b['old_suffix_commands'])
    assert m.chart(u).tobytes()!=np.tanh(u.astype(np.float32)).reshape(5,2).tobytes()
    assert not np.array_equal(np.arctanh(np.asarray(b['old_suffix_commands'],np.float64)).reshape(10),u)


def test_suffix_bits_exact5_not_full10_contract_and_signed_zero():
    a=np.zeros((5,2),np.float32);b=a.copy();b[0,0]=-0.
    assert m.suffix_bits(a)!=m.suffix_bits(b)
    with pytest.raises(ValueError):m.suffix_bits(np.zeros((10,2)))
    with pytest.raises(ValueError):m.suffix_bits(np.full((5,2),.1,np.float64))


@pytest.mark.parametrize('fault',['raw','logits','remaining','shape'])
def test_original_mapping_and_domain_identity_rejected(fault):
    b=prior.branch();_,_,u=prior.inputs()
    if fault=='raw':b['old_suffix_commands'][0][0]+=.01
    elif fault=='logits':u[12]+=.01
    elif fault=='remaining':b['packet_logit_cap']+=1e-12
    else:u=u[:10]
    with pytest.raises(ValueError):m.with_incumbent(b,u)


def test_illegal_incumbent_not_projected_or_repaired_before_neural():
    b=branch();b['packet_logit_cap']=float(np.linalg.norm(b['incumbent_logits'])/2)
    b['incumbent_domain']=m.legality(b,b['incumbent_logits']);assert not b['incumbent_domain']['legal']
    with pytest.raises(ValueError,match='legal exact incumbent'):m.evaluate(b,NS(),NS(),NS(),{})


def test_no_tolerance_for_nextafter_over_radius():
    b=branch();u=np.asarray(b['incumbent_logits']);b['packet_logit_cap']=float(np.nextafter(np.linalg.norm(u),0))
    assert not m.legality(b,u)['legal']


@pytest.mark.parametrize('fault',['unknown','drop_attempt','drop_input','drop_return','order','stage','proposal','selection','evaluation','pending'])
def test_all_native_events_strict_no_skip(completed,fault):
    b,s,r=completed;events=deepcopy(s.events)
    if fault=='unknown':events.append(dict(kind='future_unknown'))
    elif fault in ('drop_attempt','drop_input','drop_return'):
        k={'drop_attempt':'module_attempt','drop_input':'module_inputs_saved','drop_return':'module_return'}[fault]
        events.remove(next(e for e in events if e['kind']==k))
    elif fault=='order':
        j=next(j for j,e in enumerate(events) if e['kind']=='module_return');events[j-1],events[j]=events[j],events[j-1]
    elif fault=='stage':next(e for e in events if e['kind']=='stage_started')['expected_input']['latent_history'][0][0]+=1
    elif fault=='proposal':events.remove(next(e for e in events if e['kind']=='candidates_frozen'))
    elif fault=='selection':next(e for e in events if e['kind']=='selection_frozen')['index']=99
    elif fault=='evaluation':events.remove(next(e for e in events if e['kind']=='suffix_evaluation_return'))
    else:events.append(dict(kind='math_attempt',entry=dict(index=999,resource='jacobian',completed=False,stage='selected_replay')))
    with pytest.raises(ValueError):m.check_events(b,r,events)


@pytest.mark.parametrize('fault',['Q','context','module_input','old_logits','proposal','winner','cost','drop_candidate','budget','replay','guard'])
def test_saved_full_buffers_selection_cost_and_counts(completed,fault):
    b,s,original=completed;b=deepcopy(b);r=deepcopy(original)
    if fault=='Q':b['q'][0][0]=2
    elif fault=='context':b['observed_context']['latent_history'][0][0]+=1
    elif fault=='module_input':r['module_ledger'][3]['inputs']=[]
    elif fault=='old_logits':b['incumbent_logits'][0]+=.01
    elif fault=='proposal':r['proposal']['logits'][2][0]+=.01
    elif fault=='winner':r['selected_index']=0
    elif fault=='cost':r['evaluations'][1]['cost']=-1
    elif fault=='drop_candidate':r['evaluations'].pop()
    elif fault=='budget':r['resources']['predict']['completed']=6
    elif fault=='guard':r['guard_receipts'][0]['selection_sha256']='0'*64
    else:r['replay']['context']['latent_history'][-1][0]+=.01
    with pytest.raises((ValueError,IndexError,KeyError)):m.check_compile(b,r,s.load)


def test_new_witness_Q_digest_exact_selected_parent_preserved(completed):
    b,s,r=completed;_,parent,_=prior.inputs();original=deepcopy(parent)
    plan=m.seal_plan(b,r,parent,'a'*64);selected=r['evaluations'][r['selected_index']]
    assert parent==original and 'predicted_Q' not in plan and 'witness' not in plan
    assert plan['parent_predicted_Q']==original['predicted_Q'] and plan['parent_witness']==original['witness']
    assert plan['new_suffix_Q']==selected['cost'] and plan['new_suffix_context']==selected['context']
    assert plan['new_suffix_witness']['compile_row_sha256']==m.physical.saved.digest(r)
    assert plan['new_suffix_witness']['raw5']==selected['commands']==plan['commands'][5:]
    assert plan['commands'][:5]==original['commands'][:5] and plan['goal']==original['goal']
    assert plan['raw_float32_bits']==m.physical.bits(plan['commands'])
    assert plan['selection_digest']==m.physical.saved.digest({k:v for k,v in plan.items() if k!='selection_digest'})
    assert plan['safeguard']['no_modeled_worsening'] is True


class GateRAM(m.ReplayGateStore,phys.MemoryStore):
    def __init__(self,midpoints):phys.MemoryStore.__init__(self);self.midpoints=midpoints;self.matched=set();self.started=m.time.monotonic()


@pytest.fixture(scope='module')
def native():
    with pytest.MonkeyPatch.context() as patch:
        r,s,_,p,data=phys.execute(patch);strict=phys.assemble(r,s,p,data)
    return p,data,[r['rgb_encoding_inputs'][1] for r in strict['rows']]


def test_real_full10_native_runner_and_gate_schema(monkeypatch,native):
    p,data,mid=native;s=GateRAM(mid)
    r,s,env,pp,dd=phys.execute(monkeypatch,p=deepcopy(p),data=data,store=s)
    strict=phys.assemble(r,s,pp,dd)
    assert s.matched==set(range(8)) and len(strict['rows'])==8
    assert r['counts']['public_step']==dict(attempted=160,completed=160)


@pytest.mark.parametrize('field',['file_sha256','pixel_sha256','state','snapshot'])
def test_gate_does_not_execute_suffix_if_actual_mid_differs(monkeypatch,native,field):
    p,data,mid=native;mid=deepcopy(mid)
    for row in mid:
        if field=='state':row['state'][0]+=1e-10
        elif field=='snapshot':row['snapshot']['state_float64_bits']='BAD'
        else:row[field]='0'*64
    s=GateRAM(mid);r,s,env,pp,dd=phys.execute(monkeypatch,p=deepcopy(p),data=data,store=s)
    assert all(len(e.commands)==15 for e in env) and len(r['rows'])==8
    assert not s.matched


def test_unauthorized_or_illegal_prepare_opens_no_model_output(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('unexpected verify'))
    with pytest.raises(ValueError):m.compile_run('x','x','x','wrong')
    with pytest.raises(ValueError):m.run_physical('x','x','x',lambda:pytest.fail('factory'),'wrong')
    monkeypatch.setattr(m,'verify',lambda *a:dict(status='blocked_illegal_incumbent'))
    monkeypatch.setattr(m,'Store',lambda *a:pytest.fail('output on blocked domain'))
    with pytest.raises(ValueError):m.compile_run('x','x','x',m.AUTH)


def test_budget_all8_unknown_FLOPs_and_fixed_search_not_science():
    c=m.contract();assert c['plans']==8 and c['known_roots']==2 and c['total_compile']['predict']==56
    assert c['total_compile']['jacobian']==8 and c['total_compile']['svd']==0 and c['total_FLOPs'] is None
    assert not any(c[k] for k in ('P1_admission','P2_admission','fresh','physical_release','scientific_pass'))
    assert c['candidates_per_compile']==5 and tuple(c['alphas'])==m.ALPHAS


def test_MAIN_guard_receives_only_exact_model_whitelist(monkeypatch,completed):
    b,_,r=completed;called=[];original=m.guard.choose
    def record(a,c):
        assert set(a)==set(c)==m.guard.FIELDS
        assert a['context_sha256']==c['context_sha256']==m.F.core.digest(b['observed_context'])
        called.append((a,c));return original(a,c)
    monkeypatch.setattr(m.guard,'choose',record)
    assert m.choose(b,r['evaluations'])==dict(selected_index=r['selected_index'],guard_receipts=r['guard_receipts'])
    assert len(called)==4


def test_native_handoff_dispatches_selected_plans_through_original_execute(monkeypatch,completed):
    b,_,r=completed;_,parent,_=prior.inputs();plan=m.seal_plan(b,r,parent,'a'*64)
    pp=dict(plans=[plan]*8,planner_report_sha256='b'*64,strict_receipt_sha256='c'*64)
    p=dict(midpoints=['mid']*8);data=object();factory=object();seen=[]
    monkeypatch.setattr(m,'physical_material',lambda *a:(p,pp,data))
    monkeypatch.setattr(m.physical,'read',lambda *a:dict(prepared='/RAM/prepare',prepare_sha256='d'*64))
    class Store:
        def __init__(self,path,mid):assert path=='/RAM/new' and mid==p['midpoints'];self.started=None
    monkeypatch.setattr(m,'ReplayGateStore',Store)
    def execute(got,actual,store,env,**kwargs):
        assert got is pp and actual is data and env is factory
        assert kwargs['authorization']==m.physical.AUTH and store.started==kwargs['started']
        assert kwargs['binding']['feedback_compile_sha256']=='a'*64
        assert got['plans'][0]['new_suffix_Q']==r['evaluations'][r['selected_index']]['cost']
        seen.append(True);return dict(status='RAM_callback_only')
    monkeypatch.setattr(m.physical,'execute',execute)
    assert m.run_physical('/RAM/compile','a'*64,'/RAM/new',factory,m.AUTH)['status']=='RAM_callback_only'
    assert seen==[True]


def test_saved_known_only_original_logits_schema_domain_no_model():
    path=m.ROOT/'results/agent_p2_contextual_midfeedback_recompile_20261028_v1_prepare_MAIN01.json'
    if not path.exists():pytest.skip('sealed known-root metadata unavailable; not synthesized')
    assert m.sha(path)=='a415fafcf82f9f7f93aa8a9a1c68d3f205b6b4c27e5763a2f4b346947580516e'
    p=m.physical.read(path);assert m.sha(p['physical_prepare'])==p['physical_prepare_sha256']
    pp=m.physical.read(p['physical_prepare']);output=Path(pp['planner_output'])
    assert m.sha(output/'report.json')=='498b45ed9f4c580064d8f2a5dfd00ac8542b7ce9d0b86e6587d9ab6a9639d49c'
    report=m.physical.read(output/'report.json');checks=[]
    for slot in range(4):
        name=f'slot{slot}_result.json.gz';assert m.sha(output/name)==report['artifact_sha256'][name]
        result=m.physical.saved.overlay.read(output/name,m.physical.saved.EXPANDED)
        for arm in range(2):
            winner=result['results'][arm]
            selected=[a for a in winner['attempts'] if a['candidate_id']==winner['winner']['candidate_id']]
            assert len(selected)==1
            b=m.with_incumbent(p['branches'][2*slot+arm],selected[0]['logits'])
            assert b['incumbent_domain']['legal'];checks.append(b)
    assert len(checks)==8  # metadata seam check, NOT full v2 prepare or science
