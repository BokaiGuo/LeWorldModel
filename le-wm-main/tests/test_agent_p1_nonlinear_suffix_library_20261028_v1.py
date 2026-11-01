"""NumPy native-ABI fixtures only; no torch/neural/environment execution."""
from copy import deepcopy
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
import ast
import inspect
import time
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p1_nonlinear_suffix_library_20261028_v1 as m
import test_agent_allmode_prefix_20261028_v1 as upstream


class Events:
    def __init__(self):self.events=[]
    def event(self,e):self.events.append(deepcopy(e))


class RAM:
    """Linear arithmetic fixture with complete production-format tensor journal."""
    def __init__(self,task,journal,ledger):
        self.task,self.journal,self.ledger=task,journal,ledger;self.serial=0;self.j=np.zeros((192,10),np.float32)
        self.j[:10]=np.diag(np.linspace(.2,1.1,10,dtype=np.float32))
    @contextmanager
    def observe(self):yield self
    def arrays(self,kind,op=None,**values):
        self.journal.event(dict(kind=kind,scope=deepcopy(self.ledger.scope),phase=self.phase,step=0,
            operation_index=op,arrays={k:m.signal.blob(v) for k,v in values.items()}))
    def call(self,resource,fn,inputs,shape=None):
        idx=len(self.ledger.events)
        def actual():
            self.arrays('io_input',idx,**{resource+':'+k:v for k,v in inputs.items()})
            v=fn();self.arrays('io_return',idx,**{resource+':output':v});return v
        return self.ledger.call(resource,actual,shape=shape)
    def step(self,c,a,label):
        a=np.asarray(a,np.float32);self.phase=f'f{self.serial:03}';self.serial+=1
        self.ledger.scope=dict(forward=self.phase,label=label)
        self.journal.event(dict(kind='forward_started',phase=self.phase,label=label,input=deepcopy(c),commands=a.tolist()))
        z=np.asarray(c['latent_history'],np.float32)[None];h=np.asarray(c['action_history'],np.float32)[None]
        mean,std=(np.asarray(self.task[k],np.float32) for k in ('action_mean','action_std'))
        token=((a-mean)/std).reshape(10);outgoing=np.concatenate((h,token[None,None]),1)
        self.arrays('program_packet',latent=z,history=h,raw=a,mean=mean,std=std,token=token,outgoing=outgoing)
        emb=self.call('action_encoder',lambda:outgoing.copy(),{'0':outgoing},[1,3,10])
        def pred():
            y=z.copy();y[0,-1]=(z[0,-1]+self.j@a.reshape(10)).astype(np.float32)
            y=self.call('predictor',lambda:y,{'0':z,'1':emb},[1,3,192])
            return self.call('pred_proj',lambda:y.reshape(3,192),{'0':y.reshape(3,192)},[3,192]).reshape(1,3,192)
        y=self.call('predict',pred,{'latent':z,'act_emb':emb},[1,3,192])
        z=np.concatenate((z[:,1:],y[:,-1:]),1);h=outgoing[:,1:]
        self.arrays('program_boundary',latent=z,history=h)
        after=dict(c,latent_history=z[0].tolist(),action_history=h[0].tolist())
        self.journal.event(dict(kind='forward_finished',phase=self.phase,context=after));return after
    def gradient(self,c,a,label):
        a=np.asarray(a,np.float32);after=self.step(c,a,label);res=np.asarray(after['latent_history'][-1],np.float32)-np.asarray(self.task['goal'],np.float32)
        loss=np.mean(res**2).astype(np.float32);g=(2*self.j.T@res/np.float32(192)).reshape(5,2).astype(np.float32)
        e=dict(kind='AD_API_attempt',phase=self.phase,scalar_outputs=1,input_shape=[5,2],retain_graph=False,create_graph=False)
        self.journal.event(e)
        def derivative():
            def callback():self.arrays('raw_backward',len(self.ledger.events)-1,value=g);return g
            return self.ledger.call('raw_backward',callback)
        self.ledger.call('vjp',lambda:self.call('grad',derivative,{'loss':loss,'raw':a}),directions=1)
        self.journal.event(dict(e,kind='AD_API_completed'))
        return dict(context=after,native_float32_loss=float(loss),raw_gradient=g.tolist())


@pytest.fixture(scope='module')
def inputs():
    s=upstream.slot();sink=upstream.RAM();row=dict(index=0);b=upstream.ArrayBackend(row,sink)
    result=m.source.compile_arm(s['packet'],s['q'],b,lambda e:None)
    p=dict(slots=[dict(s,index=i,root_slot=m.source.ORDER[i][0],condition=m.source.ORDER[i][1]) for i in range(4)],expected_model_hash='a'*64)
    r=dict(rows=[dict(index=i,result=result) for i in range(4)])
    return m.material(p,r)


@pytest.fixture(scope='module')
def complete(inputs):
    s=inputs[0];c=s['candidates'][1];journal=Events();ledger=m.core.Ledger(m.caps(20,8),sink=journal.event)
    b=RAM(s['task'],journal,ledger);r=m.evaluate(s,c,b,journal)
    return s,c,r,ledger.receipt(),journal.events


def test_actual20forward8reverse_full9iterates(complete):
    s,c,r,receipt,events=complete;m.audit_candidate(s,c,r,receipt,events)
    assert len(r['iterates'])==9 and len(r['gradients'])==8
    assert len([e for e in events if e['kind']=='forward_started'])==20
    assert receipt['completed']==m.caps(20,8)
    assert r['selected_Q']<=r['iterates'][0]['Q']
    assert m.bits(r['commands'][:5])==c['prefix_bits']
    assert len(r['commands'])==10 and receipt['FLOPs'] is None


def test_all168_material_no_outcome_fields(inputs):
    assert len(inputs)==4 and sum(len(s['candidates']) for s in inputs)==168
    for s in inputs:
        assert [c['index'] for c in s['candidates']]==list(range(42))
        assert m.core.validate_task(s['task'])==s['task']
        for c in s['candidates']:
            assert c['incumbent_domain']['original_raw_bits']==m.bits(m.chart(c['logits']))
            assert not {'reward','success','state','physical_metrics'} & set(c)


def test_legacy_cast_outside_cap_is_not_rejected_or_repaired():
    rng=np.random.default_rng(17)
    for _ in range(1000):
        x=rng.normal(size=10);x*=np.nextafter(m.CAP,0.)/np.linalg.norm(x)
        if np.linalg.norm(x)<=m.CAP and np.linalg.norm(x.astype(np.float32).astype(float))>m.CAP:break
    else:pytest.fail('deterministic rounding fixture absent')
    c=dict(suffix_params=x.tolist(),suffix_solve=dict(params=x.tolist(),radius=m.CAP,norm=float(np.linalg.norm(x))),suffix_commands=m.chart(x).tolist())
    original=deepcopy(c);d=m.incumbent_domain(c)
    assert c==original and d['solver_float64_norm']<=m.CAP<d['postcast_float64_norm']
    assert d['original_raw_bits']==m.bits(m.chart(x))
    projected=m.proposal(x,np.zeros(10))
    assert np.linalg.norm(projected)<=m.CAP
    # Different new-update rounding is declared; iteration0 is never repaired.
    assert not np.array_equal(projected,x)


@pytest.mark.parametrize('bad',['float64_illegal','cast_changed','witness_changed'])
def test_original_incumbent_legality_still_strict(inputs,bad):
    c=deepcopy(inputs[0]['candidates'][0]);x=np.asarray(c['logits'],float)
    old=dict(suffix_params=x.tolist(),suffix_solve=dict(params=x.tolist(),radius=m.CAP,norm=float(np.linalg.norm(x))),suffix_commands=c['incumbent_raw'])
    if bad=='float64_illegal':
        x[:]=m.CAP;old.update(suffix_params=x.tolist(),suffix_commands=m.chart(x).tolist(),suffix_solve=dict(params=x.tolist(),radius=m.CAP,norm=float(np.linalg.norm(x))))
    elif bad=='cast_changed':old['suffix_commands'][0][0]=.75
    else:old['suffix_solve']['norm']+=1
    with pytest.raises(ValueError):m.incumbent_domain(old)


@pytest.mark.parametrize('bad',['dropiterate','dropgradient','prefix','Q','proposal','winner','rawbits','grad','loss','unknown','IO','op','scope','count','module_output','callback','scalar_vector'])
def test_strict_mutation_rejection(complete,bad):
    s,c,r,receipt,events=deepcopy(complete)
    if bad=='dropiterate':r['iterates'].pop()
    elif bad=='dropgradient':r['gradients'].pop()
    elif bad=='prefix':r['commands'][0][0]+=.001
    elif bad=='Q':r['iterates'][2]['Q']+=1
    elif bad=='proposal':r['iterates'][2]['logits'][0]+=.1
    elif bad=='winner':r['winner_iteration']=99
    elif bad=='rawbits':r['iterates'][1]['raw_bits']='00'*40
    elif bad=='grad':r['gradients'][0]['raw_gradient'][0][0]+=.01
    elif bad=='loss':r['gradients'][0]['native_float32_loss']+=1
    elif bad=='unknown':events.insert(1,dict(kind='native_success'))
    elif bad=='IO':events.remove(next(e for e in events if e['kind']=='io_input'))
    elif bad=='op':next(e for e in events if e['kind']=='io_input')['operation_index']=9999
    elif bad=='scope':next(e for e in events if e['kind']=='io_input')['scope']['label']='wrong'
    elif bad=='count':receipt['completed']['grad_API']-=1
    else:
        kind,key={'module_output':('io_return','predict:output'),'callback':('raw_backward','value'),'scalar_vector':('io_input','grad:loss')}[bad]
        e=next(e for e in events if e['kind']==kind and key in e['arrays']);a=m.signal.unblob(e['arrays'][key]).copy()
        if bad=='scalar_vector':a=np.repeat(a,2)
        else:a.flat[0]+=.1
        e['arrays'][key]=m.signal.blob(a)
    with pytest.raises((ValueError,IndexError)):m.audit_candidate(s,c,r,receipt,events)


def test_stable_incumbent_tie_and_entire42_library(inputs):
    s=inputs[0];c=deepcopy(s['candidates'][0]);c['logits']=[0.]*10;c['incumbent_raw']=m.chart(c['logits']).tolist()
    journal=Events();ledger=m.core.Ledger(m.caps(20,8),sink=journal.event)
    b=RAM(s['task'],journal,ledger);b.j[:]=0
    c['original_predicted_prefix']=deepcopy(s['task']['root_context']['latent_history'][-1])
    after=dict(s['task']['root_context'],latent_history=s['task']['root_context']['latent_history'][1:]+[c['original_predicted_prefix']],
        action_history=[s['task']['root_context']['action_history'][-1],m.source.kernel.normalized_history(np.asarray(c['prefix'],np.float32))[0].tolist()])
    # Determine fixture incumbent using identical pure NumPy recurrence.
    fake=RAM(s['task'],Events(),m.core.Ledger(m.caps(2,0),sink=lambda e:None));fake.j[:]=0
    mid=fake.step(s['task']['root_context'],c['prefix'],'x');last=fake.step(mid,c['incumbent_raw'],'y')
    c['original_terminal_context']=last;c['original_terminal_Q']=m.cost(s['task']['goal'],last)
    r=m.evaluate(s,c,b,journal)
    assert r['winner_iteration']==0
    with pytest.raises(ValueError):m.final_replay(s,[r],b,journal)
    rows=[dict(r,index=i) for i in range(42)];j=Events();l=m.core.Ledger(m.caps(4,0),sink=j.event);backend=RAM(s['task'],j,l);backend.j[:]=0
    p=m.final_replay(s,rows,backend,j)
    assert p['prefix_index']==p['iterate_index']==0 and l.completed==m.caps(4,0)


def test_complete_fixed_budget_and_no_global_patch():
    assert 4*(42*20+4)==m.contract()['new_predict']==3376
    assert 4*42*8==m.contract()['new_scalar_reverse']==1344
    assert len(m.roster())==168 and all(r['status']=='not_started' for r in m.roster())
    source=Path(m.__file__).read_text();tree=ast.parse(source)
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='setattr' for n in ast.walk(tree))
    assert 'loss.ndim==0' in inspect.getsource(m.Backend.gradient)
    assert 'GradGate(' not in source and m.contract()['physical_release'] is False


@pytest.mark.parametrize('move',['packet_before_start','packet_after_native','boundary_before_native','boundary_after_end','AD_before_end','AD_completed_before_reverse','reverse_after_next_start','selection_before_last_iterate','selection_after_replay_start'])
def test_cross_forward_splice_rejected_without_changing_event_content(complete,move):
    s,c,r,receipt,events=deepcopy(complete)
    def relocate(event,target,after=False):
        events.remove(event);at=events.index(target);events.insert(at+int(after),event)
    def find(kind,phase):return next(e for e in events if e['kind']==kind and e.get('phase')==phase)
    if move=='packet_before_start':relocate(find('program_packet','f000'),find('forward_started','f000'))
    elif move=='packet_after_native':relocate(find('program_packet','f000'),find('program_boundary','f000'))
    elif move=='boundary_before_native':relocate(find('program_boundary','f000'),find('program_packet','f000'),True)
    elif move=='boundary_after_end':relocate(find('program_boundary','f000'),find('forward_finished','f000'),True)
    elif move=='AD_before_end':relocate(find('AD_API_attempt','f002'),find('forward_finished','f002'))
    elif move=='AD_completed_before_reverse':relocate(find('AD_API_completed','f002'),find('AD_API_attempt','f002'),True)
    elif move=='reverse_after_next_start':relocate(find('AD_API_completed','f002'),find('forward_started','f003'),True)
    elif move=='selection_before_last_iterate':relocate(next(e for e in events if e['kind']=='candidate_selection'),find('forward_finished','f017'))
    else:relocate(next(e for e in events if e['kind']=='candidate_selection'),find('forward_started','f018'),True)
    with pytest.raises(ValueError):m.audit_candidate(s,c,r,receipt,events)


def test_authorization_before_all_IO(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('unauthorized IO'))
    with pytest.raises(ValueError):m.run('missing','a'*64,'missing','bad')


def test_prepare_verify_source_and_binding_drift(tmp_path,monkeypatch):
    marker=tmp_path/'source';marker.write_bytes(b'original')
    p=dict(schema=m.NAME+'.prepare',arguments={},bindings={str(marker):m.sha(marker)})
    def build(**kw):m.check_bindings(p['bindings']);return p
    monkeypatch.setattr(m,'build',build);path=tmp_path/'prepare.json';path.write_bytes(m.raw(p));h=m.sha(path)
    assert m.verify(path,h)==p
    marker.write_bytes(b'tampered')
    with pytest.raises(ValueError):m.verify(path,h)
    with pytest.raises(ValueError):m.verify(path,'0'*64)


def test_actual_saved_partial_consumer_retains168_and_real_used_ledger(tmp_path,monkeypatch,inputs):
    # Actual file/hashchain + the same strict consumer; only model/prepare
    # authorities are explicit test fixtures. Never a production receipt.
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    output=tmp_path/'results'/(m.NAME+'_RAM_partial');store=m.Store(output,time.monotonic())
    slots=deepcopy(inputs);p=dict(slots=slots,bindings={},original_compute=m.source.contract()['total'],expected_model_hash='a'*64)
    prepared=str(tmp_path/'RAM_prepare.json');expected='b'*64
    monkeypatch.setattr(m,'verify',lambda path,h:p if (path,h)==(prepared,expected) else pytest.fail('fixture binding'))
    store.put('binding.json',dict(prepared=prepared,prepare_sha256=expected))
    row=m.roster()[0];row['status']='attempted';journal=m.transport.NativeJournal(store,'s0_p00',lambda:None)
    ledger=m.core.Ledger(m.caps(20,8),sink=journal.event);backend=RAM(slots[0]['task'],journal,ledger)
    row['result']=m.evaluate(slots[0],slots[0]['candidates'][0],backend,journal);journal.close()
    row.update(status='sealed',resource_receipt=ledger.receipt(),journal_file='s0_p00/events.jsonl',journal_sha256=m.sha(journal.path),journal_tail=journal.tail)
    store.put('s0_p00_result.json',row);rows=m.roster();rows[0]=row
    artifacts={str(f.relative_to(output)):m.sha(f) for f in output.rglob('*') if f.is_file()}
    report=dict(schema=m.NAME+'.report',contract=m.contract(),prepared=prepared,prepare_sha256=expected,
        rows=rows,finals=[],fixed_denominator=168,slot_denominator=4,known_roots=2,status='partial_failclosed',errors=[dict(error='RAM stop')],
        artifact_sha256=artifacts,P1_admission=False,freshness=False,physical_release=False,scientific_admission=False)
    store.finalizing=True;store.put('report.json',report,final=True)
    consumed=m.consume(output,m.sha(output/'report.json'))
    assert not consumed['complete'] and consumed['plans']==[] and len(consumed['rows'])==168
    assert consumed['actual_new_compute']['completed']==m.caps(20,8)
    # A partial library cannot acquire a completed disposition just by status.
    report.update(status='all168_nonlinear_suffix_sealed_not_admitted',errors=[])
    (output/'report.json').write_bytes(m.raw(report))
    assert not m.consume(output,m.sha(output/'report.json'))['complete']
    with (output/'s0_p00/events.jsonl').open('ab') as f:f.write(b'\n')
    with pytest.raises(ValueError):m.consume(output,m.sha(output/'report.json'))


def test_same_dispatcher_all168_and_four_replays_RAM_only(tmp_path,monkeypatch,inputs):
    # Sink double suppresses durable tensor IO ONLY for this denominator test;
    # full native audit and real durable chain are exercised separately above.
    class Journal(Events):
        def __init__(self,store,namespace,check):
            super().__init__();self.path=tmp_path/(namespace+'.mock');self.tail='0'*64
        def close(self):self.path.write_bytes(b'RAM mock transport; not execution evidence')
    class Store:
        def put(self,*a,**kw):return None
    monkeypatch.setattr(m.transport,'NativeJournal',Journal)
    rows=m.roster();finals=[]
    m.dispatch(dict(slots=inputs),Store(),rows,finals,lambda:None,lambda task,journal,ledger:RAM(task,journal,ledger))
    assert len(rows)==168 and len(finals)==4 and all(r['status']=='sealed' for r in rows+finals)
    assert {k:sum(r['resource_receipt']['completed'][k] for r in rows+finals) for k in m.core.AXES}==m.caps(3376,1344)
    assert all(m.bits(row['result']['commands'][:5])==inputs[row['slot']]['candidates'][row['prefix_index']]['prefix_bits'] for row in rows)


def test_dispatch_failure_preserves_all_unreached_and_actual_attempt(tmp_path,monkeypatch,inputs):
    class Journal(Events):
        def __init__(self,*args):super().__init__();self.path=tmp_path/'mock_journal';self.tail='0'*64
        def close(self):self.path.write_bytes(b'RAM partial transport')
    class Store:
        def put(self,*args):return None
    class Failing(RAM):
        def step(self,c,a,label):
            if label=='iterate_1':raise RuntimeError('RAM injected failure')
            return super().step(c,a,label)
    monkeypatch.setattr(m.transport,'NativeJournal',Journal)
    rows=m.roster();finals=[]
    with pytest.raises(RuntimeError):m.dispatch(dict(slots=inputs),Store(),rows,finals,lambda:None,lambda t,j,l:Failing(t,j,l))
    assert rows[0]['status']=='failed_partial' and len(rows)==168 and not finals
    assert all(r['status']=='not_started' for r in rows[1:])
    assert rows[0]['resource_receipt']['completed']==m.caps(3,1)


def test_real_saved_incumbents_rounding_only_no_neural():
    report=m.source.overlay.read(m.PARENT/'report.json',m.source.CAP)
    assert m.sha(m.PARENT/'report.json')==m.PARENT_SHA
    ds=[m.incumbent_domain(c) for r in report['rows'] for c in r['result']['candidates']]
    assert len(ds)==168
    assert sum(d['postcast_float64_norm']>d['cap'] for d in ds)==81
    assert all(d['solver_float64_norm']<=d['cap'] for d in ds)
