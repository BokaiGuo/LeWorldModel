"""RAM/fake-array/tempdisk only. No model, CUDA, physics or production consume."""
from copy import deepcopy
import ast
import hashlib
import importlib
import json
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p2_intersection_host_cuda_20261029_v2 as m
from test_main_p2_intersection_full_schedules_20261029_v1 import execute
from test_agent_p2_three_program_20261028_v1 import Numeric, slot
from test_agent_p2_common_resource_grasp_cuda_20261029_v1 import ram_grasp, task


class Clock:
    def __init__(self):self.now=10.
    def __call__(self):return self.now


def test_real_horizon3_ad_return_archive_and_consumer_tempdisk(tmp_path):
    store=m.Store(tmp_path/(m.NAME+'_ad'),floor=0)
    c=slot()['packet']['root_context']
    # Three distinct full causal states, retaining actual 3x192/2x10 shape.
    contexts=deepcopy([c,c,c])
    for i,context in enumerate(contexts):context['latent_history'][0][0]=float(i+1)
    jac=store.array(np.arange(192*30,dtype=np.float32).reshape(192,30))
    event=dict(kind='arm_event',index=0,payload=dict(kind='AD_return',index=0,array=jac,contexts=contexts,completed=True))
    store.event(event);store.close()
    line=(store.output/'events.jsonl').read_bytes()
    assert 1400 < len(line) <= 4096
    decoded=list(m.Reader(store.output).events())
    assert decoded==[event]
    assert m.byte_proof()['extra_metadata_breakdown']['spectral_jacobian_events']==2*2*85*(3*1400+4096)


def test_ad_intent_full_vectors_encoded_below_existing_limit(tmp_path):
    store=m.Store(tmp_path/(m.NAME+'_intent'),floor=0)
    request=dict(index=0,tag='joint:84/J',input_context_sha256='f'*64,
                 at_logits=[-0.9999999999999999]*30,raw=[[[-0.9999999999999999]*2 for _ in range(5)] for _ in range(3)],
                 coordinates='raw_incumbent',shape=[192,30],mathematical_VJP=192,vectorize=True,completed=False)
    event=dict(kind='arm_event',index=0,payload=dict(kind='AD_intent',request=request))
    store.event(event)
    assert (store.output/'events.jsonl').stat().st_size <=1400
    assert list(m.Reader(store.output).events())==[event]


def test_internal_failure_propagates_without_dropping_other_rows():
    rows=[dict(index=i,result=dict(aborted_unit='bank' if i<2 else None),
               accounting=dict(stop='operation_failure' if i<2 else 'natural_completion')) for i in range(6)]
    failures=m.internal_failures(rows)
    assert len(rows)==6 and [e['index'] for e in failures]==[0,1]
    assert all(e['stage']=='arm_internal' for e in failures)
    assert not m.internal_failures([dict(index=0,result=dict(error=dict(type='Refused')),accounting=dict(stop='wall_cap'))])


def d(v):return dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest())


class CompatibleLedger(m.StagedLedger):
    """Run MAIN's original complete six schedule fixture with staged inputs."""
    def module(self,name,callback,capture,*,shapes,input_capture=None):
        if name=='predictor':self.predict_api()
        if input_capture is None:input_capture=lambda:(None,d(b'fixture-input'))
        def out(v):return None,capture(v)
        return super().module(name,callback,out,shapes=shapes,input_capture=input_capture)


@pytest.mark.parametrize('root,arm',m.box.ORDER)
def test_complete_six_original_schedule_with_real_staged_ledger(root,arm):
    ledger=CompatibleLedger(root,arm,clock=lambda:0.,sink=lambda _:None)
    for label in m.box.schedule(arm):execute(ledger,label)
    expected={'joint':(1902,255,85,0,489600),'composition':(1062,129,85,0,247680),
              'grasp':(1902,1896,0,632,0)}[arm]
    assert tuple(ledger.used[k] for k in ('predict_API','AD_primal','J','backward'))+(ledger.side['Jacobian_elements_requested'],)==expected
    assert ledger.used['replay_trajectories']==2 and ledger.used['mean_paid_replays']==0
    assert ledger.committed==list(m.box.schedule(arm)) and ledger.stop=='natural_completion'


def execute_packet(tape, *, fail_input=False, fail_output=False):
    def predict():
        for name in ('AE','predictor','lowproj'):
            if name=='predictor':tape.predict_api()
            def incoming():
                if fail_input:raise RuntimeError('input fixture failed')
                return None,d(b'in')
            def outgoing(v):
                if fail_output:raise RuntimeError('output fixture failed')
                return None,d(v)
            tape.module(name,lambda:b'out',outgoing,shapes=deepcopy(m.box.MODULE_SHAPES[name]),input_capture=incoming)
    return tape.call('predict',predict)


def test_exact_stage_order_input_before_dispatch_and_output_after_return():
    events=[];t=m.Tape(0,'joint',events.append,Clock());assert t.admit('nominal');execute_packet(t)
    names=[e['event'].get('operation_kind') or e['event']['kind'] for e in events if e['kind']=='budget']
    assert names.index('module_attempt') < names.index('capture_intent') < names.index('module_dispatch') < names.index('module_return') < names.index('module_capture')
    assert t.ledger.used['AE_attempts']==t.ledger.used['AE_dispatched']==t.ledger.used['AE_returned']==t.ledger.used['AE_captured']==1
    assert t.ledger.side['CPU_capture_bytes']==3*(2+3)


@pytest.mark.parametrize('failure,expected',[(dict(fail_input=True),(1,0,0,0)),(dict(fail_output=True),(1,1,1,0))])
def test_failed_capture_retains_exact_stage_and_stops(failure,expected):
    events=[];t=m.Tape(0,'joint',events.append,Clock());assert t.admit('nominal')
    with pytest.raises(RuntimeError):execute_packet(t,**failure)
    assert tuple(t.ledger.used['AE_'+x] for x in ('attempts','dispatched','returned','captured'))==expected
    assert t.ledger.used['predict_API']==0
    assert t.ledger.stop=='operation_failure'


def test_complete_tape_reexecutes_clock_and_all_budget_events():
    events=[];actual=m.Tape(0,'joint',events.append,Clock());assert actual.admit('nominal')
    for _ in range(3):execute_packet(actual)
    assert actual.commit();summary=actual.ledger.summary()
    cursor=m.Cursor(events);saved=m.SavedTape(0,'joint',cursor);assert saved.admit('nominal')
    for _ in range(3):execute_packet(saved)
    assert saved.commit();assert saved.ledger.summary()==summary;cursor.complete()
    altered=deepcopy(events);event=next(e for e in altered if e['kind']=='budget' and e['event']['kind']=='debit')
    event['event']['delta']['predict_API']+=1
    cursor=m.Cursor(altered);saved=m.SavedTape(0,'joint',cursor);assert saved.admit('nominal')
    with pytest.raises(ValueError):execute_packet(saved)


def test_real_host_module_bridge_and_saved_spectral_packet_tempdisk(tmp_path):
    """Actual HostBackend.module control path, fake native arrays only."""
    fixture=Numeric();context=slot()['packet']['root_context'];actions=np.zeros((3,5,2),np.float32)
    paths=fixture.rollout(context,actions,'nominal');row=fixture.row;store=m.Store(tmp_path/(m.NAME+'_fixture'),floor=0);clock=Clock()
    tape=m.Tape(0,'joint',store.event,clock);assert tape.admit('nominal')
    backend=object.__new__(m.HostBackend);backend.budget=tape;backend.event=store.event
    backend.check=lambda:store.event(dict(kind='memory',allocated=1,reserved=2))
    backend.array=lambda x:m.old.blob(x)
    for f in row['forwards']:
        def run_packet(f=f):
            backend.check();backend.event(dict(kind='forward_intent',request=f['request']))
            backend.event(dict(kind='forward_inputs',index=f['index'],context=f['input'],raw=f['raw']))
            backend.current=dict(index=f['index'],modules=[])
            for k,e in enumerate(f['modules']):
                if k==1:
                    tape.predict_api();backend.event(dict(kind='predict_API_intent',forward=f['index'],completed=False))
                args=tuple(m.old.unblob(d) for d in e['inputs']);out=m.old.unblob(e['output']).copy()
                names={'action_encoder':'AE','predictor':'predictor','pred_proj':'lowproj'}
                backend.module(names[e['resource']],lambda *args:out,args)
                if k==2:backend.event(dict(kind='predict_API_return',forward=f['index'],completed=True))
            assert backend.current['modules']==f['modules']
            backend.event(dict(kind='forward_return',index=f['index'],context=f['output'],completed=True))
        tape.call('predict',run_packet)
    assert tape.commit();summary=tape.ledger.summary()
    store.put('fixture.json',row);store.close();reader=m.Reader(store.output);row=reader.read('fixture.json',1024**2)
    cursor=m.Cursor(reader.events());saved=m.SavedTape(0,'joint',cursor);assert saved.admit('nominal')
    sb=m.SavedSpectral(row,saved,cursor,None);assert sb.rollout(context,actions,'nominal')==paths
    assert saved.commit();sb.complete();cursor.complete();reader.complete_archive();assert saved.ledger.summary()==summary


def test_grasp_saved_full_update_and_deadline_replay():
    """Original arithmetic fixture, staged 3-module path and saved budget replay."""
    _,_,old_events=ram_grasp(units=1);native=[e['payload'] for e in old_events if e['kind']=='g_native']
    entries=[]
    for e in native:
        if e['kind']=='module_attempt':entries.append(dict(index=e['index'],resource=e['resource'],inputs=e['inputs'],output=None))
        if e['kind']=='module_return':entries[-1]['output']=e['output']
    row=dict(neural=entries);events=[];clock=Clock()
    class BoundaryTape(m.Tape):
        def commit(self):
            result=super().commit();clock.now=460.;return result
    tape=BoundaryTape(0,'grasp',events.append,clock)
    class ProducerCursor:
        def __init__(self):self.i=0
        def get(self,kind):
            if kind=='memory':v=dict(kind='memory',allocated=1,reserved=2)
            else:
                assert self.i<len(native);v=native[self.i];self.i+=1;assert v['kind']==kind
            events.append(deepcopy(v));return deepcopy(v)
        def emit(self,e):assert self.get(e['kind'])==e
    c=ProducerCursor();b=m.SavedGrasp(task(),c,tape,row);result=m.saved_grasp(task(),0,b)
    assert c.i==len(native) and result==dict(export=None,error=None)
    assert tape.ledger.stop=='wall_cap' and tape.ledger.used['backward']==1
    cursor=m.Cursor(events);saved=m.SavedTape(0,'grasp',cursor);sb=m.SavedGrasp(task(),cursor,saved,row)
    assert m.saved_grasp(task(),0,sb)==result;cursor.complete();assert saved.ledger.summary()==tape.ledger.summary()


def test_actual_grasp_host_step_with_fake_tensor_and_saved_replay():
    """Exercise original Native.step + new staged host wrappers, no torch."""
    class Tensor:
        def __init__(self,x):self.a=np.asarray(x,np.float32);self.shape=self.a.shape
        def repeat(self,n):return Tensor(np.tile(self.a,n))
        def reshape(self,*shape):return Tensor(self.a.reshape(*shape))
        def __getitem__(self,key):return Tensor(self.a[key])
        def __sub__(self,other):return Tensor(self.a-other.a)
        def __truediv__(self,other):return Tensor(self.a/other.a)
    class FakeTorch:
        @staticmethod
        def cat(xs,axis):return Tensor(np.concatenate([x.a for x in xs],axis=axis))
    events=[];tape=m.Tape(0,'grasp',events.append,Clock());assert tape.admit('joint:0')
    b=object.__new__(m.GraspBackend);b.t=FakeTorch;b.task=task();b.event=events.append;b.budget=tape
    b.row=dict(neural=[]);b.n=0;b.active=None;b.current=None
    b.tensor=lambda a,dtype=None:Tensor(a);b.snap=lambda x:m.grasp.pack(x.a)
    b.check=lambda:events.append(dict(kind='memory',allocated=1,reserved=2))
    class FakeModel:
        def action_encoder(self,x):return b.module('AE',lambda x:Tensor(np.zeros((1,3,192),np.float32)),(x,))
        def predict(self,z,a):
            tape.predict_api()
            p=b.module('predictor',lambda z,a:Tensor(np.zeros((1,3,192),np.float32)),(z,a))
            out=b.module('lowproj',lambda x:Tensor(np.zeros((3,192),np.float32)),(p.reshape(3,192),))
            return out.reshape(1,3,192)
    b.model=FakeModel();z=Tensor(np.zeros((1,3,192)));h=Tensor(np.zeros((1,2,10)));a=Tensor(np.zeros((5,2)))
    actual=b.step(z,h,a,'lifted/0');assert tape.ledger.used['predict_API']==1
    cursor=m.Cursor(events);saved=m.SavedTape(0,'grasp',cursor);assert saved.admit('joint:0')
    sb=m.SavedGrasp(task(),cursor,saved,b.row)
    replay=sb.step(z.a,h.a,a.a,'lifted/0');cursor.complete()
    assert all(np.array_equal(x.a,y) for x,y in zip(actual,replay,strict=True))
    assert saved.ledger.summary()==tape.ledger.summary()


def test_spectral_wall_during_ae_return_rebuilds_partial_without_output():
    """One actual staged host callback crosses deadline; saved replays prefix."""
    fixture=Numeric();context=slot()['packet']['root_context'];fixture.rollout(context,np.zeros((3,5,2),np.float32),'nominal')
    complete=fixture.row['forwards'][0];events=[];clock=Clock();tape=m.Tape(0,'joint',events.append,clock)
    class FakeHost:
        def __init__(self):
            self.budget=tape;self.row=dict(forwards=[],AD=[]);self.event=events.append
            self.host=object.__new__(m.HostBackend);self.host.budget=tape;self.host.event=events.append
            self.host.check=lambda:events.append(dict(kind='memory',allocated=1,reserved=2));self.host.array=lambda x:m.old.blob(x)
        def rollout(self,c,commands,tag):
            def packet():
                f=deepcopy(complete);f.pop('output');f.update(modules=[],completed=False)
                self.row['forwards'].append(f);self.host.current=f;self.host.check()
                self.event(dict(kind='forward_intent',request=f['request']))
                self.event(dict(kind='forward_inputs',index=0,context=c,raw=f['raw']))
                e=complete['modules'][0];args=tuple(m.old.unblob(x) for x in e['inputs'])
                def ae(*args):clock.now=460.;return m.old.unblob(e['output']).copy()
                self.host.module('AE',ae,args)
            self.budget.call('predict',packet)
        def jacobian(self,*a):pytest.fail('no J after deadline')
        def cpu(self,*a):pytest.fail('no CPU after deadline')
    host=FakeHost();result=m.drive_spectral(slot(),'joint',host)
    assert result['aborted_unit']=='nominal' and result['export'] is None
    assert tape.ledger.stop=='wall_cap' and tape.ledger.used['AE_returned']==1 and tape.ledger.used['AE_captured']==0
    assert tape.ledger.used['predict_API']==0 and host.row['forwards'][0]['modules']==[]
    cursor=m.Cursor(events);saved=m.SavedTape(0,'joint',cursor);b=m.SavedSpectral(host.row,saved,cursor,None)
    assert m.drive_spectral(slot(),'joint',b)==result;b.complete();cursor.complete()
    assert saved.ledger.summary()==tape.ledger.summary()


def test_grasp_wall_before_ae_dispatch_saved_partial_full_control_path():
    _,_,old_events=ram_grasp(units=1);native=[e['payload'] for e in old_events if e['kind']=='g_native']
    first=next(e for e in native if e['kind']=='module_attempt');row=dict(neural=[dict(index=first['index'],
        resource=first['resource'],inputs=first['inputs'],output=None)])
    events=[];clock=Clock();tape=m.Tape(0,'grasp',events.append,clock)
    class ProducerCursor:
        def __init__(self):self.i=0
        def get(self,kind):
            if kind=='memory':v=dict(kind='memory',allocated=1,reserved=2)
            elif kind=='algorithm_exception':v=dict(kind=kind,type='Refused',message='wall_cap')
            else:
                v=native[self.i];self.i+=1;assert v['kind']==kind
            events.append(deepcopy(v))
            if kind=='module_attempt':clock.now=460.
            return deepcopy(v)
        def emit(self,e):assert self.get(e['kind'])==e
    c=ProducerCursor();b=m.SavedGrasp(task(),c,tape,row);result=m.saved_grasp(task(),0,b)
    assert result==dict(export=None,error=dict(type='Refused',message='wall_cap'))
    assert tape.ledger.used['AE_attempts']==1 and tape.ledger.used['AE_dispatched']==0 and tape.ledger.used['predict_API']==0
    cursor=m.Cursor(events);saved=m.SavedTape(0,'grasp',cursor);sb=m.SavedGrasp(task(),cursor,saved,row)
    assert m.saved_grasp(task(),0,sb)==result;cursor.complete();assert saved.ledger.summary()==tape.ledger.summary()


@pytest.mark.parametrize('root,arm',[(0,'joint'),(0,'composition'),(1,'joint'),(1,'composition')])
def test_actual_spectral_driver_full_schedule_fake_callbacks(root,arm):
    """Drive original proposal/selection logic all units; synthetic no learned ops."""
    class FakeBackend:
        def __init__(self):self.budget=m.Tape(root,arm,lambda e:None,lambda:0.)
        def event(self,e):pass
        def rollout(self,c,commands,tag):
            result=[]
            for _ in commands:
                def packet():
                    for name in ('AE','predictor','lowproj'):
                        if name=='predictor':self.budget.predict_api()
                        self.budget.module(name,lambda:b'out',lambda v:(None,d(v)),
                            shapes=deepcopy(m.box.MODULE_SHAPES[name]),input_capture=lambda:(None,d(b'in')))
                self.budget.call('predict',packet);result.append(deepcopy(c))
            return result
        def jacobian(self,c,u,tag,logit):
            def work():
                self.last_ad_contexts=self.rollout(c,m.old.chart(u),tag+'/forward')
                return np.zeros((192,len(u)),np.float64)
            return self.budget.call('J',work,horizon=len(u)//10,dimension=len(u))
        def cpu(self,kind,fn,*args,**kwargs):return self.budget.auxiliary(kind,lambda:fn(*args,**kwargs))
    b=FakeBackend();result=m.drive_spectral(slot(root),arm,b)
    assert result['aborted_unit'] is None and result['export'] is not None
    assert b.budget.ledger.stop=='natural_completion' and b.budget.ledger.used['replay_trajectories']==2
    assert b.budget.ledger.used['predict_API']==({'joint':1902,'composition':1062}[arm])
    assert b.budget.ledger.used['J']==85 and b.budget.ledger.used['FLOPs'] is None


def test_deadline_before_input_has_no_kernel_dispatch():
    clock=Clock();events=[];t=m.Tape(0,'joint',events.append,clock);assert t.admit('nominal')
    def predict():
        def incoming():clock.now+=450;return None,d(b'i')
        t.module('AE',lambda:pytest.fail('kernel not allowed'),lambda x:(None,d(x)),
            shapes=deepcopy(m.box.MODULE_SHAPES['AE']),input_capture=incoming)
    with pytest.raises(m.box.Refused):t.call('predict',predict)
    assert t.ledger.used['AE_attempts']==1 and t.ledger.used['AE_dispatched']==0


def test_batch_rejected_before_attempt_and_no_mutation():
    t=m.Tape(0,'joint',lambda e:None,Clock());assert t.admit('nominal')
    with pytest.raises(ValueError):
        t.call('predict',lambda:t.module('AE',lambda:None,lambda v:(None,d(b'')),shapes=[[2,3,10]],input_capture=lambda:(None,d(b''))))
    assert t.ledger.used['AE_attempts']==0


def test_actual_predict_api_cannot_be_debited_before_ae():
    t=m.Tape(0,'joint',lambda e:None,Clock());assert t.admit('nominal')
    with pytest.raises(ValueError,match='after completed AE'):
        t.call('predict',t.predict_api)
    assert t.ledger.used['predict_API']==0


def test_shapes_census_and_non_compression_byte_proof():
    p=m.byte_proof()
    assert p['packet_by_method']==dict(joint=1902,composition=1062,grasp=1902)
    assert p['packets']==9732 and p['module_floats']==3486 and p['causal_floats']==1202
    assert p['native_payload_bytes']==9732*(4*3486+8*1202)
    assert p['native_payload_bytes']+p['jacobian_bytes']+p['optimizer_payload_upper_bytes']<p['array_partition']
    assert p['total_partitions']<=1536*1024**2 and not p['compression_assumed']
    assert p['budget_event_bytes_upper']+p['packet_native_bytes_upper']+p['extra_scores_optimizer_failure_metadata_allowance']<=p['journal_partition']
    assert p['full_completion_guaranteed'] is False


@pytest.mark.parametrize('arm',['joint','composition','grasp'])
def test_complete_budget_event_census_and_per_type_bounds(arm):
    counts=dict(a=0,aux=0,ci=0,cr=0,d=0,r=0,s=0,u=0,c=0,done=0,clock=0)
    mapping={'admission':'a','auxiliary_intent':'aux','auxiliary_return':'aux','capture_intent':'ci',
             'capture_return':'cr','debit':'d','return':'r','start':'s','unit_close':'u'}
    def sink(e):
        key=mapping[e['event']['kind']] if e['kind']=='budget' else {'budget_command':'c','budget_done':'done','clock':'clock'}[e['kind']]
        counts[key]+=1;v=dict(kind='arm_event',index=5,payload=e)
        wrapper=dict(index=1234567,previous_sha256='f'*64,payload=v,sha256='f'*64)
        assert len(m.box.raw(wrapper))+1<=m.Store.event_limit(v)
    class T(m.Tape):
        def __init__(self):super().__init__(0,arm,sink,lambda:0.);self.arm=arm
        def module(self,name,callback,capture,*,shapes):
            if name=='predictor':self.predict_api()
            return super().module(name,callback,lambda v:(None,capture(v)),shapes=shapes,input_capture=lambda:(None,d(b'fixture!')))
    t=T()
    for label in m.box.schedule(arm):execute(t,label)
    assert counts==m.byte_proof()['budget_event_census'][arm]


def test_archive_exact_bytes_alias_and_full_event_chain(tmp_path):
    store=m.Store(tmp_path/(m.NAME+'_fixture'),floor=0);a=np.arange(12,dtype=np.float32).reshape(3,4)
    blob=m.old.blob(a);g=m.grasp.pack(a.astype(np.float64));store.event(dict(kind='data',a=blob,b=g,ctx=[float(i) for i in range(10)]))
    store.put('row.json',dict(a=blob,b=g));store.close()
    reader=m.Reader(store.output);events=list(reader.events());row=reader.read('row.json',1024**2)
    assert events[0]['a']==row['a']==blob and row['b']==g;reader.complete_archive()
    with (store.output/'events.jsonl').open('ab') as f:f.write(b'{}\n')
    with pytest.raises(ValueError):list(m.Reader(store.output).events())


def test_transport_preserves_mixed_int_float_identity_and_signed_zero(tmp_path):
    store=m.Store(tmp_path/(m.NAME+'_mixed'),floor=0)
    mixed=[0]+[float(i) for i in range(10)];floats=[-0.]+[float(i) for i in range(10)]
    store.put('row.json',dict(mixed=mixed,floats=floats))
    reader=m.Reader(store.output);row=reader.read('row.json',1024**2);reader.complete_archive()
    assert type(row['mixed'][0]) is int and type(row['mixed'][1]) is float
    assert np.signbit(row['floats'][0]) and m.box.raw(row)==m.box.raw(dict(mixed=mixed,floats=floats))


def test_archive_tamper_and_unknown_suffix_are_rejected(tmp_path):
    store=m.Store(tmp_path/(m.NAME+'_fixture'),floor=0);store.put('row.json',dict(a=m.old.blob(np.arange(8,dtype=np.float32))))
    r=m.Reader(store.output);r.read('row.json',1024**2);r.complete_archive()
    with (store.output/'arrays.bin').open('ab') as f:f.write(b'x')
    with pytest.raises(ValueError):r.complete_archive()


def test_six_arm_durable_budget_roundtrip_and_saved_reexecution(tmp_path):
    """Real writer/chain/reader and all six scoped budget tapes; no NN."""
    store=m.Store(tmp_path/(m.NAME+'_sixfixture'),floor=0)
    labels=[]
    class Adapter:
        def __init__(self,t,arm):self.t,self.arm=t,arm
        def admit(self,l):return self.t.admit(l)
        def commit(self):return self.t.commit()
        def call(self,k,fn,**kw):return self.t.call(k,fn,**kw)
        def auxiliary(self,k,fn):return self.t.auxiliary(k,fn)
        def module(self,name,callback,capture,*,shapes):
            if name=='predictor':self.t.predict_api()
            return self.t.module(name,callback,lambda v:(None,capture(v)),shapes=shapes,input_capture=lambda:(None,d(b'in')))
    for i,(root,arm) in enumerate(m.box.ORDER):
        store.event(dict(kind='arm_start',index=i,root=root,arm=arm))
        tape=m.Tape(root,arm,lambda e,i=i:store.event(dict(kind='arm_event',index=i,payload=e)),lambda:0.)
        label=m.box.schedule(arm)[0];labels.append(label);execute(Adapter(tape,arm),label)
        store.event(dict(kind='arm_end',index=i,used=tape.ledger.used))
    store.close();reader=m.Reader(store.output);cursor=m.Cursor(reader.events())
    for i,(root,arm) in enumerate(m.box.ORDER):
        cursor.emit(dict(kind='arm_start',index=i,root=root,arm=arm))
        class ArmCursor:
            def get(self,kind):
                e=cursor.get('arm_event');assert e['index']==i and e['payload']['kind']==kind;return e['payload']
            def emit(self,e):assert self.get(e['kind'])==e
            def clock(self):return self.get('clock')['value']
        saved=m.SavedTape(root,arm,ArmCursor());execute(Adapter(saved,arm),labels[i])
        cursor.emit(dict(kind='arm_end',index=i,used=saved.ledger.used))
    cursor.complete();reader.complete_archive()


def test_only_authenticated_sources_and_no_torch_import():
    assert m.check_sources()=={str(k):v for k,v in m.PINS.items()}
    # This fixture did not import torch; old source imports may be present but no neural forward is called.
    assert not any(k=='torch' or k.startswith('torch.') for k in sys.modules)


def test_installed_actual_lewm_predict_source_is_the_pinned_facade_abi():
    path=next(p for p in m.PINS if p.name=='lewm.py');tree=ast.parse(path.read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='LeWM')
    predict=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='predict')
    calls=[n.func.attr for n in ast.walk(predict) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
           and isinstance(n.func.value,ast.Name) and n.func.value.id=='self']
    assert calls==['predictor','pred_proj'] and m.sha(path)==m.PINS[path]


def test_cuda_authority_checked_before_any_file_read():
    with pytest.raises(ValueError,match='authorization'):m.run_cuda('/does/not/exist','bad','/does/not/exist','not-MAIN')
