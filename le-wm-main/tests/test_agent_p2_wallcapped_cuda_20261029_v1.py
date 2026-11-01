"""RAM/NumPy tests only. No real NN, torch, CUDA or physics."""
from copy import deepcopy
import io
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_wallcapped_cuda_20261029_v1 as m
from test_agent_p2_three_program_20261028_v1 import slot,Numeric


def test_own_native_request_supports_full1902_not_old1068():
    c=slot()['packet']['root_context'];a=np.zeros((5,2),np.float32)
    assert m.packet_request(1901,c,a,'raw')['index']==1901
    with pytest.raises(ValueError):m.packet_request(1902,c,a,'raw')


def test_actual_native_packet_full_io_normalizer_tamper():
    b=Numeric();b.rollout(slot()['packet']['root_context'],np.zeros((1,5,2),np.float32),'raw')
    f=b.row['forwards'][0];assert m.validate_packet(f)==f['output']
    bad=deepcopy(f);bad['output']['action_history'][0][0]+=1
    with pytest.raises(ValueError):m.validate_packet(bad)


class Control:
    def __init__(self,stop_after=3):self.started=[];self.commits=[];self.stop_after=stop_after;self.incumbent=None;self.stop='wall_cap';self.finished=[]
    def call(self,name,*args,**kw):
        if name=='start_unit':
            if len(self.started)>=self.stop_after:return False
            self.started.append(args[0]);return True
        if name=='finish_unit':self.finished.append(self.started[-1])
        if name=='commit':
            w,g,tag=args;c=m.old.objective(w['contexts'][-1],g)
            if self.incumbent is None or c<self.incumbent['cost']:self.incumbent=dict(cost=c,witness=deepcopy(w),tag=tag)
        return None
    def current(self):return dict(stop=self.stop,incumbent=deepcopy(self.incumbent))


class Backend(Numeric):
    def cpu(self,kind,fn,*args,**kw):return fn(*args,**kw)


@pytest.mark.parametrize('arm',['joint','composition'])
def test_partial_fixed_algorithm_nominal_bank_seed_retains_incumbent(arm):
    b=Backend();c=Control();r=m.drive(slot(),arm,b,c)
    assert c.started==['nominal','bank','seed:0']
    assert c.finished==c.started
    assert len(b.row['forwards'])==9 and len(b.row['AD'])==1
    assert c.incumbent is not None and r['export'] is None


def test_bank_literal_numerics_matches_original_no_free_cpu_calls():
    rng=np.random.default_rng(9);j=rng.normal(size=(192,30));n=rng.normal(size=192);g=rng.normal(size=192)
    b=Backend();assert m.old.same(m.make_bank(j,n,g,b),m.old.bank(j,n,g))


def test_auth_before_model_import_reads_or_dispatch(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('read'))
    with pytest.raises(ValueError):m.run_cuda('p','sha','out','wrong')


class RAMControl(m.Control):
    def __init__(self,s,arm,emit,stop_after=3,base=100.):
        self.fake_now=base;self.stop_after=stop_after
        super().__init__(s,arm,lambda:self.fake_now,emit)
    def call(self,name,*a,**kw):
        r=super().call(name,*a,**kw)
        if name=='finish_unit' and len(self.ledger.finished)>=self.stop_after:self.fake_now=self.ledger.start+450.
        return r


class MeteredNumeric(Numeric):
    meter_event=m.HostBackend.meter_event
    unwind=m.HostBackend.unwind
    cpu=m.HostBackend.cpu
    def __init__(self,c,emit,store=None,index=0):
        self.control=c;self.emit_native=emit;self.opened=[];self.store=store
        super().__init__(index=index,sink=self.sink_native)
    def sink_native(self,e):
        if e['kind']=='AD_return' and self.store is not None:
            n=e['array']['file'];self.store._write(n,self.arrays[n])
        self.meter_event(e)


def ram_arm(s,arm='joint',stop_after=3,store=None,index=0):
    events=[]
    def emit(e):
        events.append(deepcopy(e))
        if store is not None:store.event(dict(kind='arm_event',index=index,payload=e))
    c=RAMControl(s,arm,emit,stop_after,base=100.+1000*index);c.call('sample_memory',1024,2048)
    b=MeteredNumeric(c,emit,store,index);result=m.drive(s,arm,b,c)
    c.call('seal');account=c.call('report')
    row=dict(b.row,index=index,root_slot=s['root_slot'],arm=arm,result=result,accounting=account)
    def load(d):return np.load(io.BytesIO(b.arrays[d['file']]),allow_pickle=False)
    return row,events,load


@pytest.mark.parametrize('arm',['joint','composition'])
def test_actual_shape_partial_run_and_full_accounting_algorithm_replay(arm):
    row,events,load=ram_arm(slot(),arm)
    a=m.validate_arm(row,events,slot(),load)
    assert a['attempted']['predict']==9 and a['stop']=='wall_cap'
    assert a['export_missing'] and row['result']['export'] is None


@pytest.mark.parametrize('mutation',['unknown','raw','clock','ledger','commit','J','native_order'])
def test_resealed_data_still_rejects_semantic_mutations(mutation):
    row,events,load=ram_arm(slot())
    if mutation=='unknown':events.append(dict(kind='unknown'))
    elif mutation=='raw':row['forwards'][1]['raw']=m.old.blob(np.ones((5,2),np.float32)*.1)
    elif mutation=='clock':next(e for e in events if e['kind']=='ledger_intent')['now']=-1
    elif mutation=='ledger':row['accounting']['attempted']['predict']-=1
    elif mutation=='commit':
        e=next(e for e in events if e['kind']=='ledger_intent' and e['method']=='commit');e['args'][2]='other'
    elif mutation=='J':row['AD'][0]['request']['coordinates']='raw_incumbent'
    else:
        j=next(i for i,e in enumerate(events) if e['kind']=='native' and e['payload']['kind']=='module_inputs')
        events[j],events[j+1]=events[j+1],events[j]
    with pytest.raises(ValueError):m.validate_arm(row,events,slot(),load)


def test_real_writer_disk_reader_journal_roundtrip_and_extra_event(tmp_path):
    import time
    d=tmp_path/(m.NAME+'_RAM');store=m.Store(d,time.monotonic())
    row,events,load=ram_arm(slot(),store=store);store.put('arm0.json.gz',row);store.close()
    actual=m.journal(d)
    assert actual==[dict(kind='arm_event',index=0,payload=e) for e in events]
    saved=m.old.parent.overlay.read(d/'arm0.json.gz',32*1024**2)
    assert saved==row
    def diskload(desc):return np.load(d/desc['file'],allow_pickle=False)
    m.validate_arm(saved,[e['payload'] for e in actual],slot(),diskload)
    data=(d/'events.jsonl.gz').read_bytes();(d/'events.jsonl.gz').write_bytes(data[:-1])
    with pytest.raises(ValueError):m.journal(d)


@pytest.mark.parametrize('arm',['joint','composition'])
def test_nontrivial_local_units_same_algorithm_and_complete_contexts(arm):
    row,events,load=ram_arm(slot(),arm,stop_after=46)
    a=m.validate_arm(row,events,slot(),load)
    assert a['units_completed']==46
    assert a['attempted']['predict']==(174 if arm=='joint' else 154)
    assert a['attempted']['jacobian']==3


def test_full_composition_natural_end_and_true_double_replay_RAM():
    row,events,load=ram_arm(slot(),'composition',stop_after=1000)
    a=m.validate_arm(row,events,slot(),load)
    assert a['attempted']['predict']==1062 and a['attempted']['jacobian']==85
    assert a['stop']=='natural_completion' and a['export_complete']
    assert row['result']['export'] is not None


def saved_four(tmp_path,monkeypatch):
    import time
    policy=tmp_path/'policy';policy.mkdir();bindings={};files=[]
    for n in ('config.json','weights.pt'):
        p=policy/n;p.write_bytes(b'RAM only, never loaded');h=m.old.sha(p);bindings[str(p)]=h
        files.append(dict(path=str(p),sha256=h,bytes=p.stat().st_size,hash_buffer_sha256=h,deserialize_buffer_sha256=h,same_buffer=True))
    p=dict(slots=[slot(0),slot(1)],root_arm_order=[[0,'joint'],[0,'composition'],[1,'composition'],[1,'joint']],
        expected_model_hash='RAM_MODEL',policy=str(policy),bindings_sha256=bindings)
    prepared=tmp_path/'prepared.json';prepared.write_bytes(m.old.raw({'RAM':'not production prepare'}));ph=m.old.sha(prepared)
    d=tmp_path/(m.NAME+'_RAM');store=m.Store(d,time.monotonic())
    store.put('binding.json',dict(prepared=str(prepared),prepare_sha256=ph))
    for f in files:
        store.event(dict(kind='load_intent',path=f['path'],expected_sha256=f['sha256']));store.event(dict(kind='consumed_buffer_frozen',receipt=f))
    receipt=dict(schema=m.old.loader.NAME+'.loader_receipt',prepare_sha256=ph,adapter_source_sha256=m.old.sha(m.old.loader.__file__),
        legacy_loader_source_sha256=m.old.loader.native.SOURCES['experiments/checkpoint_compat.py'],files=files,device='cuda:0',
        remapped_keys=[],missing=[],unexpected=[],tensor_hash_before='RAM_MODEL',frozen_eval=True)
    store.put('loader.json',receipt);rows=[];stops=[]
    for i,(root,arm) in enumerate(p['root_arm_order']):
        row,_,_=ram_arm(p['slots'][root],arm,store=store,index=i)
        store.put(f'arm{i}.json.gz',row);store.event(dict(kind='arm_sealed',index=i,row_sha256=m.old.digest(row)))
        rows.append(dict(index=i,root_slot=root,arm=arm,file=f'arm{i}.json.gz'));stops.append(row['accounting']['stop'])
    store.close()
    r=dict(schema=m.NAME+'.report',prepared=str(prepared),prepare_sha256=ph,errors=[],model_before='RAM_MODEL',model_after='RAM_MODEL',
        model_unchanged=True,rows=rows,fixed_denominator=4,status='recorded_not_admitted',arm_stops=stops,
        artifact_sha256={f.name:m.old.sha(f) for f in d.iterdir()})
    store.put('report.json',r,final=True)
    monkeypatch.setattr(m,'verify',lambda *a:deepcopy(p))
    monkeypatch.setattr(m.old.parent,'check_files',lambda *a:None)
    return d


def test_actual_store_full4_saved_consume_no_model_and_exact_files(tmp_path,monkeypatch):
    d=saved_four(tmp_path,monkeypatch);r=m.consume(d,m.old.sha(d/'report.json'))
    assert len(r['plans'])==4 and r['plans']==[None]*4
    assert [a['attempted']['predict'] for a in r['accounts']]==[9]*4
    assert all(a['stop']=='wall_cap' for a in r['accounts'])
    assert len(r['comparisons'])==2 and r['total_FLOPs'] is None
    (d/'extra.txt').write_bytes(b'not allowed')
    with pytest.raises(ValueError):m.consume(d,m.old.sha(d/'report.json'))


def test_unknown_terminal_event_hash_resealed_rejected(tmp_path,monkeypatch):
    d=saved_four(tmp_path,monkeypatch)
    events=m.journal(d);events.append(dict(kind='unknown_terminal'))
    previous='0'*64;plain=b''
    for i,e in enumerate(events):
        unsigned=dict(index=i,previous_sha256=previous,payload=e);previous=m.old.digest(unsigned)
        plain+=m.old.raw(dict(unsigned,event_sha256=previous))
    import gzip,json
    (d/'events.jsonl.gz').write_bytes(gzip.compress(plain))
    r=json.loads((d/'report.json').read_bytes());r['artifact_sha256']['events.jsonl.gz']=m.old.sha(d/'events.jsonl.gz')
    (d/'report.json').write_bytes(m.old.raw(r))
    with pytest.raises(ValueError):m.consume(d,m.old.sha(d/'report.json'))


class MidJ(MeteredNumeric):
    def sink_native(self,e):
        if e['kind']=='AD_return':
            self.row['AD'][-1]['completed']=False;self.row['AD'][-1].pop('array')
            self.control.fake_now=550.;self.control.call('sample_memory',1024,2048)
            raise m.BudgetStop('wall_cap')
        super().sink_native(e)


class MidForward(MeteredNumeric):
    def jacobian(self,c,u,tag,logit):
        req=dict(index=0,tag=tag,input_context_sha256=m.old.digest(c),at_logits=np.asarray(u).tolist(),raw=m.old.chart(u).tolist(),
            coordinates='logit_zero' if logit else 'raw_incumbent',shape=[192,len(u)],mathematical_VJP=192,vectorize=True,completed=False)
        self.event(dict(kind='AD_intent',request=req));self.row['AD'].append(dict(request=req,completed=False))
        self.rollout(c,m.old.chart(u),tag+'/forward')
        pytest.fail('must stop inside first AD primal module')
    def sink_native(self,e):
        super().sink_native(e)
        if self.row['AD'] and e['kind']=='module_inputs':
            f=self.row['forwards'][-1];f['completed']=False;f.pop('output');f['modules']=[]
            self.control.fake_now=550.;self.control.call('sample_memory',1024,2048)
            raise m.BudgetStop('wall_cap')


@pytest.mark.parametrize('backend_class',[MidJ,MidForward])
def test_mid_AD_or_mid_forward_failure_unwind_full_strict_retains_non_success(backend_class,tmp_path):
    import time
    d=tmp_path/(m.NAME+'_'+backend_class.__name__);store=m.Store(d,time.monotonic());events=[]
    def emit(e):events.append(deepcopy(e));store.event(dict(kind='arm_event',index=0,payload=e))
    s=slot();c=RAMControl(s,'joint',emit,stop_after=1000);c.call('sample_memory',1024,2048)
    b=backend_class(c,emit,store=store);result=m.drive(s,'joint',b,c)
    c.call('seal');row=dict(b.row,root_slot=0,arm='joint',result=result,accounting=c.call('report'))
    def load(d):pytest.fail('missing AD matrix must not load or synthesize')
    a=m.validate_arm(row,events,s,load)
    store.put('arm0.json.gz',row);store.close()
    saved=m.old.parent.overlay.read(d/'arm0.json.gz',32*1024**2)
    m.validate_arm(saved,[e['payload'] for e in m.journal(d)],s,load)
    assert result['aborted_unit']=='bank' and a['attempted']['jacobian']==1 and a['completed']['jacobian']==0
    assert a['stop']=='wall_cap' and a['errors'] and not a['pending']
    assert a['incumbent']['tag']=='nominal' and a['export_missing']
    if backend_class is MidForward:
        assert a['attempted']['predict']==4 and a['completed']['predict']==3
        assert a['attempted']['action_encoder']==4 and a['completed']['action_encoder']==3
        bad=deepcopy(row);bad['forwards'][-1]['completed']=True
        with pytest.raises((ValueError,KeyError)):m.validate_arm(bad,events,s,load)


def test_actual_native_rollout_at1901_full3_IO_not_old1068():
    class Tensor:
        def __init__(self,a):self.a=np.asarray(a,np.float32)
        def __getitem__(self,key):return Tensor(self.a[key])
        def detach(self):return self
        def cpu(self):return self
        def numpy(self):return self.a
        def tolist(self):return self.a.tolist()
    class T:
        float32='float32'
        @staticmethod
        def tensor(a,**kw):return Tensor(a)
    b=m.HostBackend.__new__(m.HostBackend);b.t=T();b.device='RAM';b.row=dict(forwards=[None]*1901)
    b.check=lambda:None;events=[];b.event=events.append
    def transition(state,history,a):
        tokens=np.concatenate((history.a[0],m.old.kernel.normalized_history(a.a)))
        embed=np.zeros((1,3,192),np.float32);prediction=state.a.copy();projected=prediction.reshape(3,192)
        values=[([tokens[None]],embed),([state.a,embed],prediction),([projected],projected.copy())]
        for k,(ins,out) in enumerate(values):
            b.current['modules'].append(m.old.module_receipt(3*b.current['index']+k,m.protocol.MODULES[k],
                [m.old.blob(x) for x in ins],m.old.blob(out)))
        return Tensor(np.concatenate((state.a[:,1:],projected[-1:][None]),axis=1)),Tensor(tokens[None,1:])
    b.transition=transition
    _,path=b._rollout(slot()['packet']['root_context'],np.zeros((1,5,2),np.float32),'late')
    f=b.row['forwards'][-1]
    assert f['index']==1901 and [e['index'] for e in f['modules']]==[5703,5704,5705]
    assert m.validate_packet(f)==path[0]
    with pytest.raises(ValueError):b._rollout(path[0],np.zeros((1,5,2),np.float32),'over')


class MidReturned(MeteredNumeric):
    returned_count=3
    def sink_native(self,e):
        super().sink_native(e)
        if e['kind']=='module_return' and e['entry']['resource']==m.protocol.MODULES[self.returned_count-1]:
            f=self.row['forwards'][-1];f['completed']=False;f.pop('output');f['modules']=f['modules'][:self.returned_count]
            self.control.fake_now=550.;self.control.call('sample_memory',1024,2048)
            raise m.BudgetStop('wall_cap after module returned, before enclosing API return')


class FirstReturned(MidReturned):returned_count=1
class SecondReturned(MidReturned):returned_count=2


class PredictorInputs(MeteredNumeric):
    def sink_native(self,e):
        super().sink_native(e)
        if e['kind']=='module_inputs' and e['index']%3==1:
            f=self.row['forwards'][-1];f['completed']=False;f.pop('output');f['modules']=f['modules'][:1]
            self.control.fake_now=550.;self.control.call('sample_memory',1024,2048)
            raise m.BudgetStop('wall_cap with predictor entered but not returned')


@pytest.mark.parametrize('cls',[FirstReturned,SecondReturned,MidReturned,PredictorInputs])
def test_actual_module_API_prefixes_no_synthesized_returns(cls):
    events=[];emit=lambda e:events.append(deepcopy(e));s=slot()
    c=RAMControl(s,'joint',emit,stop_after=1000);c.call('sample_memory',1024,2048)
    b=cls(c,emit);result=m.drive(s,'joint',b,c);c.call('seal')
    row=dict(b.row,root_slot=0,arm='joint',result=result,accounting=c.call('report'))
    a=m.validate_arm(row,events,s,lambda d:pytest.fail('no AD'))
    assert a['attempted']['predict']==1 and a['completed']['predict']==0 and not a['pending']
    assert not any(e['kind']=='native' and e['payload']['kind']=='predict_API_return' for e in events)
    if cls is PredictorInputs:
        bad=[e for e in events if not (e['kind']=='native' and e['payload']['kind']=='predict_API_intent')]
        with pytest.raises(ValueError):m.validate_arm(row,bad,s,lambda d:None)
