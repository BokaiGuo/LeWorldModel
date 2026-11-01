"""Model-free arithmetic, native ABI and saved evidence tests only."""
from copy import deepcopy
import importlib
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import agent_raw_normalizer_ablation_20261028_v1 as m


def context():
    return dict(context_length=3,latent_dim=192,action_dim=10,layout='rolling_full_context',
        latent_history=np.zeros((3,192),np.float32).tolist(),action_history=np.full((2,10),9,np.float32).tolist())


def job():
    raw=np.arange(20,dtype=np.float32).reshape(10,2)/32
    return dict(index=0,slot=0,local_index=0,id='s0_baseline',observed_raw10=raw.tolist(),
        raw_suffix5=np.linspace(-.3,.3,10,dtype=np.float32).reshape(5,2).tolist(),
        observed_context=context(),task=dict(action_mean=np.asarray([-.0087,.0068],np.float32).tolist(),
            action_std=np.asarray([.2019,.2002],np.float32).tolist(),goal=np.zeros(192,np.float32).tolist()))


def stats():
    return dict(statistics=dict(mean=[-.007812564379916167,.006860687229453026],
        sample_std=[.20846748745157567,.2067486706124012]))


def test_only_two_fixed_arms_and_exact_caps():
    c=m.contract()
    assert c['arms']==['public','archive_sample_stats'] and c['forwards']==328 and c['pairs_per_arm']==80
    assert c['max_new_compute']==m.F.caps(328,0) and c['new_encoder']==c['new_physics']==c['new_training']==0
    assert c['output_bytes']==128*1024**2 and c['disk_floor_bytes']==200*1024**2 and c['wall_seconds']==1800
    assert c['training_provenance'] is c['deployment_allowed'] is c['P1_admission'] is False


@pytest.mark.parametrize('arm',[0,1])
def test_all_history_regenerated_once_chronological_not_reused_or_divided100(arm):
    j=job();a=m.arm_specs(j['task'],stats())[arm];p=m.packet_for(j,a)
    mean=np.asarray(a['mean'],np.float32);std=np.asarray(a['std'],np.float32)
    history=((np.asarray(j['observed_raw10'],np.float32)-mean)/std).reshape(2,10)
    token=((np.asarray(j['raw_suffix5'],np.float32)-mean)/std).reshape(10)
    assert p['history'].dtype==p['token'].dtype==np.float32
    assert p['history'].tobytes()==history.tobytes() and p['token'].tobytes()==token.tobytes()
    assert p['outgoing'].tobytes()==np.concatenate((history,token[None])).tobytes()
    assert not np.array_equal(p['history'],j['observed_context']['action_history'])
    bad=deepcopy(j);bad['observed_context']['action_history']=np.zeros((2,10)).tolist()
    assert m.packet_for(bad,a)['history'].tobytes()==history.tobytes()
    assert p['latent'].tobytes()==np.asarray(j['observed_context']['latent_history'],np.float32).tobytes()


def test_stats_float32_once_unverified_identity():
    a,b=m.arm_specs(job()['task'],stats())
    assert a['name']=='public' and a['training_provenance'] is False
    assert b['name']=='archive_sample_stats' and b['training_provenance'] is False
    assert np.asarray(b['mean'],np.float32).tobytes()==np.asarray(stats()['statistics']['mean'],np.float32).tobytes()
    assert np.asarray(b['std'],np.float32).tobytes()==np.asarray(stats()['statistics']['sample_std'],np.float32).tobytes()


def test_run_authority_before_any_read(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('unauthorized reads'))
    with pytest.raises(ValueError):m.run('/no','0'*64,'/no',None)


def test_cold_import_no_torch_or_environment():
    code="""import builtins
orig=builtins.__import__
def check(name,*a,**k):
    if name.split('.')[0] in ('torch','gymnasium','stable_worldmodel','pymunk'):raise AssertionError(name)
    return orig(name,*a,**k)
builtins.__import__=check
from experiments.spectral_reachability import agent_raw_normalizer_ablation_20261028_v1 as m
assert len(m.contract()['arms'])==2
"""
    r=subprocess.run([sys.executable,'-B','-c',code],cwd=m.ROOT,capture_output=True,text=True,timeout=30)
    assert r.returncode==0,r.stderr


from contextlib import contextmanager
from functools import lru_cache


@lru_cache(maxsize=1)
def saved():
    assert m.sha(m.PREP)==m.PREP_SHA and m.sha(m.UP/'report.json')==m.UP_SHA
    assert m.sha(m.STATS)==m.STATS_SHA and m.sha(m.PROTOCOL)==m.PROTOCOL_SHA
    f=m.signal.read(m.PREP,m.META);r=m.signal.read(m.UP/'report.json',m.META)
    p=dict(schema=m.NAME+'.prepare',contract=m.contract(),jobs=f['jobs'],manifest_sha256=f['manifest_sha256'],
        arms=m.arm_specs(f['jobs'][0]['task'],m.signal.read(m.STATS)),statistics_report=m.signal.read(m.STATS),
        reference=m.original_values(r))
    native=[m.F.load_journal(m.UP,row) for row in r['rows']]
    return p,r,native


def test_actual_all164_raw_context_goal_identity_and_rebuilt_public_tokens():
    p,_,_=saved();m.validate_roster(p)
    assert [j['index'] for j in p['jobs']]==list(range(164))
    for j in p['jobs']:
        a=m.packet_for(j,p['arms'][0]);b=m.packet_for(j,p['arms'][1])
        assert a['latent'].tobytes()==b['latent'].tobytes()
        assert a['raw'].tobytes()==b['raw'].tobytes()
        assert a['history'].tobytes()==m.FD.packet(j['observed_context']['action_history'],(2,10)).tobytes()
        assert a['history'].tobytes()!=b['history'].tobytes()


@pytest.mark.parametrize('bad',['last_index','late_raw','history','normalizer','ref_goal','arm_order','job_order'])
def test_all_roster_tamper_including_last_job(bad):
    p=deepcopy(saved()[0])
    if bad=='last_index':p['jobs'][-1]['index']=0
    elif bad=='late_raw':p['jobs'][-1]['raw_suffix5'][4][1]+=.1
    elif bad=='history':p['jobs'][-1]['observed_context']['action_history'][0][0]+=.125
    elif bad=='normalizer':p['arms'][1]['std'][1]+=.01
    elif bad=='ref_goal':p['reference'][-1]['objective']['arrays']['goal']=m.signal.blob(np.zeros(192,np.float32))
    elif bad=='arm_order':p['arms'].reverse()
    else:p['jobs'][1],p['jobs'][2]=p['jobs'][2],p['jobs'][1]
    with pytest.raises((ValueError,KeyError)):m.validate_roster(p)


class Replay:
    """Saved numerical IO fixtures, NOT a learned model or new experiment evidence."""
    def __init__(self,task,arm,journal,ledger,check):
        self.arm,self.j,self.l=arm,journal,ledger
        p,_,es=saved();panel=next(i for i in range(4) if p['jobs'][41*i]['task']['binding']==task['binding'])
        self.p=p;self.panel=panel;self.es=es[panel];self.current=None;self.last=None
    @contextmanager
    def observe(self):yield self
    def step(self,job):
        phase=f'f{job["local_index"]:03}';scope=dict(forward=phase,label=job['id']);self.l.scope=scope
        c=m.input_context(job,self.arm);packet=m.packet_for(job,self.arm);self.current=job
        local=[e for e in self.es if e.get('phase')==phase]
        def original(which,key):
            return deepcopy(next(e['arrays'][key] for e in local if e['kind']==which and key in e.get('arrays',{})))
        def emit(kind,**kw):self.j.event(dict(kind=kind,**kw))
        def arrays(kind,values,index=None):
            emit(kind,scope=scope,phase=phase,step=0,operation_index=index,arrays=values)
        def call(resource,inputs,output,callback=None):
            index=len(self.l.events)
            def invoke():
                arrays('io_input',inputs,index)
                if callback:callback()
                arrays('io_return',{resource+':output':output},index)
            first=m.signal.unblob(next(iter(inputs.values())))
            self.l.call(resource,invoke,shape=list(first.shape))
        emit('forward_started',phase=phase,label=job['id'],input=c,commands=job['raw_suffix5'])
        arrays('program_packet',{k:m.signal.blob(v[None] if k in ('latent','history','outgoing') else v) for k,v in packet.items()})
        emb=original('io_return','action_encoder:output');pred=original('io_return','predictor:output')
        proj=original('io_return','pred_proj:output');prediction=original('io_return','predict:output')
        call('action_encoder',{'action_encoder:0':m.signal.blob(packet['outgoing'][None])},emb)
        def inner():
            call('predictor',{'predictor:0':m.signal.blob(packet['latent'][None]),'predictor:1':emb},pred)
            call('pred_proj',{'pred_proj:0':m.signal.blob(m.signal.unblob(pred).reshape(3,192))},proj)
        call('predict',{'predict:latent':m.signal.blob(packet['latent'][None]),'predict:act_emb':emb},prediction,inner)
        z=np.concatenate((packet['latent'][None,1:],m.signal.unblob(prediction)[:,-1:]),axis=1)
        h=packet['outgoing'][None,1:];arrays('program_boundary',{'latent':m.signal.blob(z),'history':m.signal.blob(h)})
        after=dict(c,latent_history=z[0].tolist(),action_history=h[0].tolist());self.last=after
        emit('forward_finished',phase=phase,context=after);return after
    def objective(self,after):return deepcopy(self.p['reference'][self.current['index']]['objective']['arrays'])


class Events:
    def __init__(self):self.events=[]
    def event(self,value):self.events.append(deepcopy(value))


@lru_cache(maxsize=1)
def recorded():
    p,_,_=saved();j=Events();l=m.core.Ledger(m.F.caps(41,0),sink=j.event);row=m.new_row(0,0)
    b=Replay(p['jobs'][0]['task'],p['arms'][0],j,l,lambda:None)
    m.evaluate(p['jobs'][:41],p['arms'][0],b,row,j,p['reference'][:41]);row['resources']=l.receipt()
    return row,j.events


def test_actual_format_native_all41_public_exact_zero_AD():
    row,events=recorded();p=saved()[0];m.audit_row(p,row,events)
    assert row['resources']['attempted']==row['resources']['completed']==m.F.caps(41,0)
    for job,v,ref in zip(p['jobs'][:41],row['values'],p['reference'][:41],strict=True):m.public_exact(job,v,ref)


@pytest.mark.parametrize('bad',['unknown','old_history','double_suffix','raw','normalizer','AD','predictor_input',
    'projector_shape','public_last_output','objective','no_return','io_order'])
def test_native_saved_route_mutations_rejected(bad):
    row,events=deepcopy(recorded());p=saved()[0]
    pkt=next(e for e in events if e['kind']=='program_packet')
    if bad=='unknown':events.insert(3,dict(kind='unknown'))
    elif bad=='old_history':pkt['arrays']['history']=m.signal.blob(np.zeros((1,2,10),np.float32))
    elif bad=='double_suffix':
        a=m.signal.unblob(pkt['arrays']['token']);pkt['arrays']['token']=m.signal.blob((a-.1)/.2)
    elif bad=='raw':pkt['arrays']['raw']=m.signal.blob(np.zeros((5,2),np.float32))
    elif bad=='normalizer':pkt['arrays']['std']=m.signal.blob(np.ones(2,np.float32))
    elif bad=='AD':events.append(dict(kind='AD_API_attempt'))
    elif bad=='predictor_input':
        e=next(e for e in events if e['kind']=='io_input' and 'predictor:0' in e['arrays'])
        e['arrays']['predictor:0']=m.signal.blob(np.zeros((1,3,192),np.float32))
    elif bad=='projector_shape':
        e=next(e for e in events if e['kind']=='io_input' and 'pred_proj:0' in e['arrays'])
        e['arrays']['pred_proj:0']=m.signal.blob(np.zeros((1,3,192),np.float32))
    elif bad=='public_last_output':row['values'][-1]['context']['latent_history'][-1][0]+=.125
    elif bad=='objective':row['values'][-1]['objective']['arrays']['goal']=m.signal.blob(np.zeros(192,np.float32))
    elif bad=='no_return':events.remove(next(e for e in events if e['kind']=='request_return'))
    else:
        e=next(e for e in events if e['kind']=='io_return');events.remove(e);events.insert(0,e)
    with pytest.raises((ValueError,KeyError,StopIteration)):m.audit_row(p,row,events)


def test_all164_public_exact_including_last_not_only_baselines():
    p,_,_=saved()
    for j,ref in zip(p['jobs'],p['reference'],strict=True):
        v=dict(job_id=j['id'],context=deepcopy(ref['context']),objective=deepcopy(ref['objective']))
        m.public_exact(j,v,ref)
    v['context']['latent_history'][-1][0]+=.01
    with pytest.raises(ValueError):m.public_exact(p['jobs'][-1],v,p['reference'][-1])


@pytest.mark.parametrize('x,y,cat',[(0.,0.,'both_tie'),(1.,0.,'actual_tie'),(0.,1.,'predicted_tie'),
    (-1.,1.,'strict_reversal'),(-1.,-1.,'same_sign'),(None,0.,'missing')])
def test_scalar_ties_never_counted_as_agreement(x,y,cat):
    assert m.comparison(x,y)['category']==cat


def test_postrun_all80_each_arm_both_h_retained_and_ties_distinct():
    p,_,_=saved();assert m.sha(m.REAL)==m.REAL_SHA
    real=m.signal.read(m.REAL,m.META)
    rows=[m.new_row(a,i) for a in range(2) for i in range(4)]
    # Reusing saved model controls as fixture values is not an archive-arm result.
    for row in rows:
        i=row['panel'];row['values']=[dict(index=k,status='returned',context=p['reference'][k]['context'],
            objective=p['reference'][k]['objective']) for k in range(41*i,41*(i+1))]
    panels=m.diagnostics(p,rows,real)
    assert len(panels)==2
    for arm in panels:
        assert len(arm['endpoints'])==164 and sum(len(s['pairs']) for v in arm['panels'] for s in v['steps'])==80
        for panel in arm['panels']:
            assert [s['h'] for s in panel['steps']]==[.01,.001]
            for step in panel['steps']:assert sum(step['signs'].values())==10
        assert arm['no_winning_arm_or_h'] and not arm['training_provenance']


@pytest.mark.parametrize('bad',['index','goal','probe','latent_missing'])
def test_real_receipt_join_tamper_not_allowed(bad):
    p,_,_=saved();real=m.signal.read(m.REAL,m.META)
    if bad=='index':real['rows'][-1]['index']=0
    elif bad=='goal':real['rows'][-1]['goal192'][0]+=.1
    elif bad=='probe':real['rows'][-1]['producer_job_sha256']='0'*64
    else:real['rows'][-1]['latent_missing']=True
    with pytest.raises(ValueError):m.diagnostics(p,[m.new_row(a,i) for a in range(2) for i in range(4)],real)


def test_diagnostics_exact_zero_and_missing_pairs_stay_in_all80():
    p,_,_=saved();real=m.signal.read(m.REAL,m.META)
    real['rows'][2]['actual192']=deepcopy(real['rows'][1]['actual192'])
    z=np.asarray(real['rows'][2]['actual192'],float);g=np.asarray(real['rows'][2]['goal192'],float)
    real['rows'][2]['actual_Q']=float(np.mean((z-g)**2))
    real['rows'][3].update(latent_missing=True,actual192=None,actual_Q=None)
    rows=[m.new_row(a,i) for a in range(2) for i in range(4)]
    for row in rows:
        row['values']=[dict(index=k,status='returned',context=p['reference'][k]['context'],objective=p['reference'][k]['objective'])
            for k in range(41*row['panel'],41*(row['panel']+1))]
    diagnostics=m.diagnostics(p,rows,real)
    for arm in diagnostics:
        s=arm['panels'][0]['steps'][0];zero,missing=s['pairs'][:2]
        assert zero['vector']['exact_actual_zero_response'] and zero['vector']['response_metrics']['cosine'] is None
        assert zero['scalar_signs']['category'] in ('both_tie','actual_tie')
        assert missing['scalar_signs']['category']=='missing' and missing['vector'] is None
        assert s['actual_zero_response_pairs']>=1 and sum(s['signs'].values())==10
        assert sum(len(st['pairs']) for panel in arm['panels'] for st in panel['steps'])==80


def _fake_report(p,rows,store,prepared,ph,errors):
    h=dict(original_state_dict='a'*64)
    p.update(bindings={},original_model_before=h,original_model_after=h,upstream=dict(fixture='RAM_ONLY'),
        protocol=dict(path=str(m.PROTOCOL),sha256=m.PROTOCOL_SHA),
        deferred_real_RGB_authority=dict(path=str(m.REAL),sha256=m.REAL_SHA),inherited_compute={},inherited_resources={})
    store.put('binding.json',dict(prepared=str(prepared),prepare_sha256=ph))
    report=dict(schema=m.NAME+'.report',status='partial_failclosed' if errors else 'normalizer328_recorded_diagnostic_only',
        contract=m.contract(),prepared=str(prepared),prepare_sha256=ph,rows=rows,errors=errors,fixed_denominator_per_arm=164,
        source_verified=True,source_manifest_sha256=m.digest({}),protocol=p['protocol'],
        deferred_real_RGB_authority=p['deferred_real_RGB_authority'],model_before=h,model_after=h,
        precision_before=m.precision(),precision_after=m.precision(),CUDA_peak_bytes=[0,0],wall_seconds=1.,
        artifact_sha256={str(f.relative_to(store.path)):m.sha(f) for f in store.path.rglob('*') if f.is_file()},
        P1_admission=False,freshness=False,training_provenance=False,deployment_allowed=False,scientific_admission=False,physical_release=False)
    store.put('report.json',report,final=True);return report


@pytest.mark.parametrize('failure',[False,True])
def test_fixed328_dispatch_saved_full_consume_or_failure_preservation(tmp_path,monkeypatch,failure):
    p=dict(saved()[0]);monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    store=m.Store(tmp_path/'results'/(m.NAME+'_RAM_TEST'),m.time.monotonic())
    rows=[m.new_row(a,i) for a in range(2) for i in range(4)]
    class Fault(Replay):
        def step(self,j):
            if failure and j['index']==2:raise RuntimeError('RAM pre-call failure')
            return super().step(j)
    if failure:
        with pytest.raises(RuntimeError):m.dispatch(p,store,rows,lambda:None,lambda t,a,j,l:Fault(t,a,j,l,lambda:None))
        assert rows[0]['status']=='failed_partial' and all(v['status']=='not_started' for v in rows[1:])
        assert rows[0]['resources']['attempted']==rows[0]['resources']['completed']==m.F.caps(2,0)
    else:
        m.dispatch(p,store,rows,lambda:None,lambda t,a,j,l:Replay(t,a,j,l,lambda:None))
        assert all(row['status']=='recorded' for row in rows)
        assert sum(r['resources']['completed']['predict_API'] for r in rows)==328
    prepared=tmp_path/'RAM_fixture.json';prepared.write_bytes(b'RAM fixture only, never production')
    ph=m.sha(prepared);_fake_report(p,rows,store,prepared,ph,[dict(stage='RAM',error='kept failure')] if failure else [])
    monkeypatch.setattr(m,'verify',lambda f,h:p if (f,h)==(str(prepared),ph) else pytest.fail('RAM binding'))
    receipt=m.consume(store.path,m.sha(store.path/'report.json'))
    assert receipt['complete'] is (not failure) and receipt['pair_denominator_per_arm']==80
    assert receipt['public_exact_controls']==(2 if failure else 164)
    assert receipt['actual_new_compute']['attempted']==receipt['actual_new_compute']['completed']==m.F.caps(2 if failure else 328,0)
    assert bool(receipt['errors']) is failure and not receipt['training_provenance']
    for arm in receipt['arms']:
        assert sum(len(s['pairs']) for panel in arm['panels'] for s in panel['steps'])==80
    if failure:
        assert sum(s['signs']['missing'] for panel in receipt['arms'][1]['panels'] for s in panel['steps'])==80
    (store.path/'unexpected').write_bytes(b'extra')
    with pytest.raises(ValueError):m.consume(store.path,m.sha(store.path/'report.json'))




