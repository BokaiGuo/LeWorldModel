"""Fixed164 native-ABI RAM tests, no real environments or model execution."""
from copy import deepcopy
from functools import lru_cache
import hashlib
from pathlib import Path
import time
import numpy as np
import pytest
from experiments.spectral_reachability import agent_observed_raw_coordinate_physical_20261028_v1 as m
import test_agent_observed_suffix_physical_20261028_v1 as old


class RamStore(m.Store):
    """Native Store ABI, incremental journal bytes; no disk/physics callbacks."""
    def __init__(self):
        self.path=Path('/RAM_ONLY');self.previous=None;self.index=0;self.events=[]
        self._files={};self.chunks=[];self.journal_hash=hashlib.sha256();self.bytes=0
    @property
    def files(self):return dict(self._files,**({'journal.jsonl':b''.join(self.chunks)} if self.chunks else {}))
    def check(self,size=4096,final=False):
        m.require(self.bytes+size<=m.CAP-(0 if final else m.FINAL),'RAM own1GiB budget')
    def write(self,name,value,append=False,final=False):
        self.check(len(value),final);self.bytes+=len(value)
        if name=='journal.jsonl':
            assert append;self.chunks.append(value);self.journal_hash.update(value);return self.journal_hash.hexdigest()
        assert not append and name not in self._files
        self._files[name]=value;return hashlib.sha256(value).hexdigest()
    def emit(self,kind,**fields):
        e=dict(index=self.index,previous_sha256=self.previous,kind=kind,**deepcopy(fields))
        e['event_sha256']=m.source.digest(e);body=m.raw(e);self.write('journal.jsonl',body,append=True)
        self.events.append(m.mixed.decode(body));self.previous=e['event_sha256'];self.index+=1



def test_fixed_resource_arithmetic_not_41_independent_roots():
    c=m.contract();assert c['fixed_denominator']==164 and c['panels']==4 and c['jobs_per_panel']==41
    assert c['caps']==dict(constructor=164,close=164,reset=328,public_step=3280,physics_step=33456,
        set_state=656,goal_setter=492,goal_render=328,goal_bind=164,explicit_render=10824)
    assert c['wall_seconds']==1800 and c['output_cap_bytes']==c['disk_floor_bytes']==1024**3
    assert c['domain']=='ambient_raw_box_not_original_logit_L2' and not c['P1_admission']


def test_two_codecs_remain_distinct():
    assert m.digest([1,2])!=m.producer_digest([1,2])


def test_store_large_budget_and_exact_bytes_no_flat_rescan(tmp_path,monkeypatch):
    output=tmp_path/(m.NAME+'_RAM');monkeypatch.setattr(m.shutil,'disk_usage',lambda p:type('D',(),dict(free=3*1024**3))())
    s=m.Store(output);s.write('chunk.bin',b'12');s.write('chunk.bin',b'34',append=True)
    assert s.bytes==4 and (output/'chunk.bin').read_bytes()==b'1234'
    s.put('small.json',dict(a=1));assert s.bytes==sum(f.stat().st_size for f in output.iterdir())
    with pytest.raises(ValueError):s.write('../outside',b'bad')


def test_meter_uses_164_caps_not_old4_and_durable_failure():
    class Sink:
        def __init__(self):self.events=[]
        def check(self,*a,**k):pass
        def emit(self,kind,**fields):self.events.append(dict(kind=kind,**deepcopy(fields)))
    s=Sink();q=m.Meter(s);q.where=dict(branch=0,root_slot=0,condition='off',arm='RAM',selection_digest=old.old.h('plan'))
    for _ in range(10):q.call('constructor',lambda:None)
    assert q.counts['constructor']==dict(attempted=10,completed=10)
    with pytest.raises(RuntimeError):q.call('constructor',lambda:(_ for _ in ()).throw(RuntimeError('RAM fail')))
    assert q.counts['constructor']==dict(attempted=11,completed=10)
    assert s.events[-1]['kind']=='operation_failed'


def test_cp5_guard_rejects_mismatch_before_native_callback():
    class Sink:
        def __init__(self):self.events=[]
        def check(self,*a,**k):pass
        def emit(self,kind,**fields):self.events.append(dict(kind=kind,**deepcopy(fields)))
    p,*_=old.fixture();ref=deepcopy(p['prefix_reference'][0]);s=Sink();q=m.Meter(s)
    row=dict(index=0,prefix51=dict(complete=True,frames=deepcopy(ref['frames'])),checkpoints=[deepcopy(ref['checkpoint'])],own_returned=0)
    row['checkpoints'][0]['state'][0]+=1
    q.where=dict(branch=0,root_slot=0,condition='off',arm='RAM',selection_digest=old.old.h('plan'))
    q.guard_context(row,ref)
    q.receipts=[dict(resource='public_step',branch=0,phase='plan',step=i,status='returned') for i in range(1,6)]
    with pytest.raises(ValueError):q.call('public_step',lambda:pytest.fail('NO suffix callback'),phase='plan',step=6)
    assert s.events[-1]['kind']=='cp5_guard' and s.events[-1]['accepted'] is False
    assert not any(e['kind']=='operation_dispatch_intent' for e in s.events)


def test_authorization_precedes_any_reads(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('unauthorized read'))
    with pytest.raises(ValueError):m.run(None,None,None,lambda:pytest.fail('physics'),authorization=None)


def _rehash(j):
    j['job_sha256']=m.producer_digest({k:v for k,v in j.items() if k!='job_sha256'})


@lru_cache(maxsize=1)
def _ram_prepare():
    p,data,c,r,u,bp,br,args=old.fixture()
    jobs=[]
    for slot,plan in enumerate(bp['plans']):
        anchor=np.asarray(plan['commands'][5:],np.float32);prefix=np.asarray(plan['commands'][:5],np.float32)
        task=deepcopy(c['rows'][slot]['request']['task']);ctx=deepcopy(task['root_context'])
        schedule=[(None,None,0)]+[(co,h,s) for h in m.producer.STEPS for co in range(10) for s in (-1,1)]
        for local,(coord,h,sign) in enumerate(schedule):
            a,delta,reason=(anchor.copy(),0.,None) if coord is None else m.producer.perturb(anchor.tolist(),coord,h,sign)
            job_id=f's{slot}_baseline' if coord is None else f's{slot}_h{m.producer.STEPS.index(h)}_c{coord}_{"minus" if sign<0 else "plus"}'
            j=dict(index=len(jobs),slot=slot,local_index=local,id=job_id,root_slot=plan['root_slot'],
                condition=plan['condition'],coord=coord,h=h,sign=sign,raw_suffix5=a.tolist(),raw_bits=m.producer.F.bits(a),
                prefix5=prefix.tolist(),prefix_bits=m.producer.F.bits(prefix),full10=prefix.tolist()+a.tolist(),
                anchor_raw_suffix5=anchor.tolist(),actual_delta=delta,legal=reason is None,missing_reason=reason,
                task=deepcopy(task),task_id=task['binding']['task_id'],observed_context=deepcopy(ctx),
                observed_raw10=m.moving.warmup_commands()[5:].tolist()+prefix.tolist(),
                root_raw10=m.moving.warmup_commands().tolist(),original_AD_raw_gradient=np.zeros((5,2)).tolist(),
                baseline_reference_id=f's{slot}_baseline',
                original_AD_context=deepcopy(ctx) if coord is None else None,
                original_iterate0_context=deepcopy(ctx) if coord is None else None,
                provenance=dict(parent_selection_digest=plan['producer_selection_digest']),
                deployment_allowed=False,physical_release=False)
            _rehash(j);jobs.append(j)
    f=dict(schema=m.producer.NAME+'.prepare',contract=m.producer.contract(),full_upstream_strict_rebuilt_here=True,
        production_ready=True,jobs=jobs,manifest_sha256=m.producer_digest(jobs))
    arguments=dict(fd_prepared='/RAM/fd.json',fd_prepare_sha='a'*64,fd_source_sha='b'*64)
    return m.assemble(arguments,f,bp,br,data),data,f,bp,br


def ram_prepare():return deepcopy(_ram_prepare())


def test_pure_native164_joins_baseline_original_not_cp5_incoming_history():
    p,data,*_=ram_prepare();m.validate_execution(p)
    assert len(p['plans'])==len(p['prefix_reference'])==164
    for i in (0,41,82,123):
        assert m.mixed.raw10(p['plans'][i]['commands']).tobytes()!=m.mixed.raw10(p['jobs'][i]['observed_raw10']).tobytes()
    assert p['plans'][0]['selection_digest']==m.digest({k:v for k,v in p['plans'][0].items() if k!='selection_digest'})
    assert p['plans'][0]['producer_job_sha256']==m.producer_digest({k:v for k,v in p['jobs'][0].items() if k!='job_sha256'})


@pytest.mark.parametrize('bad',['count','order','prefix','suffix','goal','codec','legal','baseline','baseline_ref','terminal_context'])
def test_manifest_and_native_join_mutations(bad):
    p,data,f,bp,br=ram_prepare();j=f['jobs'][0]
    if bad=='count':f['jobs'].pop()
    elif bad=='order':f['jobs'][1],f['jobs'][2]=f['jobs'][2],f['jobs'][1]
    elif bad=='prefix':j['full10'][0][0]=.125
    elif bad=='suffix':f['jobs'][1]['raw_suffix5'][0][0]+=.1
    elif bad=='goal':j['task']['goal'][0]+=.125
    elif bad=='codec':j['job_sha256']=m.digest({k:v for k,v in j.items() if k!='job_sha256'})
    elif bad=='legal':j['legal']=False;j['missing_reason']='outside_rawbox'
    elif bad=='baseline_ref':j['baseline_reference_id']='s1_baseline'
    elif bad=='terminal_context':
        f['jobs'][1]['original_AD_context']=deepcopy(j['original_AD_context']);_rehash(f['jobs'][1])
    else:
        bp['plans'][0]['commands'][5][0]+=.125
        bp['plans'][0]['selection_digest']=m.digest({k:v for k,v in bp['plans'][0].items() if k!='selection_digest'})
        br['rows'][0]['plan']=deepcopy(bp['plans'][0])
    if bad!='codec':_rehash(j)
    f['manifest_sha256']=m.producer_digest(f['jobs'])
    with pytest.raises((ValueError,KeyError)):m.assemble(p['arguments'],f,bp,br,data)


from functools import lru_cache


@lru_cache(maxsize=1)
def partial_native():
    p,data,*_=ram_prepare();s=RamStore();envs=[];calls=0
    def factory():
        nonlocal calls
        calls+=1
        if calls>2:raise RuntimeError('RAM explicit constructor failure, retained')
        e=old.old.Env();envs.append(e);return e
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m.mixed,'place',old.old.fake_place);mp.setattr(m.mixed,'validate_env',lambda _:None)
        mp.setattr(m,'sha',lambda path:hashlib.sha256(s.files[Path(path).name]).hexdigest())
        r=m.execute(p,data,s,factory,authorization=m.AUTH,prepared='/RAM/prepare.json',prepare_sha256='c'*64)
    return r,s,p,data,envs


def test_real_native_ABI_first_two_jobs_and_all164_failure_denominator():
    r,s,p,data,envs=partial_native();rows=m.validate_saved(r,s.events,p)
    assert r['status']=='partial_or_failed',r['errors']
    assert len(rows)==164 and r['counts']['constructor']==dict(attempted=164,completed=2)
    assert all(x['status']=='horizon_completed' for x in rows[:2])
    assert all(x['status']=='engineering_failed' for x in rows[2:])
    assert r['counts']['public_step']['completed']==40 and r['counts']['physics_step']['completed']==408
    assert sum(x['PNG'] for x in rows)==132 and all(e.closed for e in envs)
    for i in range(2):
        assert np.asarray(envs[i].commands[10:],np.float32).tobytes()==m.mixed.raw10(p['plans'][i]['commands']).tobytes()
        g=next(e for e in s.events if e['kind']=='cp5_guard' and e['branch']==i)
        op=next(e for e in s.events if e['kind']=='operation_dispatch_intent' and e['receipt']['branch']==i
            and e['receipt']['resource']=='public_step' and e['receipt'].get('phase')=='plan' and e['receipt']['step']==6)
        assert g['accepted'] and g['index']<op['index']


@pytest.mark.parametrize('bad',['unknown','raw','cp5','missing_guard','native_label','counts','order'])
def test_native_partial_resealed_mutations(bad):
    r,s,p,*_=partial_native();r,es,p=deepcopy(r),deepcopy(s.events),deepcopy(p)
    if bad=='unknown':es[1]['kind']='unknown'
    elif bad=='raw':next(e for e in es if e['kind']=='returned_transition')['actual_bits']='00'
    elif bad=='cp5':r['rows'][0]['checkpoints'][1]['state'][0]+=1
    elif bad=='missing_guard':es.remove(next(e for e in es if e['kind']=='cp5_guard'))
    elif bad=='native_label':next(e for e in es if e['kind']=='returned_transition')['reward']+=.1
    elif bad=='counts':r['counts']['constructor']['completed']=3
    else:r['rows'][1],r['rows'][2]=r['rows'][2],r['rows'][1]
    old.old.reseal(es,r)
    with pytest.raises((ValueError,KeyError,StopIteration)):m.validate_saved(r,es,p)


def test_actual_saved_ABI_plan_jobs_to_native_assemble_no_environment():
    root=m.ROOT/'results';stem=m.producer.source.NAME
    prep=root/(stem+'_prepare_MAIN01.json');out=root/(stem+'_cuda_MAIN01');receipt=root/(stem+'_consume_MAIN01.json')
    if not receipt.exists():pytest.skip('actual saved compile authority absent')
    assert m.sha(out/'report.json')=='3de58e380acbef66269e69150cab7bca312604b7df88a0f1ae5663764d691b45'
    assert m.sha(receipt)=='09d617dbece12ce3ff2ca7246390e433454b1d8ac29042ab07eadd2be75238ec'
    c=m.read(prep,m.META_CAP);r=m.read(out/'report.json',m.META_CAP);u=m.read(receipt,m.META_CAP)
    a=dict(prepared=str(prep),prepare_sha256=m.sha(prep),output=str(out),report_sha256=m.sha(out/'report.json'),
        receipt=str(receipt),receipt_sha256=m.sha(receipt))
    jobs=m.producer.plan_jobs(c,r,u,a)
    f=dict(schema=m.producer.NAME+'.prepare',contract=m.producer.contract(),production_ready=True,
        full_upstream_strict_rebuilt_here=True,jobs=jobs,manifest_sha256=m.producer_digest(jobs))
    bp,br=m.observed.reference(c);data=m.mixed.source_material()
    p=m.assemble(dict(fd_prepared='/RAM/saved_metadata_only.json',fd_prepare_sha='d'*64,fd_source_sha=m.sha(m.producer.__file__)),f,bp,br,data)
    m.validate_execution(p)
    assert len(p['plans'])==164 and len(m.raw(p))<m.META_CAP
    for slot in range(4):
        assert m.mixed.raw10(p['plans'][41*slot]['commands']).tobytes()==m.mixed.raw10(bp['plans'][slot]['commands']).tobytes()
        assert p['jobs'][41*slot]['observed_raw10']!=p['plans'][41*slot]['commands']


def test_cold_import_does_not_load_model_or_physics():
    import subprocess,sys
    code="""
import builtins
original=builtins.__import__
def guard(name,*a,**k):
    if name.split('.')[0] in ('torch','gymnasium','stable_worldmodel','pymunk'):raise AssertionError(name)
    return original(name,*a,**k)
builtins.__import__=guard
from experiments.spectral_reachability import agent_observed_raw_coordinate_physical_20261028_v1 as m
assert m.contract()['no_neural']
"""
    p= subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,cwd=m.ROOT,timeout=30)
    assert p.returncode==0,p.stderr


def test_full164_success_saved_lifecycle_and_10824_PNG_consumer(tmp_path,monkeypatch):
    """Every callback here is the frozen NumPy RAM Env; no gym/neural imports."""
    p,data,*_=ram_prepare();s=RamStore();envs=[]
    def factory():
        e=old.old.Env();envs.append(e);return e
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m.mixed,'place',old.old.fake_place);mp.setattr(m.mixed,'validate_env',lambda _:None)
        mp.setattr(m,'sha',lambda path:hashlib.sha256(s.files[Path(path).name]).hexdigest())
        r=m.execute(p,data,s,factory,authorization=m.AUTH,prepared='/RAM/prepare.json',prepare_sha256='f'*64)
    assert r['status']==m.RECORDED,r['errors'][:3]
    assert {k:v['completed'] for k,v in r['counts'].items()}==m.contract()['caps']
    assert r['counts']['reset']==dict(attempted=328,completed=328)
    assert len(envs)==164 and all(e.closed for e in envs)
    output=tmp_path/(m.NAME+'_RAM_FULL')
    output.mkdir()
    for name,body in s.files.items():(output/name).write_bytes(body)
    monkeypatch.setattr(m,'verify',lambda prepared,expected:(p,data) if
        (prepared,expected)==('/RAM/prepare.json','f'*64) else pytest.fail('RAM authority mismatch'))
    receipt=m.consume(output,m.sha(output/'report.json'))
    assert receipt['status']==m.RECORDED and receipt['saved_PNG_verified']==10824
    assert receipt['fixed_denominator']==len(receipt['rows'])==len(receipt['metric_breakdown'])==164
    assert len([e for e in s.events if e['kind']=='cp5_guard' and e['accepted']])==164
    assert receipt['counts']['reset']['completed']==328 and receipt['P1_admission'] is False
    (output/'extra').write_bytes(b'x')
    with pytest.raises(ValueError):m.consume(output,m.sha(output/'report.json'))


def test_actual_fd_production_prepare_full_build_saved_only():
    prepared=m.ROOT/'results'/(m.producer.NAME+'_prepare_MAIN01.json')
    if not prepared.exists():pytest.skip('MAIN FD prepare not yet saved')
    assert m.sha(m.producer.__file__)=='83bf2420f5555f892ec1222bd158dde0634b096fa19f8a4bc626a2e07347970e'
    p,data=m.material(dict(fd_prepared=str(prepared),fd_prepare_sha=m.sha(prepared),fd_source_sha=m.sha(m.producer.__file__)))
    assert len(p['plans'])==len(p['jobs'])==164 and len(m.raw(p))<=m.META_CAP
    assert p['planner_authority']['full_upstream_strict_rebuilt_here'] and p['contract']['caps']['reset']==328



