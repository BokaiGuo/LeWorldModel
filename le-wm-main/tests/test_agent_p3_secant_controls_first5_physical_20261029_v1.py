"""Only RAM mock native arrays + real temporary Store; no model/native physics."""
from copy import deepcopy
from functools import lru_cache
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from PIL import Image
import pytest
from experiments.spectral_reachability import agent_p3_secant_controls_first5_physical_20261029_v1 as m
import test_agent_p3_secant_commit_physical_20261029_v1 as inherited
reference=inherited.reference
old=inherited.old
_temps=[]


def reseal(p,i):
    plan=p['plans'][i]
    plan['prefix_reference_sha256']=m.digest(p['prefix_reference'][i])
    plan['selection_digest']=m.digest({k:v for k,v in plan.items() if k!='selection_digest'})


def fixture():
    cp,data,*_=inherited.fixture()
    shared=dict(schema=m.SHARED+'.prepare',layers=[])
    for root in range(2):
        base=cp['plans'][2*root]
        task_path='/RAM/task_'+str(root)+'.json.gz';task_sha=base['base_task']['binding']['task_sha256']
        cp['bindings'][task_path]=task_sha
        shared['layers'].append(dict(arm='line_search_GN',source_slot=root,
            selection_sha256=m.SELECTIONS[root],task_id=base['task_id'],
            task_export=task_path,task_sha256=task_sha,
            commands=base['prefix_commands']+[[.125,-.25]]*5))
    cr=dict(authorities=m.authority_descriptors(),new_C_dispatch=0,**m.FLAGS)
    return m.assemble(cp,shared,cr),data,cp,shared,cr


def execute(p=None,data=None,*,fault=None,at=18):
    if p is None:p,data,*_=fixture()
    tmp=tempfile.TemporaryDirectory(prefix='ZP_RAM_');_temps.append(tmp)
    root=Path(tmp.name);prepared=root/'RAM_prepared.json';prepared.write_bytes(m.raw(p))
    store=m.Store(root/(m.NAME+'_RAM'));envs=[]
    def factory():
        e=old.Env(fault=fault,at=at);envs.append(e);return e
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m.native,'place',old.fake_place);mp.setattr(m.native,'validate_env',lambda _:None)
        r=m.execute(p,data,store,factory,authorization=m.AUTH,prepared=str(prepared),prepare_sha256=m.sha(prepared))
    events=[m.native.decode(x) for x in (store.path/'journal.jsonl').read_bytes().splitlines(keepends=True)]
    return r,store,p,data,envs,events


@lru_cache(maxsize=1)
def complete():return execute()


def test_fixed_order_literal_actions_no_model_witness():
    p,_,cp,shared,cr=fixture()
    assert [(x['root_slot'],x['arm']) for x in p['plans']]==m.ORDER
    assert p['planner_authority']==cr and cr['new_C_dispatch']==0
    for i,plan in enumerate(p['plans']):
        assert 'model_delivery' not in plan and 'witness_status' not in plan
        a=np.asarray(plan['control_raw5'],dtype='<f4')
        expected=np.zeros((5,2),dtype='<f4') if i%2==0 else np.asarray(shared['layers'][i//2]['commands'][5:],dtype='<f4')
        assert a.tobytes()==expected.tobytes()


@pytest.mark.parametrize('kind',['selection','domain','prefix','task','source_slot','arm','history','goal'])
def test_wrong_saved_controls_failclosed(kind):
    _,_,cp,shared,cr=fixture()
    if kind=='selection':shared['layers'][0]['selection_sha256']='f'*64
    elif kind=='domain':shared['layers'][0]['commands'][9][0]=1.01
    elif kind=='prefix':shared['layers'][0]['commands'][0][0]=.25
    elif kind=='task':shared['layers'][0]['task_id']='wrong'
    elif kind=='source_slot':shared['layers'][0]['source_slot']=1
    elif kind=='arm':shared['layers'][0]['arm']='reselected'
    elif kind=='history':cp['plans'][1]['model_task']['root_context']['action_history'][0][0]+=.25
    else:cp['plans'][1]['native_goal'][0]+=.25
    with pytest.raises((ValueError,KeyError)):m.assemble(cp,shared,cr)


@pytest.mark.parametrize('kind',['negativezero','modifiedP','rawhash','order','modeldelivery','sourcepointer'])
def test_resealed_but_illegal_plan_rejected(kind):
    p,*_=fixture();i=0
    if kind=='negativezero':p['plans'][0]['control_raw5'][0][0]=-0.
    elif kind=='modifiedP':i=1;p['plans'][1]['control_raw5'][0][0]=.5
    elif kind=='rawhash':p['plans'][0]['raw_float32_sha256']='e'*64
    elif kind=='order':p['plans'][0]['arm']='P'
    elif kind=='modeldelivery':p['plans'][0]['model_delivery']={}
    else:p['plans'][0]['control_source']['pointer']='/layers/1/commands'
    reseal(p,i)
    with pytest.raises((ValueError,KeyError)):m.validate_execution(p)


def test_real_full4_store_consume_census(monkeypatch):
    r,s,p,data,envs,events=complete();c=m.shape_census()
    assert r['status']==m.RECORDED,r['errors']
    assert len(m.validate_saved(r,events,p))==4 and len(envs)==4
    assert all(e.closed and len(e.commands)==20 for e in envs)
    assert r['counts']=={k:dict(attempted=v,completed=v) for k,v in m.contract()['caps'].items()}
    for i,e in enumerate(envs):
        assert np.asarray(e.commands[15:],dtype='<f4').tobytes()==np.asarray(p['plans'][i]['control_raw5'],dtype='<f4').tobytes()
    files=list(s.path.iterdir());png=[f for f in files if f.suffix=='.png']
    meta=sum(f.stat().st_size for f in files if f.suffix!='.png')
    assert len(png)==264 and max(f.stat().st_size for f in png)<=c['PNG_file_upper_bytes']
    assert meta<=c['metadata_upper_bytes'] and c['total_upper_bytes']<=m.CAP
    assert len(r['actual_operation_receipts'])==1216
    assert sum(e['kind'] in ('operation_dispatch_intent','operation_returned','operation_failed') for e in events)==2432
    assert sum(e['kind']=='fresh_observation' for e in events)==264
    assert sum(e['kind']=='branch_closed' for e in events)==4
    assert [c[k] for k in ('operation_receipt_persisted_copies','row_persisted_copies','prefix_persisted_copies')]==[3,2,2]
    monkeypatch.setattr(m,'verify',lambda *args:(deepcopy(p),deepcopy(data)))
    receipt=m.consume(s.path,m.sha(s.path/'report.json'),m.sha(s.path/'completion.json'))
    assert receipt['saved_PNG_verified']==264 and len(receipt['rows'])==4
    print('RAM_ZP_CENSUS',json.dumps(dict(PNG_count=len(png),PNG_bytes=sum(f.stat().st_size for f in png),
        metadata_bytes=meta,total_bytes=sum(f.stat().st_size for f in files),operation_receipts=1216,
        operation_events=2432,observation_events=264,journal_bytes=(s.path/'journal.jsonl').stat().st_size,
        report_bytes=(s.path/'report.json').stat().st_size,completion_bytes=(s.path/'completion.json').stat().st_size),sort_keys=True))


def test_real_store_two_events_canonical_reader_hashchain(tmp_path):
    s=m.Store(tmp_path/(m.NAME+'_two'));s.emit('RAM_one',value=1);s.emit('RAM_two',value=2)
    lines=(s.path/'journal.jsonl').read_bytes().splitlines(keepends=True)
    es=[m.native.decode(x) for x in lines]
    assert len(es)==2 and all(line==m.raw(e) for line,e in zip(lines,es,strict=True))
    assert es[1]['previous_sha256']==es[0]['event_sha256'] and es[1]['event_sha256']==s.previous
    with pytest.raises(ValueError):s.write('journal.jsonl',m.raw(es[1])+b'\n',append=True)


@pytest.mark.parametrize('kind',['state','snapshot','pixels','png','warmraw','history_frame','tokens'])
def test_exact_guard_rejects_before_control6(kind):
    p,data,*_=fixture();ref=p['prefix_reference'][0]
    if kind=='state':ref['checkpoint']['state'][0]+=.25
    elif kind=='snapshot':ref['checkpoint']['snapshot']['state_float64_bits']='wrong'
    elif kind=='pixels':ref['checkpoint']['pixel_sha256']=old.h('wrong')
    elif kind=='png':ref['frames'][-3]['receipt']['file_sha256']=old.h('wrong')
    elif kind=='warmraw':ref['warm_raw10'][0][0]=.25
    elif kind=='history_frame':ref['history_frames'][0]['pixel_sha256']=old.h('wrong')
    else:ref['history_tokens'][0][0]+=.25
    reseal(p,0);r,s,_,_,envs,events=execute(p,data)
    assert r['status']=='partial_or_failed' and len(envs)==1 and len(envs[0].commands)==15
    assert len(m.validate_saved(r,events,p))==4
    assert [x['status'] for x in r['rows']][1:]==['not_started']*3
    assert not any(e['kind']=='operation_dispatch_intent' and e['receipt']['resource']=='public_step'
        and e['receipt'].get('phase')=='plan' and e['receipt'].get('step',0)>=6 for e in events)


def test_guard_durable_before_each_commit():
    r,s,p,_,_,events=complete()
    for i in range(4):
        guard=next(e for e in events if e['kind']=='cp5_guard' and e['branch']==i)
        commit=next(e for e in events if e['kind']=='operation_dispatch_intent' and e['receipt']['branch']==i
            and e['receipt']['resource']=='public_step' and e['receipt'].get('phase')=='plan' and e['receipt'].get('step')==6)
        assert guard['accepted'] is True and guard['index']<commit['index']
    meter=m.Meter(type('Broken',(),{'emit':lambda *a,**kw:(_ for _ in ()).throw(OSError('fsync'))})())
    meter.where=dict(branch=0);meter.guard_context(r['rows'][0],p['prefix_reference'][0],p['plans'][0])
    meter.receipts=[x for x in r['actual_operation_receipts'] if x['branch']==0 and x['resource']=='public_step'
        and (x['phase']=='warmup' or x['step']<=5)]
    called=[]
    with pytest.raises(OSError):meter.call('public_step',lambda:called.append(True),phase='plan',step=6)
    assert not called and not meter.guarded


@pytest.mark.parametrize('at',[5,14,18])
def test_earlystop_all4_missing_no_padding(at):
    r,s,p,_,envs,events=execute(fault='truncate',at=at)
    assert r['status']==m.RECORDED,r['errors']
    assert len(envs)==4 and all(len(e.commands)==at for e in envs)
    assert all(x['status']=='native_stopped' and 10 in x['missing_checkpoints'] for x in r['rows'])
    assert all(x['checkpoint'] is None and x['available'] is False for x in r['endpoints'])
    assert len(m.validate_saved(r,events,p))==4


@pytest.mark.parametrize('kind',['unknown','guard_missing','guard_late','guard_proof','action','label','counter','missing_frame'])
def test_saved_mutations_rejected(kind):
    r,_,p,_,_,events=deepcopy(complete())
    if kind=='unknown':events[1]['kind']='claim'
    elif kind=='guard_missing':events[:]=[e for e in events if e['kind']!='cp5_guard']
    elif kind=='guard_late':
        i=next(i for i,e in enumerate(events) if e['kind']=='cp5_guard');e=events.pop(i);events.insert(i+2,e)
    elif kind=='guard_proof':next(e for e in events if e['kind']=='cp5_guard')['proof']['actual_raw10_float32_sha256']=old.h('wrong')
    elif kind=='action':next(e for e in events if e['kind']=='returned_transition')['actual_bits']='00'
    elif kind=='label':next(e for e in events if e['kind']=='returned_transition')['reward']+=1
    elif kind=='counter':r['counts']['physics_step']['completed']-=1
    else:events.pop(next(i for i,e in enumerate(events) if e['kind']=='fresh_observation'))
    old.reseal(events,r)
    with pytest.raises((ValueError,KeyError)):m.validate_saved(r,events,p)


@pytest.mark.parametrize('kind',['external_sha','wall','files','prepare','flags','failure','PNG','orphan'])
def test_completion_and_png_mutations(tmp_path,monkeypatch,kind):
    r,s,p,data,_,_=complete();out=tmp_path/(m.NAME+'_mutated');out.mkdir()
    for f in s.path.iterdir():(out/f.name).write_bytes(f.read_bytes())
    co=json.loads((out/'completion.json').read_bytes());expected=m.sha(out/'report.json')
    if kind=='wall':co['inclusive_wall_seconds']=901
    elif kind=='files':co['artifact_sha256'].pop(next(k for k in co['artifact_sha256'] if k.endswith('.png')))
    elif kind=='prepare':co['prepare_sha256']=old.h('wrong')
    elif kind=='flags':co['P1_admission']=True
    elif kind=='failure':(out/'completion_failure.json').write_bytes(m.raw(dict(failed=True)))
    elif kind=='PNG':next(out.glob('*.png')).write_bytes(b'not PNG')
    elif kind=='orphan':(out/'orphan.json').write_bytes(b'{}\n')
    (out/'completion.json').write_bytes(m.raw(co));h=m.sha(out/'completion.json')
    monkeypatch.setattr(m,'verify',lambda *args:(p,data))
    with pytest.raises((ValueError,KeyError)):m.consume(out,expected,old.h('wrong') if kind=='external_sha' else h)


def test_encoded_noise_hard_quotas(tmp_path):
    s=m.Store(tmp_path/(m.NAME+'_cap'));c=m.shape_census()
    pixels=np.random.default_rng(7).integers(0,256,(224,224,3),dtype=np.uint8)
    b=io.BytesIO();Image.fromarray(pixels).save(b,format='PNG');assert len(b.getvalue())<=c['PNG_file_upper_bytes']
    s.write('noise.png',b.getvalue())
    with pytest.raises(ValueError):s.write('large.png',b'x'*(c['PNG_file_upper_bytes']+1))
    with pytest.raises(ValueError):s.put('execution_binding.json',dict(x='x'*m.METADATA_CAP))


def test_all264_encoded_noise_plus_aggregate_metadata_budget(tmp_path):
    s=m.Store(tmp_path/(m.NAME+'_worstsize'));c=m.shape_census()
    pixels=np.random.default_rng(19).integers(0,256,(224,224,3),dtype=np.uint8)
    b=io.BytesIO();Image.fromarray(pixels).save(b,format='PNG');body=b.getvalue()
    for i in range(264):s.write('noise_'+str(i)+'.png',body)
    assert sum(f.stat().st_size for f in s.path.iterdir())<=c['PNG_total_upper_bytes']
    value=dict(aggregate_metadata_quota_test='')
    value['aggregate_metadata_quota_test']='x'*(m.METADATA_CAP-len(m.raw(value)))
    assert len(m.raw(value))==m.METADATA_CAP
    s.put('execution_binding.json',value)
    total=sum(f.stat().st_size for f in s.path.iterdir())
    assert total<=c['working_upper_bytes']<=m.CAP-m.FINAL
    with pytest.raises(ValueError):s.write('one_extra.png',body)
    with pytest.raises(ValueError):s.emit('cannot_exceed_metadata_quota')


def test_publication_inclusive_wall_and_failure(tmp_path):
    report=dict(execution_binding=dict(prepared='/RAM/prepare',prepare_sha256=old.h('prepare')),wall_seconds=1)
    s=m.Store(tmp_path/(m.NAME+'_pub'));s.emit('RAM',value=1)
    def clock():
        assert (s.path/'report.json').exists() and m.sha(s.path/'report.json')
        return 2
    v=m.publish(s,report,0,clock=clock)
    assert m.read(s.path/'completion.json')['report_sha256']==v['report_sha256']
    failed=m.Store(tmp_path/(m.NAME+'_fail'));failed.emit('RAM',value=1);times=iter([2,901])
    with pytest.raises(ValueError):m.publish(failed,report,0,clock=lambda:next(times))
    assert (failed.path/'completion_failure.json').exists()


def saved_C_values():
    d=m.authority_descriptors()
    return [m.read(d[k]['path'],m.META) for k in ('C_prepare','C_report','C_completion','C_receipt')]


def test_actual_saved_binding_assemble_only_no_fullstrict():
    cp,r,co,u=saved_C_values();cr=m.validate_C(cp,r,co,u)
    shared=m.read(m.authority_descriptors()['P_prepare']['path'],m.META)
    p=m.assemble(cp,shared,cr)
    assert len(p['plans'])==4 and len(cr['logical_C_slots'])==4
    assert [x['endpoint'] is None for x in cr['logical_C_slots']]==[False,True,True,True]
    assert all(x['control_role'] in ('Z','P') for x in p['plans'])
    for root,layer in enumerate(shared['layers']):
        m.validate_task_export(layer,cp['plans'][2*root]['base_task'])


def test_cp10_schema_and_missing_tamper():
    r,_,p,_,_,events=deepcopy(complete())
    assert [(x['root_slot'],x['role'],x['absolute_step'],x['available']) for x in r['endpoints']]==[
        (0,'Z',20,True),(0,'P',20,True),(1,'Z',20,True),(1,'P',20,True)]
    r['endpoints'][0]['checkpoint']=deepcopy(r['rows'][0]['checkpoints'][1])
    with pytest.raises(ValueError):m.validate_saved(r,events,p)


@pytest.mark.parametrize('field',['goal','projection','action_mean','action_std','root_context'])
def test_loaded_task_body_cannot_drift(monkeypatch,field):
    cp,_,_,_=saved_C_values()
    layer=m.read(m.authority_descriptors()['P_prepare']['path'],m.META)['layers'][0]
    interface=m.task_loader.load_task(layer['task_export'],layer['task_sha256'])
    interface['gn_task'][field]=None
    monkeypatch.setattr(m.task_loader,'load_task',lambda *args:interface)
    with pytest.raises(ValueError):m.validate_task_export(layer,cp['plans'][0]['base_task'])


@pytest.mark.parametrize('kind',['rows','count','status','completion','prepare','missing','permission','binding'])
def test_C_pinned_receipt_fields_rejected(kind):
    cp,r,co,u=saved_C_values()
    if kind=='rows':u['rows'][0]['checkpoints'][-1]['state'][0]+=.1
    elif kind=='count':u['counts']['public_step']['completed']-=1
    elif kind=='status':u['status']='partial'
    elif kind=='completion':u['completion_sha256']='a'*64
    elif kind=='prepare':co['prepare_sha256']='a'*64
    elif kind=='missing':u['rows'].pop()
    elif kind=='permission':u['P3_admission']=True
    else:u['bindings'][m.authority_descriptors()['C_report']['path']]='a'*64
    with pytest.raises((ValueError,KeyError)):m.validate_C(cp,r,co,u)


def test_no_recursive_fullstrict_cold_import_and_authorization():
    import inspect
    assert '.consume(' not in inspect.getsource(m.material)
    code=f'from experiments.spectral_reachability import {m.NAME}; import sys; assert "torch" not in sys.modules and "pymunk" not in sys.modules'
    r=subprocess.run([sys.executable,'-c',code],cwd=m.ROOT,capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    p,data,*_=fixture()
    with pytest.raises(ValueError):m.execute(p,data,None,lambda:None,authorization='wrong',prepared='/RAM/p',prepare_sha256=old.h('p'))


def test_cli_compact_run(monkeypatch,capsys):
    r,s,*_=complete()
    monkeypatch.setattr(m,'run',lambda *args,**kwargs:deepcopy(r))
    assert m.main(['run','--prepared','/RAM/p','--prepare-sha','a'*64,'--output',str(s.path),'--authorization',m.AUTH])==0
    line=capsys.readouterr().out
    result=json.loads(line)
    assert len(line)<2048 and 'rows' not in result and result['report_sha256']==m.sha(s.path/'report.json')

