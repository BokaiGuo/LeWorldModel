"""Saved provider and encoder mocks in RAM only, no files/native/neural."""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import p1_moving_extended_baseline_rgb_cuda_20261027_v1 as m
from test_p1_moving_baseline_rgb_cuda_20261027_v1 import fixture as base_fixture, GRAD, ALL
from test_p1_constructive_selected_rgb_cuda_20261027_v1 import RAMStore

GRASP='GRASP_H3_bounded'
SRA='joint_H2_spectral_packet_mapped'


def fixture(four=False):
    oldarms=ALL if four else [GRAD,'CEM']
    r,events,p,manifest,tasks,selected=base_fixture(oldarms)
    mapping=dict(zip(oldarms,[GRASP,SRA,GRAD,'CEM'] if four else [GRASP,SRA]))
    def rename(v):
        if isinstance(v,str):return mapping.get(v,v)
        if isinstance(v,list):return [rename(x) for x in v]
        if isinstance(v,dict):return {k:rename(x) for k,x in v.items()}
        return v
    r,events,p,manifest=map(rename,(r,events,p,manifest));arms=[mapping[a] for a in oldarms]
    selected={(s,mapping[a]):dict(deepcopy(row),arm=mapping[a]) for (s,a),row in selected.items()}
    manifest['schema']=m.physical.NAME+'.selection'
    manifest['authorities']=[dict(authority=k,output='/RAM/'+k,report_sha256='c'*64)
        for k,(_,_,names,_) in m.physical.AUTHORITIES.items() if any(a in names for a in arms)]
    plans=[]
    for s in (0,1):
        for arm in arms:
            entry=next(x for x in manifest['authorities'] if arm in m.physical.AUTHORITIES[x['authority']][2])
            plans.append(m.plan_from_consumed(selected[s,arm],entry,manifest['roots']))
    p.update(schema=m.physical.NAME+'.prepare',plans=plans,contract=m.physical.contract(arms,600),source_prepare_id='e'*64)
    r.update(schema=m.physical.NAME+'.report',status='moving_extended_baseline_physical_recorded',contract=p['contract'],source_prepare_id=p['source_prepare_id'])
    r['counts'].update({k:dict(attempted=len(plans),completed=len(plans)) for k in ('environment_construct','environment_close')})
    for b,(plan,row) in enumerate(zip(plans,r['rows'],strict=True)):
        row.update(branch_index=b,selected_plan=deepcopy(plan),selection_digest=plan['selection_digest'])
        for e in events:
            if e.get('branch')==b:e['selection_digest']=plan['selection_digest']
            if e['kind']=='branch_started' and e.get('branch')==b:e['plan']=deepcopy(plan)
            if e['kind']=='branch_closed' and e['receipt']['root_slot']==row['root_slot'] and e['receipt']['arm']==row['arm']:e['receipt']=deepcopy(row)
    out=[]
    for e in events:
        if e['kind']=='run_started':e['prepare']=deepcopy(p)
        if e['kind']=='run_finished':e.update(status=r['status'],counts=deepcopy(r['counts']))
        if e['kind']=='branch_started':
            scope={k:e[k] for k in ('branch','root_slot','arm','selection_digest')}
            out.extend([dict(kind=k,resource='environment_construct',**scope) for k in ('operation_attempted','operation_completed')])
        if e['kind']=='branch_closed':
            row=e['receipt'];scope=dict(branch=row['branch_index'],root_slot=row['root_slot'],arm=row['arm'],selection_digest=row['selection_digest'])
            out.extend([dict(kind=k,resource='environment_close',**scope) for k in ('operation_attempted','operation_completed')])
        out.append(e)
    previous=None
    for i,e in enumerate(out):
        e.update(index=i,previous_sha256=previous);e.pop('event_sha256',None);e['event_sha256']=m.physical.source.digest(e);previous=e['event_sha256']
    r['journal_chain_tail']=previous
    return r,out,p,manifest,tasks,selected


def test_actual_new_provider_mapping_full_arms_lifecycle_and_nativeQ():
    args=fixture();before=deepcopy(args);arms=args[3]['arms'];images=m.assemble(*args,arms)
    assert args==before and len(images)==4*len(arms)
    assert [(x['root_slot'],x['arm'],x['role']) for x in images]==[(s,a,k) for s in (0,1) for a in arms for k in ('prefix5','terminal10')]
    assert m.contract(arms)['encoder']==len(images) and m.contract(arms)['physics']==0


@pytest.mark.parametrize('bad',['schema','sourceprepare','lifecycle','selectedplan','rootsha','missing5','missing10','command','ownsha','guard','history','goal','subset','boolroot','partial'])
def test_provider_strictguards_no_dropped_arm_or_fabricated_endpoint(bad):
    r,e,p,manifest,tasks,selected=fixture();arms=manifest['arms']
    if bad=='schema':r['schema']='old.report'
    elif bad=='sourceprepare':r['source_prepare_id']='0'*64
    elif bad=='lifecycle':r['counts']['environment_close']['completed']-=1
    elif bad=='selectedplan':r['rows'][0]['selected_plan']['task_id']='wrong'
    elif bad=='rootsha':p['plans'][0]['root_sha256']='0'*64
    elif bad.startswith('missing'):r['rows'][0]['checkpoints']=[c for c in r['rows'][0]['checkpoints'] if c['step']!=int(bad[7:])]
    elif bad=='command':next(x for x in e if x['kind']=='returned_transition' and x['phase']=='plan')['command']=[0.,0.]
    elif bad=='ownsha':p['plans'][0]['own_plan_sha256']='0'*64
    elif bad=='guard':next(x for x in e if x['kind']=='guard_receipt')['accepted']=False
    elif bad=='history':tasks[0]['task_contract']['pixel_sha256'][1]='0'*64
    elif bad=='goal':tasks[0]['interface']['gn_task']['goal_source_sha256']='0'*64
    elif bad=='subset':arms=arms[:1]
    elif bad=='boolroot':manifest['roots'][0]['root_slot']=False
    else:r['rows'][0]['full_horizon']=False
    with pytest.raises(ValueError):m.assemble(r,e,p,manifest,tasks,selected,arms)


def test_frozen_physical_pin_and_unknown_scheduler_never_reach_provider_reads():
    assert m.PHYSICAL_SOURCE_SHA=='a0f7348ebe8c6e061b5d63a1ba4a8ecb064a26e7f7a1ef4be57fe8b07270d799'
    with patch.object(m,'read',side_effect=AssertionError('no fabricated actual report')):
        with pytest.raises(ValueError):m.build('/RAM/no_report','a'*64,'b'*64,'c'*64,[GRASP,SRA])
        with patch.object(m,'PHYSICAL_SOURCE_SHA',None),pytest.raises(ValueError):
            m.build('/RAM/no_report','a'*64,'b'*64,'c'*64,[GRASP,SRA])
    for arms in ([],['unknown_scheduler'],[GRASP,GRASP],[GRASP,SRA,GRAD,'CEM']):
        with pytest.raises(ValueError):m.checked_arms(arms)


def test_build_complete_ram_provider_adapter_and_external_binding_rejection(monkeypatch):
    r,e,p,manifest,tasks,selected=fixture();h='a'*64;provider_sha='f'*64
    ph='/RAM/physical/report.json';r.update(prepared='/RAM/prepare.json',prepare_sha256=h)
    saved={ph:r,'/RAM/physical/execution_binding.json':dict(prepared=r['prepared'],prepare_sha256=h),
        r['prepared']:p,p['manifest']:manifest}
    for s,path in enumerate(m.original.EXPORTS):
        saved[path]=dict(source_slot=s,blind_sha256=tasks[s]['blind_sha256'],
            blind_source_sha256=tasks[s]['blind_sha256'],task_contract=tasks[s]['task_contract'])
    pins={str(m.physical.__file__):provider_sha,str(m.moving.__file__):m.original.MOVING_SHA,
        str(m.rgb.__file__):m.original.RGB_SHA,str(m.rgb.old.__file__):m.rgb.OLD_SHA,
        **dict(zip(m.original.EXPORTS,m.original.EXPORT_SHAS)),**m.enc.MODELS,
        '/RAM/physical/journal.jsonl':r['journal_sha256']}
    pins.update({str(Path('/RAM/physical')/x['receipt']['png']):x['receipt']['file_sha256']
        for x in e if x['kind']=='fresh_observation'})
    calls=[]
    def read(path,expected=None):
        calls.append(('read',str(path)))
        assert 'sourcepack' not in str(path) and 'certificate' not in str(path)
        return deepcopy(saved[str(path)])
    def consume(entry):
        calls.append(('consume',entry['authority']))
        return deepcopy([row for row in selected.values()
            if row['arm'] in m.physical.AUTHORITIES[entry['authority']][2]]),{}
    def load(path,expected):
        calls.append(('load_task',path,expected))
        return deepcopy(tasks[m.original.EXPORTS.index(path)]['interface'])
    monkeypatch.setattr(m,'PHYSICAL_SOURCE_SHA',provider_sha)
    monkeypatch.setattr(m,'sha',lambda path:pins.get(str(path),h));monkeypatch.setattr(m,'read',read)
    monkeypatch.setattr(m.physical,'ownconsume',consume)
    monkeypatch.setattr(m.physical.kernel,'journal',lambda *a:deepcopy(e))
    monkeypatch.setattr(m.moving,'load_task',load);monkeypatch.setattr(m.moving,'source_files',lambda:[])
    monkeypatch.setattr(m.rgb.old,'png',lambda *a:np.zeros((224,224,3),np.uint8))
    monkeypatch.setattr(m.physical.kernel,'load_source',lambda *a:pytest.fail('no source/certificate decode'))
    args=dict(physical_report=ph,physical_sha=h,physical_prepare_sha=h,physical_source_sha=provider_sha,arms=manifest['arms'])
    result=m.build(**args)
    assert result['schema']==m.NAME+'.prepare' and len(result['images'])==8
    assert result['contract']['encoder']==result['contract']['projector']==8
    assert [c[1] for c in calls if c[0]=='consume']==['grasp','no_composition']
    assert len([c for c in calls if c[0]=='load_task'])==2
    r['prepare_sha256']='0'*64
    with pytest.raises(ValueError,match='execution binding'):m.build(**args)


class RawRAMStore(RAMStore):
    def write(self,name,value,append=False,final=False):
        assert not append and name not in self.files;self.files[name]=bytes(value)


def encoding_fixture(four=False):
    data=fixture(four);images=m.assemble(*data,data[3]['arms']);arms=data[3]['arms']
    rgb=np.zeros((224,224,3),np.uint8);rgb[0,0]=[65,105,225]
    from PIL import Image
    b=BytesIO();Image.fromarray(rgb).save(b,format='PNG');payload=b.getvalue();h=m.enc.digest(payload)
    for image in images:
        image['png_path']='/RAM/original.png';image['checkpoint']['file_sha256']=h;image['checkpoint']['pixel_sha256']=m.enc.digest(rgb.tobytes())
    p=dict(images=images,contract=m.contract(arms));rows=[dict(status='not_started') for _ in images]
    store=RawRAMStore();ledger=m.enc.Ledger(len(arms),store.event)
    return p,rows,store,ledger,rgb,payload,h


def test_lossless_raw_png_native_npy_metrics_actual_module_census():
    p,rows,store,ledger,rgb,payload,h=encoding_fixture()
    def encode(a):
        assert a.dtype==np.uint8 and a.tobytes()==rgb.tobytes()
        ledger.dispatch('encoder',NS(device=NS(type='cuda'),shape=(1,3,224,224)),lambda:None)
        return ledger.dispatch('projector',NS(device=NS(type='cuda'),shape=(1,192)),lambda:np.ones(192,np.float32))
    with patch.object(m.rgb.old,'png',return_value=rgb),patch.object(Path,'read_bytes',return_value=payload),\
        patch.object(Path,'stat',return_value=NS(st_size=len(payload))),patch.object(m,'sha',return_value=h):
        m.encode_images(p,encode,ledger,store,rows)
    n=len(rows);assert ledger.counts()=={k:dict(attempted=n,completed=n) for k in ('encoder','projector')}
    assert all(row['status']=='encoded' and row['rawpixel']['sha256']==h for row in rows)
    for row in rows:
        assert store.files[row['rawpixel']['file']]==payload
        z=np.load(BytesIO(store.files[int(row['latent']['file'])]),allow_pickle=False)
        assert z.dtype==np.float32 and z.shape==(192,) and np.array_equal(z,np.ones(192,np.float32))
        assert row['metrics']['actual_Q']==4. and row['metrics']['root_Q']==1.
        assert row['metrics']['rawpixel_vs_prefix5_mse']==0. and row['metrics']['predicted_Q'] is None


def test_module_error_retains_real_raw_frame_before_neural_and_full_denominator():
    p,rows,store,ledger,rgb,payload,h=encoding_fixture()
    def fail(a):
        assert 'image_00.rgb.png' in store.files
        ledger.dispatch('encoder',NS(device=NS(type='cuda'),shape=(1,3,224,224)),lambda:None)
        raise RuntimeError('between modules')
    with patch.object(m.rgb.old,'png',return_value=rgb),patch.object(Path,'read_bytes',return_value=payload),\
        patch.object(Path,'stat',return_value=NS(st_size=len(payload))),patch.object(m,'sha',return_value=h),pytest.raises(RuntimeError):
        m.encode_images(p,fail,ledger,store,rows)
    assert len(rows)==8 and rows[0]['status']=='attempted' and all(row['status']=='not_started' for row in rows[1:])
    assert store.files['image_00.rgb.png']==payload and len(store.files)==1
    assert ledger.counts()['encoder']==dict(attempted=1,completed=1)


def test_complete_native_metrics_scale_and_bad_array_guards():
    item=dict(root=[0.]*192,goal=[2.]*192)
    d=m.compare(item,np.ones(192,np.float32));assert d['actual_Q']==1 and d['root_Q']==4 and d['actual_Q_change']==-3 and d['actual_vs_root_mse']==1
    for bad in (np.ones(192,np.float64),np.ones(191,np.float32),np.full(192,np.nan,np.float32)):
        with pytest.raises(ValueError):m.compare(item,bad)


def test_maximum_budget_under_explicit_ram_provider_mock_not_authority_expansion():
    with patch.object(m.physical,'checked_arms',side_effect=lambda a:list(a)):
        assert m.contract(['mock1','mock2','mock3','mock4'])['encoder']==16
        with pytest.raises(ValueError):m.contract(['mock1','mock2','mock3','mock4','mock5'])
    with pytest.raises(ValueError):m.checked_arms(['mock1','mock2','mock3','mock4'])


def test_cap_floor_and_no_native_neural_imports():
    with patch.object(m.shutil,'disk_usage',return_value=NS(free=m.FLOOR+m.CAP-1)),pytest.raises(ValueError):m.Store('/RAM/no_create')
    store=object.__new__(m.Store);store.output=Path('/RAM')
    with patch.object(Path,'iterdir',return_value=[]),patch.object(m.shutil,'disk_usage',return_value=NS(free=m.FLOOR-1)),pytest.raises(ValueError):store.write('no_write',b'1')
    assert not any(x in sys.modules for x in ('torch','gymnasium','pymunk','stable_worldmodel'))
