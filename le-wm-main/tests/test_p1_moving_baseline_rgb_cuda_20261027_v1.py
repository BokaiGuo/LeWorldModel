"""RAM-only moving baseline RGB contracts; never neural/native/tmp output."""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import p1_moving_baseline_rgb_cuda_20261027_v1 as m
from test_p1_moving_baseline_physical_20261027_v1 import manifest_fixture, select, execute, ALL, GRAD
from test_p1_constructive_moving_history_encode_cuda_20261027_v1 import protocol
from test_p1_constructive_selected_rgb_cuda_20261027_v1 import RAMStore

TRIPLE=[GRAD,'CEM','kernel_faithful_mppi_nominal']

def fixture(arms=None):
    arms=[GRAD] if arms is None else arms
    manifest,data,rows=manifest_fixture(arms);tasks=[]
    for entry in manifest['authorities']:
        if entry['authority']=='gradient':entry['report_sha256']=m.GRADIENT_REPORT_SHA
        if entry['authority']=='sampling':entry['report_sha256']=m.SAMPLING_REPORT_SHA
    for slot,ref in enumerate(data['references']):
        pr=protocol();record=pr['slots'][slot]
        record.update(blind_sha256=ref['blind_sha256'],pixel_sha256=[m.moving.digest(a.tobytes()) for a in [*ref['history'],ref['goal_rgb']]])
        packet=dict(rgb_history=ref['history'],raw_action_history=ref['raw_history'],goal_rgb=ref['goal_rgb'])
        export=m.moving.task_export(pr,'d'*64,record,packet,[np.full(192,z,np.float32) for z in range(4)])
        interface=m.moving.task_interfaces(export)
        tasks.append(dict(task_export=m.original.EXPORTS[slot],task_sha256=m.original.EXPORT_SHAS[slot],
            blind_sha256=ref['blind_sha256'],interface=interface,task_contract=export['task_contract']))
        for values in rows.values():
            for r in values:
                if r['source_slot']==slot:r.update(task_export=tasks[-1]['task_export'],task_sha256=tasks[-1]['task_sha256'],task_id=interface['task_id'])
    plans,_=select(manifest,data,rows)
    p=dict(schema=m.physical.NAME+'.prepare',manifest='/RAM/manifest.json',manifest_sha256='a'*64,plans=plans,
        contract=m.physical.contract(arms,600),bindings={},sourcepack_sha256=m.physical.SOURCE_SHA)
    r,s,_=execute(arms=arms,p=p,data=data)
    assert r['status']=='moving_baseline_physical_recorded'
    selected={(r['source_slot'],r['arm']):r for values in rows.values() for r in values if r['arm'] in arms}
    return r,s.events,p,manifest,tasks,selected


@pytest.mark.parametrize('arms',[[GRAD],TRIPLE,ALL])
def test_gradient4_or_all16_fixed_images_same_moving_task(arms):
    args=fixture(arms);before=deepcopy(args);images=m.assemble(*args,arms)
    assert args==before and len(images)==4*len(arms)
    assert [(i['root_slot'],i['arm'],i['role']) for i in images]==[(s,a,k) for s in (0,1) for a in arms for k in ('prefix5','terminal10')]
    assert all(i['predicted'] is None and i['Q']=='identity192' for i in images)
    assert m.contract(arms)['encoder']==m.contract(arms)['projector']==len(images)
    assert all(m.contract(arms)[k]==0 for k in ('predict','AD','physics','new_independent_samples'))


@pytest.mark.parametrize('bad',['schema','missing5','missing10','boolslot','denominator','command','selected','ownsha','task','goal','history','warmraw','counts','guard','close','tail','partial','duplicate','arm','rootbool'])
def test_saved_interface_rejects_any_missing_or_wrong_binding(bad):
    r,e,p,manifest,tasks,selected=fixture();row=r['rows'][0]
    if bad=='schema':r['schema']=m.original.physical.NAME+'.report'
    elif bad.startswith('missing'):row['checkpoints']=[c for c in row['checkpoints'] if c['step']!=int(bad[7:])]
    elif bad=='boolslot':row['root_slot']=False
    elif bad=='denominator':r['fixed_branch_denominator']=8
    elif bad=='command':next(x for x in e if x['kind']=='returned_transition' and x['phase']=='plan')['command']=[0.,0.]
    elif bad=='selected':selected[0,GRAD]['commands'][0][0]=.5
    elif bad=='ownsha':p['plans'][0]['own_plan_sha256']='0'*64
    elif bad=='task':tasks[0]['task_sha256']='0'*64
    elif bad=='goal':tasks[0]['interface']['gn_task']['goal_source_sha256']='f'*64
    elif bad=='history':tasks[0]['task_contract']['pixel_sha256'][1]='0'*64
    elif bad=='warmraw':next(x for x in e if x['kind']=='warmup_return')['raw_float32_bits']='bad'
    elif bad=='counts':r['counts']['physics_step']['completed']-=1
    elif bad=='guard':next(x for x in e if x['kind']=='guard_receipt')['accepted']=False
    elif bad=='close':e[:]=[x for x in e if x['kind']!='branch_closed']
    elif bad=='tail':r['journal_chain_tail']='f'*64
    elif bad=='partial':row['full_horizon']=False
    elif bad=='duplicate':row['checkpoints'].append(deepcopy(row['checkpoints'][1]))
    elif bad=='rootbool':manifest['roots'][0]['root_slot']=False
    else:manifest['arms']=['CEM']
    with pytest.raises(ValueError):m.assemble(r,e,p,manifest,tasks,selected,[GRAD])


def test_plan_adapter_is_exact_physical_ownconsume_projection():
    _,_,p,manifest,_,selected=fixture(ALL)
    for plan in p['plans']:
        entry=next(x for x in manifest['authorities'] if x['authority']==plan['compiler_authority'])
        assert m.plan_from_consumed(selected[plan['root_slot'],plan['arm']],entry)==plan


def build_fixture(monkeypatch,arms=None):
    arms=[GRAD] if arms is None else arms
    r,e,p,manifest,tasks,selected=fixture(arms);calls=[];ph='/RAM/physical/report.json';h='a'*64;physical_source='f'*64
    saved={ph:r,'/RAM/physical/execution_binding.json':dict(prepared='/RAM/prepare.json',prepare_sha256=h),
        '/RAM/prepare.json':p,'/RAM/manifest.json':manifest}
    for s,path in enumerate(m.original.EXPORTS):saved[path]=dict(source_slot=s,blind_sha256=tasks[s]['blind_sha256'],
        blind_source_sha256=tasks[s]['blind_sha256'],task_contract=tasks[s]['task_contract'])
    pins={str(m.physical.__file__):physical_source,str(m.moving.__file__):m.original.MOVING_SHA,
        str(m.rgb.__file__):m.original.RGB_SHA,str(m.rgb.old.__file__):m.rgb.OLD_SHA,
        **dict(zip(m.original.EXPORTS,m.original.EXPORT_SHAS)),**m.enc.MODELS,
        '/RAM/physical/journal.jsonl':r['journal_sha256']}
    pins.update({str(Path('/RAM/physical')/x['receipt']['png']):x['receipt']['file_sha256'] for x in e if x['kind']=='fresh_observation'})
    def read(path,expected=None):
        calls.append(('read',str(path)));assert 'sourcepack' not in str(path) and 'certificate' not in str(path)
        return deepcopy(saved[str(path)])
    monkeypatch.setattr(m,'sha',lambda path:pins.get(str(path),h));monkeypatch.setattr(m,'read',read)
    monkeypatch.setattr(m.physical,'ownconsume',lambda entry:(deepcopy([r for r in selected.values() if r['arm'] in m.physical.AUTHORITIES[entry['authority']][2]]),{}))
    monkeypatch.setattr(m.physical.kernel,'journal',lambda *a:deepcopy(e))
    def load(path,expected):
        calls.append(('moving.load_task',path,expected));return deepcopy(tasks[m.original.EXPORTS.index(path)]['interface'])
    monkeypatch.setattr(m.moving,'load_task',load);monkeypatch.setattr(m.moving,'source_files',lambda:[])
    monkeypatch.setattr(m.rgb.old,'png',lambda *a:np.zeros((224,224,3),np.uint8))
    monkeypatch.setattr(m.rgb.planner,'load_export',lambda *a:pytest.fail('old zero-only loader'))
    monkeypatch.setattr(m.physical,'verify',lambda *a:pytest.fail('no physical evaluator reconstruction'))
    monkeypatch.setattr(m.physical.kernel,'load_source',lambda *a:pytest.fail('no sourcepack parse'))
    args=dict(physical_report=ph,physical_sha=h,physical_prepare_sha=h,physical_source_sha=physical_source,arms=arms)
    return args,saved,p,calls


def test_build_gradient_only_uses_actual_authority_and_moving_loader(monkeypatch):
    args,_,_,calls=build_fixture(monkeypatch);p=m.build(**args)
    assert len(p['images'])==4 and p['contract']['branches']==2
    assert [x for x in calls if x[0]=='moving.load_task']==[('moving.load_task',f,h) for f,h in zip(m.original.EXPORTS,m.original.EXPORT_SHAS)]
    assert p['arguments']==args and p['contract']['output_cap_bytes']==16*1024**2


def test_build_predeclared_gradient_cem_mppi12_and_sampling_actual_sha(monkeypatch):
    args,saved,_,_=build_fixture(monkeypatch,TRIPLE);p=m.build(**args)
    assert p['contract']['images']==p['contract']['encoder']==p['contract']['projector']==12 and p['contract']['branches']==6
    assert p['arguments']['arms']==TRIPLE
    sampling=next(e for e in saved['/RAM/manifest.json']['authorities'] if e['authority']=='sampling')
    sampling['report_sha256']='0'*64
    with pytest.raises(ValueError):m.build(**args)


@pytest.mark.parametrize('bad',['physicalsource','preparesha','manifestarms','modelsha','commands','source','Q'])
def test_build_failclosed_before_any_encoder(bad,monkeypatch):
    args,saved,p,_=build_fixture(monkeypatch)
    if bad=='physicalsource':args['physical_source_sha']='0'*64
    elif bad=='preparesha':saved['/RAM/physical/execution_binding.json']['prepare_sha256']='0'*64
    elif bad=='manifestarms':saved['/RAM/manifest.json']['arms']=['CEM']
    elif bad=='modelsha':saved['/RAM/manifest.json']['authorities'][0]['report_sha256']='0'*64
    elif bad=='commands':p['plans'][0]['commands'][0][0]=.5
    elif bad=='source':p['sourcepack_sha256']='0'*64
    else:
        old=m.moving.load_task
        def wrong(*a):
            v=old(*a);v['q'][0][0]=2.;return v
        monkeypatch.setattr(m.moving,'load_task',wrong)
    with pytest.raises(ValueError):m.build(**args)


@pytest.mark.parametrize('arms',[[GRAD],TRIPLE,ALL])
def test_ram_encode_exact_four_or16_module_calls_and_native_npy(arms):
    images=m.assemble(*fixture(arms),arms)
    for i in images:i['png_path']='/RAM/image.png'
    p=dict(images=images,contract=m.contract(arms));rows=[dict(status='not_started') for _ in images];store=RAMStore();ledger=m.enc.Ledger(len(arms),store.event)
    cuda_image=NS(device=NS(type='cuda'),shape=(1,3,224,224));cuda_latent=NS(device=NS(type='cuda'),shape=(1,192))
    def callback(image):
        assert image.shape==(224,224,3)
        ledger.dispatch('encoder',cuda_image,lambda:None)
        return ledger.dispatch('projector',cuda_latent,lambda:np.full(192,.25,np.float32))
    with patch.object(m.rgb.old,'png',return_value=np.zeros((224,224,3),np.uint8)):
        m.encode_images(p,callback,ledger,store,rows)
    assert all(r['status']=='encoded' for r in rows)
    assert ledger.counts()=={k:dict(attempted=len(images),completed=len(images)) for k in ('encoder','projector')}
    for payload in store.files.values():
        a=np.load(BytesIO(payload),allow_pickle=False)
        assert a.dtype==np.float32 and a.shape==(192,) and a.tobytes()==np.full(192,.25,np.float32).tobytes()
    for row in rows:
        metrics=row['metrics'];assert metrics['actual_Q']==np.mean((np.full(192,.25)-np.full(192,3.))**2)
        assert metrics['predicted_Q'] is None and metrics['prediction_status']=='unknown_not_recomputed'


@pytest.mark.parametrize('fault',['projector_failure','between_modules','missing_module'])
def test_module_failure_keeps_attempted_and_all_unstarted_rows_no_retry(fault):
    images=m.assemble(*fixture(),[GRAD])
    for i in images:i['png_path']='/RAM/image.png'
    p=dict(images=images,contract=m.contract([GRAD]));rows=[dict(status='not_started') for _ in images]
    store=RAMStore();ledger=m.enc.Ledger(1,store.event)
    def callback(a):
        ledger.dispatch('encoder',NS(device=NS(type='cuda'),shape=(1,3,224,224)),lambda:None)
        if fault=='between_modules':raise RuntimeError('between modules')
        if fault=='missing_module':return np.zeros(192,np.float32)
        def fail():raise RuntimeError('projector failure')
        return ledger.dispatch('projector',NS(device=NS(type='cuda'),shape=(1,192)),fail)
    with patch.object(m.rgb.old,'png',return_value=np.zeros((224,224,3),np.uint8)),pytest.raises((RuntimeError,ValueError)):
        m.encode_images(p,callback,ledger,store,rows)
    assert len(rows)==4 and rows[0]['status']=='attempted' and [r['status'] for r in rows[1:]]==['not_started']*3
    assert not store.files and ledger.counts()['encoder']==dict(attempted=1,completed=1)
    assert ledger.counts()['projector']==dict(attempted=int(fault=='projector_failure'),completed=0)


def test_storage_headroom_floor_cap_before_any_write():
    with patch.object(m.shutil,'disk_usage',return_value=NS(free=m.FLOOR+m.CAP-1)),pytest.raises(ValueError):m.Store('/RAM/new_baseline_rgb')
    store=object.__new__(m.Store);store.output=Path('/RAM')
    class Huge:
        def __len__(self):return m.CAP+1
    with patch.object(Path,'iterdir',return_value=[]),patch.object(m.shutil,'disk_usage',return_value=NS(free=10*m.CAP+m.FLOOR)):
        with pytest.raises(ValueError):store.write('never_created',Huge())
    with patch.object(Path,'iterdir',return_value=[]),patch.object(m.shutil,'disk_usage',return_value=NS(free=m.FLOOR-1)):
        with pytest.raises(ValueError):store.write('never_created',b'1')


def test_native_metric_rejects_wrong_scale_shape_or_dtype():
    item=dict(root=[0.]*192,goal=[2.]*192)
    out=m.compare(item,np.ones(192,np.float32));assert out['actual_Q']==1. and out['root_Q']==4. and out['actual_Q_change']==-3.
    for z in (np.ones(192,np.float64),np.ones(191,np.float32),np.full(192,np.nan,np.float32)):
        with pytest.raises(ValueError):m.compare(item,z)


def test_no_neural_native_imports_and_strict_arms():
    assert not any(x in sys.modules for x in ('torch','gymnasium','pymunk','stable_worldmodel'))
    for a in ([],[GRAD,GRAD],['GD'],['unpublished_planner']):
        with pytest.raises(ValueError):m.contract(a)
