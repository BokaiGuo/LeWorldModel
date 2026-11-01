"""Pure RAM contract/array/encoder-dispatch mocks; no neural or physics."""
from copy import deepcopy
from io import BytesIO
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
from experiments.spectral_reachability import p1_constructive_selected_rgb_cuda_20261027_v2 as m


spec=importlib.util.spec_from_file_location('rgb_v1_mock',Path(__file__).with_name('test_p1_constructive_selected_rgb_cuda_20261027_v1.py'))
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


def fixture(arms):
    p,e,prep,_,plans,interfaces=h.fixture();base=deepcopy((p,e,prep,interfaces))
    p['rows']=[];prep['plans']=[];events=[]
    for b,(slot,arm) in enumerate((s,a) for s in range(2) for a in arms):
        row=deepcopy(base[0]['rows'][0]);row.update(root_slot=slot,arm=arm)
        selection=dict(root_slot=slot,arm=arm,commands=np.zeros((10,2)).tolist())
        selection['selection_digest']=m.physical_api.source.digest(selection);row['selection_digest']=selection['selection_digest']
        p['rows'].append(row);prep['plans'].append(selection)
        for event in base[1]:
            if event.get('branch')==0 or event['kind']=='branch_closed' and event['receipt']==base[0]['rows'][0]:
                x=deepcopy(event)
                if 'branch' in x:x['branch']=b
                if x['kind']=='branch_started':x['plan']=deepcopy(selection)
                if x['kind']=='branch_closed':x['receipt']=deepcopy(row)
                events.append(x)
    p['contract']['branches']=len(p['rows']);prep['contract']=deepcopy(p['contract'])
    p['counts']=m.counts_for(len(p['rows']))
    for k,c in p['counts'].items():
        for _ in range(c['completed']):events.extend([dict(kind='operation_attempted',resource=k),dict(kind='operation_completed',resource=k)])
    events.insert(0,dict(kind='run_started',prepare=deepcopy(prep)))
    events.append(dict(kind='run_finished',status=p['status'],counts=deepcopy(p['counts']),errors=[]))
    tasks=[dict(source_slot=s,task_export=f'/mock/task_{s:02d}.json.gz',task_sha256=str(s)*64,
        blind_sha256=chr(97+s)*64,interface=deepcopy(interfaces[0])) for s in range(2)]
    return p,events,prep,tasks,arms


@pytest.mark.parametrize('arms',[['matched_ridge_GN'],list(m.planner.ARMS),['a','b','c'],['a','b','c','d']])
def test_all_fixed_arms_and_denominators(arms):
    images=m.assemble(*fixture(arms));assert len(images)==4*len(arms)
    assert [(x['source_slot'],x['arm'],x['role']) for x in images]==[(s,a,r) for s in range(2) for a in arms for r in ('prefix5','terminal10')]
    assert all(x['predicted'] is None and x['Q']=='identity192' for x in images)


@pytest.mark.parametrize('arms',[[],['a']*2,['a','b','c','d','e'],['../x'],[True]])
def test_bad_arm_contract(arms):
    with pytest.raises(ValueError):m.fixed_arms(arms)


@pytest.mark.parametrize('bad',['partial','missing5','missing10','wrongarm','boolslot','count','taskgoal','selection','guard','transition','snapshot','extra_branch'])
def test_provenance_failure_closed(bad):
    p,e,prep,tasks,arms=fixture(['matched_ridge_GN']);r=p['rows'][0]
    if bad=='partial':r['full_horizon']=False
    elif bad.startswith('missing'):r['checkpoints']=[c for c in r['checkpoints'] if c['step']!=int(bad[7:])]
    elif bad=='wrongarm':r['arm']='other'
    elif bad=='boolslot':r['root_slot']=False
    elif bad=='count':p['counts']['physics_step']['completed']-=1
    elif bad=='taskgoal':tasks[0]['interface']['gn_task']['goal_source_sha256']='f'*64
    elif bad=='selection':prep['plans'][0]['commands'][0][0]=.125
    elif bad=='guard':next(x for x in e if x['kind']=='guard_receipt' and x['step']==5)['accepted']=False
    elif bad=='transition':e.pop(next(i for i,x in enumerate(e) if x['kind']=='returned_transition'))
    elif bad=='snapshot':r['checkpoints'][1]['snapshot']['state'][0]+=1
    elif bad=='extra_branch':e.insert(1,dict(kind='branch_started',branch=99,plan={}))
    with pytest.raises(ValueError):m.assemble(p,e,prep,tasks,arms)


def test_actual_Q_only_no_imagined_prediction_substitution():
    z=np.full(192,2,np.float32);item=dict(root=[1.]*192,goal=[0.]*192,predicted=[99.]*192)
    r=m.compare(item,z);assert r['actual_Q']==4 and r['root_Q']==1 and r['actual_Q_change']==3
    assert r['predicted_Q'] is r['predicted_vs_real_mse'] is r['Q_improvement_sign_reversed'] is None
    with pytest.raises(ValueError):m.compare(item,z.astype(float))


@pytest.mark.parametrize('narms',[1,2,3,4])
def test_mock_exact_encodes_no_dedup_lossless_NPY(monkeypatch,narms):
    images=m.assemble(*fixture([f'arm{i}' for i in range(narms)]))
    for i in images:i['png_path']='/mock.png'
    monkeypatch.setattr(m.old,'png',lambda *a:np.zeros((224,224,3),np.uint8))
    s=h.RAMStore();l=m.enc.Ledger(narms,s.event);rows=[dict(status='not_started') for _ in images]
    tensor=lambda shape:NS(device=NS(type='cuda'),shape=shape)
    def callback(a):
        assert a.dtype==np.uint8 and a.shape==(224,224,3)
        for kind,shape in [('encoder',(1,3,224,224)),('projector',(1,192))]:l.dispatch(kind,tensor(shape),lambda:None)
        return np.zeros(192,np.float32)
    m.encode_images(dict(images=images),callback,l,s,rows)
    assert l.counts()=={k:dict(attempted=4*narms,completed=4*narms) for k in ('encoder','projector')}
    assert len(s.files)==len(rows)==4*narms and all(r['status']=='encoded' for r in rows)
    for raw in s.files.values():assert np.array_equal(np.load(BytesIO(raw),allow_pickle=False),np.zeros(192,np.float32))
    with pytest.raises(ValueError):l.dispatch('encoder',tensor((1,3,224,224)),lambda:None)


def test_partial_encoder_keeps_attempt_and_no_next_image(monkeypatch):
    images=m.assemble(*fixture(['matched_ridge_GN']))
    for i in images:i['png_path']='/mock.png'
    monkeypatch.setattr(m.old,'png',lambda *a:np.zeros((224,224,3),np.uint8))
    s=h.RAMStore();l=m.enc.Ledger(1,s.event);rows=[dict(status='not_started') for _ in images]
    def callback(a):
        l.dispatch('encoder',NS(device=NS(type='cuda'),shape=(1,3,224,224)),lambda:None)
        raise RuntimeError('mock projector missing')
    with pytest.raises(RuntimeError):m.encode_images(dict(images=images),callback,l,s,rows)
    assert l.counts()['encoder']==dict(attempted=1,completed=1) and not s.files
    assert rows[0]['status']=='attempted' and rows[1]['status']=='not_started'


def test_unmetered_callback_rejected(monkeypatch):
    images=m.assemble(*fixture(['matched_ridge_GN']))
    for i in images:i['png_path']='/mock.png'
    monkeypatch.setattr(m.old,'png',lambda *a:np.zeros((224,224,3),np.uint8))
    s=h.RAMStore();l=m.enc.Ledger(1,s.event)
    with pytest.raises(ValueError):m.encode_images(dict(images=images),lambda a:np.zeros(192,np.float32),l,s,[{} for _ in images])
    assert not s.files


def test_verify_external_hash_and_full_rebuild(monkeypatch):
    p=dict(arguments=dict(arms=['a']),images=[])
    monkeypatch.setattr(m,'read',lambda path,h:p if h=='a'*64 else (_ for _ in ()).throw(ValueError('SHA')))
    monkeypatch.setattr(m,'build',lambda **kw:deepcopy(p))
    assert m.verify('/RAM','a'*64)==p
    with pytest.raises(ValueError):m.verify('/RAM','b'*64)
    monkeypatch.setattr(m,'build',lambda **kw:dict(p,images=[{}]))
    with pytest.raises(ValueError):m.verify('/RAM','a'*64)


@pytest.mark.parametrize('arms',[['matched_ridge_GN'],list(m.planner.ARMS)])
def test_build_generic_compiler_pointers_no_privileged_read(monkeypatch,arms):
    p,e,prep,tasks,_=fixture(arms);hval='d'*64
    paths=[f'/mock/task_{s:02d}.json.gz' for s in range(2)]
    for s,t in enumerate(tasks):
        t['task_export']=paths[s];t['task_sha256']=hval
        t['blind_sha256']=list(m.enc.APPROVED_INPUTS.values())[s]
    compiler=dict(schema='new_opaque.report',status='model_sealed_not_admitted',errors=[],rows=[])
    for b,sel in enumerate(prep['plans']):
        slot=sel['root_slot'];compiler['rows'].append(dict(source_slot=slot,arm=sel['arm'],commands=sel['commands'],blind_sha256=tasks[slot]['blind_sha256']))
        sel.update(compiler_report='/mock/compiler.json',compiler_report_SHA=hval,compiler_schema=compiler['schema'],
            compiler_status=compiler['status'],commands_pointer=f'/rows/{b}/commands',source_slot_pointer=f'/rows/{b}/source_slot',
            blind_sha_pointer=f'/rows/{b}/blind_sha256',arm_pointer=f'/rows/{b}/arm')
        sel['selection_digest']=m.physical_api.source.digest({k:v for k,v in sel.items() if k!='selection_digest'})
        p['rows'][b]['selection_digest']=sel['selection_digest']
        next(x for x in e if x['kind']=='branch_started' and x['branch']==b)['plan']=deepcopy(sel)
        closed=[x for x in e if x['kind']=='branch_closed'][b];closed['receipt']=deepcopy(p['rows'][b])
    manifest=dict(schema=m.physical_api.NAME+'.selection',sourcepack='/forbidden/sourcepack.json',sourcepack_sha256=prep['sourcepack_sha256'],
        source_prepared='/forbidden/source_prepare.json',source_prepare_sha256=hval,roots=prep['source_slots'],arms=arms,
        plans=deepcopy(prep['plans']),selection_fixed_before_physics=True,physical_outcome_selection=False,wall_seconds=180)
    prep.update(manifest='/mock/selection.json',manifest_sha256=hval,bindings={})
    e[0]['prepare']=deepcopy(prep);p['journal_sha256']=hval
    saved={'/mock/report.json':p,'/mock/execution_binding.json':dict(prepared='/mock/prepared.json',prepare_sha256=hval),
        '/mock/prepared.json':prep,'/mock/selection.json':manifest,'/mock/compiler.json':compiler}
    for s,path in enumerate(paths):saved[path]=dict(source_slot=s,blind_sha256=tasks[s]['blind_sha256'],blind_source_sha256=tasks[s]['blind_sha256'])
    opened=[]
    def read(path,expected=None):
        path=str(path);opened.append(path)
        assert '/forbidden/' not in path
        return deepcopy(saved[path])
    def sha(path):
        path=str(path)
        if path==m.old.__file__:return m.OLD_SHA
        if path==m.physical_api.__file__:return m.PHYSICAL_SHA
        if path.endswith('.png'):return '1'*64
        return m.enc.MODELS.get(path,hval)
    monkeypatch.setattr(m,'read',read);monkeypatch.setattr(m,'sha',sha)
    monkeypatch.setattr(m.physical_api,'read',read);monkeypatch.setattr(m.physical_api,'sha',sha)
    monkeypatch.setattr(m.planner,'load_export',lambda path,h:deepcopy(tasks[paths.index(path)]['interface']))
    monkeypatch.setattr(m.planner,'source_files',lambda:[])
    monkeypatch.setattr(m.old,'journal',lambda *a:deepcopy(e))
    monkeypatch.setattr(m.old,'png',lambda *a:np.zeros((224,224,3),np.uint8))
    result=m.build('/mock/report.json',hval,paths,[hval,hval],arms)
    assert len(result['images'])==4*len(arms) and '/mock/compiler.json' in result['bindings']
    assert all('/forbidden/' not in path for path in opened)
