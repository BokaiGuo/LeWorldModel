"""RAM-only fixtures; no model, simulator, disk fixtures or actual prepare."""
from copy import deepcopy
from io import BytesIO
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
from experiments.spectral_reachability import p1_constructive_mppi_rgb_cuda_20261027_v1 as m

spec=importlib.util.spec_from_file_location('mppi_rgb_generic_fixture',Path(__file__).with_name('test_p1_constructive_selected_rgb_cuda_20261027_v2.py'))
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


def fixture():
    report,events,prep,tasks,_=h.fixture([m.adapter.ARM])
    prep['schema']=m.adapter.NAME+'.prepare';events[0]['prepare']=deepcopy(prep)
    return report,events,prep,tasks


def test_schema_view_is_ram_only_and_all_four_images_kept():
    data=fixture();before=deepcopy(data);images=m.assemble(*data)
    assert data==before and len(images)==4
    assert [(i['source_slot'],i['role']) for i in images]==[(s,r) for s in (0,1) for r in ('prefix5','terminal10')]
    assert all(i['arm']==m.adapter.ARM and i['predicted'] is None for i in images)


@pytest.mark.parametrize('bad',['generic_schema','journal_schema','journal_content','partial','order','wrongarm','missing5','guard','goal','command','counts'])
def test_saved_input_failure_closed(bad):
    r,e,p,t=fixture()
    if bad=='generic_schema':p['schema']=m.adapter.physical.NAME+'.prepare'
    elif bad=='journal_schema':e[0]['prepare']['schema']=m.adapter.physical.NAME+'.prepare'
    elif bad=='journal_content':e[0]['prepare']['plans'][0]['commands'][0][0]=.3
    elif bad=='partial':r['rows'][1]['full_horizon']=False
    elif bad=='order':r['rows'].reverse()
    elif bad=='wrongarm':r['rows'][0]['arm']='GN'
    elif bad=='missing5':r['rows'][0]['checkpoints'].pop(1)
    elif bad=='guard':next(x for x in e if x['kind']=='guard_receipt' and x['step']==5)['accepted']=False
    elif bad=='goal':t[0]['interface']['gn_task']['goal_source_sha256']='f'*64
    elif bad=='command':next(x for x in e if x['kind']=='returned_transition')['command']=[.1,.2]
    elif bad=='counts':r['counts']['public_step']['completed']-=1
    with pytest.raises(ValueError):m.assemble(r,e,p,t)


def saved_build_fixture(monkeypatch):
    r,e,p,t=fixture();hv='d'*64
    p.update(manifest='/RAM/selection.json',manifest_sha256=hv,bindings={},sourcepack_sha256=m.adapter.SOURCEPACK_SHA,
        contract=m.adapter.physical.contract(2,m.adapter.WALL),
        adapter=dict(source_sha256=m.ADAPTER_SHA,compiler_report=str(m.adapter.REPORT),
        compiler_report_sha256=m.adapter.REPORT_SHA,compiler_prepare_sha256=m.adapter.PREPARE_SHA,
        generic_executor_sha256=m.adapter.PHYSICAL_SHA,
        command_authority='frozen MPPI v2 own consume final nominal only',official_MPPI=False,P1_admission=False,
        paper_strong_MPPI_requirement_closed=False,search_cost_per_root=m.adapter.compiler.budget(),search_contract=m.adapter.compiler.contract()))
    r.update(sourcepack_sha256=m.adapter.SOURCEPACK_SHA,contract=deepcopy(p['contract']),journal_sha256=hv)
    manifest=dict(plans=deepcopy(p['plans']),roots=deepcopy(p['source_slots']),arms=[m.adapter.ARM])
    e[0]['prepare']=deepcopy(p)
    paths=m.EXPORTS;hashes=m.EXPORT_SHAS
    saved={str(m.REPORT):r,str(m.REPORT.parent/'execution_binding.json'):dict(prepared='/RAM/prepared.json',prepare_sha256=m.PHYSICAL_PREPARE_SHA),
        '/RAM/prepared.json':p,'/RAM/selection.json':manifest}
    for s,path in enumerate(paths):
        t[s]['interface']['task_id']=m.adapter.TASK_IDS[s]
        saved[path]=dict(source_slot=s,blind_sha256=m.adapter.BLIND_SHAS[s],blind_source_sha256=m.adapter.BLIND_SHAS[s])
    opened=[]
    def read(path,expected=None):
        path=str(path);opened.append(path)
        assert 'sourcepack' not in path and 'certificate' not in path and 'source_prepare' not in path
        if expected is not None:
            known={str(m.REPORT):m.REPORT_SHA,'/RAM/prepared.json':m.PHYSICAL_PREPARE_SHA,'/RAM/selection.json':hv,**dict(zip(paths,hashes))}
            assert expected==known[path]
        return deepcopy(saved[path])
    def sha(path):
        path=str(path)
        known={str(m.adapter.__file__):m.ADAPTER_SHA,str(m.rgb.__file__):m.RGB_SHA,
            str(m.rgb.old.__file__):m.rgb.OLD_SHA,str(m.adapter.physical.__file__):m.adapter.PHYSICAL_SHA,
            str(m.REPORT):m.REPORT_SHA,'/RAM/prepared.json':m.PHYSICAL_PREPARE_SHA,**dict(zip(paths,hashes)),**m.enc.MODELS}
        return known.get(path,'1'*64 if path.endswith('.png') else hv)
    monkeypatch.setattr(m,'read',read);monkeypatch.setattr(m,'sha',sha)
    monkeypatch.setattr(m.adapter,'build',lambda:(deepcopy(manifest),{}))
    monkeypatch.setattr(m.adapter,'verify',lambda *a:pytest.fail('do not load physical source'))
    monkeypatch.setattr(m.adapter.physical,'verify',lambda *a:pytest.fail('wrong generic verify'))
    monkeypatch.setattr(m.adapter.physical,'load_source',lambda *a:pytest.fail('privileged source loader'))
    monkeypatch.setattr(m.rgb.planner,'load_export',lambda path,h:deepcopy(t[paths.index(path)]['interface']))
    monkeypatch.setattr(m.rgb.planner,'source_files',lambda:[])
    monkeypatch.setattr(m.rgb.old,'journal',lambda *a:deepcopy(e))
    monkeypatch.setattr(m.rgb.old,'png',lambda *a:np.zeros((224,224,3),np.uint8))
    return saved,p,manifest,opened


def test_build_accepts_adapter_prepare_not_generic_and_no_privileged_reader(monkeypatch):
    _,_,_,opened=saved_build_fixture(monkeypatch);p=m.build()
    assert p['schema']==m.NAME+'.prepare' and len(p['images'])==4
    assert p['physical_report_sha256']==m.REPORT_SHA and p['physical_prepare_sha256']==m.PHYSICAL_PREPARE_SHA
    assert p['contract']['encoder']==p['contract']['projector']==4
    assert p['contract']['predict']==p['contract']['AD']==p['contract']['physics']==0
    assert p['contract']['no_dedup'] and not p['contract']['selection']
    assert str(m.REPORT) in opened


@pytest.mark.parametrize('bad',['binding','schema','commands','compiler','source','contract','blind'])
def test_build_rejects_changed_bindings(monkeypatch,bad):
    saved,p,manifest,_=saved_build_fixture(monkeypatch)
    if bad=='binding':saved[str(m.REPORT.parent/'execution_binding.json')]['prepare_sha256']='f'*64
    elif bad=='schema':p['schema']=m.adapter.physical.NAME+'.prepare'
    elif bad=='commands':p['plans'][0]['commands'][0][0]=.5
    elif bad=='compiler':p['adapter']['compiler_report_sha256']='f'*64
    elif bad=='source':p['sourcepack_sha256']='f'*64
    elif bad=='contract':p['contract']['with_target']=True
    elif bad=='blind':saved[m.EXPORTS[0]]['blind_sha256']='f'*64
    with pytest.raises(ValueError):m.build()


def test_verify_uses_own_full_rebuild(monkeypatch):
    p=dict(images=[],schema=m.NAME+'.prepare')
    monkeypatch.setattr(m,'read',lambda path,h:deepcopy(p));monkeypatch.setattr(m,'build',lambda:deepcopy(p))
    assert m.verify('/RAM/p','a'*64)==p
    monkeypatch.setattr(m,'build',lambda:dict(p,changed=True))
    with pytest.raises(ValueError):m.verify('/RAM/p','a'*64)


def test_four_encodes_no_dedup_durable_native_NPY(monkeypatch):
    images=m.assemble(*fixture())
    for i in images:i['png_path']='/RAM/same.png'
    monkeypatch.setattr(m.rgb.old,'png',lambda *a:np.zeros((224,224,3),np.uint8))
    store=h.h.RAMStore();ledger=m.enc.Ledger(1,store.event);rows=[dict(status='not_started') for _ in images]
    def callback(rgb):
        assert rgb.shape==(224,224,3)
        for k,s in [('encoder',(1,3,224,224)),('projector',(1,192))]:
            ledger.dispatch(k,NS(device=NS(type='cuda'),shape=s),lambda:None)
        return np.zeros(192,np.float32)
    m.rgb.encode_images(dict(images=images),callback,ledger,store,rows)
    assert ledger.counts()=={k:dict(attempted=4,completed=4) for k in ('encoder','projector')}
    assert len(store.files)==4 and all(r['status']=='encoded' for r in rows)
    for raw in store.files.values():
        a=np.load(BytesIO(raw),allow_pickle=False);assert a.shape==(192,) and a.dtype==np.float32


def test_generic_partial_keeps_attempt_and_does_not_advance(monkeypatch):
    images=m.assemble(*fixture())
    for i in images:i['png_path']='/RAM/a.png'
    monkeypatch.setattr(m.rgb.old,'png',lambda *a:np.zeros((224,224,3),np.uint8))
    store=h.h.RAMStore();ledger=m.enc.Ledger(1,store.event);rows=[dict(status='not_started') for _ in images]
    def callback(a):
        ledger.dispatch('encoder',NS(device=NS(type='cuda'),shape=(1,3,224,224)),lambda:None)
        raise RuntimeError('mock projector failure')
    with pytest.raises(RuntimeError):m.rgb.encode_images(dict(images=images),callback,ledger,store,rows)
    assert rows[0]['status']=='attempted' and rows[1]['status']=='not_started' and not store.files
    assert ledger.counts()['encoder']==dict(attempted=1,completed=1)


def test_actual_metrics_are_same_identity_meanQ_not_predicted():
    d=m.rgb.compare(dict(root=[1.]*192,goal=[0.]*192),np.full(192,2,np.float32))
    assert d['actual_Q']==4 and d['root_Q']==1 and d['actual_Q_change']==3
    assert d['predicted_Q'] is None and d['predicted_vs_real_mse'] is None


def test_run_rejects_binding_before_store_or_model_import(monkeypatch):
    def reject(*a):raise ValueError('external binding failed')
    monkeypatch.setattr(m,'verify',reject)
    monkeypatch.setattr(m.rgb,'Store',lambda *a:pytest.fail('no output before verify'))
    with pytest.raises(ValueError,match='external binding'):m.run('/RAM/prepared','bad','/RAM/output')


def test_prepare_uses_exclusive_small_metadata_writer(monkeypatch):
    p=dict(schema=m.NAME+'.prepare',images=[]);calls=[]
    monkeypatch.setattr(m,'build',lambda:deepcopy(p))
    monkeypatch.setattr(m.adapter,'write_new',lambda path,data:(calls.append((path,data)) or dict(sha256='a'*64)))
    assert m.prepare('/RAM/new')['sha256']=='a'*64 and calls==[('/RAM/new',p)]


@pytest.mark.parametrize('bad',('official','admitted','paper','cost','nominal','search'))
def test_mppi_nominal_and_inherited_cost_contract_cannot_be_relabelled(monkeypatch,bad):
    _,p,_,_=saved_build_fixture(monkeypatch);a=p['adapter']
    if bad=='official':a['official_MPPI']=True
    elif bad=='admitted':a['P1_admission']=True
    elif bad=='paper':a['paper_strong_MPPI_requirement_closed']=True
    elif bad=='cost':a['search_cost_per_root']['predict']=66
    elif bad=='nominal':a['command_authority']='best sample'
    else:a['search_contract']['dispatch']='best sample'
    with pytest.raises(ValueError):m.build()
