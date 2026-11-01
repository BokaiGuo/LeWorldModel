"""RAM mocks + real temporary build/Store/saved consumer. No torch or physics."""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p3_same_time_cp10_rgb_bridge_20261029_v1 as m
import test_agent_p2_midfeedback_terminal_rgb_20261028_v1 as enc


@lru_cache(maxsize=1)
def actual_values():
    d=m.descriptors()
    return {k:m.controls.read(d[k]['path'],m.META) for k in
        ('C_prepare','C_report','C_completion','C_receipt','ZP_prepare','ZP_report','ZP_completion','ZP_receipt')}


def rows_fixture():
    v=deepcopy(actual_values())
    return m.fixed_rows(v['C_prepare'],v['C_receipt'],v['ZP_prepare'],v['ZP_receipt'])


def fixture():
    return dict(schema=m.NAME+'.prepare',contract=m.contract(),rows=rows_fixture(),expected_calls=m.contract()['new_calls'])


def encode_fixture(p,store,*,tokens=257,forbidden=None):
    rows=deepcopy(p['rows']);model=enc.Model(tokens);torch=enc.fake_torch();errors=[]
    trace=m.bridge.Trace(model,torch,store,lambda:None,p['expected_calls'])
    guard=m.Ledger(lambda e:store.event(dict(kind='guard_event',event=e)),no_grad=lambda:True)
    def encode(pix,row,scope,ordinal):
        guard.image=ordinal
        with m.physical.saved.overlay.traced_scope(trace,scope,row):
            if forbidden=='predict':model.predict(None)
            if forbidden=='AD':torch.autograd.functional.jacobian(None)
            if forbidden=='action_encoder':model.action_encoder(None)
            pre=m.bridge.pixels.preprocess_pixels(pix)
            row['preprocessed']=trace.save(enc.Tensor(pre),f'r{row["index"]}_preprocessed')
            return model.projector(model.encoder(enc.Tensor(pre.reshape(1,3,224,224)))[:,0]).a[0].copy()
    with trace,m.bridge.native.guard(model,torch,guard,errors):
        m.process(p,rows,store,encode)
    assert not errors
    return rows,trace.entries,guard.entries,guard.blocked


def test_fixed_actual_five_identity_history_and_raw_tokens():
    rows=rows_fixture()
    assert [(x['root_slot'],x['role']) for x in rows]==m.ORDER
    for row in rows:
        assert row['absolute_steps']==[10,15,20]
        assert [x['absolute_step'] for x in row['history_sources']]==[10,15,20]
        a=np.asarray(row['causal_raw10'],np.float32)
        t=row['cp5_task']
        expected=((a-np.asarray(t['action_mean'],np.float32))/np.asarray(t['action_std'],np.float32)).reshape(2,10)
        assert expected.tobytes()==np.asarray(row['normalized_action_history'],np.float32).tobytes()
        assert all(m.sha(f['png'])==f['file_sha256'] for f in row['history_sources'])


@pytest.mark.parametrize('kind',['missing','prefix','history','goal','arm','rawhash','state','root','cp5_not_cp10'])
def test_actual_source_join_rejects_identity_drift(kind):
    v=deepcopy(actual_values());p=v['ZP_prepare'];s=v['ZP_receipt']
    if kind=='missing':s['rows'][0]['status']='native_stopped'
    elif kind=='prefix':s['rows'][0]['prefix_reference_exact']=False
    elif kind=='history':p['prefix_reference'][0]['history_frames'][1]['absolute_step']=5
    elif kind=='goal':p['plans'][0]['native_goal'][0]+=.25
    elif kind=='arm':s['rows'][0]['arm']='P'
    elif kind=='rawhash':p['plans'][0]['raw_float32_sha256']='f'*64
    elif kind=='state':s['rows'][0]['trajectory'][-1]['state'][0]+=.25
    elif kind=='root':s['rows'][0]['root_slot']=1
    else:s['rows'][0]['checkpoints'][-1]['step']=5
    with pytest.raises((ValueError,KeyError)):
        m.fixed_rows(v['C_prepare'],v['C_receipt'],p,s)


def disk_build(tmp_path,mp):
    """Complete build via temp authority bytes, no consumer/NN. Real original PNG copies."""
    v=deepcopy(actual_values());original=m.descriptors();d=deepcopy(original)
    for prefix in ('C','ZP'):
        directory=tmp_path/prefix;directory.mkdir()
        p,r,c,u=[v[prefix+'_'+k] for k in ('prepare','report','completion','receipt')]
        p['bindings']={}
        pp=directory/'prepare.json';pp.write_bytes(m.raw(p))
        r['execution_binding']['prepared']=str(pp);r['execution_binding']['prepare_sha256']=m.sha(pp)
        rp=directory/'report.json';rp.write_bytes(m.raw(r))
        c.update(prepared=str(pp),prepare_sha256=m.sha(pp),report_sha256=m.sha(rp))
        co=directory/'completion.json';co.write_bytes(m.raw(c))
        u.update(completion=deepcopy(c),physical_report_sha256=m.sha(rp),completion_sha256=m.sha(co),
            bindings={str(pp):m.sha(pp),str(rp):m.sha(rp),str(co):m.sha(co)})
        up=directory/'receipt.json';up.write_bytes(m.raw(u))
        for key,path in (('prepare',pp),('report',rp),('completion',co),('receipt',up)):
            d[prefix+'_'+key]=dict(path=str(path),sha256=m.sha(path))
        for row in r['rows']:
            for cp in row['checkpoints']:
                if cp['step']==10:
                    src=Path(original[prefix+'_report']['path']).parent/cp['png']
                    (directory/cp['png']).write_bytes(src.read_bytes())
    policy=tmp_path/'policy';policy.mkdir()
    config=policy/'config.json';config.write_bytes((m.bridge.rgb.POLICY/'config.json').read_bytes())
    weights=policy/'weights.pt';weights.write_bytes(b'RAM fixture, never model loaded')
    mp.setattr(m,'descriptors',lambda:deepcopy(d))
    mp.setattr(m.bridge.rgb,'POLICY',policy)
    mp.setattr(m.bridge.rgb,'MODELS',{str(config):m.sha(config),str(weights):m.sha(weights)})
    mp.setattr(m.bridge,'source_files',lambda:[])
    p=m.build()
    prepared=tmp_path/'own_prepare.json';prepared.write_bytes(m.raw(p))
    assert m.verify(prepared,m.sha(prepared))==p
    return p,prepared


def completed_disk(tmp_path,mp):
    p,prepared=disk_build(tmp_path,mp)
    store=m.Store(tmp_path/(m.NAME+'_RAM'),m.Budget())
    store.put('binding.json',dict(prepared=str(prepared),prepare_sha256=m.sha(prepared)))
    store.put('initial_roster.json',p['rows'])
    rows,modules,guards,blocked=encode_fixture(p,store)
    seal=dict(rows=rows,modules=modules,guards=guards,blocked=blocked,errors=[])
    h=store.put('native_seal.json',seal,final=True)
    store.event(dict(kind='all5_native_sealed',sha256=h,encoded=5),final=True)
    load=lambda e:m.bridge.rgb.load_array(e,lambda n:(store.path/n).read_bytes())
    tasks=m.assemble_tasks(p,rows,load)
    th=store.put('cp10_tasks.json',tasks,final=True)
    store.event(dict(kind='cp10_tasks_published',sha256=th,all5=True),final=True)
    inventory=[dict(type='RAMfixture',path=str(Path(m.__file__).resolve()),sha256=m.sha(m.__file__))]
    r=dict(schema=m.NAME+'.report',status=m.RECORDED,prepared=str(prepared),prepare_sha256=m.sha(prepared),
        contract=m.contract(),**seal,counts=m.bridge.counts(modules),model_hash_before=p['expected_model_hash'],
        model_hash_after=p['expected_model_hash'],model_unchanged=True,source_verified=True,native_seal_sha256=h,
        artifact_sha256=deepcopy(store.hashes),event_count=store.index,elapsed_seconds=0.,
        precision_before=m.planner.precision_contract(),precision_after=m.planner.precision_contract(),
        model_inventory_before=inventory,model_inventory_after=inventory,CUDA_peak_bytes=[1,1],
        tasks_sha256=th,known_roots=2,fixed_denominator=5,**m.FLAGS)
    publication=m.publish(store,r,store.budget.started)
    return p,store,r,publication


def test_complete_build_real_store_consumer_disk_ABI(tmp_path,monkeypatch):
    p,s,r,pub=completed_disk(tmp_path,monkeypatch)
    out=m.consume(s.path,pub['report_sha256'],pub['completion_sha256'])
    assert len(out['tasks'])==5 and len(list(s.path.glob('*.npy.gz')))==35
    assert s.index==52 and len(list(s.path.iterdir()))==93
    assert out['actual_module_IO_bytes']==m.shape_census()['native_module_IO_bytes']
    assert out['actual_metadata_bytes']<=m.METADATA
    assert out['actual_array_file_bytes']<=m.shape_census()['gzip_NPY_upper_bytes']
    for spec,result in zip(p['rows'],out['tasks'],strict=True):
        task=result['task'];old=spec['cp5_task']
        assert task['binding']['task_id']!=old['binding']['task_id']
        assert np.asarray(task['root_context']['latent_history'][:2],np.float32).tobytes()==np.asarray(old['root_context']['latent_history'][1:],np.float32).tobytes()
        for key in ('goal','projection','action_mean','action_std'):assert task[key]==old[key]
        assert task['root_context']['action_history']==spec['normalized_action_history']
        assert all(result[k] is False for k in m.FLAGS)
    print('RAM_CP10_CENSUS',json.dumps(dict(array_files=35,event_files=s.index,total_files=93,
        metadata_bytes=out['actual_metadata_bytes'],array_file_bytes=out['actual_array_file_bytes'],
        native_module_IO_bytes=out['actual_module_IO_bytes']),sort_keys=True))


@pytest.mark.parametrize('forbidden',['predict','AD','action_encoder'])
def test_forbidden_module_or_AD_rejected(forbidden):
    p=fixture();store=enc.old.Memory()
    with pytest.raises(ValueError):encode_fixture(p,store,forbidden=forbidden)


@pytest.mark.parametrize('tokens',[197,256])
def test_wrong_native_hidden_rejected(tokens):
    with pytest.raises(ValueError):encode_fixture(fixture(),enc.old.Memory(),tokens=tokens)


def test_failed_encode_preserves_all5_no_task_subset():
    p=fixture();store=enc.old.Memory();rows=deepcopy(p['rows'])
    with pytest.raises(RuntimeError):
        m.process(p,rows,store,lambda *args:(_ for _ in ()).throw(RuntimeError('RAM failure')))
    assert len(rows)==5 and [x['status'] for x in rows]==['attempted']+['not_started']*4
    assert m.assemble_tasks(p,rows,store.load)==[None]*5


def test_exact_five_guard_not_native_eight_image_capacity():
    guard=m.Ledger(lambda e:None)
    guard.entries=[{}]*10;guard.image=5
    with pytest.raises(ValueError):guard.dispatch('encoder',None,lambda:None)


def test_full_shape_bound_noise_and_metadata_quotas(tmp_path):
    s=m.Store(tmp_path/(m.NAME+'_noise'),m.Budget())
    rng=np.random.default_rng(7)
    for i in range(5):
        for j,(shape,dtype) in enumerate(m.SHAPES):
            a=rng.integers(0,256,shape,dtype=np.uint8) if dtype=='uint8' else rng.standard_normal(shape,dtype=np.float32)
            e=s.array(f'{i}_{j}',a)
            b=m.bridge.rgb.load_array(e,lambda n:(s.path/n).read_bytes())
            assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
    c=m.shape_census()
    assert s.arrays==35 and sum(f.stat().st_size for f in s.path.iterdir())<=c['gzip_NPY_upper_bytes']
    empty=dict(quota='');empty['quota']='x'*(m.METADATA-len(m.raw(empty)))
    s.put('quota.json',empty)
    assert sum(f.stat().st_size for f in s.path.iterdir())<=c['working_upper_bytes']<m.CAP-m.FINAL
    with pytest.raises(ValueError):s.put('extra.json',{})
    with pytest.raises(ValueError):s.array('extra',np.zeros(192,np.float32))
    assert c['total_upper_bytes']==32945904 and c['native_module_IO_bytes']==4005120


@pytest.mark.parametrize('kind',['events','task','array','orphan','completion','publication','precision','model','counter'])
def test_consumer_tamper_rejected(tmp_path,monkeypatch,kind):
    p,s,r,pub=completed_disk(tmp_path,monkeypatch)
    if kind=='events':
        path=s.path/'event_0050.json';v=m.controls.read(path,m.META);v['kind']='not_sealed';path.write_bytes(m.raw(v))
    elif kind=='task':
        path=s.path/'cp10_tasks.json';v=m.controls.read(path,m.META);v[0]['task']['root_context']['latent_history'][0][0]+=.5;path.write_bytes(m.raw(v))
    elif kind=='array':next(s.path.glob('*.npy.gz')).write_bytes(b'corrupt')
    elif kind=='orphan':(s.path/'hidden.json').write_bytes(m.raw({}))
    elif kind=='completion':pub['completion_sha256']='f'*64
    elif kind=='publication':(s.path/'completion_failure.json').write_bytes(m.raw({}))
    elif kind=='precision':r['precision_after']['math']=False;(s.path/'report.json').write_bytes(m.raw(r))
    elif kind=='model':r['model_hash_after']='a'*64;(s.path/'report.json').write_bytes(m.raw(r))
    else:r['counts']['encoder']['completed']-=1;(s.path/'report.json').write_bytes(m.raw(r))
    with pytest.raises((ValueError,KeyError)):m.consume(s.path,pub['report_sha256'],pub['completion_sha256'])


@pytest.mark.parametrize('kind',['history','source','rows','guard','actions','modules'])
def test_full_saved_validation_rejects_resealed_semantics(tmp_path,monkeypatch,kind):
    p,s,r,pub=completed_disk(tmp_path,monkeypatch)
    events=[m.controls.read(s.path/f'event_{i:04}.json',m.META) for i in range(s.index)]
    if kind=='history':r['rows'][0]['cp5_task']['root_context']['latent_history'][0][0]+=.25
    elif kind=='source':r['rows'][0]['source']['pixel_sha256']='a'*64
    elif kind=='rows':r['rows'].pop()
    elif kind=='guard':r['guards'][0]['image']=1
    elif kind=='actions':r['rows'][0]['causal_raw10'][0][0]=0.
    else:r['rows'][0]['modules'][0]['scope']['step']=5
    with pytest.raises((ValueError,KeyError)):
        m.validate(p,r,events,lambda e:m.bridge.rgb.load_array(e,lambda n:(s.path/n).read_bytes()))


def test_native_seal_must_precede_task_publication(tmp_path,monkeypatch):
    p,s,r,pub=completed_disk(tmp_path,monkeypatch)
    events=[m.controls.read(s.path/f'event_{i:04}.json',m.META) for i in range(s.index)]
    events[-1],events[-2]=events[-2],events[-1]
    with pytest.raises(ValueError):m.validate(p,r,events,lambda e:m.bridge.rgb.load_array(e,lambda n:(s.path/n).read_bytes()))


def test_cold_import_no_torch_physics_and_wrong_authority():
    c=f'from experiments.spectral_reachability import {m.NAME}; import sys; assert "torch" not in sys.modules and "pymunk" not in sys.modules'
    r=subprocess.run([sys.executable,'-c',c],cwd=m.ROOT,capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    with pytest.raises(ValueError):m.run('/not/read','f'*64,'/not/write','wrong')

