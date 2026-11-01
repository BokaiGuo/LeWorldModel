"""RAM callbacks/temp disk only. No torch, physics, weights or upstream consume."""
from copy import deepcopy
from functools import lru_cache
import hashlib
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p3_same_time_cp15_rgb_bridge_20261029_v1 as m
import test_agent_p2_midfeedback_terminal_rgb_20261028_v1 as fake


@lru_cache(maxsize=1)
def actual_abi():
    a=m.actual_authorities()
    # Only the exact already named saved JSON ABI. Not build/verify/fullstrict.
    for v in a.values():assert m.sha(v['path'])==v['sha256']
    p=m.read(a['physical_prepare']['path']);r=m.read(a['physical_receipt']['path'])
    w=m.read(p['authority']['full_witness']['path'])
    out=str(Path(a['physical_batch']['path']).parent)
    return p,r,w,out


def rows_fixture(missing=()):
    p,r,w,out=actual_abi();rows=m.fixed_rows(p,r,w,out)
    for i in missing:
        rows[i].update(source=None,physical_response=None,missing_reason='fixture_missing',status='missing_physical_cp15')
    return rows


def prepared(missing=()):
    rows=rows_fixture(missing)
    return dict(schema=m.NAME+'.prepare',rows=rows,contract=m.contract(),fixture_only=True,
        expected_calls=m.calls(10-len(missing)),expected_model_hash=m.bridge.rgb.MODEL_TENSOR_SHA,
        bindings={},authorities=m.actual_authorities())


def encode_fixture(p,store,forbidden=None,tokens=257):
    rows=deepcopy(p['rows']);model=fake.Model(tokens);torch=fake.fake_torch();errors=[]
    trace=m.bridge.Trace(model,torch,store,lambda:None,p['expected_calls'])
    guard=m.Ledger(lambda e:store.event(dict(kind='guard_event',event=e)),no_grad=lambda:True,available=p['expected_calls']['encoder'])
    def encode(pix,row,scope,ordinal):
        guard.image=ordinal
        with m.physical.saved.overlay.traced_scope(trace,scope,row):
            if forbidden=='predict':model.predict(None)
            if forbidden=='AD':torch.autograd.functional.jacobian(None)
            if forbidden=='action_encoder':model.action_encoder(None)
            pre=m.bridge.pixels.preprocess_pixels(pix)
            row['preprocessed']=trace.save(fake.Tensor(pre),f'r{row["index"]}_preprocessed')
            return model.projector(model.encoder(fake.Tensor(pre.reshape(1,3,224,224)))[:,0]).a[0].copy()
    with trace,m.bridge.native.guard(model,torch,guard,errors):m.process(p,rows,store,encode)
    assert not errors
    return rows,trace.entries,guard.entries,guard.blocked


def make_output(tmp_path,missing=()):
    p=prepared(missing);store=m.Store(str(tmp_path/'rgb'),m.Budget())
    store.put('binding.json',dict(prepared='/fixture/prepare.json',prepare_sha256='e'*64))
    store.put('initial_roster.json',p['rows'])
    rows,modules,guards,blocked=encode_fixture(p,store)
    seal=dict(rows=rows,modules=modules,guards=guards,blocked=blocked,errors=[])
    h=store.put('native_seal.json',seal,final=True)
    store.event(dict(kind='all10_native_sealed',sha256=h,complete=True),final=True)
    load=lambda e:m.bridge.rgb.load_array(e,lambda n:(store.path/n).read_bytes())
    tasks=m.tasks_for(p,rows,load);th=store.put('cp15_tasks.json',tasks,final=True)
    store.event(dict(kind='cp15_tasks_published',sha256=th,complete=True),final=True)
    r=dict(schema=m.NAME+'.report',status=m.RECORDED,prepared='/fixture/prepare.json',prepare_sha256='e'*64,
        contract=m.contract(),**seal,counts=m.bridge.counts(modules),model_hash_before=p['expected_model_hash'],
        model_hash_after=p['expected_model_hash'],model_inventory_before=[dict(path='/fixture/kernel.py',sha256='a'*64)],
        model_inventory_after=[dict(path='/fixture/kernel.py',sha256='a'*64)],source_verified=True,
        precision_before=m.planner.precision_contract(),precision_after=m.planner.precision_contract(),
        CUDA_peak_bytes=[0,0],native_seal_sha256=h,tasks_sha256=th,artifact_sha256=deepcopy(store.hashes),
        event_count=store.index,elapsed_seconds=0.0,**m.FLAGS)
    # Fake CUDA metadata above is only a RAM ABI fixture. Production verify does
    # not accept this fixture prepare or these fabricated module source paths.
    s=m.publish(store,r,store.budget.started)
    return p,r,store,s


def test_actual_named_abi_all_ten_no_upstream_consume():
    rows=rows_fixture()
    assert [(r['root_slot'],r['role'],r['probe']) for r in rows]==m.ORDER
    assert len(rows)==10 and all(r['source'] is not None for r in rows)
    for r in rows:
        incoming=np.asarray(r['raw15'],dtype='<f4')[-10:]
        t=r['cp10_task']
        expected=((incoming-np.asarray(t['action_mean'],np.float32))/np.asarray(t['action_std'],np.float32)).reshape(2,10)
        assert incoming.tobytes().hex()==r['causal_raw10_bits']
        assert expected.tobytes()==np.asarray(r['normalized_action_history'],np.float32).tobytes()
        assert r['history_absolute_steps']==[15,20,25]
        assert r['physical_response']['state']==r['source']['state']


@pytest.mark.parametrize('fault',['drop','order','source','guard','task','checkpoint','native','model','receipt_domain'])
def test_saved_join_mutations(fault):
    p,r,w,out=deepcopy(actual_abi())
    if fault=='drop':r['jobs'].pop()
    elif fault=='order':p['jobs'][0]['probe']='Z2'
    elif fault=='source':p['jobs'][0]['source_witness_sha256']='0'*64
    elif fault=='guard':r['jobs'][0]['row']['guards']=[51]
    elif fault=='task':w['rows'][0]['stageA_task']['task']['projection'][0][1]=1.0
    elif fault=='checkpoint':r['jobs'][0]['row']['endpoint15']['step']=10
    elif fault=='native':r['jobs'][0]['row']['responses'][-1]['state'][0]+=1
    elif fault=='model':w['rows'][0]['stageA_task']['task']['binding']['model_sha256']='0'*64
    else:r['source_authorities']={}
    with pytest.raises((ValueError,KeyError)):m.fixed_rows(p,r,w,out)


def test_budget_has_no_tight_compressbound():
    c=m.census();assert c['maximum_arrays']==70 and c['total_upper_bytes']==42908314<m.CAP
    assert m.META==16*1024**2 and m.FINAL==8*1024**2 and m.WALL==600


@pytest.mark.parametrize('i',range(10))
def test_actual_report_completion_metadata_exact_abi(i):
    p,r,w,out=actual_abi();a=m.actual_authorities();batch=m.read(a['physical_batch']['path'])
    directory=Path(out)/f'job{i:02}';report=m.read(directory/'report.json');completion=m.read(directory/'completion.json')
    entry=r['jobs'][i]
    assert m.sha(str(directory/'report.json'))==entry['report_sha256']
    assert m.sha(str(directory/'completion.json'))==entry['completion_sha256']
    m.job_authority(p,a['physical_prepare'],i,batch['jobs'][i],entry,report,completion)
    bad=deepcopy(completion);bad['artifact_sha256']['journal.jsonl']='0'*64
    with pytest.raises(ValueError,match='completion/journal'):
        m.job_authority(p,a['physical_prepare'],i,batch['jobs'][i],entry,report,bad)


@pytest.mark.parametrize('missing',[(),(0,),(0,3,9),tuple(range(10))])
def test_real_tempdisk_full_chain(tmp_path,monkeypatch,missing):
    p,r,store,seals=make_output(tmp_path,missing)
    p['bindings']['/fixture/kernel.py']='a'*64
    # Test-only seam replaces authentication, never the production CLI/prepare.
    monkeypatch.setattr(m,'verify',lambda *a:p)
    monkeypatch.setattr(m.bridge,'verify_bindings',lambda b:None)
    result=m.consume(str(store.path),seals['report_sha256'],seals['completion_sha256'])
    assert len(result['rows'])==len(result['tasks'])==10
    assert result['encoded']==10-len(missing) and result['missing']==len(missing)
    assert len(list(store.path.glob('*.npy.gz')))==7*(10-len(missing))
    for i in missing:assert result['tasks'][i] is None and result['rows'][i]['latent'] is None
    for i,t in enumerate(result['tasks']):
        if t is not None:
            assert t['identity']['namespace']==m.NAME
            assert t['task']['root_context']['latent_history'][:2]==p['rows'][i]['cp10_task']['root_context']['latent_history'][1:]
            assert t['task']['goal']==p['rows'][i]['cp10_task']['goal']
    assert not result['scientific_admission'] and result['consume_new_neural_calls']==0


@pytest.mark.parametrize('bad',['predict','AD','action_encoder'])
def test_real_guard_blocks_forbidden_apis(tmp_path,bad):
    p=prepared();store=m.Store(str(tmp_path/'bad'),m.Budget())
    with pytest.raises((ValueError,AssertionError)):encode_fixture(p,store,forbidden=bad)


def test_wrong_hidden_not_shape_relaxed(tmp_path):
    with pytest.raises(ValueError):encode_fixture(prepared(),m.Store(str(tmp_path/'bad'),m.Budget()),tokens=197)


@pytest.mark.parametrize('fault',['event','pre','latent','guard','count','row','task','arrayfile','extra'])
def test_independent_tempdisk_semantic_and_byte_failures(tmp_path,monkeypatch,fault):
    p,r,store,seals=make_output(tmp_path)
    load=lambda e:m.bridge.rgb.load_array(e,lambda n:(store.path/n).read_bytes())
    events=[m.old.controls.native.decode((store.path/f'event_{i:04}.json').read_bytes()) for i in range(r['event_count'])]
    if fault=='arrayfile':
        f=next(store.path.glob('*.npy.gz'));f.write_bytes(b'corrupt')
        p['bindings']['/fixture/kernel.py']='a'*64
        monkeypatch.setattr(m,'verify',lambda *a:p)
        with pytest.raises(ValueError):m.consume(str(store.path),seals['report_sha256'],seals['completion_sha256'])
        return
    if fault=='extra':
        (store.path/'extra.bin').write_bytes(b'unknown')
        with pytest.raises(ValueError):m.consume(str(store.path),seals['report_sha256'],seals['completion_sha256'])
        return
    if fault=='event':events.pop()
    elif fault=='pre':r['rows'][0]['preprocessed']=r['rows'][1]['preprocessed']
    elif fault=='latent':
        # An adversary can reseal bytes outside Store. Do not relax the real
        # writer's seventy-array cap merely to construct this negative fixture.
        memory=fake.old.Memory();d=memory.array('bad_latent',np.zeros(192,np.float32))
        (store.path/d['file']).write_bytes(memory.files[d['file']]);r['rows'][0]['latent']=d
    elif fault=='guard':r['guards'].pop()
    elif fault=='count':r['counts']['encoder']['completed']-=1
    elif fault=='row':r['rows'][0]['raw15'][0][0]+=.01
    else:r['rows'][0]['cp10_task']['goal'][0]+=.01
    with pytest.raises((ValueError,KeyError)):m.validate(p,r,events,load)


def test_failclosed_before_io_and_no_placeholder(tmp_path):
    with pytest.raises(ValueError,match='explicit MAIN'):m.run('/none','0'*64,str(tmp_path/'out'),'wrong')
    with pytest.raises(ValueError,match='three explicit'):m.build({})
    a=m.actual_authorities();a['physical_receipt']['sha256']='0'*64
    with pytest.raises(ValueError,match='only sealed actual'):m.build(a)
    f=tmp_path/'fixture.json';f.write_bytes(b'{}')
    with pytest.raises(ValueError,match='placeholder'):m.descriptor(str(f),'0'*64)


def test_no_upstream_consume_no_global_patch_no_neural_top_import():
    import ast
    tree=ast.parse(Path(m.__file__).read_text())
    assert not any(isinstance(x,ast.Global) for x in ast.walk(tree))
    assert not any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute)
        and x.func.attr in ('consume_job','reconstruct','native_job','consume_failure') for x in ast.walk(tree))
    assert not any(isinstance(x,(ast.Import,ast.ImportFrom)) and 'torch' in ast.unparse(x) for x in tree.body)


@pytest.mark.parametrize('available',[0,1,5,8,9,10])
def test_fakecuda_exact_available_ledger_all20_and_21st_rejected(available):
    events=[];callbacks=[];ledger=m.Ledger(events.append,no_grad=lambda:True,available=available)
    for i in range(available):
        ledger.image=i
        for kind,shape in [('encoder',(1,3,224,224)),('projector',(1,192))]:
            x=fake.Tensor(np.zeros(shape,np.float32))
            def callback():callbacks.append((i,kind));return x
            assert ledger.dispatch(kind,x,callback) is x
    assert len(callbacks)==len(ledger.entries)==2*available
    assert all(e['completed'] for e in ledger.entries) and len(events)==4*available
    ledger.image=available
    with pytest.raises(ValueError,match='exact available'):
        ledger.dispatch('encoder',fake.Tensor(np.zeros((1,3,224,224),np.float32)),lambda:callbacks.append('forbidden21'))
    assert len(callbacks)==2*available
    if available==10:assert ledger.n==3 and len(callbacks)==20
