"""RAM tensors + actual saved ABI/temp disk only; no NN/physics/fullstrict."""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import hashlib
import gzip
import io
import json
import time
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_full30_feedback_rgb_20261029_v1 as m
import test_agent_p2_midfeedback_terminal_rgb_20261028_v1 as enc


@lru_cache(maxsize=1)
def actual_values():
    v={k:m.read(d['path']) for k,d in m.PINS.items()}
    sources=[]
    for i in range(4):
        u=v[f'job{i}'];b=u['bindings']
        pp=next(path for path,h in b.items() if h==u['prepare_sha256'])
        rp=next(path for path,h in b.items() if h==u['physical_report_sha256'])
        sources.append((m.read(pp),m.read(rp),Path(rp).parent))
    return v,sources


def fixture():
    v,sources=actual_values();rows=[]
    for i,(p,_,directory) in enumerate(sources):rows.extend(m.fixed_rows(p,v[f'job{i}'],directory,i))
    return dict(rows=rows,contract=m.contract(),expected_calls=deepcopy(m.contract()['max_calls']))


def encode_fixture(p,store,*,tokens=257,forbidden=None):
    rows=deepcopy(p['rows']);model=enc.Model(tokens);torch=enc.fake_torch();errors=[]
    trace=m.bridge.Trace(model,torch,store,lambda:None,p['expected_calls'])
    guard=m.Ledger(lambda e:store.event(dict(kind='guard_event',event=e)),no_grad=lambda:True)
    original=model.predict
    def encode(pixels,row,scope,ordinal):
        guard.image=ordinal
        with m.physical.saved.overlay.traced_scope(trace,scope,row):
            if forbidden=='predict':model.predict(None)
            if forbidden=='AD':torch.autograd.functional.jacobian(None)
            if forbidden=='action_encoder':model.action_encoder(None)
            pre=m.bridge.pixels.preprocess_pixels(pixels)
            row['preprocessed']=trace.save(enc.Tensor(pre),f'r{row["index"]}_preprocessed')
            return model.projector(model.encoder(enc.Tensor(pre.reshape(1,3,224,224)))[:,0]).a[0].copy()
    with trace,m.bridge.native.guard(model,torch,guard,errors):m.process(p,rows,store,encode)
    assert not errors and model.predict==original
    return rows,trace.entries,guard.entries,guard.blocked


def disk_build(tmp_path,mp):
    """Complete NEW build/verify through real files; never upstream fullstrict."""
    values,sources=actual_values();v=deepcopy(values);pins=deepcopy(m.PINS);us=[]
    def put(path,value,body=False):
        path.write_bytes(value if body else m.raw(value))
        return dict(path=str(path),sha256=m.sha(path))
    for i,(original,report,directory) in enumerate(sources):
        d=tmp_path/f'job{i}';d.mkdir();p=deepcopy(original);r=deepcopy(report);u=v[f'job{i}']
        p['bindings']={}
        pp=put(tmp_path/f'physical_prepare{i}.json',p)
        # Retain ALL source files, not just selected PNGs, with real saved bytes.
        for f in directory.iterdir():(d/f.name).write_bytes(f.read_bytes())
        r['execution_binding']['prepared']=pp['path'];r['execution_binding']['prepare_sha256']=pp['sha256']
        rp=put(d/'report.json',r);co=m.read(d/'completion.json');co['report_sha256']=rp['sha256']
        completion=put(d/'completion.json',co)
        u.update(prepare_sha256=pp['sha256'],physical_report_sha256=rp['sha256'],
            completion_sha256=completion['sha256'],completion=co,
            bindings={str(f):m.sha(f) for f in d.iterdir()}|{pp['path']:pp['sha256']})
        pins[f'job{i}']=put(tmp_path/f'receipt{i}.json',u);us.append(u)
    v['joined']['job_authorities']=[pins[f'job{i}'] for i in range(4)]
    pins['joined']=put(tmp_path/'joined.json',v['joined'])
    pins['O_report']=put(tmp_path/'O_report.json',v['O_report'])
    v['O_receipt']['report_sha256']=pins['O_report']['sha256']
    pins['O_receipt']=put(tmp_path/'O_receipt.json',v['O_receipt'])
    c=m.read(m.MODEL_PINS['prepared']['path'])
    policy=tmp_path/'policy';policy.mkdir()
    config=put(policy/'config.json',(m.bridge.rgb.POLICY/'config.json').read_bytes(),True)
    weights=put(policy/'fixture_weights.bin',b'FAKE bytes, never loaded',True)
    c['policy']=str(policy);c['bindings_sha256']={}
    modelpins=dict(prepared=put(tmp_path/'model_prepare.json',c))
    for key in ('report','completion','receipt'):modelpins[key]=put(tmp_path/f'model_{key}.json',{})
    shapes=m.bridge.rgb.config_shapes(m.bridge.rgb.read(policy/'config.json'))
    mp.setattr(m,'PINS',pins);mp.setattr(m,'MODEL_PINS',modelpins)
    mp.setattr(m,'model_metadata',lambda:(str(policy),shapes,{x['path']:x['sha256'] for x in (config,weights)}))
    mp.setattr(m,'source_files',lambda:m.own_files()+[Path(f) for f in m.CODE_PINS])
    p=m.build();prepared=tmp_path/(m.NAME+'_prepare.json');m.write(prepared,p)
    assert m.raw(m.verify(prepared,m.sha(prepared)))==m.raw(p)
    return p,prepared


def completed_disk(tmp_path,mp):
    p,prepared=disk_build(tmp_path,mp);s=m.Store(tmp_path/(m.NAME+'_RAM'),m.Budget())
    s.put('binding.json',dict(prepared=str(prepared),prepare_sha256=m.sha(prepared)))
    s.put('initial_roster.json',p['rows'])
    rows,modules,guards,blocked=encode_fixture(p,s)
    seal=dict(rows=rows,modules=modules,guards=guards,blocked=blocked,errors=[])
    h=s.put('native_seal.json',seal,final=True)
    s.event(dict(kind='all24_native_sealed',sha256=h,encoded=24),final=True)
    inventory=[dict(type='RAM_ONLY',path=str(Path(m.__file__).resolve()),sha256=m.sha(m.__file__))]
    r=dict(schema=m.NAME+'.report',status=m.RECORDED,prepared=str(prepared),prepare_sha256=m.sha(prepared),
        contract=m.contract(),**seal,counts=m.bridge.counts(modules),model_hash_before=p['expected_model_hash'],
        model_hash_after=p['expected_model_hash'],model_unchanged=True,source_verified=True,native_seal_sha256=h,
        artifact_sha256=deepcopy(s.hashes),event_count=s.index,elapsed_seconds=0.,
        precision_before=m.precision_contract(),precision_after=m.precision_contract(),
        model_inventory_before=inventory,model_inventory_after=inventory,CUDA_peak_bytes=[1,1],
        known_roots=2,fixed_denominator=24,full_upstream_rebuilt_here=False,**m.FLAGS)
    publication=m.publish(s,r,s.budget.started)
    return p,s,r,publication


def test_actual_all24_join_not_O_or_dedup():
    p=fixture()
    assert [(r['base_index'],r['policy'],r['checkpoint']) for r in p['rows']]==[
        (base,policy,cp) for base in range(4) for policy in ('I','F') for cp in m.CP]
    assert [r['index'] for r in p['rows']]==list(range(24))
    assert all(r['source'] is not None and r['absolute_step']==r['checkpoint']+10 for r in p['rows'])
    assert all('feedback_physical' in r['source']['path'] for r in p['rows'])
    assert all(m.sha(r['source']['path'])==r['source']['file_sha256'] for r in p['rows'])


@pytest.mark.parametrize('mutation',['raw','guard','goal','checkpoint','state','time','policy','root','model','order'])
def test_fixed_rows_real_saved_ABI_mutations(mutation):
    values,sources=actual_values();p=deepcopy(sources[0][0]);u=deepcopy(values['job0'])
    if mutation=='raw':u['rows'][0]['plan']['commands'][20][0]+=.01
    elif mutation=='guard':u['rows'][0]['feedback_guard']['no_state_restore']=False
    elif mutation=='goal':p['plans'][0]['new_model_plan']['goal'][0]+=.01
    elif mutation=='checkpoint':u['rows'][0]['checkpoints'][4]['step']=19
    elif mutation=='state':u['rows'][0]['trajectory'][-1]['state'][0]+=.01
    elif mutation=='time':u['rows'][0]['own_returned']=True
    elif mutation=='policy':u['rows'][0]['policy']='F'
    elif mutation=='root':u['rows'][0]['root_slot']=1
    elif mutation=='model':p['expected_model_hash']='f'*64
    else:u['rows'].reverse()
    with pytest.raises(ValueError):m.fixed_rows(p,u,sources[0][2],0)


def test_complete_disk_build_store_consumer_and_exact_full_transport(tmp_path,monkeypatch):
    p,s,r,pub=completed_disk(tmp_path,monkeypatch)
    out=m.consume(s.path,pub['report_sha256'],pub['completion_sha256'])
    assert out['encoded']==24 and out['missing']==0 and len(out['rows'])==24
    assert out['actual_native_module_IO_bytes']==19224576
    assert s.arrays==168 and s.index==241 and len(list(s.path.iterdir()))==414
    assert out['counts']=={k:dict(attempted=v,completed=v) for k,v in p['expected_calls'].items()}
    assert out['byte_census']['all_metadata_bytes']<m.METADATA
    print('FULL24_DISK_METADATA',out['byte_census'],'PREPARE_BYTES',len(m.raw(p)))
    assert all('model_Q' not in row and 'actual_encoded_Q' not in row for row in out['rows'])


@pytest.mark.parametrize('kind',['unknown','lifecycle','scope','raw','native_input','CLS','projector','counter'])
def test_coherently_resealed_native_consumer_mutations(tmp_path,monkeypatch,kind):
    p,s,r,pub=completed_disk(tmp_path,monkeypatch)
    files={f.name:f.read_bytes() for f in s.path.iterdir()}
    events=[m.producer.native.decode(files[f'event_{i:04}.json']) for i in range(s.index)]
    if kind=='unknown':events.insert(0,dict(kind='unknown'))
    elif kind=='lifecycle':events[1],events[2]=events[2],events[1]
    elif kind=='scope':r['rows'][0]['modules'][0]['scope']['row']=1
    elif kind=='raw':r['rows'][0]['raw30_float32_bits']='0'*480
    elif kind=='counter':r['counts']['encoder']['completed']-=1
    else:
        pos={'native_input':(0,'inputs'),'CLS':(1,'inputs'),'projector':(1,'output')}[kind]
        ref=r['rows'][0]['modules'][pos[0]][pos[1]]
        ref=ref[0] if type(ref) is list else ref
        x=m.codec.load_array(ref,lambda n:files[n]);x.flat[0]+=.1
        # Consumer raw arithmetic/link checks, independent of file descriptor seals.
        def load(e):
            return x if e==ref else m.codec.load_array(e,lambda n:files[n])
        with pytest.raises(ValueError):m.validate(p,r,events,load)
        return
    with pytest.raises(ValueError):m.validate(p,r,events,lambda e:m.codec.load_array(e,lambda n:files[n]))


@pytest.mark.parametrize('kind',['extra','artifact','receipt','prepare','source'])
def test_complete_build_tamper_rejected(tmp_path,monkeypatch,kind):
    p,prepared=disk_build(tmp_path,monkeypatch)
    if kind=='extra':
        (tmp_path/'job0'/'orphan.json').write_bytes(m.raw({}))
    elif kind=='artifact':next((tmp_path/'job0').glob('*.png')).write_bytes(b'changed image')
    elif kind=='receipt':Path(m.PINS['job0']['path']).write_bytes(m.raw({}))
    elif kind=='prepare':prepared.write_bytes(prepared.read_bytes()+b' ')
    else:
        path=tmp_path/'policy'/'fixture_weights.bin';path.write_bytes(b'changed frozen weight fixture')
    with pytest.raises(ValueError):m.verify(prepared,m.sha(prepared) if kind!='prepare' else 'f'*64)


def reseal_bundle(s,r):
    """Attacker coherently updates outer hashes; scientific IO checks still apply."""
    r['artifact_sha256']={p.name:m.sha(p) for p in s.path.iterdir() if p.name not in ('report.json','completion.json')}
    (s.path/'report.json').write_bytes(m.raw(r))
    c=m.read(s.path/'completion.json');c['report_sha256']=m.sha(s.path/'report.json')
    c['artifact_sha256']={p.name:m.sha(p) for p in s.path.iterdir() if p.name!='completion.json'}
    (s.path/'completion.json').write_bytes(m.raw(c))
    return m.sha(s.path/'report.json'),m.sha(s.path/'completion.json')


@pytest.mark.parametrize('kind',['unknown_event','extra','missing_array','precision','model','seal','completion_wall','latent_value'])
def test_complete_saved_consumer_resealed_mutations(tmp_path,monkeypatch,kind):
    p,s,r,pub=completed_disk(tmp_path,monkeypatch)
    if kind=='unknown_event':
        path=s.path/'event_0000.json';v=m.read(path);v['kind']='unknown';path.write_bytes(m.raw(v))
    elif kind=='extra':(s.path/'extra.json').write_bytes(m.raw({}))
    elif kind=='missing_array':next(s.path.glob('*.npy.gz')).unlink()
    elif kind=='precision':r['precision_after']['autocast']=True
    elif kind=='model':r['model_inventory_after'][0]['sha256']='0'*64
    elif kind=='seal':
        path=s.path/'native_seal.json';v=m.read(path);v['rows'][0]['root_slot']=1
        path.write_bytes(m.raw(v));r['native_seal_sha256']=m.sha(path)
    elif kind=='latent_value':
        old=deepcopy(r['rows'][0]['latent']);x=m.codec.load_array(old,lambda n:(s.path/n).read_bytes())
        x[0]+=.25;stream=io.BytesIO();np.save(stream,x,allow_pickle=False)
        (s.path/old['file']).write_bytes(gzip.compress(stream.getvalue(),mtime=0))
        new=dict(old,sha256=m.sha(s.path/old['file']),array_sha256=hashlib.sha256(x.tobytes()).hexdigest())
        def change(v):
            if type(v) is list:return [change(x) for x in v]
            if type(v) is dict:
                return deepcopy(new) if v==old else {k:change(x) for k,x in v.items()}
            return v
        r=change(r)
        for path in s.path.glob('*.json'):
            if path.name not in ('report.json','completion.json'):
                path.write_bytes(m.raw(change(m.read(path))))
        r['native_seal_sha256']=m.sha(s.path/'native_seal.json')
        path=s.path/'event_0240.json';event=m.read(path);event['sha256']=r['native_seal_sha256'];path.write_bytes(m.raw(event))
    report_sha,completion_sha=reseal_bundle(s,r)
    if kind=='completion_wall':
        path=s.path/'completion.json';c=m.read(path);c['inclusive_wall_seconds']=601.
        path.write_bytes(m.raw(c));completion_sha=m.sha(path)
    with pytest.raises((ValueError,KeyError)):m.consume(s.path,report_sha,completion_sha)


def test_shape_full168_noise_and_aggregate_metadata_cap(tmp_path):
    c=m.shape_census();assert c['gzip_NPY_upper_bytes']==37344384 and c['total_upper_bytes']==62510208
    assert c['raw_stored_array_bytes']==37306368 and c['NPY_header_bytes']==21504
    s=m.Store(tmp_path/(m.NAME+'_noise'),m.Budget());rng=np.random.default_rng(7)
    for i in range(24):
        for j,(shape,dtype) in enumerate(m.SHAPES):
            x=rng.integers(0,256,shape,dtype=np.uint8) if dtype=='uint8' else rng.standard_normal(shape,dtype=np.float32)
            ref=s.array(f'r{i}_a{j}',x);y=m.codec.load_array(ref,lambda n:(s.path/n).read_bytes())
            assert x.tobytes()==y.tobytes()
    v={'padding':''};v['padding']='x'*(m.METADATA-len(m.raw(v)))
    s.put('full_metadata.json',v,final=True)
    assert s.metadata_bytes==m.METADATA and s.budget.used<=c['gzip_NPY_upper_bytes']+m.METADATA
    with pytest.raises(ValueError):s.put('extra.json',{})
    with pytest.raises(ValueError):s.array('extra',np.zeros(192,np.float32))
    print('FULL168_NOISE_AND16MiB_METADATA_BYTES',s.budget.used)


@pytest.mark.parametrize('forbidden',['predict','AD','action_encoder'])
def test_forbidden_no_retry_and_restored_hooks(forbidden):
    p=fixture();s=enc.old.Memory()
    with pytest.raises(ValueError):encode_fixture(p,s,forbidden=forbidden)


@pytest.mark.parametrize('tokens',[197,256])
def test_wrong_original_native_hidden_rejected(tokens):
    with pytest.raises(ValueError):encode_fixture(fixture(),enc.old.Memory(),tokens=tokens)


def test_missing_source_null_and_failure_keeps_full_roster():
    p=fixture();p['rows'][0].update(source=None,status='missing_physical_checkpoint')
    rows=deepcopy(p['rows']);s=enc.old.Memory()
    with pytest.raises(RuntimeError):
        m.process(p,rows,s,lambda *a:(_ for _ in ()).throw(RuntimeError('fixture encoder failure')))
    assert len(rows)==24 and rows[0]==p['rows'][0] and rows[1]['status']=='attempted'
    assert all(row['status']=='not_started' for row in rows[2:])


def test_authorization_wall_before_any_output_or_model(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('I/O before auth'))
    with pytest.raises(ValueError):m.run('x','y',tmp_path/'out','bad')
    monkeypatch.setattr(m,'verify',lambda *a:fixture())
    clock=iter([0.,601.]);monkeypatch.setattr(m.time,'monotonic',lambda:next(clock,601.))
    out=tmp_path/(m.NAME+'_expired')
    with pytest.raises(ValueError):m.run('x','y',out,m.AUTH)
    assert not out.exists()


def test_exact24_ledger_cap():
    ledger=m.Ledger(lambda e:None);ledger.entries=[{}]*48;ledger.image=24
    with pytest.raises(ValueError):ledger.dispatch('encoder',None,lambda:pytest.fail('25th'))


def test_real_store_exclusive_codec_and_publication_wall(tmp_path):
    s=m.Store(tmp_path/(m.NAME+'_codec'),m.Budget())
    s.event(dict(kind='a'));s.event(dict(kind='b'))
    for i in range(2):
        b=(s.path/f'event_{i:04}.json').read_bytes()
        assert b==m.raw(m.producer.native.decode(b)) and b'\n\n' not in b
    with pytest.raises(FileExistsError):s.put('event_0000.json',{})
    r=dict(prepared='/RAM/prepare',prepare_sha256='a'*64,status='partial_or_failed')
    with pytest.raises(ValueError):m.publish(s,r,0.,clock=lambda:601.)
    assert (s.path/'report.json').is_file() and not (s.path/'completion.json').exists()
