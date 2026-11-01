"""Non-neural arrays, serialized provenance and constant-interface failure tests."""
from contextlib import nullcontext
from copy import deepcopy
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import p5_jepa_fresh_fidelity_main_20261027_v12 as m


def fake_panel(monkeypatch,tmp_path,fail_predict=None,source_missing=False,remove_failure=False):
    """No torch modules, neural arithmetic, model weights, simulator or GPU."""
    class Tensor:
        is_cuda = True  # Only mimics the interface; never a CUDA execution claim.
        def __init__(self,value): self.value=np.asarray(value)
        @property
        def shape(self): return self.value.shape
        def permute(self,*dims): return Tensor(self.value.transpose(dims))
        def to(self,*args): return self
        def __truediv__(self,value): return Tensor(self.value/value)
        def __getitem__(self,key): return Tensor(self.value[key])
        def reshape(self,*shape): return Tensor(self.value.reshape(shape))
        def unsqueeze(self,axis): return Tensor(np.expand_dims(self.value,axis))
        def expand(self,*shape): return Tensor(np.broadcast_to(self.value,shape))
        def cpu(self): return self
        def numpy(self): return self.value
        def double(self): return Tensor(self.value.astype(np.float64))
        def __sub__(self,other): return Tensor(self.value-other.value)
        def square(self): return Tensor(self.value**2)
        def mean(self): return Tensor(self.value.mean())
        def item(self): return float(self.value)
    class Handle:
        def __init__(self,mapping,fn): self.mapping,self.fn=mapping,fn
        def remove(self):
            self.mapping.remove(self.fn)
            if remove_failure: raise RuntimeError('injected removal error')
    source=tmp_path/'bound_source.txt'
    source.write_text('array interface fixture, not a neural run')
    class Interface:
        def __init__(self,kind): self.kind,self.pre,self.post,self.calls=kind,[],[],0
        def register_forward_pre_hook(self,fn):
            self.pre.append(fn);return Handle(self.pre,fn)
        def register_forward_hook(self,fn):
            self.post.append(fn);return Handle(self.post,fn)
        def __call__(self,*args,**kwargs):
            for fn in list(self.pre): fn(self,args)
            self.calls+=1
            if self.kind=='predictor' and self.calls==fail_predict:
                if source_missing: source.unlink()
                raise RuntimeError('injected forward failure')
            if self.kind=='official_encoder':
                output=dict(x_norm_patchtokens=Tensor(np.zeros((4,256,384),np.float32)))
            elif self.kind=='proprio_encoder':
                output=Tensor(np.zeros((1,3,1,16),np.float32))
            else:
                output=(Tensor(np.zeros((1,3,256,384),np.float32)),)
            for fn in list(self.post): fn(self,args,output)
            return output
    torch=ModuleType('torch')
    torch.from_numpy=Tensor
    torch.float32=np.float32
    torch.isfinite=lambda value:np.isfinite(value.value)
    torch.inference_mode=nullcontext
    attention=ModuleType('torch.nn.attention')
    attention.sdpa_kernel=lambda *a:nullcontext()
    attention.SDPBackend=SimpleNamespace(MATH='constant-interface')
    transforms=ModuleType('app.plan_common.datasets.transforms')
    transforms.make_transforms=lambda **kw:lambda x:x
    monkeypatch.setitem(sys.modules,'torch',torch)
    monkeypatch.setitem(sys.modules,'torch.nn.attention',attention)
    monkeypatch.setitem(sys.modules,'app.plan_common.datasets.transforms',transforms)
    monkeypatch.delenv('PYTHONPATH',raising=False)
    models={key:Interface(key) for key in m.EXPECTED}
    receipt={key:{'fixture':True} for key in models}
    v4=SimpleNamespace(model_receipt=lambda model:{'fixture':True})
    v5=SimpleNamespace(import_v4=lambda root:v4)
    monkeypatch.setattr(m.parent,'helper',lambda root:SimpleNamespace(dependencies=lambda root:(None,v5,None)))
    monkeypatch.setattr(m.parent,'require_cuda',lambda:None)
    monkeypatch.setattr(m.parent,'load_cuda',lambda *args:(models,{}, {},receipt))
    inputs={}
    roots=[]
    stats=dict(ACTION_MEAN=[-.0087,.0068],ACTION_STD=[.2019,.2002],
        PROPRIO_MEAN=[0.]*4,PROPRIO_STD=[1.]*4)
    for i,seed in enumerate(m.SEEDS):
        rgb=np.full((4,224,224,3),i,np.uint8)
        prop=np.full((4,4),i,np.float32)
        commands=m.SCHEDULE.reshape(15,2).copy()
        _,packets=m.normalized_inputs(rgb,prop,commands,stats)
        inputs[seed]=(rgb,prop,commands,{})
        roots.append(dict(seed=seed,root_sha256=m.root_identity(seed,rgb,prop),
            factual=f'fixture-{seed}',stats=stats,controls={k:v.tolist() for k,v in m.controls(packets).items()}))
    manifest=dict(roots=roots,factual_paths={str(s):f'fixture-{s}' for s in m.SEEDS},
        collector_manifest='fixture',parent_manifest={},bindings={str(source):m.file_sha(source)})
    monkeypatch.setattr(m,'prepare',lambda *args:manifest)
    monkeypatch.setattr(m,'factual_inputs',lambda path,seed,*args:inputs[seed])
    return manifest,models


def test_three_fixed_identities_not_argument_order(tmp_path):
    entries=[f'{s}={tmp_path/s.__str__()}' for s in m.SEEDS[::-1]]
    paths=m.seed_paths(entries)
    assert set(paths)==set(m.SEEDS)
    with pytest.raises(ValueError): m.seed_paths(entries[:2])
    with pytest.raises(ValueError): m.seed_paths(entries+[entries[0]])
    with pytest.raises(ValueError): m.seed_paths([str(2026102701)+'='+str(tmp_path)])

def test_protocol_all_roots_no_selection_or_admission():
    p=m.protocol()
    assert p['seeds']==[2026102702,2026102703,2026102704]
    assert p['counts_total']==dict(official_encoder=3,proprio_encoder=3,predictor=12)
    assert p['P5_admission'] is p['target_optimized'] is p['cpu_neural_fallback'] is False
    assert p['arms']==['actual','normalized_zero','negative_normalized','reverse_steps']

def test_native_dataset_projection_and_context():
    rgb=np.zeros((4,224,224,3),np.uint8)
    prop=np.arange(16,dtype=np.float32).reshape(4,4)
    stats=dict(ACTION_MEAN=[.1,-.1],ACTION_STD=[.2,.1],PROPRIO_MEAN=[1.]*4,PROPRIO_STD=[2.]*4)
    commands=m.SCHEDULE.reshape(15,2)
    p,a=m.normalized_inputs(rgb,prop,commands,stats)
    reference=(((100*commands).astype(np.float32)/np.float32(100)
        -np.array(stats['ACTION_MEAN'],np.float32))/np.array(stats['ACTION_STD'],np.float32)).reshape(3,10)
    np.testing.assert_array_equal(a,reference)
    np.testing.assert_array_equal(p,(prop-1)/2)
    for arm,values in m.controls(a).items():
        np.testing.assert_array_equal(values[:2],a[:2])

def test_complete_three_root_arrays_and_totals(monkeypatch,tmp_path):
    manifest,models=fake_panel(monkeypatch,tmp_path)
    out=tmp_path/'cuda-fixture'
    result=m.run_cuda(manifest,out)
    assert result['status']=='fresh_native_panel_fidelity_recorded'
    assert [r['seed'] for r in result['rows']]==list(m.SEEDS)
    assert result['counts']==result['attempted_counts']==m.protocol()['counts_total']
    assert result['frame_counts']==dict(official_encoder=12,proprio_encoder=9,predictor=36)
    assert all(r['counts_match'] and r['frames_match'] and r['model_unchanged'] for r in result['rows'])
    assert sum(len(r['artifacts']) for r in result['rows'])==15
    for row in result['rows']:
        for artifact in row['artifacts'].values():
            assert m.file_sha(artifact['path'])==artifact['sha256']
            array=np.load(artifact['path'],allow_pickle=False)
            assert array.dtype==np.float32
    assert (out/'report.json').is_file() and list(out.glob('checkpoint_*.json'))
    assert all(not model.pre and not model.post for model in models.values())
    with pytest.raises(ValueError): m.run_cuda(manifest,out)

def test_second_predict_failure_retains_all_three_roots_and_actual(monkeypatch,tmp_path):
    manifest,_=fake_panel(monkeypatch,tmp_path,fail_predict=2)
    result=m.run_cuda(manifest,tmp_path/'partial')
    first,second,third=result['rows']
    assert first['status']=='failed' and second['status']==third['status']=='completed'
    assert first['attempted_counts']['predictor']==2 and first['counts']['predictor']==1
    assert set(first['costs'])=={'actual'}
    assert set(first['artifacts'])=={'official_patch_features','actual_successor'}
    assert 'injected forward failure' in first['error']
    assert result['attempted_counts']['predictor']==10 and result['counts']['predictor']==9
    assert result['status']=='fresh_panel_fidelity_failed'
    assert len(list((tmp_path/'partial').glob('*_successor.npy')))==9

def test_missing_source_keeps_original_failure_and_receipts(monkeypatch,tmp_path):
    manifest,_=fake_panel(monkeypatch,tmp_path,fail_predict=2,source_missing=True)
    result=m.run_cuda(manifest,tmp_path/'source-failure')
    assert 'injected forward failure' in result['rows'][0]['error']
    assert not result['sources_unchanged'] and result['source_check_errors']
    assert result['rows'][0]['artifacts']['actual_successor']

def test_hook_remove_failure_preserved_without_losing_arrays(monkeypatch,tmp_path):
    manifest,_=fake_panel(monkeypatch,tmp_path,remove_failure=True)
    result=m.run_cuda(manifest,tmp_path/'remove-failure')
    assert result['status']=='fresh_panel_fidelity_failed'
    assert len(result['finalization_errors'])==6
    assert sum(len(r['artifacts']) for r in result['rows'])==15

def test_loader_failure_retains_full_census_and_zero_neural_counts(monkeypatch,tmp_path):
    manifest,_=fake_panel(monkeypatch,tmp_path)
    def fail(*args): raise RuntimeError('load rejected')
    monkeypatch.setattr(m.parent,'load_cuda',fail)
    result=m.run_cuda(manifest,tmp_path/'load-failure')
    assert 'load rejected' in result['error']
    assert [r['status'] for r in result['rows']]==['not_run']*3
    assert all(v==0 for v in result['counts'].values())

def test_preflight_failure_materializes_report_without_cuda(monkeypatch,tmp_path):
    manifest,_=fake_panel(monkeypatch,tmp_path)
    def fail(*args): raise ValueError('preflight drift')
    monkeypatch.setattr(m,'prepare',fail)
    monkeypatch.setattr(m.parent,'require_cuda',lambda:pytest.fail('must not inspect CUDA'))
    result=m.run_cuda(manifest,tmp_path/'preflight')
    assert not result['cuda_executed']
    assert 'preflight drift' in result['error']
    assert (tmp_path/'preflight/report.json').is_file()

@pytest.mark.parametrize('bad',[np.zeros((1,256,384),np.float64),
    np.zeros((256,384),np.float32),np.full((1,256,384),np.nan,np.float32)])
def test_nonoriginal_or_nonfinite_array_rejected(tmp_path,bad):
    with pytest.raises(ValueError):m.save_array(tmp_path,m.SEEDS[0],'actual_successor',bad)

def test_duplicate_json_key_refused(tmp_path):
    p=tmp_path/'ambiguous.json';p.write_text('{"seed":1,"seed":2}')
    with pytest.raises(ValueError,match='duplicate'):m.load_json(p)


STATS = dict(ACTION_MEAN=[-.0087,.0068],ACTION_STD=[.2019,.2002],
    PROPRIO_MEAN=[236.6155,264.5674,-2.93032027,2.54307914],
    PROPRIO_STD=[101.1202,87.0112,74.84556075,74.14009094])


def fixture_panel(tmp_path):
    """Serialize v3 source schema, never invoke collector/physics/neural methods."""
    panel = tmp_path/'factual_main_v1'
    panel.mkdir()
    stats = {k:np.asarray(v,np.float32).tolist() for k,v in STATS.items()}
    proto = dict(seeds=list(m.SEEDS),schedule=m.SCHEDULE.tolist())
    frozen = tmp_path/'manifest_frozen.json'
    m.save_json(frozen,dict(schema=m.COLLECTOR,status='prepared_no_physics',
        protocol=proto,native_stats=stats,seals={'fixture':'not-a-real-source-seal'}))
    manifest_sha = m.file_sha(frozen)
    panel_rows = []
    for idx,seed in enumerate(m.SEEDS):
        directory = panel/f'seed_{seed}'
        directory.mkdir()
        m.save_json(directory/'fixture_marker.json',dict(no_physics=True,no_neural=True))
        np.save(directory/'schedule_env_u.npy',m.SCHEDULE.reshape(15,2),allow_pickle=False)
        events = [dict(phase='panel_registered_before_runtime',seed=seed,manifest_file_sha256=manifest_sha),
            dict(phase='preflight_intent',seed=seed,protocol=proto,
                schedule_file_sha256=m.file_sha(directory/'schedule_env_u.npy'))]
        records = {}
        exports = {}
        selected = []
        for role in ('a','b'):
            branch = directory/role
            branch.mkdir()
            for op in ('construct','seed','reset'):
                detail = dict(seed=seed) if op in ('seed','reset') else {}
                events += [dict(phase='intent',role=role,operation=op,details=detail),
                    dict(phase='returned',role=role,operation=op)]
            for i in range(16):
                frame = branch/f'frame_{i:02d}'
                frame.mkdir()
                state = np.array([idx+i,2.,3.,4.,.5,6.,7.],np.float32)
                prop = state[[0,1,5,6]]
                rgb = np.full((224,224,3),idx+i,np.uint8)
                if role=='a' and i in m.FRAMES:
                    selected.append((rgb,prop))
                for name,arr in dict(rgb=rgb,state=state,proprio=prop).items():
                    np.save(frame/(name+'.npy'),arr,allow_pickle=False)
                for name in ('full_public_state.json','transition.json'):
                    m.save_json(frame/name,dict(fixture=True,frame=i))
                files = {p.name:dict(sha256=m.file_sha(p),bytes=p.stat().st_size) for p in frame.iterdir()}
                record = dict(frame=i,files=files,state=state.tolist(),proprio=prop.tolist())
                if i:
                    u = m.SCHEDULE.reshape(15,2)[i-1]
                    command = dict(frame=i,action_index=i-1,packet_index=(i-1)//5,
                        step_within_packet=(i-1)%5,raw_env_argument=u.tolist(),
                        pixel_displacement=(100*u).tolist(),agent_xy_before=[1.,2.],
                        absolute_target=(np.array([1.,2.])+100*u).tolist(),expected_substeps=10,dt=.01,
                        pre_public_sha256=records[role,i-1]['files']['full_public_state.json']['sha256'])
                    m.save_json(branch/f'command_{i:02d}.json',command)
                    events += [dict(phase='intent',role=role,operation='step',details=command),
                        dict(phase='returned',role=role,operation='step')]
                records[role,i] = record
                events.append(dict(phase='assets_durable',role=role,record=record))
            events += [dict(phase='intent',role=role,operation='close',details={}),
                dict(phase='returned',role=role,operation='close')]
            consumer = branch/'consumer'
            consumer.mkdir()
            rgb,prop = map(np.stack,zip(*selected))
            norm_prop,packets = m.normalized_inputs(rgb,prop,m.SCHEDULE.reshape(15,2),stats)
            arrays = dict(normalized_packets=packets,controls_packets=np.stack(list(m.controls(packets).values())),
                raw_proprio_frames=prop,normalized_proprio_frames=norm_prop,context_proprio=norm_prop[:3])
            for name,arr in arrays.items(): np.save(consumer/(name+'.npy'),arr,allow_pickle=False)
            exported = dict(schema=m.COLLECTOR,seed=seed,role=role,frame_order=list(m.FRAMES),
                context_frames=[0,5,10],target_frame=15,arms=list(m.ARMS),native_stats=stats,
                file_hashes={p.name:dict(sha256=m.file_sha(p),bytes=p.stat().st_size) for p in consumer.iterdir()})
            m.save_json(consumer/'inputs_manifest.json',exported)
            exports[role] = dict(path=f'{role}/consumer/inputs_manifest.json',
                sha256=m.file_sha(consumer/'inputs_manifest.json'))
            events.append(dict(phase='consumer_assets_durable',role=role,consumer=exports[role]))
        report = dict(schema=m.COLLECTOR,layout_schema='p5_jepa_factual_nonzero_20261027_v2',
            seed=seed,status='fresh_native_panel_root_recorded',errors=[],role_status={'a':'completed','b':'completed'},
            trajectory_frames={'a':16,'b':16},source_hashes_unchanged=True,protocol=proto,
            manifest_file_sha256=manifest_sha,P5_admission=False,neural_executed=False,
            native_call_receipts=dict(python_call_entries=m.PHYSICAL_COUNTS,
                space_step_events=[dict(dt=.01)]*302),
            rows=[dict(frame=i,all_exact=True,exact={name:True for name in records['a',i]['files']}) for i in range(16)],
            consumer_inputs=exports)
        events.append(dict(phase='terminal',status=report['status'],summary_sha256=m.digest(report)))
        write_fixture_journal(directory/'journal.jsonl',events)
        report['journal_sha256'] = m.file_sha(directory/'journal.jsonl')
        m.save_json(directory/'report.json',report)
        panel_rows.append(dict(seed=seed,root=f'seed_{seed}',report=f'seed_{seed}/report.json',
            journal=f'seed_{seed}/journal.jsonl',report_sha256=m.file_sha(directory/'report.json'),
            journal_sha256=report['journal_sha256'],status=report['status']))
    m.save_json(panel/'report.json',dict(schema=m.COLLECTOR,status='fresh_native_nonzero_panel_recorded',
        source_hashes_unchanged=True,runtime_error=None,source_error=None,protocol=proto,
        manifest_file_sha256=manifest_sha,rows=panel_rows))
    return panel,frozen


def write_fixture_journal(path, events):
    previous = None
    lines = []
    for i,event in enumerate(events):
        entry = dict(sequence=i,previous_sha256=previous,event=event)
        entry['entry_sha256'] = m.digest(entry)
        previous = entry['entry_sha256']
        lines.append(json.dumps(entry))
    path.write_text('\n'.join(lines)+'\n')


def resign_fixture(panel, seed, events):
    """Tests may rehash serialization; semantic validation must still reject."""
    directory = panel/f'seed_{seed}'
    report = json.loads((directory/'report.json').read_text())
    report.pop('journal_sha256')
    events[-1]['summary_sha256'] = m.digest(report)
    write_fixture_journal(directory/'journal.jsonl',events)
    report['journal_sha256'] = m.file_sha(directory/'journal.jsonl')
    (directory/'report.json').write_text(json.dumps(report))
    overall = json.loads((panel/'report.json').read_text())
    row = next(r for r in overall['rows'] if r['seed']==seed)
    row['report_sha256'] = m.file_sha(directory/'report.json')
    row['journal_sha256'] = report['journal_sha256']
    (panel/'report.json').write_text(json.dumps(overall))


def test_v3_exact_schema_reader_all_roots(tmp_path):
    panel,frozen = fixture_panel(tmp_path)
    for seed in m.SEEDS:
        rgb,prop,commands,bindings = m.factual_inputs(panel/f'seed_{seed}',seed,frozen)
        assert rgb.shape==(4,224,224,3) and prop.shape==(4,4)
        np.testing.assert_array_equal(commands,m.SCHEDULE.reshape(15,2))
        assert len(bindings)>200


@pytest.mark.parametrize('defect',['asset','chain','step_units','seed','interrupted','wrong_frame','drop_root'])
def test_v3_tamper_refusal_including_rehashed_semantics(tmp_path,defect):
    panel,frozen = fixture_panel(tmp_path)
    seed = m.SEEDS[0]
    directory = panel/f'seed_{seed}'
    events = [json.loads(line)['event'] for line in (directory/'journal.jsonl').read_text().splitlines()]
    if defect=='asset':
        np.save(directory/'a/frame_02/state.npy',np.ones(7,np.float32),allow_pickle=False)
    elif defect=='chain':
        with (directory/'journal.jsonl').open('a') as stream:stream.write('{}\n')
    elif defect=='drop_root':
        overall = json.loads((panel/'report.json').read_text())
        overall['rows'].pop()
        (panel/'report.json').write_text(json.dumps(overall))
    else:
        event = next(e for e in events if e.get('operation')=='step' and e['phase']=='intent')
        if defect=='step_units':event['details']['raw_env_argument']=[99.,99.]
        elif defect=='seed':
            next(e for e in events if e.get('operation')=='seed' and e['phase']=='intent')['details']['seed']=2026102701
        elif defect=='interrupted':
            events.remove(next(e for e in events if e.get('operation')=='step' and e['phase']=='returned'))
        elif defect=='wrong_frame':event['details']['frame']=0
        resign_fixture(panel,seed,events)
    with pytest.raises(ValueError):
        m.factual_inputs(directory,seed,frozen)


def test_prepare_missing_roots_refuses_before_parent_or_cuda(monkeypatch,tmp_path):
    monkeypatch.setattr(m.parent,'require_cuda',lambda:pytest.fail('no CUDA in prepare'))
    with pytest.raises(ValueError,match='three'):
        m.prepare({str(m.SEEDS[0]):str(tmp_path)},tmp_path/'missing')


def test_source_and_data_path_symlink_refused(tmp_path):
    p=tmp_path/'real';p.write_text('source')
    link=tmp_path/'link';link.symlink_to(p)
    with pytest.raises(ValueError,match='nonsymlink'):m.file_sha(link)


