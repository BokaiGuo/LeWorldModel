"""Three fixed independent-root native factual fidelity consumer; prepare is non-neural.

Only main may invoke cuda. Never train, optimize targets, select roots, execute
physics, or modify the immutable v11/v8 sources. No CPU neural fallback exists.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p5_jepa_fresh_fidelity_main_20261027_v11 as previous

NAME = 'p5_jepa_fresh_fidelity_main_20261027_v12'
COLLECTOR = 'p5_jepa_factual_panel_20261027_v3'
ROOT = Path(__file__).resolve().parents[2]
SEEDS = (2026102702, 2026102703, 2026102704)
ARMS, FRAMES = previous.ARMS, previous.FRAMES
PARENT_SHA = 'c40c9bee86cd5487ca9c8785e517677605e0a55bad9a2bcf2f07b2f1260edaa6'
parent = previous.parent
numeric, normalized_inputs, controls = previous.numeric, previous.normalized_inputs, previous.controls
require, digest = previous.require, previous.digest
SCHEDULE = np.array([
    [[.12,.04],[.08,-.06],[-.05,.09],[.11,.03],[-.04,-.07]],
    [[-.07,.10],[.03,.13],[.09,-.02],[-.11,.05],[.06,-.08]],
    [[.05,-.12],[-.09,.07],[.13,.02],[.02,-.05],[-.06,.11]],
], dtype=np.float64)
EXPECTED = dict(official_encoder=1, proprio_encoder=1, predictor=4)
PHYSICAL_COUNTS = dict(env_construct=2,seed=4,reset=2,control_step=30,
    space_step=302,native_render=32,native_get_obs=32,close=2)


def protocol():
    return dict(schema=NAME,seeds=list(SEEDS),excluded_first_seed=2026102701,
        arms=list(ARMS),frames=list(FRAMES),context_frames=[0,5,10],target_frame=15,
        counts_per_root=EXPECTED,counts_total={k:3*v for k,v in EXPECTED.items()},
        successful_frame_counts_per_root=dict(official_encoder=4,proprio_encoder=3,predictor=12),
        ordering='fixed seed order; all three rows retained, no replacement, screening, retries or subset summary',
        target='observed factual frame15 native patch tokens; no counterfactual ground truth',
        action='native u64; dataset_pixel32=(100*u64).astype(float32); normalize pixel32/100 with source float32 stats',
        proprio='raw state[x,y,vx,vy] at0/5/10; never target proprio in predictor',
        intervention='four immutable v11 arms, change last packet only, preserve first two and context proprio',
        metric='global float64 patch MSE on CUDA, original256x384 output, no target optimization',
        weights='one immutable v8 CUDA loader, same frozen eval state before and after each root and final',
        backend='SDPA MATH, TF32 off, CUDA model/inputs only; offline cached weights, no CPU neural fallback',
        accounting='pre hooks attempted; post hooks completed; separate per-root and total counters',
        artifacts='all four-frame patches and all four successors/root, original float32, exclusive SHA-bound npy',
        failure_policy='persist patches/each successor immediately, checkpoint partial costs/statuses, preserve original and finalization errors separately',
        prospective_control=False,training=False,target_optimized=False,root_selection_changed=False,
        P5_admission=False,cpu_neural_fallback=False,new_physical_steps=0)


def strict_object(pairs):
    out = {}
    for key,value in pairs:
        require(key not in out,'duplicate JSON key')
        out[key] = value
    return out


def file_sha(path):
    path = Path(path)
    require(path.is_file() and not any(p.is_symlink() for p in (path,*path.parents)),
        'regular nonsymlink evidence required: '+str(path))
    return parent.sha(path)


def load_json(path, bindings=None, expected=None):
    value = file_sha(path)
    path = Path(path).resolve()
    require(expected is None or value==expected,'input SHA mismatch: '+str(path))
    if bindings is not None:
        require(str(path) not in bindings or bindings[str(path)]==value,'input changed while reading')
        bindings[str(path)] = value
    return json.loads(path.read_text(),object_pairs_hook=strict_object,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError('nonfinite JSON '+token)))


def verify_bindings(bindings):
    require(isinstance(bindings,dict) and bool(bindings),'nonempty hash closure required')
    errors = []
    for path,expected in bindings.items():
        try:
            if file_sha(path)!=expected:
                errors.append(str(path)+': digest changed')
        except Exception as exc:
            errors.append(str(path)+': '+type(exc).__name__+': '+str(exc))
    return errors


def save_json(path, value):
    """Exclusive durable receipt; caller never overwrites a completed checkpoint."""
    path = Path(path)
    with path.open('xb') as stream:
        stream.write(json.dumps(value,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n')
        stream.flush()
        os.fsync(stream.fileno())


def save_array(directory, seed, name, value):
    a = np.asarray(value)
    shape = (4,256,384) if name=='official_patch_features' else (1,256,384)
    require(a.dtype==np.float32 and a.shape==shape and np.isfinite(a).all(),
        'original finite float32 patch/successor required')
    path = Path(directory)/f'seed_{seed}_{name}.npy'
    with path.open('xb') as stream:
        np.save(stream,a,allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    return dict(path=str(path.resolve()),sha256=file_sha(path),shape=list(a.shape),
        dtype=str(a.dtype),bytes=path.stat().st_size)


def root_identity(seed, rgb, proprio):
    def array_sha(a):
        a = np.ascontiguousarray(a)
        h = hashlib.sha256()
        h.update(str(a.dtype).encode())
        h.update(json.dumps(list(a.shape)).encode())
        h.update(a.tobytes())
        return h.hexdigest()
    return digest(dict(seed=seed,initial_rgb_sha256=array_sha(rgb[0]),
        initial_proprio_sha256=array_sha(proprio[0])))


def seed_paths(values):
    """Explicit seed identity, never positional input order."""
    out = {}
    for text in values:
        seed,path = text.split('=',1)
        seed = int(seed)
        require(seed in SEEDS and seed not in out,'unknown/duplicate fixed seed')
        out[seed] = Path(path).resolve()
    require(set(out)==set(SEEDS),'all three fixed seed paths required')
    return out


def empty_row(seed, identity):
    return dict(seed=seed,root_sha256=identity,status='not_run',error=None,
        counts={k:0 for k in EXPECTED},attempted_counts={k:0 for k in EXPECTED},
        frame_counts={k:0 for k in EXPECTED},attempted_frame_counts={k:0 for k in EXPECTED},
        arm_status={arm:'not_run' for arm in ARMS},costs={},artifacts={},
        finalization_errors=[])


def execute_root(models, inputs, prepared, row, directory, checkpoint):
    """CUDA-only tensor work. Interface mocks in tests are constant NumPy arrays."""
    import torch
    from app.plan_common.datasets.transforms import make_transforms
    from torch.nn.attention import sdpa_kernel, SDPBackend
    rgb,prop,commands = inputs
    norm_prop,packets = normalized_inputs(rgb,prop,commands,prepared['stats'])
    transform = make_transforms(img_size=224,random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.,1.),random_resize_scale=(1.,1.),
        reprob=0.,auto_augment=False,motion_shift=False)
    with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
        pixels = torch.from_numpy(rgb.copy()).permute(0,3,1,2).to('cuda',torch.float32)/255.
        patch = models['official_encoder'](transform(pixels),is_training=True)['x_norm_patchtokens']
        require(patch.shape==(4,256,384) and patch.is_cuda and torch.isfinite(patch).all(),
            'CUDA factual patch contract')
        row['artifacts']['official_patch_features'] = save_array(directory,row['seed'],
            'official_patch_features',patch.cpu().numpy())
        checkpoint()
        visual,target = patch[:3].reshape(1,3,1,16,16,384),patch[3:4]
        prop_tensor = torch.from_numpy(norm_prop[:3]).unsqueeze(0).to('cuda')
        features = models['proprio_encoder'](prop_tensor)
        require(features.shape==(1,3,1,16) and features.is_cuda and torch.isfinite(features).all(),
            'CUDA context proprio contract')
        features = features.expand(1,3,256,16)
        for arm,acts in controls(packets).items():
            require(acts.tolist()==prepared['controls'][arm],'sealed control drift')
            row['arm_status'][arm] = 'attempted'
            checkpoint()
            output = models['predictor'](visual,
                torch.from_numpy(acts).unsqueeze(0).to('cuda'),features)
            require(output[0].shape==(1,3,256,384) and output[0].is_cuda
                and torch.isfinite(output[0]).all(),'CUDA successor contract')
            successor = output[0][:,-1]
            row['artifacts'][arm+'_successor'] = save_array(directory,row['seed'],
                arm+'_successor',successor.cpu().numpy())
            checkpoint()
            cost = (successor.double()-target.double()).square().mean().item()
            require(np.isfinite(cost),'finite factual MSE required')
            row['costs'][arm] = cost
            row['arm_status'][arm] = 'completed'
            checkpoint()
        row['actual_minus_control'] = {a:row['costs']['actual']-row['costs'][a] for a in ARMS[1:]}


def run_cuda(manifest, output):
    """Main-only entry; all error paths materialize retained per-seed receipts."""
    output = Path(output).resolve()
    require(not output.exists(),'exclusive fresh CUDA output required')
    output.mkdir(parents=False,exist_ok=False)
    rows = [empty_row(seed,next((r['root_sha256'] for r in manifest.get('roots',[])
        if r['seed']==seed),None)) for seed in SEEDS]
    result = dict(schema=NAME,status='fresh_panel_fidelity_failed',protocol=protocol(),
        manifest_sha256=digest(manifest),cuda_executed=False,new_physical_steps=0,
        P5_admission=False,rows=rows,error=None,finalization_errors=[],source_check_errors=[])
    serial = 0
    def checkpoint():
        nonlocal serial
        save_json(output/f'checkpoint_{serial:04d}.json',result)
        serial += 1
    models,handles,current,v4 = None,[],None,None
    before = None
    try:
        require(manifest==prepare(manifest['factual_paths'],manifest['collector_manifest']),
            'prepared source/input/protocol drift')
        require(not os.environ.get('PYTHONPATH'),'physics PYTHONPATH overlay forbidden')
        parent.require_cuda()
        result['cuda_executed'] = True
        checkpoint()
        models,strict,sources,before = parent.load_cuda(ROOT,manifest['parent_manifest'])
        _,v5,_ = parent.helper(ROOT).dependencies(ROOT)
        v4 = v5.import_v4(ROOT)
        result.update(model_before=before,strict_loads=strict,imported_sources=sources)
        require(set(models)==set(EXPECTED),'exact frozen three-module closure required')
        def hook(key,phase):
            def record(module,args,*rest):
                require(current is not None,'unattributed neural call prohibited')
                counter = 'attempted_counts' if phase=='attempted' else 'counts'
                frame_counter = 'attempted_frame_counts' if phase=='attempted' else 'frame_counts'
                current[counter][key] += 1
                shape = args[0].shape
                current[frame_counter][key] += int(shape[0] if key=='official_encoder' else shape[0]*shape[1])
            return record
        for key,model in models.items():
            handles.append(model.register_forward_pre_hook(hook(key,'attempted')))
            handles.append(model.register_forward_hook(hook(key,'completed')))
        for row in rows:
            current = row
            prepared = next(r for r in manifest['roots'] if r['seed']==row['seed'])
            row['status'] = 'attempted'
            try:
                rgb,prop,commands,_ = factual_inputs(prepared['factual'],row['seed'],
                    manifest['collector_manifest'])
                require(root_identity(row['seed'],rgb,prop)==row['root_sha256'],'root identity drift')
                checkpoint()
                execute_root(models,(rgb,prop,commands),prepared,row,output,checkpoint)
                row['status'] = 'completed'
            except Exception as exc:
                row['status'],row['error'] = 'failed',type(exc).__name__+': '+str(exc)
            finally:
                try:
                    row['model_after'] = {k:v4.model_receipt(m) for k,m in models.items()}
                    row['model_unchanged'] = row['model_after']==before
                except Exception as exc:
                    row['finalization_errors'].append('model_receipt: '+str(exc))
                    row['model_unchanged'] = False
                row['counts_match'] = row['counts']==row['attempted_counts']==EXPECTED
                row['frames_match'] = row['frame_counts']==row['attempted_frame_counts']==protocol()['successful_frame_counts_per_root']
                if not all((row['counts_match'],row['frames_match'],row['model_unchanged'],
                    all(v=='completed' for v in row['arm_status'].values()))) or row['finalization_errors']:
                    row['status'] = 'failed'
                checkpoint()
    except Exception as exc:
        result['error'] = type(exc).__name__+': '+str(exc)
    finally:
        current = None
        for handle in handles:
            try:
                handle.remove()
            except Exception as exc:
                result['finalization_errors'].append('hook_remove: '+str(exc))
        if models is not None and v4 is not None:
            try:
                result['model_after'] = {k:v4.model_receipt(m) for k,m in models.items()}
                result['model_unchanged'] = result['model_after']==before
            except Exception as exc:
                result['finalization_errors'].append('model_receipt: '+str(exc))
        try:
            result['source_check_errors'] = verify_bindings(manifest['bindings'])
        except Exception as exc:
            result['source_check_errors'].append('source closure: '+str(exc))
        result['sources_unchanged'] = not result['source_check_errors']
        for name in ('counts','attempted_counts','frame_counts','attempted_frame_counts'):
            result[name] = {k:sum(r[name][k] for r in rows) for k in EXPECTED}
        if (result['error'] is None and all(r['status']=='completed' for r in rows)
            and result.get('model_unchanged') is True and result['sources_unchanged']
            and not result['finalization_errors']):
            result['status'] = 'fresh_native_panel_fidelity_recorded'
        try:
            checkpoint()
        except Exception as exc:
            result['finalization_errors'].append('final_checkpoint: '+str(exc))
            result['status'] = 'fresh_panel_fidelity_failed'
        save_json(output/'report.json',result)
    return result


def verify_collector(path, bindings):
    """Import only the collector's declared non-neural prepare/verify interface."""
    import importlib.util
    path = Path(path).resolve()
    manifest = load_json(path,bindings)
    require(manifest['schema']==COLLECTOR and manifest['status']=='prepared_no_physics'
        and manifest['protocol']['seeds']==list(SEEDS),'fixed collector manifest required')
    numeric(manifest['protocol']['schedule'],(3,5,2))
    require(np.array_equal(manifest['protocol']['schedule'],SCHEDULE),'sealed v2 action schedule required')
    require(isinstance(manifest['seals'],dict) and bool(manifest['seals']),'collector source closure required')
    for source,h in manifest['seals'].items():
        require(file_sha(source)==h,'collector source drift: '+source)
        bindings[str(Path(source).resolve())] = h
    source = ROOT/'experiments/spectral_reachability'/f'{COLLECTOR}.py'
    require(str(source) in manifest['seals'],'collector source missing from manifest')
    spec = importlib.util.spec_from_file_location(NAME+'_readonly_collector',source)
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    collector.verify(ROOT,manifest)
    return manifest


def read_array(path, shape, dtype, bindings, expected=None):
    h = file_sha(path)
    path = Path(path).resolve()
    require(expected is None or h==expected,'array source receipt mismatch')
    require(path.stat().st_size<=max(200_000,int(np.prod(shape))*np.dtype(dtype).itemsize+1024),
        'bounded array bytes required')
    a = np.load(path,allow_pickle=False,max_header_size=1024)
    require(a.shape==shape and a.dtype==np.dtype(dtype) and np.isfinite(a).all(),'array shape/dtype/finite mismatch')
    require(str(path) not in bindings or bindings[str(path)]==h,'array changed while reading')
    bindings[str(path)] = h
    return a


def factual_inputs(directory, seed, collector_manifest):
    """Parameterize the immutable v11 input contract for the exact v3 root schema."""
    directory = Path(directory).resolve()
    require(seed in SEEDS and directory.name==f'seed_{seed}','explicit fixed seed directory required')
    bindings = {}
    manifest = load_json(collector_manifest,bindings)
    require(manifest['schema']==COLLECTOR and manifest['protocol']['seeds']==list(SEEDS)
        and np.array_equal(manifest['protocol']['schedule'],SCHEDULE),'collector protocol mismatch')
    manifest_sha = bindings[str(Path(collector_manifest).resolve())]
    panel = load_json(directory.parent/'report.json',bindings)
    require(panel['schema']==COLLECTOR and panel['status']=='fresh_native_nonzero_panel_recorded'
        and panel['source_hashes_unchanged'] is True and panel['runtime_error'] is None
        and panel['source_error'] is None and panel['protocol']==manifest['protocol']
        and panel['manifest_file_sha256']==manifest_sha,'complete source-bound panel required')
    require(len(panel['rows'])==3 and {r['seed'] for r in panel['rows']}==set(SEEDS),'all three unique seeds required')
    indexed = {r['seed']:r for r in panel['rows']}
    require(all(indexed[s]['status']=='fresh_native_panel_root_recorded'
        and indexed[s]['root']==f'seed_{s}' and indexed[s]['report']==f'seed_{s}/report.json'
        and indexed[s]['journal']==f'seed_{s}/journal.jsonl' for s in SEEDS),'panel root identity/status mismatch')
    row = indexed[seed]
    report = load_json(directory/'report.json',bindings,row['report_sha256'])
    require(report['schema']==COLLECTOR and report['layout_schema']=='p5_jepa_factual_nonzero_20261027_v2'
        and report['seed']==seed and report['status']=='fresh_native_panel_root_recorded'
        and report['errors']==[] and report['role_status']=={'a':'completed','b':'completed'}
        and report['trajectory_frames']=={'a':16,'b':16} and report['source_hashes_unchanged'] is True
        and report['protocol']==manifest['protocol'] and report['manifest_file_sha256']==manifest_sha
        and report['P5_admission'] is False and report['neural_executed'] is False,
        'complete paired source-native root required')
    counts = report['native_call_receipts']['python_call_entries']
    require(all(counts.get(k)==v for k,v in PHYSICAL_COUNTS.items()),'302-step per-root physical census mismatch')
    space_steps = report['native_call_receipts']['space_step_events']
    require(len(space_steps)==302 and all(e['dt']==.01 for e in space_steps),'physical substep timing mismatch')
    require(len(report['rows'])==16 and all(r['frame']==i and r['all_exact'] is True
        and set(r['exact'])=={'rgb.npy','state.npy','proprio.npy','full_public_state.json','transition.json'}
        and all(v is True for v in r['exact'].values()) for i,r in enumerate(report['rows'])),
        'sixteen exact repeated frames required')
    journal = directory/'journal.jsonl'
    require(file_sha(journal)==report['journal_sha256']==row['journal_sha256'],'root journal SHA mismatch')
    bindings[str(journal)] = report['journal_sha256']
    events,previous_hash = [],None
    for sequence,line in enumerate(journal.read_text().splitlines()):
        entry = json.loads(line,object_pairs_hook=strict_object)
        require(entry['sequence']==sequence and entry['previous_sha256']==previous_hash
            and entry['entry_sha256']==digest({k:v for k,v in entry.items() if k!='entry_sha256'}),
            'hashchain sequence/previous/digest mismatch')
        previous_hash = entry['entry_sha256']
        events.append(entry['event'])
    require(len(events)>2 and events[0]['phase']=='panel_registered_before_runtime'
        and events[0]['seed']==seed and events[0]['manifest_file_sha256']==manifest_sha
        and events[1]['phase']=='preflight_intent' and events[1]['seed']==seed
        and events[1]['protocol']==manifest['protocol'],'root preflight binding mismatch')
    require(events[-1]['phase']=='terminal' and events[-1]['status']==report['status']
        and events[-1]['summary_sha256']==digest({k:v for k,v in report.items() if k!='journal_sha256'}),
        'root terminal summary mismatch')
    read_array(directory/'schedule_env_u.npy',(15,2),np.float64,bindings,
        events[1]['schedule_file_sha256'])
    require(np.array_equal(np.load(directory/'schedule_env_u.npy',allow_pickle=False),SCHEDULE.reshape(15,2)),
        'factual schedule differs from frozen v2')
    records,steps,exports = {},{},{}
    pending = None
    operations = []
    for event in events[2:-1]:
        phase = event['phase']
        if phase=='intent':
            require(pending is None,'interrupted/overlapping native intent')
            pending = event
            operations.append((event['role'],event['operation']))
            if event['operation']=='step':
                key = event['role'],event['details']['frame']
                require(key not in steps,'duplicate step intent')
                steps[key] = event['details']
            if event['operation'] in ('seed','reset'):
                require(event['details']['seed']==seed,'wrong native reset seed')
        elif phase=='returned':
            require(pending is not None and (event['role'],event['operation'])==
                (pending['role'],pending['operation']),'unpaired native return')
            pending = None
        elif phase=='assets_durable':
            key = event['role'],event['record']['frame']
            require(pending is None and key not in records,'duplicate/indeterminate frame receipt')
            records[key] = event['record']
        elif phase=='consumer_assets_durable':
            require(event['role'] not in exports and pending is None,'duplicate consumer export')
            exports[event['role']] = event['consumer']
        else:
            raise ValueError('unexpected completed-root journal phase: '+phase)
    require(pending is None and operations==[(role,op) for role in ('a','b')
        for op in ['construct','seed','reset',*['step']*15,'close']],
        'complete fixed native operation order required')
    require(set(records)=={(r,i) for r in ('a','b') for i in range(16)}
        and set(steps)=={(r,i) for r in ('a','b') for i in range(1,16)}
        and set(exports)=={'a','b'},'complete frame/command/export census required')
    frames = {}
    for role in ('a','b'):
        for i in range(16):
            frame = directory/role/f'frame_{i:02d}'
            record = records[role,i]
            require(set(record['files'])=={'rgb.npy','state.npy','proprio.npy','full_public_state.json','transition.json'},
                'full frame asset closure required')
            for name,info in record['files'].items():
                p = frame/name
                h = file_sha(p)
                require(h==info['sha256'] and p.stat().st_size==info['bytes'],'durable frame file mismatch')
                bindings[str(p)] = h
                require(h==records['a',i]['files'][name]['sha256'],'independent repeat asset mismatch')
            state = read_array(frame/'state.npy',(7,),np.float32,bindings)
            prop = read_array(frame/'proprio.npy',(4,),np.float32,bindings)
            rgb = read_array(frame/'rgb.npy',(224,224,3),np.uint8,bindings)
            require(np.array_equal(prop,state[[0,1,5,6]]) and state.tolist()==record['state']
                and prop.tolist()==record['proprio'],'frame native state/proprio mismatch')
            if role=='a' and i in FRAMES:
                frames[i] = rgb,prop
            if i:
                command = load_json(directory/role/f'command_{i:02d}.json',bindings)
                require(command==steps[role,i] and command['frame']==i and command['action_index']==i-1
                    and command['packet_index']==(i-1)//5 and command['step_within_packet']==(i-1)%5
                    and np.array_equal(numeric(command['raw_env_argument'],(2,)),SCHEDULE.reshape(15,2)[i-1])
                    and np.array_equal(command['pixel_displacement'],100*SCHEDULE.reshape(15,2)[i-1])
                    and command['expected_substeps']==10 and command['dt']==.01
                    and command['pre_public_sha256']==records[role,i-1]['files']['full_public_state.json']['sha256'],
                    'executed command/step-intent/frame/units mismatch')
                require(np.array_equal(numeric(command['absolute_target'],(2,)),
                    numeric(command['agent_xy_before'],(2,))+100*SCHEDULE.reshape(15,2)[i-1]),
                    'native relative target mismatch')
        receipt = exports[role]
        require(receipt==report['consumer_inputs'][role]
            and receipt['path']==f'{role}/consumer/inputs_manifest.json','root export receipt mismatch')
        exported = load_json(directory/receipt['path'],bindings,receipt['sha256'])
        require(exported['schema']==COLLECTOR and exported['seed']==seed and exported['role']==role
            and exported['frame_order']==list(FRAMES) and exported['context_frames']==[0,5,10]
            and exported['target_frame']==15 and exported['arms']==list(ARMS)
            and exported['native_stats']==manifest['native_stats'],'source-native export time/stats mismatch')
        for name,info in exported['file_hashes'].items():
            require(Path(name).name==name,'export filename must be local')
            p = directory/role/'consumer'/name
            require(file_sha(p)==info['sha256'] and p.stat().st_size==info['bytes'],'export file receipt mismatch')
            bindings[str(p)] = info['sha256']
    rgb = np.stack([frames[i][0] for i in FRAMES])
    prop = np.stack([frames[i][1] for i in FRAMES])
    commands = SCHEDULE.reshape(15,2).copy()
    normalized_prop,packets = normalized_inputs(rgb,prop,commands,manifest['native_stats'])
    banks = np.stack(list(controls(packets).values()))
    for role in ('a','b'):
        consumer = directory/role/'consumer'
        checks = dict(normalized_packets=(packets,(3,10)),
            controls_packets=(banks,(4,3,10)),raw_proprio_frames=(prop,(4,4)),
            normalized_proprio_frames=(normalized_prop,(4,4)),context_proprio=(normalized_prop[:3],(3,4)))
        for name,(expected,shape) in checks.items():
            path = consumer/(name+'.npy')
            require(str(path) in bindings,'required export missing from source receipt')
            actual = read_array(path,shape,np.float32,bindings)
            require(np.array_equal(actual,expected),'native dataset export/consumer mapping mismatch: '+name)
        # Bind all remaining constructor/RNG and supplementary export files too.
        for path in sorted((directory/role).rglob('*')):
            require(not path.is_symlink(),'symlink factual asset rejected')
            if path.is_file():
                value = file_sha(path)
                require(str(path) not in bindings or bindings[str(path)]==value,'asset changed while reading')
                bindings[str(path)] = value
    require(not verify_bindings(bindings),'factual bytes changed during validation')
    return rgb,prop,commands,bindings


def prepare(factual_paths, collector_manifest):
    """All-root source-bound prepare. Missing/failed source evidence refuses CUDA."""
    require(file_sha(previous.__file__)==PARENT_SHA,'immutable v11 helper identity rejected')
    require(isinstance(factual_paths,dict),'explicit seed-to-directory mapping required')
    paths = {int(seed):Path(path).resolve() for seed,path in factual_paths.items()}
    require(len(paths)==len(factual_paths)==3 and set(paths)==set(SEEDS),'all three fixed unique seeds required')
    require(len({p.parent for p in paths.values()})==1,'one complete source panel required')
    bindings = {}
    collector_manifest = Path(collector_manifest).resolve()
    collector = verify_collector(collector_manifest,bindings)
    frozen_path = ROOT/'results/p5_jepa_native_action_response_20261027_v8/manifest_frozen.json'
    frozen = load_json(frozen_path,bindings)
    parent.verify_manifest(ROOT,frozen)
    for path,h in frozen['seals'].items():
        require(file_sha(path)==h,'frozen model source drift')
        bindings[str(Path(path).resolve())] = h
    _,v5,_ = parent.helper(ROOT).dependencies(ROOT)
    v4 = v5.import_v4(ROOT)
    stats_path = v4.native_root(ROOT)/'app/plan_common/datasets/pusht_dset.py'
    stats = v4.source_stats(stats_path)
    roots = []
    for seed in SEEDS:
        rgb,prop,commands,root_bindings = factual_inputs(paths[seed],seed,collector_manifest)
        for path,h in root_bindings.items():
            require(path not in bindings or bindings[path]==h,'cross-root input hash differs')
            bindings[path] = h
        norm_prop,packets = normalized_inputs(rgb,prop,commands,stats)
        require(set(stats)==set(collector['native_stats']) and all(np.array_equal(
            np.asarray(stats[k],np.float32),np.asarray(collector['native_stats'][k],np.float32))
            for k in stats),'model/native factual normalization mismatch')
        roots.append(dict(seed=seed,root_sha256=root_identity(seed,rgb,prop),factual=str(paths[seed]),
            stats=deepcopy(stats),input_sha256=dict(rgb=v4.array_sha(rgb),
                normalized_proprio=v4.array_sha(norm_prop),packets=v4.array_sha(packets)),
            controls={k:v.tolist() for k,v in controls(packets).items()}))
    require(len({r['root_sha256'] for r in roots})==3,'distinct stable root identities required')
    files = [Path(__file__).resolve(),ROOT/'tests'/f'test_{NAME}.py',
        ROOT/'innerdocs'/f'{NAME}.md',Path(previous.__file__).resolve(),
        Path(parent.__file__).resolve(),stats_path]
    for path in files:
        bindings[str(path)] = file_sha(path)
    require(not verify_bindings(bindings),'prepare source/input drift')
    return dict(schema=NAME,status='prepared_non_neural',protocol=protocol(),P5_admission=False,
        cuda_executed=False,factual_paths={str(s):str(paths[s]) for s in SEEDS},
        collector_manifest=str(collector_manifest),parent_manifest=frozen,roots=roots,bindings=bindings)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','cuda'))
    parser.add_argument('--panel-run',type=Path,default=ROOT/f'results/{COLLECTOR}/factual_main_v1')
    parser.add_argument('--collector-manifest',type=Path,default=ROOT/f'results/{COLLECTOR}/manifest_frozen.json')
    parser.add_argument('--manifest',type=Path,default=ROOT/f'results/{NAME}/manifest_frozen.json')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args(argv)
    if args.command=='prepare':
        require(not args.manifest.exists(),'new exclusive prepared manifest required')
        result = prepare({str(s):str(args.panel_run/f'seed_{s}') for s in SEEDS},args.collector_manifest)
        args.manifest.parent.mkdir(parents=True,exist_ok=True)
        save_json(args.manifest,result)
    else:
        require(args.output is not None,'explicit fresh CUDA output required')
        result = run_cuda(load_json(args.manifest),args.output)
    print(json.dumps(dict(status=result['status'],P5_admission=False)))
    if result['status']=='fresh_panel_fidelity_failed':
        raise SystemExit(1)


if __name__=='__main__':
    main()


