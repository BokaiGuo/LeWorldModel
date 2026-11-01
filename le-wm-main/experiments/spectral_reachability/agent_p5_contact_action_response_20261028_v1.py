"""Blind-public native JEPA action words. Saved prepare; MAIN-only HOST CUDA.

No contact/pose/suffix receipts are accepted by this interface. No physics,
training, gradients, goal scoring, selection, readout or CPU neural fallback.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

import numpy as np

from experiments.spectral_reachability import agent_p5_native_action_words_20261028_v2 as owner

TAG = 'agent_p5_contact_action_response_20261028_v1'
ROOT = Path(__file__).resolve().parents[2]
OWNER_SHA = '5dc5d158ff817d9f39dca0f82ea9eb562f0f9566cc3e74a2abfd2be1af4d74ce'
PUBLIC = ROOT / 'results/agent_p5_contact_source_native_20261028_v1_public_MAIN01_payload.json'
PUBLIC_SHA = 'fdc063a6475577071fb74cf4ccd0853c4b2e4e5c061693d73efbc437686ffec3'
PUBLIC_SCHEMA = 'agent_p5_contact_source_native_20261028_v1.public_context'
PUBLIC_UNITS = 'raw native xy/vxy; downstream must use frozen native preprocessing, not refit'
WORDS = owner.WORDS
PATCHES, FEATURES, PROP = 256, 384, 16
EXPECTED = dict(official_encoder=1, proprio_encoder=1, predictor=8)
BATCH = dict(official_encoder=3, proprio_encoder=1, predictor=1)
FRAMES = dict(official_encoder=3, proprio_encoder=3, predictor=3)
CAP, FLOOR, WALL = 64*1024**2, 200*1024**2, 600
HELPERS = tuple('p5_jepa_' + s for s in (
    'strict_metadata_20261027_v3', 'real_forward_20261027_v4',
    'action_fidelity_20261027_v5', 'official_equivalence_20261027_v6',
    'official_action_fidelity_20261027_v7', 'native_action_response_20261027_v8'))
# Exact projected model authority; generated from pinned v12 JSON, not a new model.
MODEL_SPEC_SHA = 'a4757f79a4a3e699e239892bdad0273f162a6f38ef61c3df96c0e9f7a9188cb5'
require, sha, digest, read_json, write_new = owner.require, owner.sha, owner.digest, owner.read_json, owner.write_new


def protocol():
    return dict(schema=TAG, context_frames=[0,5,10], words=list(WORDS), denominator=4,
        suffix_raw_steps=10, packets=2, raw_dtype='float64', relative_scale=100,
        context='public RGB only, raw agent xy/vxy normalized by frozen source stats; real prefix actions',
        feedback='native visual AND patchwise proprio predictions append; independent branches; no teacher forcing',
        official_encoder='one API with batch3 independent public RGB frames, not one sample',
        counts_api=EXPECTED, counts_forward_samples={k:EXPECTED[k]*BATCH[k] for k in EXPECTED},
        counts_context_frames={k:EXPECTED[k]*FRAMES[k] for k in EXPECTED}, batch=BATCH,
        scope='includes context encoding and all predictor attempts; failure charged; no replay/warmup/padding',
        shared='RGB/proprio context encoded once for all four branches, no duplicate refund',
        backward=0, AD=0, JVP=0, VJP=0, SVD=0, physics=0, FLOPs=None,
        goal_encoded=False, goal_scoring=False, candidate_selection=False, training=False,
        fitting=False, adapter=False, cpu_neural_fallback=False,
        metrics='float64 feature word-order/cancellation MSE only; no native-unit predictions or physical efficacy',
        constructed_OOD_calibration=True, independent_architecture=False, fresh=False,
        scientific_admission=False, P5_admission=False, action_witness_established=False,
        task_success=None, physical_release=False,
        output_cap_bytes=CAP, free_floor_bytes=FLOOR, wall_seconds=WALL,
        free_cuda_scratch_bytes=4*1024**3)


def model_projection(manifest):
    """No old factual arrays, labels, private results or root metadata in CUDA plan."""
    parent = manifest['parent_manifest']
    external = (ROOT/'external/jepa-wms',
                ROOT/'external/dinov2_official_7764ea0f912e53c92e82eb78a2a1631e92725fc8')
    explicit = {ROOT/'experiments/spectral_reachability'/(name+'.py') for name in HELPERS}
    explicit.update((ROOT/'.cache/dinov2/dinov2-small/model.safetensors',
                     ROOT/'data/jepa_wm_checkpoints/jepa_wm_pusht.pth.tar'))
    seals = {p:h for p,h in parent['seals'].items()
             if Path(p) in explicit or (Path(p).suffix in ('.py','.yaml')
                                        and any(Path(p).is_relative_to(d) for d in external))}
    require(explicit <= {Path(p) for p in seals}, 'model loader/source/weights missing')
    stats = manifest['roots'][0]['stats']
    require(all(r['stats']==stats for r in manifest['roots']), 'shared frozen statistics')
    return dict(loader_manifest={**{k:parent[k] for k in (
        'expected_dynamics_schema','expected_model_receipts','official_weight_mapping')}, 'seals':seals},
        stats=stats)


def model_authority():
    require(sha(owner.__file__)==OWNER_SHA, 'sealed owner source')
    m = read_json(owner.MANIFEST, owner.PINS[str(owner.MANIFEST)])
    r = read_json(owner.REPORT, owner.PINS[str(owner.REPORT)])
    owner.verify_v12_manifest_digest(m,r)
    require(m['schema']==r['schema']==owner.PARENT and r['status']=='fresh_native_panel_fidelity_recorded'
            and r['cuda_executed'] is True and r['error'] is None
            and r['finalization_errors']==r['source_check_errors']==[]
            and r['sources_unchanged'] is True and r['model_unchanged'] is True
            and r['model_before']==r['model_after']==m['parent_manifest']['expected_model_receipts'],
            'actual frozen model authority required')
    spec = model_projection(m)
    require(digest(spec)==MODEL_SPEC_SHA, 'single model projection authority digest')
    owner.verify_bindings(spec['loader_manifest']['seals'])
    return spec


def public_inputs(path, expected_sha, stats):
    """Only MAIN external actual public SHA is authority; no private metadata input."""
    require(expected_sha==PUBLIC_SHA, 'MAIN actual public payload SHA required')
    p = read_json(path, expected_sha)
    require(set(p)=={'schema','inputs','proprio_units','ready_for_numeric_context_loading',
                    'source_witness_included','scientific_admission'}, 'public payload exact keys')
    require(p['schema']==PUBLIC_SCHEMA and p['ready_for_numeric_context_loading'] is True
            and p['source_witness_included'] is False and p['scientific_admission'] is False
            and p['proprio_units']==PUBLIC_UNITS, 'blind publication ABI')
    shapes = {**{f'context_rgb_{i:02d}':((224,224,3),'uint8') for i in (0,5,10)},
        'context_proprio':((3,4),'float32'), 'history_raw_u':((10,2),'float64'),
        'history_normalized':((2,10),'float32'), 'goal_rgb':((224,224,3),'uint8')}
    require(set(p['inputs'])==set(shapes), 'seven public arrays only')
    dirs = {Path(x['path']).parent for x in p['inputs'].values()}
    require(len(dirs)==1, 'one separate public directory')
    directory = next(iter(dirs))
    require(directory.is_absolute() and not any(x.is_symlink() for x in (directory,*directory.parents)),
            'absolute nonsymlink public directory')
    require({x.name for x in directory.iterdir()}=={k+'.npy' for k in shapes}, 'public allowlist no extras')
    arrays = {}
    for k,(shape,dtype) in shapes.items():
        rec = p['inputs'][k]
        require(set(rec)=={'path','sha256','bytes','shape','dtype'}
                and Path(rec['path'])==directory/(k+'.npy'), 'public member identity')
        arrays[k] = owner.array(rec,shape,np.dtype(dtype))
    require(np.array_equal(arrays['history_raw_u'],np.zeros((10,2),np.float64)), 'fixed prefix only')
    require(np.array_equal(arrays['history_normalized'],owner.normalize(
        arrays['history_raw_u'].reshape(2,5,2),stats)), 'actual native prefix normalization')
    return p, arrays


def normalized_proprio(raw,stats):
    raw = owner.check_array(raw,(3,4))
    mean,std = (np.asarray(stats[k],np.float32) for k in ('PROPRIO_MEAN','PROPRIO_STD'))
    require(mean.shape==std.shape==(4,) and np.isfinite(mean).all()
            and np.isfinite(std).all() and np.all(std>0), 'frozen proprio statistics')
    result = (raw-mean)/std
    return owner.check_array(result,(3,4))


def self_bindings():
    return {str(p):sha(p) for p in (Path(__file__).resolve(),Path(owner.__file__).resolve(),
        ROOT/'tests'/('test_'+TAG+'.py'),ROOT/'innerdocs'/(TAG+'.md'))}


def prepare(payload, payload_sha):
    spec = model_authority()
    p,arrays = public_inputs(payload,payload_sha,spec['stats'])
    normalized_proprio(arrays['context_proprio'],spec['stats'])
    return dict(schema=TAG,status='prepared_public_model_only_for_MAIN',protocol=protocol(),
        public_payload_path=str(Path(payload).absolute()),public_payload_sha256=payload_sha,
        inputs=p['inputs'],model_spec=spec,bindings=self_bindings(),
        plans={k:dict(raw=v.tolist(),normalized=owner.normalize(v,spec['stats']).tolist())
               for k,v in owner.programs().items()},scientific_admission=False,physical_release=False)


def verify(m):
    require(sha(owner.__file__)==OWNER_SHA, 'sealed owner source')
    require(set(m)=={'schema','status','protocol','public_payload_path','public_payload_sha256',
        'inputs','model_spec','bindings','plans','scientific_admission','physical_release'}, 'exact deployment plan')
    require(m['schema']==TAG and m['status']=='prepared_public_model_only_for_MAIN'
            and m['protocol']==protocol() and m['scientific_admission'] is False
            and m['physical_release'] is False, 'fixed protocol')
    require(digest(m['model_spec'])==MODEL_SPEC_SHA and m['bindings']==self_bindings(), 'authority/self seals')
    owner.verify_bindings(m['model_spec']['loader_manifest']['seals'])
    p,a = public_inputs(m['public_payload_path'],m['public_payload_sha256'],m['model_spec']['stats'])
    require(p['inputs']==m['inputs'] and m['plans']=={k:dict(raw=v.tolist(),
        normalized=owner.normalize(v,m['model_spec']['stats']).tolist()) for k,v in owner.programs().items()},
        'public/word seals')
    normalized_proprio(a['context_proprio'],m['model_spec']['stats'])
    return a


class Meter:
    """One in-flight operation; predispatch debit, no failed-attempt refund."""
    def __init__(self, event, guard=lambda:None):
        self.attempted = dict.fromkeys(EXPECTED,0)
        self.completed = dict.fromkeys(EXPECTED,0)
        self.pending = None
        self.event,self.guard = event,guard

    def before(self,key,batch):
        self.guard()
        require(key in EXPECTED and type(batch) is int and batch==BATCH[key], 'fixed batch ABI')
        require(self.pending is None and self.attempted[key]<EXPECTED[key], 'predispatch cap/inflight')
        self.attempted[key] += 1
        self.pending = key
        self.event('attempted',self.snapshot())

    def after(self,key,batch):
        require(self.pending==key and type(batch) is int and batch==BATCH[key], 'paired actual return')
        self.completed[key] += 1
        self.pending = None
        self.event('returned',self.snapshot())

    def snapshot(self):
        return {**{f:dict(a) for f,a in (('attempted',self.attempted),('completed',self.completed))},
            **{f'{unit}_{phase}':{k:a[k]*factors[k] for k in EXPECTED}
               for unit,factors in (('samples',BATCH),('frames',FRAMES))
               for phase,a in (('attempted',self.attempted),('completed',self.completed))},'pending':self.pending}


class Store:
    def __init__(self,output):
        self.output = Path(output).absolute()
        require(self.output.parent==ROOT/'results' and self.output.name.startswith(TAG+'_')
                and not self.output.exists() and not any(p.is_symlink() for p in (self.output,*self.output.parents)),
                'exclusive new results child')
        self.output.mkdir(exist_ok=False)
        self.started,self.bytes,self.serial,self.previous = time.monotonic(),0,0,None

    def guard(self,reserve=0):
        require(time.monotonic()-self.started<WALL, 'wall cap')
        require(self.bytes+reserve+1024**2<=CAP, 'output cap with final reserve')
        require(shutil.disk_usage(self.output).free>=FLOOR+reserve+1024**2, 'free floor')

    def event(self,kind,value):
        e = dict(sequence=self.serial,previous_sha256=self.previous,kind=kind,value=value)
        data_size = len(json.dumps(e,allow_nan=False).encode())*2+1024
        self.guard(data_size)
        new_sha = digest(e)
        path = self.output/f'event_{self.serial:04d}.json'
        write_new(path,{**e,'event_sha256':new_sha})
        self.previous,self.serial = new_sha,self.serial+1
        self.bytes += path.stat().st_size

    def emit(self,word,step,name,value):
        a = np.asarray(value)
        require(a.dtype==np.float32 and a.size<=max(30,3*PATCHES*FEATURES,3*PATCHES*PROP)
                and np.isfinite(a).all(), 'bounded original array')
        self.guard(a.nbytes+1024)
        path = self.output/f'{word}_{step}_{name}.npy'
        with path.open('xb') as f:
            np.save(f,a,allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
        self.bytes += path.stat().st_size
        rec = dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,shape=list(a.shape),dtype='float32')
        self.event('artifact',dict(key=f'{word}_{step}_{name}',record=rec))
        return rec


def cuda_case(models,arrays,stats,emit):
    import torch
    from app.plan_common.datasets.transforms import make_transforms
    from torch.nn.attention import sdpa_kernel,SDPBackend
    def original(t,shape):
        require(t.is_cuda and t.dtype==torch.float32 and tuple(t.shape)==shape, 'original CUDA float32 ABI')
        a = t.detach().cpu().numpy().copy()
        require(np.isfinite(a).all(), 'finite actual output')
        return a
    with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
        rgb = np.stack([arrays[f'context_rgb_{f:02d}'] for f in (0,5,10)])
        pixels = torch.from_numpy(rgb).permute(0,3,1,2).to('cuda',torch.float32)/255.
        transform = make_transforms(img_size=224,random_horizontal_flip=False,
            random_resize_aspect_ratio=(1.,1.),random_resize_scale=(1.,1.),
            reprob=0.,auto_augment=False,motion_shift=False)
        v = original(models['official_encoder'](transform(pixels),is_training=True)['x_norm_patchtokens'],(3,256,384))
        emit('shared',-1,'visual_features',v)
        props = normalized_proprio(arrays['context_proprio'],stats)
        emit('shared',-1,'normalized_proprio',props)
        p = models['proprio_encoder'](torch.from_numpy(props.copy()).to('cuda').unsqueeze(0))
        require(tuple(p.shape)==(1,3,1,16), 'native proprio feature ABI')
        p = original(p.expand(1,3,256,16),(1,3,256,16))[0]
        emit('shared',-1,'proprio_features',p)
        def predict(v,a,p):
            out = models['predictor'](torch.from_numpy(v.copy()).to('cuda').reshape(1,3,1,16,16,384),
                torch.from_numpy(a.copy()).to('cuda').unsqueeze(0),torch.from_numpy(p.copy()).to('cuda').unsqueeze(0))
            require(len(out)==3 and out[1] is None, 'native AdaLN output tuple')
            return original(out[0],(1,3,256,384))[0],original(out[2],(1,3,256,16))[0]
        return owner.contrasts(owner.rollout(predict,v,p,arrays['history_normalized'],
            {k:owner.normalize(raw,stats) for k,raw in owner.programs().items()},emit))


def run_cuda(m,output):
    arrays = verify(m)
    require(not os.environ.get('PYTHONPATH') and not any(k in sys.modules for k in ('pymunk','pymunk.space')),
            'fresh process without physics overlay')
    store = Store(output)
    r = dict(schema=TAG,status='failed',manifest_sha256=digest(m),protocol=protocol(),
        words=[dict(word=w,status='not_run',error=None) for w in WORDS],artifacts={},errors=[],
        finalization_errors=[],cuda_executed=False,scientific_admission=False,physical_release=False)
    meter = Meter(store.event,store.guard)
    models,handles,v4,before = None,[],None,None
    old_alarm = signal.getsignal(signal.SIGALRM)
    def timeout(signum,frame):
        raise TimeoutError('fixed 600s wall cap; no retry')
    signal.signal(signal.SIGALRM,timeout)
    signal.alarm(WALL)
    try:
        store.guard()
        import torch
        require(torch.cuda.is_available(), 'HOST CUDA required; no CPU neural fallback')
        require(torch.cuda.mem_get_info()[0]>=4*1024**3, '4GiB free CUDA scratch')
        loader = importlib.import_module('experiments.spectral_reachability.'+owner.LOADER)
        require(sha(loader.__file__)==owner.LOADER_SHA, 'sealed loader')
        models,strict,sources,before = loader.load_cuda(ROOT,m['model_spec']['loader_manifest'])
        _,v5,_ = loader.helper(ROOT).dependencies(ROOT)
        v4 = v5.import_v4(ROOT)
        r.update(cuda_executed=True,model_before=before,strict_loads=strict,imported_sources=sources)
        require(set(models)==set(EXPECTED), 'exact frozen modules')
        def pre(key):
            def hook(module,args):
                shape = {'official_encoder':(3,3,224,224),'proprio_encoder':(1,3,4),
                         'predictor':(1,3,1,16,16,384)}[key]
                require(args[0].is_cuda and tuple(args[0].shape)==shape, 'native batch/context ABI')
                meter.before(key,int(args[0].shape[0]))
            return hook
        def post(key):
            return lambda module,args,out:meter.after(key,int(args[0].shape[0]))
        for k,model in models.items():
            handles.extend((model.register_forward_pre_hook(pre(k)),model.register_forward_hook(post(k))))
        def emit(word,step,name,value):
            if word in WORDS and step==0 and name=='visual_in':
                r['words'][WORDS.index(word)]['status']='attempted'
            rec = store.emit(word,step,name,value)
            r['artifacts'][f'{word}_{step}_{name}'] = rec
            if word in WORDS and step==1 and name=='proprio_out':
                r['words'][WORDS.index(word)]['status']='completed'
        r['contrasts'] = cuda_case(models,arrays,m['model_spec']['stats'],emit)
        require(meter.attempted==meter.completed==EXPECTED and meter.pending is None, 'complete resource census')
        r['status']='public_native_action_response_recorded'
    except Exception as exc:
        error = type(exc).__name__+': '+str(exc)
        r['errors'].append(error)
        for row in r['words']:
            if row['status']=='attempted':
                row.update(status='failed',error=error)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM,old_alarm)
        for handle in handles:
            try:
                handle.remove()
            except Exception as exc:
                r['finalization_errors'].append('hook cleanup: '+str(exc))
        if models is not None and v4 is not None:
            try:
                r['model_after']={k:v4.model_receipt(v) for k,v in models.items()}
                require(r['model_after']==before==m['model_spec']['loader_manifest']['expected_model_receipts'], 'model unchanged')
            except Exception as exc:
                r['finalization_errors'].append('model receipt: '+str(exc))
        try:
            verify(m)
            store.guard()
        except Exception as exc:
            r['finalization_errors'].append('source/resources: '+str(exc))
        if r['errors'] or r['finalization_errors']:
            r['status']='failed'
        r.update(meter=meter.snapshot(),wall_seconds=time.monotonic()-store.started,
                 journal_terminal_sha256=store.previous)
        final_bytes = len(json.dumps(r,sort_keys=True,indent=2,allow_nan=False).encode())+1
        require(final_bytes<=1024**2 and store.bytes+final_bytes<=CAP
                and shutil.disk_usage(store.output).free>=FLOOR+final_bytes, 'terminal reserve/floor')
        write_new(store.output/'report.json',r)
    return r


def consume(m,report_path,report_sha):
    arrays = verify(m)
    path = Path(report_path).absolute()
    r = read_json(path,report_sha)
    require(r['schema']==TAG and r['protocol']==protocol() and r['manifest_sha256']==digest(m), 'report contract')
    require(r['status']=='public_native_action_response_recorded' and r['errors']==r['finalization_errors']==[]
            and r['cuda_executed'] is True and r['scientific_admission'] is False
            and r['physical_release'] is False, 'complete actual CUDA required')
    require(r['words']==[dict(word=w,status='completed',error=None) for w in WORDS], 'all4 fixed denominator')
    require(r['model_before']==r['model_after']==m['model_spec']['loader_manifest']['expected_model_receipts'], 'frozen model')
    require(np.isfinite(r['wall_seconds']) and 0<=r['wall_seconds']<WALL, 'wall receipt')
    meter = Meter(lambda k,v:None)
    records,previous = {},None
    events = sorted(path.parent.glob('event_*.json'))
    require(0<len(events)<=128, 'bounded event stream')
    operation_order = []
    for i,p in enumerate(events):
        require(p.name==f'event_{i:04d}.json', 'event gap')
        e = read_json(p)
        require(e['sequence']==i and e['previous_sha256']==previous
                and e['event_sha256']==digest({k:v for k,v in e.items() if k!='event_sha256'}), 'journal chain')
        previous = e['event_sha256']
        if e['kind'] in ('attempted','returned'):
            value = e['value']
            phase = 'attempted' if e['kind']=='attempted' else 'completed'
            changed = [k for k in EXPECTED if value[phase][k]!=getattr(meter,phase)[k]]
            require(len(changed)==1, 'one actual operation/event')
            k = changed[0]
            (meter.before if phase=='attempted' else meter.after)(k,BATCH[k])
            require(value==meter.snapshot(), 'API/sample/frame attempts/returns')
            operation_order.append((e['kind'],k))
        else:
            require(e['kind']=='artifact', 'unknown event')
            v = e['value']
            require(v['key'] not in records, 'duplicate artifact')
            records[v['key']]=v['record']
    expected_order = [k for k in EXPECTED for _ in range(EXPECTED[k])]
    require(operation_order==[(phase,k) for k in expected_order for phase in ('attempted','returned')], 'fixed actual execution order')
    require(previous==r['journal_terminal_sha256'] and records==r['artifacts']
            and meter.snapshot()==r['meter'] and meter.attempted==meter.completed==EXPECTED, 'complete durable census')
    used = set()
    def take(word,step,name,shape):
        key = f'{word}_{step}_{name}'
        rec = records[key]
        require(Path(rec['path'])==path.parent/(key+'.npy'), 'exact local original artifact')
        used.add(key)
        return owner.array(rec,shape)
    visual = take('shared',-1,'visual_features',(3,PATCHES,FEATURES))
    prop = take('shared',-1,'proprio_features',(3,PATCHES,PROP))
    require(np.array_equal(take('shared',-1,'normalized_proprio',(3,4)),
                           normalized_proprio(arrays['context_proprio'],m['model_spec']['stats'])),
            'original frozen proprio preprocessing')
    cursor = [None,None]
    def emit(word,step,name,value):
        cursor[:] = word,step
        require(np.array_equal(take(word,step,name,value.shape),value), 'exact causal action/feedback/context')
    def predict(v,a,p):
        w,s = cursor
        return take(w,s,'visual_out',v.shape),take(w,s,'proprio_out',p.shape)
    terminal = owner.rollout(predict,visual,prop,arrays['history_normalized'],
        {k:owner.normalize(v,m['model_spec']['stats']) for k,v in owner.programs().items()},emit)
    require(used==set(records) and owner.contrasts(terminal)==r['contrasts'], 'all original arrays/contrasts')
    wanted = {p.name for p in events}|{Path(rec['path']).name for rec in records.values()}|{'report.json'}
    require({p.name for p in path.parent.iterdir()}==wanted and path.name=='report.json', 'exact run inventory')
    require(sum(p.stat().st_size for p in path.parent.iterdir())<=CAP, 'saved output cap')
    return dict(schema=TAG+'.consume',status='all4_saved_public_action_words_verified',
        manifest_sha256=digest(m),report_sha256=sha(path),public_payload_sha256=m['public_payload_sha256'],
        contrasts=r['contrasts'],meter=meter.snapshot(),scientific_admission=False,
        physical_release=False,action_witness_established=False,task_success=None,
        interpretation='saved causal prediction consistency, not physical efficacy or noncommutative object capability')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','verify','cuda','consume'))
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--manifest-sha256')
    p.add_argument('--payload',type=Path)
    p.add_argument('--payload-sha256')
    p.add_argument('--report',type=Path)
    p.add_argument('--report-sha256')
    p.add_argument('--output',type=Path)
    p.add_argument('--confirm-host-cuda',action='store_true')
    a = p.parse_args(argv)
    if a.command=='prepare':
        require(a.payload is not None and a.payload_sha256, 'actual public payload path/SHA')
        value = prepare(a.payload,a.payload_sha256)
        write_new(a.manifest,value)
    else:
        require(a.manifest_sha256, 'actual manifest FILE SHA')
        m = read_json(a.manifest,a.manifest_sha256)
        verify(m)
        if a.command=='verify':
            value = dict(status='public_model_plan_verified_non_neural')
        elif a.command=='cuda':
            require(a.confirm_host_cuda and a.output is not None, 'MAIN explicit HOST CUDA confirmation')
            value = run_cuda(m,a.output)
        else:
            require(a.report is not None and a.report_sha256 and a.output is not None, 'actual report SHA/new receipt')
            value = consume(m,a.report,a.report_sha256)
            write_new(a.output,value)
    print(json.dumps(dict(status=value['status'],scientific_admission=False),sort_keys=True))
    return int(value['status']=='failed')


if __name__=='__main__':
    raise SystemExit(main())
