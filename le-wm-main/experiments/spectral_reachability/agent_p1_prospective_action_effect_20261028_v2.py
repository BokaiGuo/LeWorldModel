"""MAIN native replay20 -> CUDA teacher20 -> sealed fixed5 -> factual25.

Single-process production adapter. Import/prepare/consume never instantiate a
model/environment. The worker runs synthetic tests only; MAIN owns run.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from copy import deepcopy
import gzip
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import agent_p1_prospective_action_effect_20261028_v1 as old
from experiments.spectral_reachability import agent_p1_teacher_cross_library_physical_20261028_v1 as phys

NAME='agent_p1_prospective_action_effect_20261028_v2'
ROOT=old.ROOT
teacher=phys.teacher
require,sha,canonical,digest=phys.require,phys.sha,phys.canonical,phys.digest
PARENT=ROOT/'results/agent_p1_teacher_cross_library_physical_20261028_v1_prepare_MAIN02.json'
PARENT_SHA='b125d28c4d931dca7661233d5a81e0da890ef53e2a55ceef55172134d0b08f5a'
CAPS=dict(constructor=40,close=40,reset=80,public_step=1000,physics_step=10160,
          set_state=160,goal_setter=120,goal_render=80,goal_bind=40,explicit_render=3240,
          encode_API=88,predict_API=40)
WALL=3600


def contract():
    return dict(schema=NAME+'.contract',anchors=8,slots=40,pairs=80,known_roots=2,fresh_roots=0,
        teacher15='all actual5/10/15; full v1 saved history rebuild required',
        teacher20='actual10/15/20; actual raw11:20; full3 H1 native model.predict',
        recipe='warm10 + original prefix5 + original model-selected suffix5 + new fixed word5',
        selection='seal all5 float64 identity192 MSE argmin before first extension; smallest index exact ties',
        branches='five independent reset/replay20; leader reused only for word0; no teleport/retry/padding',
        renderer='existing terminal off/on/off triplet, full same native state; no task intervention',
        caps=CAPS,wall_seconds=WALL,output_bytes=phys.CAP,disk_floor_bytes=phys.FLOOR,
        reserve_bytes=phys.FINAL,precision='float32 native tensors; float64 MSE /192 once',
        model_inputs=['RGB','latent_history','normalized_action_history','raw_actions','latent_goal'],
        old_panel=dict(fixed_pairs=80,comparable=76,unknown=4,never_replaced_by_new=True),
        single_process=True,training=False,Qfit=False,oracle_reselection=False,
        P1_admission=False,independent_confirmation=False,equal_FLOPs=False,
        total_FLOPs='not measured; actual API/module/sample counts and native callbacks reported separately')


def validate_request(r):
    require(isinstance(r,dict) and r.get('schema')==NAME+'.forward','own forward schema')
    v=deepcopy(r);v['schema']=old.NAME+'.forward';old.validate_forward(v)


def actual_forward(z,z20,bits):
    return old.forward20(z,z20,bits,teacher.kernel.ACTION_MEAN.tolist(),teacher.kernel.ACTION_STD.tolist())


def selection(predicted,goal):
    p=old.literal(predicted,(5,192)).astype(np.float64);g=old.literal(goal,(192,)).astype(np.float64)
    costs=np.mean((p-g)**2,axis=1).tolist()
    s=dict(predictions=p.astype(np.float32).tolist(),goal=g.astype(np.float32).tolist(),
           costs=costs,selected_word=min(range(5),key=lambda w:(costs[w],w)),
           selection_uses_outcome=False)
    s['seal_sha256']=digest(s);return s


def same_start(a,b):require(canonical(a)==canonical(b),'full native replay20 including signed state/PNG/pixels/raw')


def exact_array(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.dtype.str==b.dtype.str and a.shape==b.shape and a.tobytes()==b.tobytes()


def new_row(job):
    return dict(**deepcopy(job),status='unstarted',public={k:dict(attempted=0,completed=0) for k in ('warmup','prefix','suffix')},
        error=None,secondary_errors=[],root_state=None,goal=None,end_state=None,last_state=None,last_return=None,
        initial_exact=False,history_exact=False,start_verified=False,start_state=None,teacher_history_verified=False,
        reference_frames_verified=0,reference_raw_verified=0,rgb_guard_accepted=False,native_success=None,
        checkpoints=[],missing_checkpoints=[0,5,10],trajectory=[],last_observation=None,PNG=0,actual_latent_Q=None,
        endpoint=None,alternate_endpoint=None,alternate_error=None,replay20=None,terminal_triplet=None)


def production_jobs(package,parent):
    jobs=[]
    for j in package['jobs']:
        for word in range(5):
            job=deepcopy(parent['jobs'][j['first_selected_original_index']]);job.update(index=len(jobs),word=word,
                replay_original_index=j['first_selected_original_index'],extension_raw5_bits=j['words'][word]['raw5_bits'])
            job['job_digest']=digest({k:v for k,v in job.items() if k!='job_digest'});jobs.append(job)
    return jobs


class Store(phys.Store):
    """Original bounded exclusive physical writer, plus original tensor serializer."""
    def __init__(self,path):
        self.path=Path(path).absolute();self.output=self.path
        require(self.path.resolve()==self.path and self.path.name.startswith(NAME+'_') and not self.path.exists()
                and shutil.disk_usage(self.path.parent).free>=phys.FLOOR+phys.CAP,'new own1GiB output +floor')
        self.path.mkdir();phys.adapter.paired.fsync_dir(self.path.parent)
        self.used=0;self.files={};self.previous=None;self.index=0;self.terminal=False;self.drift=False
        self.observations=[]
    def emit(self,kind,**fields):
        super().emit(kind,**fields)
        if kind=='fresh_observation':self.observations.append(deepcopy(dict(kind=kind,**fields)))
    def event(self,event):
        e=deepcopy(event);kind=e.pop('kind');self.emit(kind,**e)
    array=teacher.Store.array


class Meter(phys.Meter):
    def __init__(self,store,started=None):
        super().__init__(store,started)
        self.counts={k:dict(attempted=0,completed=0) for k in CAPS};self.active={k:0 for k in CAPS}
    def call(self,kind,fn,*,cleanup=False,**fields):
        require(kind in CAPS and (not cleanup or kind=='close'),'known callback; close-only cleanup')
        if not cleanup:
            require(not self.fault and time.monotonic()-self.started<WALL,'fault/wall before callback');self.store.check()
        c=self.counts[kind];require(c['attempted']<CAPS[kind],'actual predispatch cap '+kind)
        i=self.index;self.index+=1
        kw=dict(**self.where,operation_index=i,resource='internal_physics_step' if kind=='physics_step' else kind,fields=deepcopy(fields))
        terminal=self.store.terminal
        if cleanup:self.store.terminal=True
        try:
            try:self.store.emit('operation_attempted',**kw)
            except BaseException as e:
                self.fault=True
                if not cleanup:raise
                self.secondary.append(phys.error('close_intent',e))
            c['attempted']+=1;self.active[kind]+=1
            try:result=fn();c['completed']+=1
            except BaseException as e:
                try:self.store.emit('operation_failed',**kw,error=phys.error(kind,e))
                except BaseException:self.fault=True
                raise
            finally:self.active[kind]-=1
            try:self.store.emit('operation_returned',**kw)
            except BaseException as e:
                self.fault=True
                if not cleanup:raise
                self.secondary.append(phys.error('close_return',e))
            return result
        finally:self.store.terminal=terminal


def build():
    v=old.build();parent,p,data=phys.verify(PARENT,PARENT_SHA)
    prep=teacher.read(ROOT/old.INPUTS['teacher_prepare'][0],old.INPUTS['teacher_prepare'][1])
    bindings=deepcopy(parent['bindings'])
    phys.adapter.merge(bindings,{str(PARENT):PARENT_SHA,**v['source_files'],**v['frozen_seams']})
    for item in v['inputs'].values():phys.adapter.merge(bindings,{item['path']:item['sha256']})
    for file in (Path(__file__),ROOT/'tests'/f'test_{NAME}.py',ROOT/'innerdocs'/f'{NAME}.md'):
        phys.adapter.merge(bindings,{str(file.resolve()):sha(file)})
    for file in (Path(prep['policy'])/'config.json',Path(prep['policy'])/'weights.pt'):
        require(str(file) in bindings,'parent binds actual checkpoint file')
    compat=ROOT/'experiments/checkpoint_compat.py'
    phys.adapter.merge(bindings,{str(compat):sha(compat)})
    out=dict(schema=NAME+'.prepare',contract=contract(),jobs=v['jobs'],bindings=bindings,
        policy=prep['policy'],expected_model_hash=prep['expected_model_hash'],parent=str(PARENT),parent_sha256=PARENT_SHA,
        teacher15_actual_context_proof=v['teacher15_actual_context_proof'],P1_admission=False)
    phys.probe.verify_bindings(out);return out,parent,p,data


def verify(path,expected):
    saved=teacher.read(path,expected);now,parent,p,data=build()
    require(canonical(saved)==canonical(now),'whole own prepare reconstruction');return now,parent,p,data


class NativeTrace(teacher.overlay.NativeTrace):
    """Own88/40 module caps; saved actual CUDA float32 inputs/outputs."""
    def __enter__(self):
        def pre(kind):
            def hook(module,args):
                self.check();require(self.scope is not None and module not in self.pending,'scoped native module')
                image='image_index' in self.scope
                require((kind in ('encoder','projector'))==image,'image/forward scopes separated')
                require(sum(e['resource']==kind for e in self.entries)<(88 if image else 40),'own module cap')
                require(not any(e['scope']==self.scope and e['resource']==kind for e in self.entries),'unique module per scope')
                e=dict(index=len(self.entries),resource=kind,scope=deepcopy(self.scope),completed=False)
                self.entries.append(e);self.pending[module]=e;self.store.event(dict(kind='tensor_module_attempt',entry=deepcopy(e)))
                shape={'encoder':(1,3,224,224),'projector':(1,192),'action_encoder':(1,3,10),
                       'predictor':(1,3,192),'pred_proj':(3,192)}[kind]
                require(len(args)==(2 if kind=='predictor' else 1) and tuple(args[0].shape)==shape,'real full3 native tensor ABI')
                if kind=='predictor':require(tuple(args[1].shape)==(1,3,192),'encoded action full3 ABI')
                e['inputs']=[self.save(x,f'module_{e["index"]:04}_input{j}') for j,x in enumerate(args)]
            return hook
        def post(module,args,out):
            e=self.pending.pop(module);require(out is not None,'actual returned module')
            if hasattr(out,'last_hidden_state'):out=out.last_hidden_state
            shapes={'encoder':(1,257,192),'projector':(1,192),'action_encoder':(1,3,192),
                    'predictor':(1,3,192),'pred_proj':(3,192)}
            require(tuple(out.shape)==shapes[e['resource']],'actual output shape')
            e['output']=self.save(out,f'module_{e["index"]:04}_output');self.torch.cuda.synchronize()
            self.check();e['completed']=True;self.store.event(dict(kind='tensor_module_completed',entry=deepcopy(e)))
        try:
            for kind in ('encoder','projector','action_encoder','predictor','pred_proj'):
                module=getattr(self.model,kind);self.hooks.extend((module.register_forward_pre_hook(pre(kind)),module.register_forward_hook(post)))
            return self
        except BaseException:self.__exit__(None,None,None);raise


def loaded_model(p,torch,device,store):
    """Same bytes hashed and consumed by JSON/torch; no separate loader reopen."""
    from hydra.utils import instantiate
    from experiments import checkpoint_compat as compat
    blobs={};files=[]
    for name in ('config.json','weights.pt'):
        path=Path(p['policy'])/name;store.emit('load_intent',path=str(path),expected_sha256=p['bindings'][str(path)])
        require(path.is_file() and not path.is_symlink(),'regular checkpoint')
        raw=path.read_bytes();h=hashlib.sha256(raw).hexdigest();require(h==p['bindings'][str(path)],'same consumed checkpoint bytes')
        blobs[name]=raw;files.append(dict(path=str(path),sha256=h,bytes=len(raw),same_buffer=True))
        store.emit('consumed_buffer_frozen',receipt=files[-1])
    model=instantiate(json.loads(blobs['config.json']))
    state=torch.load(BytesIO(blobs['weights.pt']),map_location='cpu',weights_only=True)
    remapped,changed=compat.remap_legacy_vit_keys(state);missing,unexpected=model.load_state_dict(remapped,strict=False)
    require(not missing and not unexpected,'original compatible strict tensor load')
    model=model.to(device).eval().requires_grad_(False);model.interpolate_pos_encoding=True
    teacher.enc.frozen(model);before=teacher.enc.tensor_hash(model);require(before==p['expected_model_hash'],'frozen tensor identity')
    store.put('loader.json',dict(files=files,remapped_keys=changed,missing=[],unexpected=[],device=str(device),tensor_hash=before))
    return model,before


class CudaModel:
    """No environment/evaluator handles are passed to these model methods."""
    def __init__(self,model,torch,trace,store,meter):
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        self.model,self.torch,self.trace,self.store,self.meter=model,torch,trace,store,meter
        self.p0=p0;self.transform=p0.image_transform();self.images=[];self.forwards=[]
    def encode(self,pixels,label):
        a=np.asarray(pixels);require(a.shape==(224,224,3) and a.dtype==np.uint8,'only RGB model input')
        i=len(self.images);e=dict(index=i,label=label,completed=False,pixels=self.store.array(f'image{i}_pixels',a))
        self.images.append(e);self.store.emit('image_attempt',entry=deepcopy(e))
        with teacher.overlay.traced_scope(self.trace,dict(image_index=i),e),self.torch.inference_mode():
            x=self.p0._image_tensor(a,transform=self.transform,device=self.torch.device('cuda:0'))
            e['preprocessed']=self.trace.save(x,f'image{i}_preprocessed')
            result=self.meter.call('encode_API',lambda:self.model.encode({'pixels':x}),image_index=i)
            z=result['emb'][0,0].detach().cpu().numpy().copy()
        teacher.enc.native(z);e['latent']=self.store.array(f'image{i}_latent',z)
        e['completed']=True;self.store.emit('image_return',entry=deepcopy(e));return z.tolist()
    def predict(self,r):
        validate_request(r);e=dict(index=len(self.forwards),request=deepcopy(r),completed=False);self.forwards.append(e)
        self.store.emit('forward_attempt',entry=deepcopy(e))
        with teacher.overlay.traced_scope(self.trace,dict(forward_index=e['index']),e),self.torch.inference_mode():
            z=self.torch.tensor(r['latents'],device='cuda:0',dtype=self.torch.float32)[None]
            outgoing=np.concatenate((old.literal(r['history'],(2,10)),teacher.kernel.normalized_history(old.rawbits(r['raw5_bits']))))
            a=self.torch.tensor(outgoing,device='cuda:0',dtype=self.torch.float32)[None]
            encoded_actions=self.model.action_encoder(a)
            out=self.meter.call('predict_API',lambda:self.model.predict(z,encoded_actions),forward_index=e['index'])
            require(tuple(out.shape)==(1,3,192),'real predict full3 output')
            e['full_output']=self.trace.save(out,f'forward{e["index"]}_full_output')
            endpoint=out[0,-1].detach().cpu().numpy().copy()
        teacher.enc.native(endpoint);e['endpoint']=endpoint.tolist()
        e['completed']=True;self.store.emit('forward_return',entry=deepcopy(e));return endpoint.tolist()


def validate_triplet(trip):
    require(len(trip)==3 and [e['receipt']['overlay_index'] for e in trip]==[0,1,2],'complete off/on/off')
    fixed=trip[0]['pre'];state=trip[0]['receipt']['state'];snap=trip[0]['receipt']['snapshot']
    for i,e in enumerate(trip):
        r=e['receipt'];require(canonical(e['pre'])==canonical(e['post'])==canonical(fixed) and
            canonical(r['state'])==canonical(state) and canonical(r['snapshot'])==canonical(snap) and
            r['condition']==('on' if i==1 else 'off'),'same-state render invariants')
    require(all(trip[0]['receipt'][k]==trip[2]['receipt'][k] for k in ('file_sha256','pixel_sha256')),'off restore exact')


def pixels(store,receipt):
    return teacher.rgb(store.path/receipt['png'],receipt['file_sha256'],receipt['pixel_sha256'])


def replay20(env,parent,p,data,j,row,meter,store):
    ai=j['anchor_index'];original=parent['jobs'][j['first_selected_original_index']]
    phys.episode(env,data['roots'][j['source_slot']],data['source_plan'],p['tasks'][original['task']],original,
                 parent['references'][ai],parent['teacher_inputs'][ai],parent['anchors'][ai],row,meter,store)
    selected=row['trajectory'][-1]
    raw=np.concatenate((phys.moving.warmup_commands(),old.rawbits(j['prefix5_bits']),old.rawbits(j['first_suffix5_bits'])))
    f=dict(raw20_bits=raw.tobytes().hex(),state7=deepcopy(selected['state']),snapshot=deepcopy(selected['snapshot']),
           png_sha256=selected['file_sha256'],pixel_sha256=selected['pixel_sha256'])
    row['replay20']=deepcopy(f);store.emit('replay20_ready',**meter.where,fingerprint=f)
    return pixels(store,selected),f


def extend25(env,j,word,row,meter,store,backend,seal):
    require(seal is not None,'no new outcome before model seal')
    goal=np.asarray(row['goal'],dtype=np.float64);body=env.unwrapped
    guard=phys.rgb_guard.RgbObservabilityGuard(root_id=row['root_sha256'],agent_rgb=(65,105,225))
    image=pixels(store,row['trajectory'][-1]);guard.observe(image,primitive_step=0)
    fixed=phys.invariant(body);metadata={k:v for k,v in fixed.items() if k!='snapshot'}
    store.emit('extension_started',**meter.where,seal_sha256=seal['seal_sha256'],raw5_bits=j['words'][word]['raw5_bits'])
    for i,a in enumerate(old.rawbits(j['words'][word]['raw5_bits']),1):
        require(canonical(phys.invariant(body))==canonical(fixed),'no task/physical change before dispatch')
        obs,t,tr=phys.dispatch(env,a,'suffix',5+i,row,meter,store,goal,(guard,'extension'))
        image,frame=phys.observe(env,obs,'suffix',5+i,row,meter,store,None,(guard,'extension',i))
        fixed=phys.invariant(body)
        require(canonical({k:v for k,v in fixed.items() if k!='snapshot'})==canonical(metadata),'same actual20 goal/renderer/task through25')
        if (t or tr) and i<5:raise phys.helper.NativeStop('early extension stop; factual25 unknown')
    trip=deepcopy(store.observations[-3:]);validate_triplet(trip);row['terminal_triplet']=trip
    original=trip[1 if j['condition']=='on' else 0]['receipt'];alternate=trip[0 if j['condition']=='on' else 1]['receipt']
    row['endpoint']=backend.encode(pixels(store,original),f'a{j["anchor_index"]}w{word}_original25')
    row.update(status='terminal25_original_encoded',end_state=deepcopy(frame['state']),terminal_return=deepcopy(row['last_return']))
    try:
        row['alternate_endpoint']=backend.encode(pixels(store,alternate),f'a{j["anchor_index"]}w{word}_alternate25')
        row['status']='terminal25_encoded'
    except Exception as exc:
        row['alternate_error']=phys.error('alternate_encode',exc)
        store.emit('alternate_encode_failed',**meter.where,error=row['alternate_error'])
    store.emit('terminal25_encoded',**meter.where,endpoint=row['endpoint'],alternate_endpoint=row['alternate_endpoint'],
               terminal_return=row['terminal_return'])


def native_factory():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    spec=gym.spec('swm/PushT-v1')
    require(spec.entry_point=='stable_worldmodel.envs.pusht.env:PushT' and spec.max_episode_steps is None and spec.additional_wrappers==(),
            'original inert native constructor lineage')
    return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False,render_action=False)


def execute_panel(package,parent,p,data,store,meter,backend,rows,anchors,factory=native_factory):
    """Actual physical adapter; injectable boundary is used only in tests."""
    seen=[]
    for j,anchor in zip(package['jobs'],anchors,strict=True):
        seal=None;leader=None
        for word in range(5):
            row=rows[j['anchor_index']*5+word];meter.where=dict(anchor_index=j['anchor_index'],word=word,job_digest=row['job_digest'])
            env=None
            try:
                if word and seal is None:row['status']='unknown_unsealed_anchor';continue
                env=meter.call('constructor',factory)
                require(all(env is not oldenv and env.unwrapped is not oldenv.unwrapped for oldenv in seen),'fresh independent environment each branch')
                seen.append(env);phys.validate_env(env.unwrapped)
                require(env.unwrapped.render_action is False and env.unwrapped.render_size==224 and env.unwrapped.render_mode=='rgb_array','actual renderer224')
                with phys.native_hooks(env,meter,row['secondary_errors']):
                    rgb,fingerprint=replay20(env,parent,p,data,j,row,meter,store)
                    if word==0:
                        leader=deepcopy(fingerprint);anchor['replay20']=leader
                        z20=backend.encode(rgb,f'a{j["anchor_index"]}_actual20')
                        z15=parent['teacher_inputs'][j['anchor_index']]['context']['latent_history']
                        forward=actual_forward(z15,z20,j['prefix5_bits']+j['first_suffix5_bits'])
                        anchor.update(forward=forward,actual20=z20,replay20=leader)
                        predicted=[]
                        for w in range(5):
                            request=dict(schema=NAME+'.forward',anchor_index=j['anchor_index'],word=w,
                                         **deepcopy(forward),raw5_bits=j['words'][w]['raw5_bits'])
                            predicted.append(backend.predict(request))
                        seal=selection(predicted,parent['anchors'][j['anchor_index']]['goal'])
                        store.put(f'seal_{j["anchor_index"]}.json',seal)
                        store.emit('model_selection_sealed',anchor_index=j['anchor_index'],seal=seal)
                        anchor['seal']=deepcopy(seal)
                    else:same_start(leader,fingerprint)
                    extend25(env,j,word,row,meter,store,backend,seal)
                require(not row['secondary_errors'],'native hook cleanup retained')
            except Exception as e:
                row.update(status='unknown',error=phys.error('branch',e),endpoint=None,alternate_endpoint=None,end_state=None)
                store.emit('branch_error',**meter.where,error=row['error'])
            finally:
                if env is not None:
                    try:meter.call('close',env.close,cleanup=True)
                    except BaseException as e:
                        row['secondary_errors'].append(phys.error('close',e));row.update(status='unknown',endpoint=None,alternate_endpoint=None)
                store.put(f'branch_{row["index"]:02}.json',row)
            if meter.fault:raise RuntimeError('durability/native ledger fault; stop all dispatch')


def run(prepared,expected,output):
    start=time.monotonic();package,parent,p,data=verify(prepared,expected);store=Store(output);meter=Meter(store,start)
    rows=[new_row(j) for j in production_jobs(package,parent)];anchors=[dict(anchor_index=i,seal=None) for i in range(8)]
    errors=[];model=None;trace=None;backend=None;before=after=None;source_ok=False;host=None
    binding=dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected)
    try:
        store.put('binding.json',binding);store.emit('run_started',binding=binding,contract=contract())
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model,before=loaded_model(package,torch,device,store)
        def check():require(time.monotonic()-start<WALL,'3600s wall including validation/load');store.check()
        trace=NativeTrace(model,torch,store,check,[1,257,192]);backend=CudaModel(model,torch,trace,store,meter)
        with sdpa_kernel(SDPBackend.MATH),torch.autocast(device_type='cuda',enabled=False),trace:
            execute_panel(package,parent,p,data,store,meter,backend,rows,anchors)
    except BaseException as e:errors.append(phys.error('execution',e))
    finally:
        if model is not None:
            try:teacher.enc.frozen(model);after=teacher.enc.tensor_hash(model)
            except BaseException as e:errors.append(phys.error('final_model',e))
        try:phys.probe.verify_bindings(package);source_ok=True
        except BaseException as e:errors.append(phys.error('final_bindings',e))
        errors.extend(meter.secondary);store.terminal=True
        complete=not errors and source_ok and before==after==package['expected_model_hash'] and all(r['status']=='terminal25_encoded' for r in rows)
        report=dict(schema=NAME+'.report',status='complete_unadmitted' if complete else 'partial',binding=binding,
            contract=contract(),rows=rows,anchors=anchors,images=backend.images if backend else [],forwards=backend.forwards if backend else [],
            native_module_attempts=trace.entries if trace else [],counts=meter.receipt(),errors=errors,
            module_counts=module_counts(trace.entries if trace else []),
            source_verified=source_ok,model_hash_before=before,model_hash_after=after,host=host,
            wall_seconds=time.monotonic()-start,total_FLOPs=None,P1_admission=False,artifacts={})
        try:
            store.emit('run_finished',status=report['status'],counts=report['counts'],errors=errors)
            report['artifacts']=store.census();report['output_bytes_before_report']=store.used
            store.put('report.json',report,final=True);store.census()
        except BaseException as e:
            errors.append(phys.error('final_report',e));report['status']='partial'
            try:store.put('terminal_failure.json',dict(errors=errors,counts=meter.receipt(),rows=rows),final=True)
            except BaseException:pass
    return dict(output=str(store.path),report_sha256=sha(store.path/'report.json') if (store.path/'report.json').is_file() else None,
                status=report['status'],errors=errors)


def load_array(output,r):
    file=Path(output)/r['file'];require(file.parent==Path(output) and not file.is_symlink() and sha(file)==r['file_sha256'],'tensor file binding')
    with gzip.open(file,'rb') as f:raw=f.read(4*1024**2+1)
    require(len(raw)<=4*1024**2 and hashlib.sha256(raw).hexdigest()==r['npy_sha256'],'bounded NPY binding')
    a=np.load(BytesIO(raw),allow_pickle=False)
    require(str(a.dtype)==r['dtype'] and list(a.shape)==r['shape'] and np.isfinite(a).all() and
            hashlib.sha256(a.tobytes()).hexdigest()==r['array_sha256'],'actual tensor values/dtype/shape binding');return a


def read_events(path):
    require(Path(path).stat().st_size<=phys.JOURNAL_CAP,'bounded journal');events=[];previous=None
    with Path(path).open('rb') as f:
        for line in f:
            require(line.endswith(b'\n'),'complete durable line')
            e=phys.adapter.decode(line);require(line==canonical(e)+b'\n','canonical journal line');h=e.pop('event_sha256')
            require(e['index']==len(events) and e['previous_sha256']==previous and digest(e)==h,'journal chain')
            e['event_sha256']=h;events.append(e);previous=h
    return events


def count_operations(events):
    counts={};attempts={};returned=set();failed=set()
    for e in events:
        if e['kind'] not in ('operation_attempted','operation_returned','operation_failed'):continue
        i=e['operation_index'];k=e['resource'];require(k in CAPS or k=='internal_physics_step','known actual resource')
        identity={x:e.get(x) for x in ('resource','fields','anchor_index','word','job_digest')}
        c=counts.setdefault(k,dict(attempted=0,completed=0))
        if e['kind']=='operation_attempted':
            require(i not in attempts and i==len(attempts),'unique contiguous callback attempts');attempts[i]=identity;c['attempted']+=1
        else:
            require(i in attempts and attempts[i]==identity and i not in returned and i not in failed,'paired callback identity')
            if e['kind']=='operation_returned':returned.add(i);c['completed']+=1
            else:failed.add(i)
    for k,c in counts.items():require(c['attempted']<=CAPS['physics_step' if k=='internal_physics_step' else k],'actual cap')
    return counts


def module_counts(entries):
    out={}
    for e in entries:
        c=out.setdefault(e['resource'],dict(attempted=0,completed=0));c['attempted']+=1;c['completed']+=bool(e['completed'])
    return out


def validate_scopes(report,events):
    known={row['job_digest']:(row['anchor_index'],row['word']) for row in report['rows']}
    require(len(known)==len(report['rows']),'unique owned physical branches')
    for e in events:
        if 'job_digest' in e:
            require(e['job_digest'] in known and known[e['job_digest']]==(e.get('anchor_index'),e.get('word')),'no foreign or crossed physical/model scope')


def native20_projection(rows,events):
    """Read-only projection for ORIGINAL physical validators, not new evidence.

    The original20 calls/51 references/guards are validated by their unchanged
    saved consumer. New extension/model events remain in the v2 full ledger.
    Operation indices are reindexed only in this explicitly derived view.
    """
    cut={e['job_digest']:e['index'] for e in events if e['kind']=='replay20_ready'}
    filtered=[];opids={}
    for e in events:
        if 'job_digest' not in e or e.get('resource') in ('encode_API','predict_API'):continue
        if e['job_digest'] in cut and e['index']>cut[e['job_digest']]:continue
        e=deepcopy(e)
        if 'operation_index' in e:
            oldid=e['operation_index']
            if e['kind']=='operation_attempted':opids[oldid]=len(opids)
            require(oldid in opids,'native projection operation pairing');e['operation_index']=opids[oldid]
        filtered.append(e)
    projected=[]
    for original in rows:
        row=deepcopy(original);ready=row['replay20'] is not None
        row['status']='horizon_completed' if ready else ('unstarted' if row['status']=='unstarted' else 'failed')
        row['end_state']=deepcopy(row['replay20']['state7']) if ready else None
        if ready:
            row['error']=None;row['PNG']=66;row['trajectory']=row['trajectory'][:22]
            row['public']['suffix']=dict(attempted=5,completed=5)
        projected.append(row)
    return projected,filtered


def validate_native20(parent,p,report,events,output):
    projected,filtered=native20_projection(report['rows'],events)
    jobs=[]
    for row in projected:
        original=parent['jobs'][row['replay_original_index']]
        jobs.append({k:deepcopy(row[k]) for k in original})
    pkg=dict(parent,jobs=jobs)
    start=dict(kind='run_started',contract=phys.contract(),execution_binding={})
    view=dict(schema=phys.NAME+'.report',status='partial',contract=phys.contract(),fixed_denominator=40,rows=projected,
        P1_admission=False,C2_non_greedy_premise=False,new_neural_calls=0,execution_binding={},counts=count_operations(filtered))
    phys.validate_trace(pkg,p,view,[start]+filtered)
    phys.validate_payloads(pkg,view,filtered,lambda name:(output/name).read_bytes())


def native_labels(goal,state):
    """Literal saved-only mirror of frozen PushT.eval_state, not a new metric."""
    g=phys.source.vector(goal);s=phys.source.vector(state)
    pos=np.linalg.norm(g[:4]-s[:4]);angle=np.abs(g[4]-s[4]);angle=np.minimum(angle,2*np.pi-angle)
    return dict(native_success=bool(pos<20 and angle<np.pi/9),native_distance=float(np.linalg.norm(g-s)))


def consume(output,report_sha):
    """Saved-only audit/attribution. No model loading, physics or new selector."""
    output=Path(output).absolute();r=teacher.read(output/'report.json',report_sha,phys.REPORT_CAP)
    package,parent,p,data=verify(r['binding']['prepared'],r['binding']['prepare_sha256'])
    require(r['schema']==NAME+'.report' and r['contract']==contract() and r['P1_admission'] is False and
            len(r['rows'])==40 and len(r['anchors'])==8,'fixed new40/80 and own schema')
    for row,job in zip(r['rows'],production_jobs(package,parent),strict=True):
        require(all(canonical(row[k])==canonical(v) for k,v in job.items()),'new immutable jobs rebuilt from actual sealed source, not report')
    require(set(x.name for x in output.iterdir())==set(r['artifacts'])|{'report.json'},'exact complete artifact census')
    for name,h in r['artifacts'].items():
        require(Path(name).name==name and not (output/name).is_symlink() and sha(output/name)==h,'all saved artifact SHA')
    require(sum((output/n).stat().st_size for n in r['artifacts'])==r['output_bytes_before_report'],'actual output bytes')
    events=read_events(output/'journal.jsonl');require(count_operations(events)==r['counts'],'reconstructed actual callback ledger')
    validate_scopes(r,events)
    require(events[0]['kind']=='run_started' and events[0]['binding']==r['binding'] and events[-1]['kind']=='run_finished' and
            events[-1]['counts']==r['counts'] and events[-1]['status']==r['status'] and events[-1]['errors']==r['errors'] and
            teacher.read(output/'binding.json')==r['binding'],'durable start/end/binding')
    validate_native20(parent,p,r,events,output)
    require(module_counts(r['native_module_attempts'])==r['module_counts'],'actual module ledger distinct from API counts')
    # Every saved image/forward/module return must agree with the independent journal.
    for field,attempt,returned in [('images','image_attempt','image_return'),('forwards','forward_attempt','forward_return'),
                                  ('native_module_attempts','tensor_module_attempt','tensor_module_completed')]:
        entries=r[field];ae=[e['entry'] for e in events if e['kind']==attempt];re=[e['entry'] for e in events if e['kind']==returned]
        require([e['index'] for e in ae]==list(range(len(entries))),'all actual attempts retained')
        require(re==[e for e in entries if e['completed']],'actual returned entries, not fabricated counts')
    for image in r['images']:
        if image['completed']:require(image['modules']==[e for e in r['native_module_attempts'] if e['scope']==dict(image_index=image['index'])],'global/image module trace join')
    for f in r['forwards']:
        if f['completed']:require(f['modules']==[e for e in r['native_module_attempts'] if e['scope']==dict(forward_index=f['index'])],'global/forward module trace join')
    for entry in r['native_module_attempts']:
        require(entry['resource'] in ('encoder','projector','action_encoder','predictor','pred_proj'),'known native module')
        for receipt in entry.get('inputs',[])+([entry['output']] if 'output' in entry else []):
            require(load_array(output,receipt).dtype==np.float32,'original module float32')
    image_by_label={}
    for image in r['images']:
        if not image['completed']:continue
        require(image['label'] not in image_by_label,'unique image label');image_by_label[image['label']]=image
        a=load_array(output,image['pixels']);z=load_array(output,image['latent']);modules=image['modules']
        require(a.shape==(224,224,3) and a.dtype==np.uint8 and z.shape==(192,) and z.dtype==np.float32,'RGB/latent ABI')
        require([e['resource'] for e in modules]==['encoder','projector'],'one actual image path')
        enc,proj=modules
        require(exact_array(load_array(output,image['preprocessed']).reshape(1,3,224,224),load_array(output,enc['inputs'][0])),'image tensor -> encoder')
        require(exact_array(load_array(output,enc['output'])[:,0],load_array(output,proj['inputs'][0])) and
                exact_array(load_array(output,proj['output'])[0],z),'CLS -> projector -> latent bridge')
    for f in r['forwards']:
        validate_request(f['request'])
        if not f['completed']:continue
        req=f['request'];modules=f['modules'];require([e['resource'] for e in modules]==['action_encoder','predictor','pred_proj'],'one full3 H1')
        ae,pr,pp=modules
        a=np.concatenate((old.literal(req['history'],(2,10)),teacher.kernel.normalized_history(old.rawbits(req['raw5_bits']))))[None]
        require(exact_array(load_array(output,ae['inputs'][0]),a) and
            exact_array(load_array(output,pr['inputs'][0]),old.literal(req['latents'],(3,192))[None]) and
            exact_array(load_array(output,pr['inputs'][1]),load_array(output,ae['output'])),'request -> action/latent -> predictor')
        require(exact_array(load_array(output,pr['output']).reshape(3,192),load_array(output,pp['inputs'][0])) and
            exact_array(load_array(output,pp['output']).reshape(1,3,192),load_array(output,f['full_output'])) and
            exact_array(load_array(output,f['full_output'])[0,-1],old.literal(f['endpoint'],(192,))),'predictor -> pred_proj -> full endpoint')
    analyses=[]
    for ai,(j,anchor) in enumerate(zip(package['jobs'],r['anchors'],strict=True)):
        rows=r['rows'][ai*5:ai*5+5];seal=anchor['seal'];predicted=None
        if seal is not None:
            predicted=seal['predictions'];require(selection(predicted,parent['anchors'][ai]['goal'])==seal,'recompute model-only stable seal')
            se=[e for e in events if e['kind']=='model_selection_sealed' and e['anchor_index']==ai]
            require(len(se)==1 and se[0]['seal']==seal and teacher.read(output/f'seal_{ai}.json')==seal,'durable unique seal')
            fs=[f for f in r['forwards'] if f['request']['anchor_index']==ai]
            require(len(fs)==5 and all(f['completed'] and f['request']['word']==w and f['request']['raw5_bits']==j['words'][w]['raw5_bits'] and
                f['endpoint']==predicted[w] for w,f in enumerate(fs)),'sealed original five raw actions, no extra predict')
            returned=[e for e in events if e['kind']=='forward_return' and e['entry']['request']['anchor_index']==ai]
            require(len(returned)==5 and all(e['index']<se[0]['index'] for e in returned),'all model returns before seal')
            z20=load_array(output,image_by_label[f'a{ai}_actual20']['latent']).tolist()
            require(actual_forward(parent['teacher_inputs'][ai]['context']['latent_history'],z20,j['prefix5_bits']+j['first_suffix5_bits'])==anchor['forward'] and
                all(all(canonical(f['request'][k])==canonical(anchor['forward'][k]) for k in ('latents','history')) for f in fs),'actual20 context/history exact')
            leader=r['rows'][ai*5]['trajectory'][21]
            require(exact_array(load_array(output,image_by_label[f'a{ai}_actual20']['pixels']),teacher.rgb(output/leader['png'],leader['file_sha256'],leader['pixel_sha256'])),
                    'actual encoded20 is exact replay20 RGB, not true-mid15 again')
            for e in events:
                if e['kind']=='extension_started' and e['anchor_index']==ai:
                    require(e['index']>se[0]['index'] and e['seal_sha256']==seal['seal_sha256'] and
                            e['raw5_bits']==j['words'][e['word']]['raw5_bits'],'strict seal before any outcome')
        for word,row in enumerate(rows):
            require(row['index']==ai*5+word and row['word']==word and row['anchor_index']==ai,'all fixed row identities')
            path=output/f'branch_{row["index"]:02}.json'
            if path.exists():require(teacher.read(path)==row,'durable branch agrees')
            if row['replay20'] is not None:
                same_start(anchor['replay20'],row['replay20'])
                ready=[e for e in events if e['kind']=='replay20_ready' and e.get('job_digest')==row['job_digest']]
                frame=row['trajectory'][21];raw20=phys.expected_raw(parent['jobs'][j['first_selected_original_index']])
                recomputed=dict(raw20_bits=b''.join(a.tobytes() for _,_,a in raw20).hex(),state7=frame['state'],snapshot=frame['snapshot'],
                    png_sha256=frame['file_sha256'],pixel_sha256=frame['pixel_sha256'])
                same_start(recomputed,row['replay20'])
                require(len(ready)==1 and canonical(ready[0]['fingerprint'])==canonical(recomputed),'actual20 ready event and complete start')
                require(row['reference_frames_verified']==51 and row['reference_raw_verified']==15 and
                    row['teacher_history_verified'] is True,'actual teacher15 physical replay proofs')
            if row['status'] in ('terminal25_encoded','terminal25_original_encoded'):
                require(seal is not None and not row['secondary_errors'] and row['error'] is None,'known terminal requires clean sealed branch')
                trip=row['terminal_triplet'];validate_triplet(trip)
                for e in trip:teacher.rgb(output/e['receipt']['png'],e['receipt']['file_sha256'],e['receipt']['pixel_sha256'])
                raw=[e for e in events if e['kind']=='raw_return' and e.get('anchor_index')==ai and e.get('word')==word]
                expected=phys.expected_raw(parent['jobs'][j['first_selected_original_index']])+[
                    ('suffix',i+6,a) for i,a in enumerate(old.rawbits(j['words'][word]['raw5_bits']))]
                require(len(raw)==25 and all((e['phase'],e['step'],e['raw_bits'],e['actual_bits'],e['actual_dtype'])==
                    (phase,step,a.tobytes().hex(),a.tobytes().hex(),'<f4') for e,(phase,step,a) in zip(raw,expected,strict=True)),
                    'all25 exact raw actions and chronology')
                require(all(not e['terminated'] and not e['truncated'] for e in raw[:-1]),'no padding after native stop')
                scoped=[e for e in events if e.get('job_digest')==row['job_digest']]
                intents=[e for e in scoped if e['kind']=='operation_attempted' and e['resource']=='public_step']
                require(len(intents)==25 and all(e['fields']==dict(phase=phase,step=step,raw_bits=a.tobytes().hex()) for e,(phase,step,a) in zip(intents,expected,strict=True)),
                    'actual25 public dispatch intents, no hidden extra action')
                frames=[e for e in scoped if e['kind']=='fresh_observation']
                require(len(frames)==81 and row['PNG']==81 and len(row['trajectory'])==27,'all27 actual view triplets')
                for k in range(0,len(frames),3):
                    validate_triplet(frames[k:k+3])
                    for e in frames[k:k+3]:
                        fr=e['receipt'];teacher.rgb(output/fr['png'],fr['file_sha256'],fr['pixel_sha256'])
                        require(canonical(fr['snapshot'])==canonical(e['post']['snapshot']) and canonical(fr['state'])==canonical(fr['snapshot']['state']),
                                'fresh rendered full native state/snapshot join')
                require(canonical(frames[-3:])==canonical(trip) and
                    canonical([e['receipt'] for e in frames if e['receipt']['overlay_index']==(1 if j['condition']=='on' else 0)])==canonical(row['trajectory']),
                    'all trajectory frames and terminal triplet join global journal')
                for e in raw:
                    labels=native_labels(row['goal'],e['state'])
                    require(all(e[k]==v for k,v in labels.items()) and canonical(e['snapshot']['state'])==canonical(e['state']),
                            'same-task official native flags and full7 distance, including velocity')
                last=raw[-1];terminal=row['terminal_return']
                require(all(terminal[k]==last[k] for k in ('phase','step','state','reward','terminated','truncated','native_success','native_distance')) and
                    terminal['diagnostics']==phys.pose.pose_diagnostics(row['goal'],last['state']) and
                    canonical(row['end_state'])==canonical(last['state'])==canonical(trip[0]['receipt']['state']),
                    'actual25 terminal labels/state, not stale20 outcomes')
                guard=row['guards']['extension']
                require(guard['primitive_commands_authorized']==5 and guard['abstained'] is False and
                    [o['primitive_step'] for o in guard['observations']]==list(range(6)) and all(o['canonical_agent_pixels']>0 for o in guard['observations']),
                    'actual RGB extension guard all five commands')
                for kind,key in [('original','endpoint'),('alternate','alternate_endpoint')]:
                    if key=='alternate_endpoint' and row[key] is None:
                        require(row['status']=='terminal25_original_encoded' and row['alternate_error'] is not None,'explicit alternate-only failure')
                        continue
                    im=image_by_label[f'a{ai}w{word}_{kind}25'];latent=load_array(output,im['latent']).tolist()
                    view=1 if (j['condition']=='on')==(kind=='original') else 0
                    require(canonical(row[key])==canonical(latent) and exact_array(load_array(output,im['pixels']),
                        teacher.rgb(output/trip[view]['receipt']['png'],trip[view]['receipt']['file_sha256'],trip[view]['receipt']['pixel_sha256'])),
                        'same-state actual rendered endpoint encoding')
            else:require(row['endpoint'] is row['alternate_endpoint'] is None,'unknown never recycled as observed')
        actual=[x['endpoint'] for x in rows];alternate=[x['alternate_endpoint'] for x in rows]
        analyses.append(dict(anchor_index=ai,selected_word=None if seal is None else seal['selected_word'],
            closed_loop_return=None if seal is None or rows[seal['selected_word']]['status'] not in ('terminal25_encoded','terminal25_original_encoded') else rows[seal['selected_word']].get('terminal_return'),
            effects=None if predicted is None else old.effects(predicted,actual,parent['anchors'][ai]['goal']),
            renderer=old.renderer_effects(actual,alternate)))
    available=sum(sum(p['available'] for p in a['effects']['pairs']) for a in analyses if a['effects'] is not None)
    complete=r['status']=='complete_unadmitted'
    if complete:
        require(not r['errors'] and r['source_verified'] and r['model_hash_before']==r['model_hash_after']==package['expected_model_hash'] and
            all(row['status']=='terminal25_encoded' for row in r['rows']) and r['counts']['predict_API']==dict(attempted=40,completed=40) and
            r['counts']['encode_API']==dict(attempted=88,completed=88),'whole complete resource/model/source closure')
        require(r['counts']=={('internal_physics_step' if k=='physics_step' else k):dict(attempted=v,completed=v) for k,v in CAPS.items()} and
            r['module_counts']=={k:dict(attempted=v,completed=v) for k,v in dict(encoder=88,projector=88,action_encoder=40,predictor=40,pred_proj=40).items()} and
            r['wall_seconds']<WALL,'complete exact actual callback/module axes; not FLOPs equality')
        loader=teacher.read(output/'loader.json');require(loader['tensor_hash']==package['expected_model_hash'] and loader['device']=='cuda:0' and
            loader['missing']==loader['unexpected']==[] and len(loader['files'])==2,'actual frozen loader')
        for f in loader['files']:require(f['same_buffer'] and f['sha256']==package['bindings'][f['path']] and f['bytes']==Path(f['path']).stat().st_size,'same-buffer loadedbytes')
    return dict(schema=NAME+'.consume',status='saved_adapter_audited_not_P1_admission' if complete else 'partial_saved_audit',
        report_sha256=report_sha,fixed_slots=40,fixed_pairs=80,available_pairs=available,unknown_pairs=80-available,
        analyses=analyses,actual_counts=r['counts'],total_FLOPs=None,P1_admission=False,
        interpretation='common-zero bias vs action-conditioned error and renderer effects; not unique hidden-state cause or deployed correction')


def write_json(path,value):
    path=Path(path).absolute();require(path.resolve()==path and path.name.startswith(NAME+'_') and path.suffix=='.json','own exclusive JSON stem')
    raw=canonical(value);require(len(raw)<=phys.PREP_CAP,'bounded JSON')
    with path.open('xb') as f:require(f.write(raw)==len(raw),'short write');f.flush();os.fsync(f.fileno())
    phys.adapter.paired.fsync_dir(path.parent);return dict(path=str(path),sha256=sha(path))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','run','consume','verify'))
    for name in ('prepared','prepare-sha','output','report-sha','receipt'):parser.add_argument('--'+name)
    args=parser.parse_args()
    if args.command=='prepare':require(args.output,'--output');out=write_json(args.output,build()[0])
    elif args.command=='verify':verify(args.prepared,args.prepare_sha);out=dict(verified=True)
    elif args.command=='run':out=run(args.prepared,args.prepare_sha,args.output)
    else:
        out=consume(args.output,args.report_sha)
        if args.receipt:out=write_json(args.receipt,out)
    print(canonical(out).decode())


if __name__=='__main__':main()
