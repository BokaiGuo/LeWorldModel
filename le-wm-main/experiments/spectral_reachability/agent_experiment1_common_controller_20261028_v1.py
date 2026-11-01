"""Calibration-only E1 common controller; MAIN-only CUDA, no physical execution.

Two known roots, two dispatched target arms, two learned slots UNKNOWN. All
72 H2 candidates/group are saved; 144 predict/API/module cap, no best replay.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import shutil
import time
import zlib
import numpy as np
from experiments.spectral_reachability import agent_kinodynamic_matched_domain_20261028_v1 as beam

NAME='agent_experiment1_common_controller_20261028_v1'
ROOT,sampling,core,enc=beam.ROOT,beam.sampling,beam.core,beam.enc
require,sha,canonical,read=beam.require,beam.sha,beam.canonical,beam.read
ARMS=('arbitrary_latent_waypoint','dynamics_compiled_witness')
LEARNED='existing_learned_prior'
ALL_ARMS=(ARMS[0],LEARNED,ARMS[1])
SEEDS=(2028102811,2028102812)
LATENT_SEEDS=(2028102801,2028102802)
BEAM_SHA='8e6d03cdbae69382cd87bcb7f443fb882023b7930ef138bd02aa37f0950d46b8'
REPORT=ROOT/'results/agent_kinodynamic_matched_domain_20261028_v1_cuda_MAIN01/report.json'
REPORT_SHA='d4976aeca357b5324c74be8b2b7e3bfc254e82b834b3acfe1b0ab90a1ef4614c'
SPEC=ROOT/'innerdocs/agent_experiment1_next_step_JASON_20261028_v1.md'
SPEC_SHA='b06787399646bc0066a005e0cef10bfa4637df32170a29d0c4b9719088d372d2'
CAP,FLOOR,FINAL=64*1024**2,200*1024**2,512*1024
BLOB_CAP,META_CAP,WALL=32*1024**2,24*1024**2,900
RESOURCES=sampling.RESOURCES
budget=sampling.budget


def vec(value):
    a=np.asarray(value,dtype=np.float64)
    require(a.shape==(192,) and np.isfinite(a).all() and np.array_equal(a,a.astype(np.float32)),
        'finite native float32 vector192; no padding')
    return a


def arbitrary(task,slot):
    require(type(slot) is int and slot in (0,1),'fixed source slot')
    z=vec(task['root_context']['latent_history'][-1]);g=vec(task['goal'])
    rng=np.random.Generator(np.random.PCG64(LATENT_SEEDS[slot]))
    draws=rng.normal(size=(20,192));scale=float(np.linalg.norm(g-z)/np.sqrt(192))
    points=(z+scale*draws).astype(np.float32)
    require(np.isfinite(points).all(),'finite latent proposals')
    costs=np.mean((points.astype(float)-g)**2,axis=1)
    winner=min(range(20),key=lambda i:(float(costs[i]),i))
    return dict(seed=LATENT_SEEDS[slot],draws=draws.tolist(),scale=scale,
        candidates=points.astype(float).tolist(),costs=costs.tolist(),winner_index=winner,
        target=points[winner].astype(float).tolist(),model_calls=0,learned=False)


def contract():
    return dict(roots=2,dispatch_arms=list(ARMS),dispatch_denominator=4,
        scientific_slots=6,learned_unknown_slots=2,calibration_only=True,P1_admission=False,
        iterations=3,samples=24,candidates_per_group=72,horizon=2,raw_steps=10,
        seed=list(SEEDS),latent_seed=list(LATENT_SEEDS),latent_candidates=20,
        objective='0.5 mean192(pred5-target)^2 + 0.5 mean192(pred10-original_goal)^2',
        CEM=dict(elite=4,mean0=0.,std0=1.,std_clip=[.05,1.],ddof=0,smoothing=False),
        action_map='original projected logit ball then float32 tanh; original raw->token ONCE',
        packet_logit_cap=beam.CAP,history='three real moving states and TWO nonzero incoming tokens',
        per_group=budget(),total=budget(576),predictor_samples=576,
        image_encoder=0,image_projector=0,AD=0,physics=0,training=False,Qfit=False,
        extra_replay=0,retries=0,padding=0,with_target=False,
        Q='unchanged identity192, mean not sum; original task.goal NOT overwritten',
        target_seal='ALL targets before model loading and any controller dispatch',
        matched='common controller API/sample/module budget only; proposal costs not equal',
        equal_total_compute=False,official_learned_comparison=False,independent_confirmation=False,
        inherited_encoding_once=sampling.contract()['inherited_encoding_once'],
        inherited_compiled_search=dict(report_sha256=REPORT_SHA,predict_API_per_root=144,
            predictor_samples_per_root=144,selected_from_H2=120,charged_to_current_controller=False),
        storage=dict(output_cap=CAP,blob_cap=BLOB_CAP,metadata_uncompressed_cap=META_CAP,
            final_reserve=FINAL,format='lossless content-addressed little-endian float64 arrays; gzip metadata'),
        floor_bytes=FLOOR,wall_seconds=WALL,
        wall_scope='from Store construction, checks at writes/dispatch; excludes prepare/verify; no native preemption',
        CUDA_only=True,MATH=True,TF32=False,autocast=False,CPU_neural_fallback=False)


def target_summaries(p):
    out=[]
    for slot,compiled in zip(p['slots'],p['compiled'],strict=True):
        s=slot['source_slot'];task=sampling.old.load_export(slot['task_export'],slot['task_sha256'])['gn_task']
        a=arbitrary(task,s)
        out.append(dict(source_slot=s,task_id=slot['task_id'],arbitrary_sha256=core.digest(a),
            targets={ARMS[0]:a['target'],ARMS[1]:compiled['target']}))
    return out


def build():
    require(sha(beam.__file__)==BEAM_SHA and sha(SPEC)==SPEC_SHA,'frozen helpers and accepted specification')
    require(sha(REPORT)==REPORT_SHA,'external compiled report')
    consumed=beam.consume(REPORT.parent,REPORT_SHA)
    p=deepcopy(sampling.build());p.update(schema=NAME+'.prepare',arguments={},contract=contract())
    cfg=read(Path(p['policy'])/'config.json')
    require(all(cfg['predictor'][k]==192 for k in ('input_dim','hidden_dim','output_dim')) and
        cfg['predictor']['num_frames']==3 and cfg['action_encoder']['input_dim']==10 and
        cfg['action_encoder']['emb_dim']==192 and cfg['pred_proj']['input_dim']==cfg['pred_proj']['output_dim']==192,
        'actual checkpoint configuration supports fixed192 receipt/storage bound')
    p['compiled']=[]
    source=read(REPORT);upstream=beam.verify(source['prepared'],source['prepare_sha256'])
    require(p['slots']==upstream['slots'] and p['expected_model_hash']==upstream['expected_model_hash'],
        'same entire root/task/model/history/normalizer cohort')
    require(len(consumed['plans'])==2,'both compiled plans')
    for slot,plan,row in zip(p['slots'],consumed['plans'],source['rows'],strict=True):
        s=slot['source_slot'];require(plan['source_slot']==s and plan['task_id']==slot['task_id'] and
            plan['task_sha256']==slot['task_sha256'] and plan['blind_sha256']==slot['blind_sha256'], 'compiled identity')
        result=read(REPORT.parent/row['result_file'])
        prefix=beam.checked_context(result['whole_contexts'][0])
        require(result['whole_contexts']==result['segmented_contexts'] and
            result['commands']==plan['commands'] and result['selected_path']==plan['selected_path'], 'certified action-generated target')
        vec(prefix['latent_history'][-1])
        p['compiled'].append(dict(source_slot=s,target=prefix['latent_history'][-1],prefix_context=prefix,
            raw10=plan['commands'],path=plan['selected_path'],selection_sha256=plan['selection_sha256'],
            result_path=str(REPORT.parent/row['result_file']),result_sha256=source['artifact_sha256'][row['result_file']]))
    p['targets']=target_summaries(p)
    p['bindings_sha256'].update(upstream['bindings_sha256'])
    for f in (REPORT,Path(source['prepared']),SPEC,Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):
        p['bindings_sha256'][str(f.resolve())]=sha(f)
    for name,h in source['artifact_sha256'].items():p['bindings_sha256'][str(REPORT.parent/name)]=h
    for path,h in p['bindings_sha256'].items():require(sha(path)==h,'source/asset final closure')
    return p


def prepare(output):
    p=build();raw=canonical(p);path=Path(output).absolute()
    require(len(raw)<=256*1024 and not path.exists() and not path.is_symlink() and
        shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'exclusive <=256KiB prepare / floor')
    with path.open('xb') as f:require(f.write(raw)==len(raw),'short prepare');f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(raw))


def verify(path,expected):
    core.hash_value(expected);require(sha(path)==expected,'external prepare SHA')
    p=read(path);require(p==build(),'entire own prepare and authority closure');return p


class ArrayArchive:
    """Lossless JSON float arrays -> raw f64 blob refs. No pickle or quantization."""
    def __init__(self,append):self.append=append;self.entries={};self.size=0
    def pack(self,v):
        if type(v) is list:
            try:a=np.asarray(v)
            except ValueError:a=np.asarray([],dtype=int)
            if a.dtype==np.float64 and a.size>=10 and a.ndim<=4:
                require(a.size<=4096 and np.isfinite(a).all(),'bounded finite array')
                # Every packed leaf was a Python float; no mixed int coercion.
                def floats(x):return all(floats(y) for y in x) if type(x) is list else type(x) is float
                if floats(v):
                    raw=a.astype('<f8').tobytes();h=hashlib.sha256(raw).hexdigest()
                    if h not in self.entries:
                        require(self.size+len(raw)<=BLOB_CAP,'32MiB lossless array bound')
                        self.append(raw);self.entries[h]=(self.size,len(raw));self.size+=len(raw)
                    offset,n=self.entries[h]
                    return dict(__f64__=h,shape=list(a.shape),offset=offset,nbytes=n)
            return [self.pack(x) for x in v]
        if type(v) is dict:
            require('__f64__' not in v,'reserved archive key');return {k:self.pack(x) for k,x in v.items()}
        return v


class ArrayReader:
    def __init__(self,raw):
        require(len(raw)<=BLOB_CAP,'bounded array archive');self.raw=raw;self.entries={};self.used=set()
    def unpack(self,v):
        if type(v) is dict and '__f64__' in v:
            require(set(v)=={'__f64__','shape','offset','nbytes'},'exact lossless array ref')
            h=v['__f64__'];core.hash_value(h);shape=v['shape'];offset=v['offset'];n=v['nbytes']
            require(type(shape) is list and 1<=len(shape)<=4 and all(type(x) is int and x>0 for x in shape), 'array shape')
            size=int(np.prod(shape,dtype=np.int64))
            require(10<=size<=4096 and type(offset) is int and type(n) is int and offset>=0 and
                n==size*8 and offset+n<=len(self.raw),'array range/size')
            raw=self.raw[offset:offset+n];require(hashlib.sha256(raw).hexdigest()==h,'array byte SHA')
            require(h not in self.entries or self.entries[h]==(offset,n),'one storage location per content')
            self.entries[h]=(offset,n);self.used.add((offset,n))
            a=np.frombuffer(raw,dtype='<f8').reshape(shape);require(np.isfinite(a).all(),'finite array')
            return a.tolist()
        if type(v) is dict:return {k:self.unpack(x) for k,x in v.items()}
        if type(v) is list:return [self.unpack(x) for x in v]
        return v
    def finish(self):
        end=0
        for start,n in sorted(self.used):require(start==end,'no unreferenced/overlapping array bytes');end+=n
        require(end==len(self.raw),'all archived arrays referenced')


class Store:
    def __init__(self,output):
        self.started=time.monotonic();self.output=Path(output).absolute()
        require(not self.output.exists() and not self.output.is_symlink() and
            shutil.disk_usage(self.output.parent).free>=FLOOR+CAP,'NEW 64MiB +200MiB floor')
        self.output.mkdir();self.closed=False;self.broken=False;self.meta=0;self.event_count=0;self.previous='0'*64
        self.comp=zlib.compressobj(6,zlib.DEFLATED,31)
        self._write('arrays.bin',b'');self._write('events.jsonl.gz',b'')
        self.archive=ArrayArchive(lambda raw:self._write('arrays.bin',raw,append=True))
    def check_time(self):require(time.monotonic()-self.started<=WALL,'900s dispatch/write limit')
    def _write(self,name,raw,append=False,final=False):
        if not final:self.check_time()
        require(Path(name).name==name and not (self.output/name).is_symlink(),'flat regular output')
        used=sum(f.stat().st_size for f in self.output.iterdir())
        require(used+len(raw)<=CAP-(0 if final else FINAL) and shutil.disk_usage(self.output).free>=FLOOR+len(raw), 'cap/floor')
        with (self.output/name).open('ab' if append else 'xb') as f:
            require(f.write(raw)==len(raw),'short write');f.flush();os.fsync(f.fileno())
        fd=os.open(self.output,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    def metadata(self,v):
        raw=canonical(self.archive.pack(v));require(self.meta+len(raw)<=META_CAP,'24MiB uncompressed metadata cap')
        self.meta+=len(raw);return raw
    def event(self,v):
        require(not self.closed and not self.broken,'journal cannot resume')
        try:
            body=dict(index=self.event_count,previous=self.previous,event=self.archive.pack(v))
            ident=core.digest(body);raw=canonical(dict(body,sha256=ident)).rstrip(b'\n')+b'\n'
            require(self.meta+len(raw)<=META_CAP,'journal metadata cap');self.meta+=len(raw)
            self._write('events.jsonl.gz',self.comp.compress(raw)+self.comp.flush(zlib.Z_SYNC_FLUSH),append=True)
            self.previous=ident;self.event_count+=1
        except BaseException:self.broken=True;raise
    def put(self,name,value,final=False):
        raw=canonical(value) if final else self.metadata(value)
        if final:require(len(raw)<=FINAL,'final reserved report size')
        self._write(name,zlib.compress(raw) if name.endswith('.z') else raw,final=final)
    def close(self):
        if not self.closed:
            self.closed=True;self._write('events.jsonl.gz',self.comp.flush(zlib.Z_FINISH),append=True,final=True)


class Dispatch(beam.Dispatch):
    def __call__(self,before,raw,phase,path):
        require(self.index<72,'no extra candidate or best replay');return super().__call__(before,raw,phase,path)


class SavedDispatch(beam.SavedDispatch):
    def __call__(self,before,raw,phase,path):
        require(self.index<72,'no extra saved candidate');return super().__call__(before,raw,phase,path)
    def done(self):
        require(self.index==72 and len(self.operations)==576 and all(
            sum(o['resource']==r and o['status']=='returned' for o in self.operations)==144 for r in RESOURCES),
            '72 H2 candidates and all144 actual API/modules per group')


def optimize(task,slot,arm,target,dispatch,sink):
    core.validate_task(task);require(type(slot) is int and slot in (0,1) and arm in ARMS,'fixed group')
    w=vec(target);g=vec(task['goal']);require(task['packet_logit_cap']==beam.CAP,'same action ball')
    rng=np.random.Generator(np.random.PCG64(SEEDS[slot]));mean=np.zeros((2,10));std=np.ones((2,10))
    bank=[];updates=[];root=beam.checked_context(task['root_context'])
    for it in range(3):
        params=rng.normal(mean,std,size=(24,2,10))
        xs=[sampling.gn.screen.projected_logits(x,beam.CAP) for x in params]
        sink(dict(kind='bank_frozen',iteration=it,mean=mean.tolist(),std=std.tolist(),parameters=params.tolist(),logits=[x.tolist() for x in xs]))
        chunk=[]
        for j,x in enumerate(xs):
            raw=np.tanh(x.astype(np.float32)).reshape(10,2).astype(float).tolist()
            cs=dispatch(root,raw,'whole_H2',[it,j])
            require(len(cs)==2,'full H2, never a single raw step')
            prefix=float(np.mean((vec(cs[0]['latent_history'][-1])-w)**2))
            terminal=float(np.mean((vec(cs[1]['latent_history'][-1])-g)**2))
            c=dict(index=len(bank),iteration=it,parameters=params[j].tolist(),logits=x.tolist(),
                commands=raw,contexts=cs,prefix_target_cost=prefix,model_terminal_cost=terminal,cost=.5*(prefix+terminal))
            bank.append(c);chunk.append(c);sink(dict(kind='candidate_saved',candidate=c))
        elite=sorted(chunk,key=lambda c:(c['cost'],c['index']))[:4]
        xx=np.asarray([c['logits'] for c in elite]);mean=xx.mean(axis=0);std=np.clip(xx.std(axis=0,ddof=0),.05,1.)
        update=dict(iteration=it,mean=mean.tolist(),std=std.tolist(),elite_indices=[c['index'] for c in elite])
        updates.append(update);sink(dict(kind='distribution_update',update=update))
    winner=min(range(72),key=lambda i:(bank[i]['cost'],i))
    sink(dict(kind='winner_sealed_no_replay',index=winner,commands=bank[winner]['commands'],cost=bank[winner]['cost']))
    return dict(task_id=core.task_id(task),source_slot=slot,arm=arm,seed=SEEDS[slot],target=list(target),
        candidates=bank,updates=updates,winner_index=winner)


def journal_check(task,slot,arm,target,result,events):
    c=beam.Cursor(events);d=SavedDispatch(task,c)
    rebuilt=optimize(task,slot,arm,target,d,c.expect);d.done()
    require(c.index==len(events) and rebuilt==result,'exact whole algorithm/context/module/RNG/selection evidence')
    return rebuilt


def initial_rows(p):
    return [dict(source_slot=s['source_slot'],arm=a,task_id=s['task_id'],blind_sha256=s['blind_sha256'],
        status='unknown' if a==LEARNED else 'not_started',commands=None,resources=None if a==LEARNED else {},
        predictor_samples=None if a==LEARNED else dict(attempted=0,completed=0),
        error='no certified learned assets; H10 is official fallback' if a==LEARNED else None)
        for s in p['slots'] for a in ALL_ARMS]


def target_payload(p):
    require(target_summaries(p)==p['targets'],'unchanged targets/task goal')
    return dict(schema=NAME+'.targets',targets=p['targets'],compiled=p['compiled'],
        arbitrary=[arbitrary(sampling.old.load_export(s['task_export'],s['task_sha256'])['gn_task'],s['source_slot']) for s in p['slots']])


def model_plan(slot,arm,result):
    c=result['candidates'][result['winner_index']]
    plan=dict(schema=NAME+'.model_plan',source_slot=slot['source_slot'],arm=arm,task_id=slot['task_id'],
        task_export=slot['task_export'],task_sha256=slot['task_sha256'],blind_sha256=slot['blind_sha256'],
        task_contract_sha256=slot['task_contract_sha256'],commands=c['commands'],logits=c['logits'],
        target=result['target'],contexts=c['contexts'],prediction=c['contexts'][-1],selected_index=c['index'],
        model_terminal_cost=c['model_terminal_cost'],prefix_target_cost=c['prefix_target_cost'],tracking_cost=c['cost'],
        with_target=False,selection_uses_physics=False,calibration_only=True,P1_admission=False)
    return dict(plan,selection_sha256=core.digest(plan))


def execute(task,slot,arm,target,model,device,torch,sink,row,check):
    m=sampling.Meter(sink,lambda:torch.cuda.synchronize(device),check)
    try:
        require(not torch.is_autocast_enabled('cuda'),'no autocast')
        backend=beam.NativeEdge(model,device,task)
        with torch.inference_mode(),backend.guard(m):
            d=Dispatch(task,backend,m,sink);result=optimize(task,slot,arm,target,d,sink)
        require(d.index==72 and m.counts()==budget() and m.predictor_samples()==dict(attempted=144,completed=144),'full controller exact budget')
        return result
    finally:row['resources']=m.counts();row['predictor_samples']=m.predictor_samples()


def plan_all(p,store,runner,rows):
    for row in rows:
        if row['arm']==LEARNED:continue
        s=row['source_slot'];arm=row['arm'];slot=p['slots'][s];target=p['targets'][s]['targets'][arm]
        row['status']='attempted';task=sampling.old.load_export(slot['task_export'],slot['task_sha256'])['gn_task']
        def sink(e):store.event(dict(source_slot=s,arm=arm,task_id=slot['task_id'],event=e))
        sink(dict(kind='arm_started',seed=SEEDS[s],target_sha256=core.digest(target),task_sha256=slot['task_sha256']))
        result=runner(task,s,arm,target,sink,row)
        require(row['resources']==budget() and row['predictor_samples']==dict(attempted=144,completed=144),'all actual returns')
        stem=f'root_{s}_{arm}';store.put(stem+'.result.json.z',result)
        plan=model_plan(slot,arm,result);store.put(stem+'.plan.json',plan)
        sink(dict(kind='plan_sealed',selection_sha256=plan['selection_sha256'],commands=plan['commands']))
        row.update(status='model_plan_sealed',commands=plan['commands'],selection_sha256=plan['selection_sha256'],
            result_file=stem+'.result.json.z',plan_file=stem+'.plan.json')


def aggregate(rows):
    active=[r for r in rows if r['arm']!=LEARNED]
    return ({k:{phase:sum(r['resources'].get(k,{}).get(phase,0) for r in active) for phase in ('attempted','completed')} for k in RESOURCES},
        {phase:sum(r['predictor_samples'][phase] for r in active) for phase in ('attempted','completed')})


def run(prepared,expected,output):
    p=verify(prepared,expected);store=Store(output);rows=initial_rows(p)
    errors=[];model=None;before=after=host=None;source_ok=False;complete=False
    try:
        print('E1 calibration only: four common-controller groups; learned2 UNKNOWN; no physics/admission.',flush=True)
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        payload=target_payload(p);store.put('targets.json',payload)
        store.event(dict(kind='ALL_targets_sealed_before_model',targets_sha256=core.digest(payload)))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        enc.frozen(model);before=enc.tensor_hash(model);require(before==p['expected_model_hash'],'original frozen model')
        with sdpa_kernel(SDPBackend.MATH):
            plan_all(p,store,lambda t,s,a,w,k,r:execute(t,s,a,w,model,device,torch,k,r,store.check_time),rows)
    except BaseException as exc:
        errors.append(enc.error('execution',exc))
        try:store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as secondary:errors.append(enc.error('error_journal',secondary))
    finally:
        for row in rows:
            if row['status']=='attempted':row['status']='failed_partial'
        if model is not None:
            try:enc.frozen(model);after=enc.tensor_hash(model)
            except BaseException as exc:errors.append(enc.error('final_model',exc))
        try:verify(prepared,expected);source_ok=True
        except BaseException as exc:errors.append(enc.error('final_source',exc))
        try:store.close()
        except BaseException as exc:errors.append(enc.error('journal_finalize',exc))
        hashes={}
        try:files=list(store.output.iterdir())
        except BaseException as exc:errors.append(enc.error('listing',exc));files=[]
        for f in files:
            try:require(f.is_file() and not f.is_symlink(),'regular artifact');hashes[f.name]=sha(f)
            except BaseException as exc:errors.append(enc.error('hash',exc))
        resources,samples=aggregate(rows)
        complete=not errors and source_ok and before is not None and before==after and resources==budget(576) and samples==dict(attempted=576,completed=576) and all(r['status']==('unknown' if r['arm']==LEARNED else 'model_plan_sealed') for r in rows)
        report=dict(schema=NAME+'.report',status='calibration_common_controller_plans_sealed' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,contract=p['contract'],rows=rows,
            resources=resources,predictor_samples=samples,errors=errors,first_error=errors[0] if errors else None,
            artifact_sha256=hashes,source_verified=source_ok,model_hash_before=before,model_hash_after=after,
            model_unchanged=before==after if before else None,host=host,commands_released=complete,
            journal_events=store.event_count,journal_final_sha256=store.previous,array_bytes=store.archive.size,
            metadata_uncompressed_bytes=store.meta,elapsed_seconds=time.monotonic()-store.started,
            inherited_encoding_once=p['contract']['inherited_encoding_once'],
            inherited_compiled_search=p['contract']['inherited_compiled_search'],
            dispatch_denominator=4,scientific_slots=6,learned_unknown_slots=2,training=False,new_physics=0,
            calibration_only=True,P1_admission=False,equal_total_compute=False)
        try:store.put('report.json',report,final=True)
        except BaseException as exc:complete=False;print(canonical(dict(report_write_error=enc.error('final_report',exc),errors=errors)).decode(),flush=True)
    require(complete,'partial evidence preserved, no retry: '+str(errors));return report


def bounded_z(raw,limit,wbits=zlib.MAX_WBITS):
    d=zlib.decompressobj(wbits);out=d.decompress(raw,limit+1)
    require(len(out)<=limit and not d.unconsumed_tail and d.eof and not d.unused_data,'single complete bounded compressed stream')
    return out


def decode(raw):return sampling.base.decode(raw)


def saved_payloads(output,report):
    """Authenticate every file before following refs; verify metadata hashchain."""
    output=Path(output);names=set(report['artifact_sha256'])
    order=[(s,a) for s in (0,1) for a in ARMS]
    expected={'binding.json','arrays.bin','events.jsonl.gz','targets.json'}|{f'root_{s}_{a}.{x}' for s,a in order for x in ('result.json.z','plan.json')}
    require(names==expected and {f.name for f in output.iterdir()}==names|{'report.json'},'all12 artifacts plus report')
    require(sum(f.stat().st_size for f in output.iterdir())<=CAP,'total output cap')
    for name,h in report['artifact_sha256'].items():
        f=output/name;require(f.is_file() and not f.is_symlink() and sha(f)==h,'artifact bytes')
    ar=ArrayReader((output/'arrays.bin').read_bytes());meta=0;payload={}
    for name in sorted(names-{'arrays.bin','events.jsonl.gz'}):
        raw=(output/name).read_bytes()
        if name.endswith('.z'):raw=bounded_z(raw,META_CAP)
        meta+=len(raw);require(meta<=META_CAP,'bounded total expanded metadata')
        payload[name]=ar.unpack(decode(raw))
    raw=bounded_z((output/'events.jsonl.gz').read_bytes(),META_CAP,31);meta+=len(raw)
    require(meta<=META_CAP and raw.endswith(b'\n'),'complete metadata census')
    events=[];previous='0'*64
    for i,line in enumerate(raw.splitlines()):
        e=decode(line);require(set(e)=={'index','previous','event','sha256'} and e['index']==i and e['previous']==previous,'journal order/hashchain')
        body={k:e[k] for k in ('index','previous','event')};require(core.digest(body)==e['sha256'],'journal SHA')
        previous=e['sha256'];events.append(ar.unpack(e['event']))
    ar.finish()
    require(report['journal_events']==len(events) and report['journal_final_sha256']==previous and
        report['array_bytes']==len(ar.raw) and report['metadata_uncompressed_bytes']==meta,'all byte/event counters')
    return payload,events


def validate_saved(p,r,payload,events):
    """RAM-callable strict consumer core; NEVER runs neural inference."""
    require(r['schema']==NAME+'.report' and r['status']=='calibration_common_controller_plans_sealed' and
        r['errors']==[] and r['first_error'] is None and r['commands_released'] is True and r['source_verified'] is True and
        r['model_unchanged'] is True and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'] and
        r['contract']==p['contract'] and r['resources']==budget(576) and r['predictor_samples']==dict(attempted=576,completed=576) and
        r['inherited_encoding_once']==p['contract']['inherited_encoding_once'] and
        r['inherited_compiled_search']==p['contract']['inherited_compiled_search'] and
        r['dispatch_denominator']==4 and r['scientific_slots']==6 and r['learned_unknown_slots']==2 and
        r['calibration_only'] is True and r['P1_admission'] is False and r['training'] is False and
        r['new_physics']==0 and r['equal_total_compute'] is False,'complete common-controller-only contract')
    require(payload['binding.json']==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'prepare binding')
    tp=target_payload(p);require(payload['targets.json']==tp and events and events[0]==dict(kind='ALL_targets_sealed_before_model',targets_sha256=core.digest(tp)),'targets sealed first')
    initial=initial_rows(p);require(len(r['rows'])==6,'all scientific slots including unknown learned')
    at=1;plans=[];first_draws={}
    for row,start in zip(r['rows'],initial,strict=True):
        s=start['source_slot'];a=start['arm'];slot=p['slots'][s]
        if a==LEARNED:require(row==start,'unknown learned remains null, not zero/fallback');continue
        stem=f'root_{s}_{a}'
        require({k:row[k] for k in ('source_slot','arm','task_id','blind_sha256')}=={k:start[k] for k in ('source_slot','arm','task_id','blind_sha256')} and
            row['status']=='model_plan_sealed' and row['error'] is None and row['resources']==budget() and row['predictor_samples']==dict(attempted=144,completed=144) and
            row['result_file']==stem+'.result.json.z' and row['plan_file']==stem+'.plan.json','exact group identity/resources')
        group=[]
        while at<len(events) and set(events[at])=={'source_slot','arm','task_id','event'} and events[at]['source_slot']==s and events[at]['arm']==a:
            require(events[at]['task_id']==slot['task_id'],'event task join');group.append(events[at]['event']);at+=1
        target=p['targets'][s]['targets'][a]
        require(len(group)>2 and group[0]==dict(kind='arm_started',seed=SEEDS[s],target_sha256=core.digest(target),task_sha256=slot['task_sha256']),'arm start target/seed')
        task=sampling.old.load_export(slot['task_export'],slot['task_sha256'])['gn_task'];result=payload[row['result_file']]
        journal_check(task,s,a,target,result,group[1:-1]);plan=model_plan(slot,a,result)
        require(payload[row['plan_file']]==plan and row['commands']==plan['commands'] and row['selection_sha256']==plan['selection_sha256'] and group[-1]==dict(kind='plan_sealed',selection_sha256=plan['selection_sha256'],commands=plan['commands']),'own already-evaluated selected raw10')
        first_draws[(s,a)]=group[1]['parameters'];plans.append(plan)
    require(at==len(events) and aggregate(r['rows'])==(r['resources'],r['predictor_samples']),'no stray events/counts')
    require(all(first_draws[(s,ARMS[0])]==first_draws[(s,ARMS[1])] for s in (0,1)),'same first round random draws')
    return plans


def consume(output,expected):
    core.hash_value(expected);output=Path(output).absolute();require(sha(output/'report.json')==expected,'external report SHA')
    r=read(output/'report.json');p=verify(r['prepared'],r['prepare_sha256']);payload,events=saved_payloads(output,r)
    plans=validate_saved(p,r,payload,events)
    for path,h in p['bindings_sha256'].items():require(sha(path)==h,'final source/model/task closure')
    for name,h in r['artifact_sha256'].items():require(sha(output/name)==h,'final artifact SHA')
    require(sha(output/'report.json')==expected,'final report SHA')
    return dict(schema=NAME+'.consume',status='common_controller_calibration_verified',plans=[dict(x,whole_run_report_sha256=expected) for x in plans],
        rows=r['rows'],resources=r['resources'],predictor_samples=r['predictor_samples'],dispatch_denominator=4,
        scientific_slots=6,learned_unknown_slots=2,calibration_only=True,P1_admission=False,equal_total_compute=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('prepared','prepare-sha','output','report-sha'):p.add_argument('--'+k)
    a=p.parse_args()
    if a.command=='prepare':result=prepare(a.output)
    elif a.command=='verify':result=dict(status='verified_no_execution',contract=verify(a.prepared,a.prepare_sha)['contract'])
    elif a.command=='run':result=run(a.prepared,a.prepare_sha,a.output)
    else:result=consume(a.output,a.report_sha)
    print(canonical(result).decode(),flush=True)


if __name__=='__main__':main()
