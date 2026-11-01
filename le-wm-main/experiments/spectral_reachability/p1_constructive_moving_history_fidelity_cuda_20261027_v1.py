"""MAIN-only factual moving-history fidelity. Not a planner or auditor.

Two existing slots, two step15 encodings, four fresh predict calls per slot.
Saved source/exports are read in prepare; no model/physics execution there.
"""
from __future__ import annotations
import argparse
import base64
from copy import deepcopy
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import zlib
import numpy as np
from PIL import Image
from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as moving
from experiments.spectral_reachability import p1_constructive_offline_fidelity_cuda_20261027_v2 as numerics

NAME='p1_constructive_moving_history_fidelity_cuda_20261027_v1'
SOURCE_NAME='p1_constructive_moving_history_source_20261027_v1'
ROOT,low=moving.ROOT,moving.low
require,sha,read,canonical=moving.require,moving.sha,moving.read,moving.canonical
SOURCE_SHA='c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6'
PREPARE_SHA='c1292f0ebfb1827e5da92b32ab13d2e4a12c025c4741277ab4c4f478d6ee8cb9'
ENCODER_SHA='e7fb554327b3bb5a96190f6291f8450b2b0723cd981817715cfb7e50aa266bf6'
TASK_SHAS=('a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b','101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74')
KINDS=('root_H1','observed_prefix_H1','open_H2_first','open_H2_terminal')
CAP,FLOOR,FINAL=2*1024**2,200*1024**2,256*1024


def bits(a,b): return a.dtype.str==b.dtype.str and a.shape==b.shape and a.tobytes()==b.tobytes()
def digest_json(x): return moving.core.digest(x)


class Inputs:
    def __init__(self): self.bindings={}
    def raw(self,path,expected=None,limit=8*1024**2):
        p=Path(path).absolute(); require(p.is_file() and not p.is_symlink() and p.stat().st_size<=limit,'bounded regular saved input')
        b=p.read_bytes(); h=low.digest(b)
        require(expected is None or h==expected,'saved source external SHA '+str(p))
        require(str(p) not in self.bindings or self.bindings[str(p)]==h,'same-path source identity')
        self.bindings[str(p)]=h; return b
    def json(self,path,expected=None):
        value=json.loads(self.raw(path,expected)); canonical(value); return value
    def png(self,parent,event):
        name=event['png']; require(Path(name).name==name and name not in ('','.','..'),'flat source PNG')
        b=self.raw(Path(parent)/name,event['file_sha256'],1024**2)
        with Image.open(BytesIO(b)) as im:
            require(im.mode=='RGB' and im.size==(224,224) and im.format=='PNG','original RGB224 PNG')
            a=np.array(im)
        require(a.dtype==np.uint8 and low.digest(a.tobytes())==event['pixel_sha256'],'actual source decoded pixel SHA')
        return a,b
    def finish(self):
        for p,h in self.bindings.items(): require(sha(p)==h,'saved/source/model seal drift '+p)


def source_chain(raw,tail):
    require(raw.endswith(b'\n'),'complete saved source journal')
    es=[json.loads(line) for line in raw.splitlines()]; previous=None
    for i,e in enumerate(es):
        require(e['index']==i and e['previous_sha256']==previous and e['event_sha256']==digest_json({k:v for k,v in e.items() if k!='event_sha256'}),'source journal identity')
        previous=e['event_sha256']
    require(es[0]['kind']=='run_started' and es[-1]['kind']=='run_finished' and previous==tail,'source terminal chain bound by pack')
    return es


def executed_actions(events,slot):
    selected=lambda kind:[e for e in events if e['kind']==kind and e.get('resource')=='public_step' and e.get('slot')==slot and e.get('phase')=='replay0']
    attempts,returns=selected('operation_attempted'),selected('operation_completed')
    require([e['step'] for e in attempts]==[e['step'] for e in returns]==list(range(1,21)),'full20 actual replay public-step attempts/returns')
    values=[]
    for a,b in zip(attempts,returns,strict=True):
        raw=np.asarray(a['raw_action']); x=raw.astype(np.float32)
        require(raw.shape==(2,) and np.isfinite(raw).all() and np.array_equal(raw,x) and np.array_equal(b['raw_action'],x),'literal completed raw float32 action')
        values.append(x)
    actual=np.stack(values)
    warm=[e for e in events if e['kind']=='warmup_return' and e.get('slot')==slot and e.get('phase')=='replay0']
    native=[e for e in events if e['kind']=='native_return' and e.get('slot')==slot and e.get('phase')=='replay0']
    require([e['step'] for e in warm]==[e['step'] for e in native]==list(range(1,11)),'all10 warmup and witness returned transitions')
    for i,e in enumerate(warm):
        require(np.array_equal(e['raw_action'],actual[i]) and e['raw_float32_bits']==actual[i].tobytes().hex(),'actual latest-action warmup bits')
    require(bits(actual[:10],moving.fixed_raw()),'actual moving history, never zero padding')
    return actual[:10],actual[10:]


def load_saved(sourcepack,source_sha,encoder_report,encoder_sha):
    require(source_sha==SOURCE_SHA and encoder_sha==ENCODER_SHA,'fixed two actual source/export receipts')
    io=Inputs(); sp=Path(sourcepack).absolute(); ep=Path(encoder_report).absolute()
    pack=io.json(sp,source_sha); report=io.json(ep,encoder_sha)
    require(pack['schema']==SOURCE_NAME+'.sourcepack' and pack['sourcepack_complete'] is True and pack['status']=='fixed_slots_processed' and pack['errors']==[] and pack['fixed_denominator']==2 and len(pack['rows'])==2,'complete original moving source')
    binding=io.json(sp.parent/'execution_binding.json')
    require(binding['prepare_file_sha256']==PREPARE_SHA,'explicit original source prepare SHA')
    envelope=io.json(binding['prepared'],PREPARE_SHA)
    require(envelope['schema']==SOURCE_NAME+'.envelope','moving source prepare envelope')
    dec=zlib.decompressobj(31); raw=dec.decompress(base64.b64decode(envelope['gzip_base64'],validate=True),2*1024**2+1)
    require(len(raw)<=2*1024**2 and dec.eof and not dec.unused_data and not dec.unconsumed_tail,'bounded original source metadata')
    plan=json.loads(raw)
    require(plan['schema']==SOURCE_NAME+'.prepare' and digest_json(plan)==envelope['plan_sha256']==pack['prepare_id'],'source original prepare identity')
    events=source_chain(io.raw(sp.parent/'journal.jsonl'),pack['journal_chain_tail'])
    require(events[0]['prepare']==plan and plan['with_target'] is False,'source plan/journal render binding')
    for path,h in plan['bindings'].items(): io.bindings[path]=h
    require(report['schema']==moving.NAME+'.report' and report['status']=='moving_blind_tasks_encoded_and_sealed' and report['errors']==[] and report['source_verified'] is True and report['model_unchanged'] is True and report['model_hash_before']==report['model_hash_after']==low.MODEL_TENSOR_SHA and len(report['rows'])==2,'complete native moving encoder receipt')
    require(report['contract']==moving.contract() and report['resources']=={k:dict(attempted=8,completed=8) for k in ('encoder','projector')},'original encoder contract/counts')
    eb=io.json(ep.parent/'binding.json'); require(eb['prepare_sha256']==report['prepare_sha256'],'encoder source prepare binding')
    encoder_plan=io.json(eb['prepared'],eb['prepare_sha256'])
    require(encoder_plan['schema']==moving.NAME+'.prepare' and encoder_plan['normalization']==moving.normalization() and encoder_plan['expected_model_hash']==low.MODEL_TENSOR_SHA,'new moving encoder prepare')
    for path,h in encoder_plan['bindings_sha256'].items(): io.bindings[path]=h
    payload=[]; slots=[]
    for slot,(row,erow) in enumerate(zip(pack['rows'],report['rows'],strict=True)):
        require(row['slot']==plan['slots'][slot] and row['slot']['index']==slot and row['reachable'] is True and row['executed_replay_steps']==[10,10] and row['status']=='reachable_native_replayed','fixed original source slots, not outcome-filtered')
        require((erow['order'],erow['source_slot'],erow['status'])==(slot,slot,'task_sealed') and erow['blind_source_sha256']==row['blind_payload_sha256'],'own moving exported source slot')
        task_path=ep.parent/f'task_{slot:02d}.json.gz'; task_sha=TASK_SHAS[slot]
        require(erow['task_path']==str(task_path) and erow['task_file_sha256']==task_sha and report['artifact_sha256'][task_path.name]==task_sha,'MAIN exact task export SHA/path')
        io.raw(task_path,task_sha)
        task=moving.load_task(task_path,task_sha)['gn_task']  # MUST use new moving interface, not low.load_task.
        require(moving.core.task_id(task)==erow['task_id'],'moving own task identity')
        blind_path=sp.parent/row['blind_payload']; io.raw(blind_path,row['blind_payload_sha256'])
        packet=moving.load_blind(blind_path,row['blind_payload_sha256'])
        require(encoder_plan['slots'][slot]['blind_npz']==str(blind_path) and encoder_plan['slots'][slot]['blind_sha256']==row['blind_payload_sha256'],'encoder actual source payload binding')
        warm,commands=executed_actions(events,slot)
        commit=io.json(sp.parent/f's{slot}_replay0_commit.json')
        require(np.array_equal(commit['commands'],commands) and bits(warm,packet['raw_action_history']) and np.array_equal(task['root_raw_action_history'],warm),'witness committed AND actually executed; not proposed planner actions')
        root=np.asarray(task['root_context']['latent_history'],np.float32); goal=np.asarray(task['goal'],np.float32)
        for j,expected in enumerate([*root,goal]):
            image_index=4*slot+j; entry=erow['images'][j]; name=f'image_{image_index:02d}.native192.npy'; path=ep.parent/name
            require(entry['path']==str(path) and entry['sha256']==report['artifact_sha256'][name] and entry['shape']==[192] and entry['dtype']=='float32','native exported entry binding')
            stream=BytesIO(io.raw(path,entry['sha256'])); z=np.load(stream,allow_pickle=False)
            low.native(z); require(stream.tell()==len(stream.getvalue()) and bits(z,expected),'original encoded native arrays equal new task')
        observations=[e for e in events if e['kind']=='fresh_observation' and e.get('slot')==slot and e.get('phase')=='replay0' and e.get('step') in (0,5,10,15,20)]
        require([e['step'] for e in observations]==[0,5,10,15,20],'all source factual boundary images')
        selected=[]
        for j,e in enumerate(observations):
            guards=[g for g in events if g['kind']=='fresh_guard_passed' and (g.get('slot'),g.get('phase'),g.get('step'))==(slot,'replay0',e['step'])]
            require(len(guards)==1,'source selected frame fresh guard')
            a,b=io.png(sp.parent,e)
            expected=packet['rgb_history'][j] if j<3 else packet['goal_rgb'] if j==4 else None
            if expected is not None: require(bits(a,expected),'source PNG equals actual encoder blind input/goal')
            selected.append((a,b))
        prefix,png=selected[3]; require(len(png)<=512*1024,'small prefix PNG output budget')
        slots.append(dict(slot=slot,task_path=str(task_path),task_sha256=task_sha,task_id=erow['task_id'],root_sha256=row['root_sha256'],
            source_phase='replay0',prefix_step=15,prefix_png=str(sp.parent/observations[3]['png']),prefix_file_sha256=observations[3]['file_sha256'],prefix_pixel_sha256=observations[3]['pixel_sha256'],
            raw_history_sha256=low.digest(warm.tobytes()),witness_executed_sha256=low.digest(commands.tobytes()),witness_raw10=commands.astype(float).tolist()))
        payload.append(dict(slot=slot,task_id=erow['task_id'],root_history=root,raw_history=warm,commands=commands,prefix_rgb=prefix,prefix_png_bytes=png,goal_latent=goal))
    io.finish(); return slots,payload,io.bindings,report['host']


def contract():
    return dict(slots=2,phase='factual evaluator only, not planner candidates or selection',calls_per_slot=list(KINDS),
        counts=dict(encoder=2,projector=2,predict=8,predictor=8,action_encoder=8,pred_proj=8),
        tokens=dict(per_predict=3,latent_input=24,action_packet_input=24,pred_proj_samples=24),
        prefix_source='replay0 real step15 PNG; goal step20 already encoded by NEW moving task',
        warmup='actual nonzero raw10 from source journal + moving load_task, never old zero loader',
        H2='fresh root->predicted15->predicted20, never reuse H1 or actual z15 in H2',
        normalizer=moving.normalization(),SDP='MATH',matmul_TF32=False,cudnn_TF32=False,
        AD=0,grad=0,engine=0,physics=0,training=False,selection=False,CPU_neural=False,no_retries=True,no_dedup=True,
        wall_seconds=600,allocated_bytes=3221225472,reserved_bytes=4294967296,output_cap_bytes=CAP,disk_floor_bytes=FLOOR,
        final_reserve_bytes=FINAL,scientific_threshold=None,P1_admission=False,independent_confirmation=False)


def material(sourcepack,source_sha,encoder_report,encoder_sha):
    slots,payload,bindings,host=load_saved(sourcepack,source_sha,encoder_report,encoder_sha)
    files=moving.source_files()+[Path(numerics.__file__),Path(numerics.old.__file__),Path(numerics.old.rgb.__file__),Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    for path in files: bindings[str(Path(path).resolve())]=sha(path)
    bindings.update(moving.MODELS)
    p=dict(schema=NAME+'.prepare',arguments=dict(sourcepack=str(Path(sourcepack).absolute()),source_sha=source_sha,encoder_report=str(Path(encoder_report).absolute()),encoder_sha=encoder_sha),
        slots=slots,bindings=bindings,contract=contract(),policy=str(moving.POLICY),expected_model_hash=low.MODEL_TENSOR_SHA,source_host=host)
    for path,h in bindings.items(): require(sha(path)==h,'prepare final source/model closure')
    return p,payload


def prepare(output,**arguments):
    p,_=material(**arguments); b=canonical(p); path=Path(output).absolute()
    require(not path.exists() and len(b)<256*1024 and shutil.disk_usage(path.parent).free>=FLOOR+len(b),'exclusive small prepare+200MiB')
    with path.open('xb') as f: f.write(b); f.flush(); os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(b))


def verify(path,expected):
    require(sha(path)==expected,'external new prepare SHA'); p=read(path); current,payload=material(**p['arguments'])
    require(p==current,'entire source/model/export/PNG/action reconstruction'); return p,payload


class Store(numerics.Store):
    def __init__(self,output):
        self.output=Path(output).absolute(); require(not self.output.exists() and shutil.disk_usage(self.output.parent).free>=FLOOR+CAP,'exclusive2MiB+200MiB')
        self.output.mkdir(); self.comp=zlib.compressobj(6,zlib.DEFLATED,31); self.closed=self.broken=False; self._write('events.jsonl.gz',b'')
    def _write(self,name,raw,append=False,final=False):
        require(Path(name).name==name and name not in ('','.','..') and not (self.output/name).is_symlink(),'flat nonsymlink output')
        used=sum(p.stat().st_size for p in self.output.iterdir() if p.is_file())
        require(used+len(raw)<=CAP-(0 if final else FINAL) and shutil.disk_usage(self.output).free>=FLOOR+len(raw),'allbytes2MiB/final reserve/200MiB floor')
        with (self.output/name).open('ab' if append else 'xb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
        fd=os.open(self.output,os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)


def save_array(store,name,a): return numerics.save_array(store,name,a)


class Visual(low.Ledger):
    def __init__(self,sink,limits): super().__init__(1,sink); self.limits=limits
    def dispatch(self,kind,tensor,callback,*args,**kwargs):
        self.limits.check(); expected=[(i,k) for i in range(2) for k in ('encoder','projector')]
        require(len(self.entries)<4 and (self.image,kind)==expected[len(self.entries)],'fixed two prefix images cap/order')
        require(self.no_grad() and all(e['completed'] for e in self.entries),'inference only, no retry')
        require(tensor.device.type=='cuda' and tuple(tensor.shape)==((1,3,224,224) if kind=='encoder' else (1,192)),'CUDA batch1 native visual')
        e=dict(index=len(self.entries),image=self.image,resource=kind,completed=False); self.entries.append(e)
        try:
            self.sink(dict(kind='visual_attempt',entry=deepcopy(e))); out=callback(*args,**kwargs); self.sync(); self.limits.check(); e['completed']=True
            self.sink(dict(kind='visual_return',entry=deepcopy(e))); return out
        except BaseException as exc: e['error']=low.error(kind,exc); raise


class Predict(numerics.PredictLedger):
    def dispatch(self,kind,fn,*args,**kwargs):
        self.limits.check(); require(not self.failed and len(self.entries)<32 and kind==self.ORDER[len(self.entries)%4],'fixed8 predict calls/order')
        require(all(e['completed'] or e['resource']=='predict' for e in self.entries),'no retry after incomplete child')
        e=dict(index=len(self.entries),request=len(self.entries)//4,resource=kind,tokens=3,completed=False); self.entries.append(e)
        try:
            self.sink(dict(kind='predict_attempt',entry=deepcopy(e))); out=fn(*args,**kwargs); self.sync(); self.limits.check(); e['completed']=True
            self.sink(dict(kind='predict_return',entry=deepcopy(e))); return out
        except BaseException as exc: self.failed=True; e['error']=low.error(kind,exc); raise


REQUEST_KEYS={'slot','kind','latent_history','raw_packets','tokens'}


def request(slot,kind,history,past,outgoing):
    require(type(slot) is int and slot in (0,1) and kind in KINDS,'fixed request identity')
    require(history.dtype==past.dtype==outgoing.dtype==np.float32 and history.shape==(3,192) and past.shape==(10,2) and outgoing.shape==(5,2),'native history/actual raw shapes')
    require(all(np.isfinite(a).all() for a in (history,past,outgoing)),'finite native inputs')
    raw=np.concatenate([past,outgoing]).reshape(3,5,2)
    return dict(slot=slot,kind=kind,latent_history=history.copy(),raw_packets=raw,tokens=numerics.normalized(raw))


def native_predict(model,torch,device,r):
    require(set(r)==REQUEST_KEYS,'backend whitelist excludes labels/goal/task/witness metadata')
    mapped=dict(slot=r['slot'],episode=r['slot'],window=10 if r['kind'] in ('root_H1','open_H2_first') else 15,condition='F',
        latent_history=r['latent_history'],raw_packets=r['raw_packets'],tokens=r['tokens'])
    return numerics.native_predict(model,torch,device,mapped)


def evaluate(payload,prefixes,backend,store,rows,limits):
    require(len(payload)==len(prefixes)==2 and len(rows)==8,'fixed two roots/eight calls')
    results=[]
    def dispatch(r,index):
        require(set(r)==REQUEST_KEYS,'no labels in model request'); limits.check(); row=rows[index]; row['status']='attempted'
        inputs={k:save_array(store,f'request_{index:02d}.{k}.npy',r[k]) for k in ('latent_history','raw_packets','tokens')}
        store.event(dict(kind='request',index=index,slot=r['slot'],call=r['kind'],inputs=inputs))
        out=backend(deepcopy(r)); require(set(out)=={'prediction','embedding_before','embedding_after'},'native forward outputs')
        for a in out.values(): require(a.dtype==np.float32 and a.shape==(3,192) and np.isfinite(a).all(),'native float32 full prediction/AE embeddings')
        require(bits(out['embedding_before'],out['embedding_after']),'no embedding mask in factual evaluator')
        receipts={k:save_array(store,f'request_{index:02d}.{k}.npy',v) for k,v in out.items()}
        receipts['last_native192']=save_array(store,f'request_{index:02d}.last_native192.npy',out['prediction'][-1])
        row.update(status='saved',slot=r['slot'],kind=r['kind'],inputs=inputs,outputs=receipts)
        store.event(dict(kind='output_saved',row=deepcopy(row))); return out['prediction'].copy()
    for slot,(d,prefix) in enumerate(zip(payload,prefixes,strict=True)):
        require(d['slot']==slot,'own source slot'); low.native(prefix)
        z,h,a=d['root_history'],d['raw_history'],d['commands']; base=4*slot
        p0=dispatch(request(slot,KINDS[0],z,h,a[:5]),base)
        shifted_raw=np.concatenate([h[5:],a[:5]])
        teacher=np.concatenate([z[1:],prefix[None]])
        p1=dispatch(request(slot,KINDS[1],teacher,shifted_raw,a[5:]),base+1)
        p2=dispatch(request(slot,KINDS[2],z,h,a[:5]),base+2)  # fresh, NEVER reuse p0
        imagined=np.concatenate([z[1:],p2[-1:]])
        p3=dispatch(request(slot,KINDS[3],imagined,shifted_raw,a[5:]),base+3)
        results.append(dict(slot=slot,predictions=[p0,p1,p2,p3]))
    return results


def score(results,payload,prefixes):
    require(len(results)==len(payload)==len(prefixes)==2,'fixed scoring denominator')
    rows=[]
    for slot,(r,d,prefix) in enumerate(zip(results,payload,prefixes,strict=True)):
        require(r['slot']==d['slot']==slot and len(r['predictions'])==4,'all four own factual predicts')
        target=[prefix,d['goal_latent'],prefix,d['goal_latent']]; root=d['root_history'][-1]
        current=[root,prefix,root,root]
        mse=lambda a,b:float(np.mean((a.astype(float)-b.astype(float))**2))
        rows.append(dict(slot=slot,task_id=d['task_id'],evaluations=[dict(kind=k,target_step=15 if j in (0,2) else 20,
            mse=mse(p[-1],t),persistence_mse=mse(c,t),target_array=numerics.array_receipt(t))
            for j,(k,p,t,c) in enumerate(zip(KINDS,r['predictions'],target,current,strict=True))],
            root_H1_vs_open_H2_first_bitwise=bits(r['predictions'][0],r['predictions'][2])))
    return dict(rows=rows,slot_denominator=2,predict_call_denominator=8,
        mean_mse={k:float(np.mean([r['evaluations'][j]['mse'] for r in rows])) for j,k in enumerate(KINDS)},
        labels='actually executed factual source successors only; no planner candidates',training_provenance_verified=False,scientific_threshold=None,P1_admission=False)


def run(prepared,expected,output):
    limits=numerics.Limits(free=lambda:shutil.disk_usage(Path(output).absolute().parent).free)
    p,payload=verify(prepared,expected); limits.check(); store=Store(output)
    errors=[]; forbidden=[]; model=None; before=after=host=None; source_ok=False
    images=[dict(slot=i,status='not_started') for i in range(2)]; rows=[dict(index=i,status='not_started') for i in range(8)]
    visual=Visual(store.event,limits); pred=Predict(store.event,limits)
    def ad_event(e): forbidden.append(deepcopy(e)); store.event(e)
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        store.put('source_binding.json',dict(arguments=p['arguments'],slots=p['slots']))
        store.put('request_recipe.json',dict(slot_order=[0,1],call_order=list(KINDS),H2_prefix='fresh prediction from open_H2_first, NOT true prefix'))
        os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        host=require_cuda_allocation('cuda:0'); device=torch.device('cuda:0')
        for key in ('name','torch_cuda','torch_version'): require(host[key]==p['source_host'][key],'same original native CUDA numerics '+key)
        limits.memory=lambda:(torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device)); limits.free=lambda:shutil.disk_usage(store.output).free
        limits.check(); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        limits.check(); low.frozen(model); before=low.tensor_hash(model); require(before==p['expected_model_hash'],'original frozen model tensor SHA')
        visual.sync=pred.sync=lambda:torch.cuda.synchronize(device); visual.no_grad=lambda:not torch.is_grad_enabled()
        transform=p0.image_transform(); prefixes=[]
        with numerics.old.no_ad(torch,ad_event,errors),sdpa_kernel(SDPBackend.MATH):
            with low.guard(model,torch,visual,errors):
                for slot,d in enumerate(payload):
                    limits.check(); row=images[slot]; row['status']='attempted'; visual.image=slot
                    store._write(f'slot_{slot}_step15.png',d['prefix_png_bytes'])
                    row['png']=dict(file=f'slot_{slot}_step15.png',sha256=low.digest(d['prefix_png_bytes']),pixels=numerics.array_receipt(d['prefix_rgb']))
                    store.event(dict(kind='prefix_encode_attempt',row=deepcopy(row)))
                    with torch.inference_mode():
                        x=p0._image_tensor(d['prefix_rgb'].copy(),transform=transform,device=device)
                        require(x.device.type=='cuda' and tuple(x.shape)==(1,1,3,224,224),'single native CUDA prefix pixels')
                        z=model.encode({'pixels':x})['emb']; require(z.device.type=='cuda' and z.dtype==torch.float32 and tuple(z.shape)==(1,1,192),'native projected prefix192')
                        value=z[0,0].detach().cpu().numpy().copy(); low.native(value)
                    require([(e['resource'],e['completed']) for e in visual.entries if e['image']==slot]==[('encoder',True),('projector',True)],'two actual visual module returns')
                    row.update(status='encoded',latent=save_array(store,f'slot_{slot}_step15.native192.npy',value)); prefixes.append(value)
                    save_array(store,f'slot_{slot}_step20.existing_native192.npy',d['goal_latent'])
                    store.event(dict(kind='prefix_encode_saved',row=deepcopy(row))); visual.image=None
            with numerics.prediction_guard(model,torch,pred,errors):
                results=evaluate(payload,prefixes,lambda r:native_predict(model,torch,device,r),store,rows,limits)
        store.event(dict(kind='all8_predictions_durable_before_scoring')); store.put('diagnostics.json',score(results,payload,prefixes)); limits.check()
    except BaseException as exc:
        errors.insert(0,low.error('execution',exc))
        try: store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as secondary: errors.append(low.error('error_journal',secondary))
    finally:
        for r in images+rows:
            if r['status']=='attempted': r['status']='failed_partial'
        if model is not None:
            try: low.frozen(model); after=low.tensor_hash(model)
            except BaseException as exc: errors.append(low.error('final_model',exc))
        try: verify(prepared,expected); source_ok=True
        except BaseException as exc: errors.append(low.error('final_source',exc))
        try: limits.check()
        except BaseException as exc: errors.append(low.error('final_limits',exc))
        try: store.close()
        except BaseException as exc: errors.append(low.error('journal_finalize',exc))
        counts=dict(**visual.counts(),**pred.counts())
        complete=not errors and not forbidden and not visual.blocked and not pred.failed and source_ok and before is not None and before==after and all(r['status']=='encoded' for r in images) and all(r['status']=='saved' for r in rows) and all(counts[k]['attempted']==counts[k]['completed']==v for k,v in contract()['counts'].items())
        hashes={}
        for f in store.output.iterdir():
            if f.is_file():
                try: hashes[f.name]=sha(f)
                except BaseException as exc: errors.append(low.error('artifact_hash',exc)); complete=False
        report=dict(schema=NAME+'.report',status='moving_factual_fidelity_saved_not_admitted' if complete else 'partial_or_failed',prepared=str(Path(prepared).absolute()),prepare_sha256=expected,
            source_binding=p['arguments'],slots=p['slots'],images=images,requests=rows,resources=counts,module_ledger=dict(visual=visual.entries,prediction=pred.entries),
            errors=errors,first_error=errors[0] if errors else None,forbidden_AD_attempts=forbidden,blocked_visual=visual.blocked,
            contract=p['contract'],source_verified=source_ok,model_hash_before=before,model_hash_after=after,model_unchanged=before==after if before else None,
            host=host,limits=limits.receipt(),artifact_sha256=hashes,AD=0 if complete else None,physics=0,selection=False,P1_admission=False)
        try: store.put('report.json',report,final=True)
        except BaseException as exc:
            complete=False; errors.append(low.error('final_report',exc)); print(canonical(dict(errors=errors)).decode(),flush=True)
    require(complete,'partial retained; no retry: '+str(errors)); return report


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('command',choices=('prepare','verify','run'))
    for key in ('sourcepack','source-sha','encoder-report','encoder-sha','prepared','prepare-sha','output'): parser.add_argument('--'+key)
    a=parser.parse_args()
    if a.command=='prepare': r=prepare(a.output,sourcepack=a.sourcepack,source_sha=a.source_sha,encoder_report=a.encoder_report,encoder_sha=a.encoder_sha)
    elif a.command=='verify': r=dict(status='saved_inputs_verified_no_neural',slots=len(verify(a.prepared,a.prepare_sha)[1]))
    else: r=run(a.prepared,a.prepare_sha,a.output)
    print(canonical(r).decode(),end='')


if __name__=='__main__': main()
