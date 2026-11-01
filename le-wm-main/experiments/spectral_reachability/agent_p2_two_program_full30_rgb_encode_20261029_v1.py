"""Four full30 plans x seven actual checkpoints, MAIN HOST CUDA encode only.

Own physical authority and encoder consumer; original frozen native192/Q.
No predictions, training, physics or scientific admission in this stage.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import inspect
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import agent_p2_contextual_midfeedback_recompile_20261028_v1 as intervention

bridge=intervention.bridge
physical=bridge.physical
require,sha,raw=physical.require,physical.sha,physical.raw
ROOT=bridge.ROOT
from experiments.spectral_reachability import agent_p2_two_program_full30_physical_20261029_v1 as producer
from experiments.spectral_reachability import agent_p2_two_program_postcontext_20261029_v1_kernel as K
import gzip
import io
import zlib
META=16*1024**2
CAP,ARRAY_CAP,EVENT_CAP,FINAL,FLOOR,WALL=64*1024**2,48*1024**2,8*1024**2,8*1024**2,200*1024**2,600
CP=(0,5,10,15,20,25,30)
PRODUCER_SHA='df65ccb93a800dea8b8e810b488b4818a70f058ee721b685dd49ef10d680a4a9'
PROTOCOL=ROOT/'innerdocs/agent_p2_two_program_full30_rgb_protocol_20261029_v1.md'
PROTOCOL_SHA='c14a4cf207dc4c19eea6502ba198e5ce3da35d1c79974cc146c604ce66da8608'
ACTUAL=dict(physical_prepared=str(ROOT/'results/agent_p2_two_program_full30_physical_20261029_v1_prepare_MAIN01.json'),
    physical_prepare_sha='04d4a887e09381cc9869d76b259af41159d6678269c48e5a3cf0caa42df66100',
    physical_output=str(ROOT/'results/agent_p2_two_program_full30_physical_20261029_v1_MAIN01'),
    physical_report_sha='39dae04ac59d9fd1b1e3e65dfe675a4627c13ada00d8c065e655053018b35c90',
    physical_receipt=str(ROOT/'results/agent_p2_two_program_full30_physical_20261029_v1_consume_MAIN01.json'),
    physical_receipt_sha='aabc5cce3855021eba0fe5751c0ae87809cfefb4c2ed54b0d2899048fd6b3cd8',
    physical_source_sha=PRODUCER_SHA)


def read(path):return producer.read(path,META)
NAME='agent_p2_two_program_full30_rgb_encode_20261029_v1'
AUTH='MAIN_FULL30_28_RGB_HOST_CUDA_ONLY'


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def precision_contract():
    return dict(matmul_TF32=False,cudnn_TF32=False,autocast=False,math=True,flash=False,efficient=False,cudnn_sdp=False)


def contract():
    return dict(schema=NAME+'.contract',fixed_denominator=28,plan_denominator=4,known_roots=2,fresh_roots=0,
        checkpoints=list(CP),raw_control_steps=30,anchor=0,forecasts=24,
        max_calls={k:28 if k in ('encoder','projector') else 0 for k in bridge.KINDS},
        AD=0,new_physics=0,new_optimizer=0,training=False,Qfit=False,batch=1,no_dedup=True,no_padding=True,
        native_hidden=[1,257,192],latent=192,encoding='original encode CLS/projector; no extra latent normalization',
        Q='original saved goal192 identity mean-square',raw_normalizer='original float32 subtract/divide once',
        CUDA_only=True,TF32=False,autocast=False,SDP='MATH',total_FLOPs=None,equal_FLOPs=False,equal_total_compute=False,
        wall_seconds=WALL,output_cap_bytes=CAP,array_cap_bytes=ARRAY_CAP,event_metadata_cap_bytes=EVENT_CAP,
        final_reserve_bytes=FINAL,disk_floor_bytes=FLOOR,cuda_allocated_bytes=3*1024**3,cuda_reserved_bytes=4*1024**3,
        metadata_cap_bytes=META,full_upstream_rebuilt_here=False,
        P1_admission=False,P2_admission=False,P2_mechanism=False,C3=False,scientific_pass=False,physical_release=False)


def join_actions(prep,report,events):
    require(report['rows'] and len(report['rows'])==4,'all4 physical source rows')
    require(events and events[0]['kind']=='run_started' and events[-1]['kind']=='run_finished','native journal bookends')
    tail=None
    allowed={'run_started','run_finished','operation_dispatch_intent','operation_returned','operation_failed',
        'placement_recipe','goal_attached','native_history_guard','invalid_observation','fresh_observation',
        'returned_transition','own_response','branch_closed'}
    for i,e in enumerate(events):
        require(e['kind'] in allowed and type(e['index']) is int and e['index']==i
            and e['previous_sha256']==tail and e['event_sha256']==producer.source.digest(
                {k:v for k,v in e.items() if k!='event_sha256'}),'all physical source events/chain no unknown filtering')
        tail=e['event_sha256']
    require(tail==report['journal_chain_tail'],'source native journal tail')
    for i,(plan,row) in enumerate(zip(prep['plans'],report['rows'],strict=True)):
        scoped=[e for e in events if e.get('branch')==i]
        guards=[e for e in scoped if e['kind']=='native_history_guard']
        require(len(guards)==int(row['initial_exact']),'source native history guard presence')
        if guards:
            g=guards[0]
            require(g['guard']['root_context_sha256']==K.digest(plan['planner_plan']['root_context'])
                and g['guard']['task_id']==plan['task_id']
                and g['guard']['warm_raw_float32_bits']==producer.moving.warmup_commands().tobytes().hex()
                and g['guard']['history_controls']==[-10,-5,0]
                and g['guard']['history_absolute_steps']==[0,5,10],'actual root guard/task/context/raw source join')
        for phase,n,cmd in (('warmup',row['warmup_returned'],producer.moving.warmup_commands()),
                            ('plan',row['own_returned'],producer.raw30(plan['commands']))):
            transitions=[e for e in scoped if e['kind']=='returned_transition' and e['phase']==phase]
            require([e['step'] for e in transitions]==list(range(1,n+1)),'all actual same-action source returns')
            for e in transitions:
                want=cmd[e['step']-1].tobytes().hex()
                require(e['actual_dtype']==np.dtype(np.float32).str and e['requested_bits']==e['actual_bits']==want,
                    'actual raw bits same frozen action/time, no repair')
        intents=[e for e in scoped if e['kind']=='operation_dispatch_intent' and
                 e['receipt']['resource']=='public_step' and e['receipt'].get('phase')=='plan']
        require(not intents or (len(guards)==1 and all(g['index']<e['index'] for e in intents)),
                'source history accepted before plan intent')


def fixed_rows(prep,strict,modelprep,output):
    require(strict['schema']==producer.NAME+'.consume' and strict['status']==producer.RECORDED
        and strict['fixed_denominator']==4 and strict['known_roots']==2 and strict['errors']==[]
        and strict['own_native_saved_reconstructed_here'] is True
        and all(strict[k] is False for k in ('P1_admission','P2_admission','P2_mechanism','physical_release')),
        'MAIN complete own full30 physical receipt, not an old schema')
    producer.validate_execution(prep)
    require(len(strict['rows'])==len(strict['plans'])==4 and strict['plans']==prep['plans']
        and modelprep['schema']==K.NAME+'.prepare' and modelprep['slots']==prep['model_slots']
        and modelprep['expected_model_hash']==prep['expected_model_hash'],'all4 frozen model/physical plans')
    rows=[]
    for i,(plan,actual) in enumerate(zip(prep['plans'],strict['rows'],strict=True)):
        key=producer.ORDER[i];w=plan['planner_plan'];slot=modelprep['slots'][key[0]]
        require((actual['index'],actual['root_slot'],actual['condition'],actual['arm'])==(i,*key)
            and actual['plan']==plan and actual['selection_digest']==plan['selection_digest']
            and actual['task_id']==plan['task_id'] and actual['status'] in ('horizon_completed','native_stopped'),
            'fixed same raw30 physical row/plan identity')
        producer.check_planner_plan(w,slot,producer.PINS['planner_report_sha'],modelprep['expected_model_hash'])
        commands=producer.raw30(plan['commands']);warm=producer.moving.warmup_commands()
        require(type(actual['own_returned']) is int and 0<=actual['own_returned']<=30,'literal actual eventtime')
        require(actual['missing_checkpoints']==[cp for cp in CP if cp not in [c['step'] for c in actual['checkpoints']]],
            'canonical missing checkpoints')
        for j,step in enumerate(CP):
            cp=[c for c in actual['checkpoints'] if c['step']==step]
            frames=[f for f in actual['trajectory'] if f['phase']=='plan' and f['step']==step and f['overlay_index']==0]
            responses=[v for v in actual['responses'] if v['step']==step]
            present=actual['initial_exact'] and step<=actual['own_returned']
            require(len(cp)==len(frames)==int(present) and len(responses)==int(present and step>0),
                'unique real checkpoint, cp0 has no plan response; missing not padded')
            source=native_metric=response=None
            if present:
                f=frames[0]
                require(cp[0]=={k:f[k] for k in ('step','png','file_sha256','pixel_sha256','state','snapshot')}
                    and f['condition']=='off' and f['state']==f['snapshot']['state']
                    and Path(f['png']).name==f['png'],'same-time off PNG/native state')
                if step==0:
                    require(f['state']==actual['root_state']==prep['source_roots'][key[0]]['root_state']
                        and actual['root_sha256']==plan['native_root_sha256'],'authenticated cp0 anchor not warm frame')
                else:
                    response=responses[0]
                    require(response['state']==f['state'],'exact same-time public response')
                    native_metric=producer.native.metrics(plan['native_goal'],response['state'])
                    require(response['native_metrics']==native_metric['native']
                        and response['reward']==native_metric['native7_reward']
                        and response['native_success'] is response['terminated'] is native_metric['native_success'],
                        'unchanged original native goal labels')
                source=deepcopy(cp[0]);source['path']=str(Path(output).absolute()/source['png'])
            history=np.concatenate((warm,commands[:step]))[-10:]
            pred=None if step==0 else bridge.latent(w['contexts'][j-1]['latent_history'][-1]).tolist()
            rows.append(dict(index=len(rows),plan_index=i,root_slot=key[0],condition='off',arm=key[2],
                task_id=plan['task_id'],checkpoint=step,absolute_step=10+step,selection_digest=plan['selection_digest'],
                witness_sha256=w['witness_sha256'],model_hash=modelprep['expected_model_hash'],
                goal192=bridge.latent(w['goal']).tolist(),predicted192=pred,
                observed_anchor192=bridge.latent(w['root_context']['latent_history'][-1]).tolist() if step==0 else None,
                model_Q=None if pred is None else bridge.mse(pred,w['goal']),full30=plan['commands'],
                raw_float32_bits=commands.tobytes().hex(),causal_raw10=history.tolist(),
                causal_raw10_bits=history.tobytes().hex(),
                normalized_action_history=producer.normalization.normalized_history(history).tolist(),
                physical_status=actual['status'],actual_returned=actual['own_returned'],
                native_metrics=native_metric,response=response,source=source,
                status='not_started' if present else 'missing_physical_checkpoint',latent=None))
    return rows


def build(**arguments):
    require(arguments==ACTUAL,'only pinned MAIN actual physical three authorities before input reads')
    require(sha(producer.__file__)==PRODUCER_SHA and sha(PROTOCOL)==PROTOCOL_SHA,'frozen source/protocol pins')
    paths={'physical_prepared':'physical_prepare_sha','physical_receipt':'physical_receipt_sha'}
    for k,h in paths.items():require(sha(arguments[k])==arguments[h],'actual physical authority SHA')
    output=Path(arguments['physical_output']);rp=output/'report.json'
    require(sha(rp)==arguments['physical_report_sha'],'actual native report SHA')
    prep=read(arguments['physical_prepared']);strict=read(arguments['physical_receipt']);report=read(rp)
    require(report['schema']==producer.NAME+'.report' and report['rows']==strict['rows']
        and report['execution_binding']['prepared']==arguments['physical_prepared']
        and report['execution_binding']['prepare_sha256']==strict['prepare_sha256']==arguments['physical_prepare_sha']
        and strict['physical_report_sha256']==arguments['physical_report_sha'],'complete actual three-authority join')
    mp=producer.PINS
    for path,h in ((mp['planner_prepared'],mp['planner_prepare_sha']),
                   (str(Path(mp['planner_output'])/'report.json'),mp['planner_report_sha']),
                   (mp['planner_receipt'],mp['planner_receipt_sha'])):
        require(sha(path)==h,'actual frozen model authority')
    modelprep=read(mp['planner_prepared']);mr=read(Path(mp['planner_output'])/'report.json')
    require(sha(output/'journal.jsonl')==report['journal_sha256'],'original native history/action source bytes')
    require((output/'journal.jsonl').stat().st_size<=producer.JOURNAL_CAP,'bounded source journal before read')
    journal=(output/'journal.jsonl').read_bytes()
    require(len(journal)<=producer.JOURNAL_CAP and journal.endswith(b'\n') and b'\n\n' not in journal,'bounded native singleLF journal')
    join_actions(prep,report,[producer.native.decode(line) for line in journal.splitlines()])
    rows=fixed_rows(prep,strict,modelprep,output)
    bindings={};producer.native.merge(bindings,strict['bindings']);producer.native.merge(bindings,prep['bindings'])
    producer.native.merge(bindings,{str(rp):arguments['physical_report_sha'],
        arguments['physical_prepared']:arguments['physical_prepare_sha'],
        arguments['physical_receipt']:arguments['physical_receipt_sha'],str(PROTOCOL):PROTOCOL_SHA})
    producer.native.merge(bindings,{str(Path(f).resolve()):sha(f) for f in bridge.source_files()+own_files()+producer.own_files()+[Path(K.__file__)]})
    producer.native.merge(bindings,bridge.rgb.MODELS)
    require({str(f.absolute()) for f in output.iterdir()}=={f for f in bindings if Path(f).parent==output},
        'entire physical inventory is bound, no unknown source artifacts')
    for row in rows:
        if row['source'] is not None:bridge.load_pixels(row['source'])
    producer.rehash(bindings);n=sum(x['source'] is not None for x in rows)
    require(Path(modelprep['policy']).resolve()==bridge.rgb.POLICY.resolve(),'original policy/encoder/transform pairing')
    shapes=bridge.rgb.config_shapes(bridge.rgb.read(bridge.rgb.POLICY/'config.json'))
    return dict(schema=NAME+'.prepare',arguments=deepcopy(arguments),contract=contract(),rows=rows,bindings=bindings,
        plans=deepcopy(prep['plans']),shapes=shapes,expected_calls={k:n if k in ('encoder','projector') else 0 for k in bridge.KINDS},
        policy=modelprep['policy'],expected_model_hash=modelprep['expected_model_hash'],array_budget=array_budget(),
        inherited_physical_counts=strict['counts'],inherited_model_resources=mr['actual_resources'],
        inherited_model_wall_seconds=mr['wall_seconds'],physical_report_sha256=arguments['physical_report_sha'],
        physical_receipt_sha256=arguments['physical_receipt_sha'],full_upstream_rebuilt_here=False,
        inherited_evidence='MAIN pinned complete model/native receipts; no upstream fullstrict rerun')

def write(path,value):
    path=Path(path).absolute();payload=raw(value)
    require(path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink() and len(payload)<=META
        and shutil.disk_usage(path.parent).free>=FLOOR+len(payload),'own exclusive bounded metadata')
    with path.open('xb') as f:require(f.write(payload)==len(payload),'full metadata');f.flush();os.fsync(f.fileno())
    bridge.rgb.syncdir(path.parent);return dict(path=str(path),sha256=sha(path),status=value.get('status','prepared'))


def verify(path,expected):
    require(sha(path)==expected,'external own prepare SHA');p=read(path)
    require(raw(p)==raw(build(**p['arguments'])),'entire prepare reconstruction');return p


class Budget:
    def __init__(self,clock=None):
        self.clock=clock or time.monotonic;self.started=self.clock();self.used=0
        self.groups=dict(arrays=0,metadata=0,final=0)
        self.free=lambda:shutil.disk_usage(ROOT).free
    def check(self,size=0,final=False):
        require(final or self.clock()-self.started<WALL,'whole600s before dispatch/write')
        require(self.used+size<=CAP-(0 if final else FINAL) and self.free()>=FLOOR+size,'64MiB/reserve8MiB/floor200MiB')


class Store(bridge.rgb.Artifacts):
    def __init__(self,output,budget):
        self.path=Path(producer.native.absolute(str(output)))
        require(self.path.name.startswith(NAME+'_') and not self.path.exists() and not self.path.is_symlink()
            and self.path.parent.is_dir() and shutil.disk_usage(self.path.parent).free>=CAP+FLOOR,'new own64MiB output')
        self.path.mkdir();bridge.rgb.syncdir(self.path.parent);self.hashes={};self.budget=budget;self.index=0
        budget.free=lambda:shutil.disk_usage(self.path).free
    def check(self,size=0,final=False):self.budget.check(size,final)
    def write(self,name,value,final=False):
        require(type(value) is bytes and Path(name).name==name and name not in ('','.','..')
            and not (self.path/name).is_symlink(),'flat original bytes')
        group='arrays' if name.endswith('.npy.gz') else ('final' if final else 'metadata')
        require(not (group=='arrays' and final),'arrays cannot borrow final reserve')
        limit=ARRAY_CAP if group=='arrays' else EVENT_CAP
        require(self.budget.groups[group]+len(value)<=limit,'group byte cap '+group)
        self.check(len(value),final)
        path=self.path/name
        try:
            with path.open('xb') as f:
                require(f.write(value)==len(value),'complete write');f.flush();os.fsync(f.fileno())
        finally:
            if path.is_file() and name not in self.hashes:
                size=path.stat().st_size;self.budget.used+=size;self.budget.groups[group]+=size
        bridge.rgb.syncdir(self.path);h=hashlib.sha256(value).hexdigest();self.hashes[name]=h;return h
    def array(self,name,a):
        checked_array(a)
        return super().array(name,a)
    def event(self,e):
        self.put(f'event_{self.index:04}.json',e);self.index+=1


SHAPES={(224,224,3),(1,1,3,224,224),(1,3,224,224),(1,257,192),(1,192),(192,)}


def checked_array(a):
    require(isinstance(a,np.ndarray) and tuple(a.shape) in SHAPES
        and a.dtype==(np.uint8 if a.shape==(224,224,3) else np.float32) and np.isfinite(a).all(),
        'exact finite native array dtype/shape, no scalar codec promotion')
    return a


def load_array(ref,blob):
    require(type(ref) is dict and set(ref)=={'file','sha256','array_sha256','shape','dtype'},'exact native array descriptor')
    name=ref['file'];require(Path(name).name==name and name.endswith('.npy.gz'),'flat array name')
    payload=blob(name);require(len(payload)<=2*1024**2 and hashlib.sha256(payload).hexdigest()==ref['sha256'],'array file seal')
    dec=zlib.decompressobj(31);data=dec.decompress(payload,2*1024**2+1)
    require(len(data)<=2*1024**2 and dec.eof and not dec.unused_data and not dec.unconsumed_tail,
        'one complete gzip member, no tail')
    stream=io.BytesIO(data);require(np.lib.format.read_magic(stream)==(1,0),'NPY1 header')
    shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
    require(shape in SHAPES and not fortran and dtype==(np.dtype('uint8') if shape==(224,224,3) else np.dtype('<f4'))
        and ref['shape']==list(shape) and ref['dtype']==str(dtype)
        and len(data)-stream.tell()==int(np.prod(shape))*dtype.itemsize,'bounded exact NPY shape before allocation')
    a=np.frombuffer(data,offset=stream.tell(),dtype=dtype).reshape(shape).copy();checked_array(a)
    require(hashlib.sha256(a.tobytes()).hexdigest()==ref['array_sha256'],'array literal bytes seal')
    return a


def byte_census(files):
    arrays=sum(len(b) for n,b in files.items() if n.endswith('.npy.gz'))
    final_names={'native_seal.json','report.json','completion.json','terminal_failure.json'}
    final=sum(len(b) for n,b in files.items() if n in final_names)
    metadata=sum(len(b) for n,b in files.items() if not n.endswith('.npy.gz') and n not in final_names)
    require(arrays<=ARRAY_CAP and metadata<=EVENT_CAP and final<=FINAL and arrays+metadata+final<=CAP,
        'actual full arrays48MiB/eventsmetadata8MiB/final8MiB census')
    return dict(arrays=arrays,metadata=metadata,final=final,total=arrays+metadata+final)


def array_budget():
    shapes=((224,224,3),(1,1,3,224,224),(1,3,224,224),(1,257,192),(1,192),(1,192),(192,))
    payload=npy=compressed_bound=0
    for shape in shapes:
        a=np.zeros(shape,np.uint8 if shape==(224,224,3) else np.float32)
        stream=io.BytesIO();np.save(stream,a,allow_pickle=False);n=len(stream.getvalue())
        payload+=a.nbytes;npy+=n
        # Conservative zlib compressBound plus gzip wrapper allowance; no assumed compressibility.
        compressed_bound+=n+(n>>12)+(n>>14)+(n>>25)+32
    require(28*compressed_bound<=ARRAY_CAP,'all196 arrays including NPY header and gzip worst bound fit48MiB')
    return dict(images=28,arrays=196,payload=28*payload,npy=28*npy,gzip_upper_bound=28*compressed_bound,
                array_cap=ARRAY_CAP,metadata_cap=EVENT_CAP,final_reserve=FINAL)


def process(p,rows,store,encode,check=lambda:None):
    require(len(rows)==28,'all28 source rows including missing')
    require(rows==p['rows'],'immutable complete predispatch roster')
    ordinal=0
    for row in rows:
        if row['source'] is None:continue
        check();scope=dict(row=row['index'],step=row['checkpoint'],phase='encode');row['status']='attempted'
        store.event(dict(kind='image_attempt',scope=scope,guard_image=ordinal))
        pixels=bridge.load_pixels(row['source']);row['pixels']=store.array(f'r{row["index"]}_pixels',pixels)
        z=encode(pixels,row,scope,ordinal);bridge.latent(z)
        row['latent']=store.array(f'r{row["index"]}_latent',z);row['status']='encoded'
        store.event(dict(kind='image_return',scope=scope,row=deepcopy(row)));ordinal+=1


def precision_state(torch):
    return dict(matmul_TF32=torch.backends.cuda.matmul.allow_tf32,cudnn_TF32=torch.backends.cudnn.allow_tf32,
        autocast=torch.is_autocast_enabled('cuda'),math=torch.backends.cuda.math_sdp_enabled(),
        flash=torch.backends.cuda.flash_sdp_enabled(),efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
        cudnn_sdp=torch.backends.cuda.cudnn_sdp_enabled())


def model_inventory(model,bindings):
    result=[]
    for module in model.modules():
        path=str(Path(inspect.getfile(type(module))).resolve())
        require(path in bindings and sha(path)==bindings[path],'presealed installed model class source')
        result.append(dict(type=type(module).__module__+'.'+type(module).__qualname__,path=path,sha256=bindings[path]))
    require(result,'complete module inventory');return result


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN explicit CUDA before I/O')
    budget=Budget();p=verify(prepared,expected);budget.check();store=Store(output,budget)
    rows=deepcopy(p['rows']);precision=None;errors=[];trace=guard=None;model=None;before=after=None;source_ok=False
    numeric_before=numeric_after=None;classes_before=classes_after=None;peak=None
    def check():
        store.check()
        require(torch.cuda.memory_allocated(device)<=contract()['cuda_allocated_bytes'] and
            torch.cuda.memory_reserved(device)<=contract()['cuda_reserved_bytes'],'actual CUDA memory caps')
        require(precision_state(torch)==precision_contract(),'actual TF32/autocast/MATH contract')
    store.put('binding.json',dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected));store.put('initial_roster.json',rows)
    try:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        require_cuda_allocation('cuda:0');device=torch.device('cuda:0');torch.cuda.reset_peak_memory_stats(device)
        precision=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        bridge.native.frozen(model);before=bridge.native.tensor_hash(model);require(before==p['expected_model_hash'],'original frozen model')
        classes_before=model_inventory(model,p['bindings'])
        trace=bridge.Trace(model,torch,store,check,p['expected_calls'])
        guard=bridge.native.Ledger(7,lambda e:store.event(dict(kind='guard_event',event=e)),
            lambda:torch.cuda.synchronize(device),lambda:not torch.is_grad_enabled())
        transform=p0.image_transform()
        def encode(pixels,row,scope,ordinal):
            guard.image=ordinal
            with physical.saved.overlay.traced_scope(trace,scope,row):
                x=p0._image_tensor(pixels,transform=transform,device=device)
                row['preprocessed']=trace.save(x,f'r{row["index"]}_preprocessed')
                bridge.same(x.detach().cpu().numpy(),bridge.pixels.preprocess_pixels(pixels),'original preprocessing exactly')
                out=model.encode({'pixels':x})['emb'];require(tuple(out.shape)==(1,1,192),'native192 only')
                return out[0,0].detach().cpu().numpy().copy()
        with trace,bridge.native.guard(model,torch,guard,errors),torch.inference_mode(),\
            torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            numeric_before=precision_state(torch);check()
            process(p,rows,store,encode,check)
            torch.cuda.synchronize(device);check();numeric_after=precision_state(torch)
            peak=[torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device)]
    except BaseException as e:errors.append(physical.kernel.error('encode',e))
    finally:
        if precision is not None:
            torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=precision
        try:
            if model is not None:
                bridge.native.frozen(model);after=bridge.native.tensor_hash(model);classes_after=model_inventory(model,p['bindings'])
            bridge.verify_bindings(p['bindings']);require(sha(prepared)==expected,'prepare unchanged');budget.check();source_ok=True
        except BaseException as e:errors.append(physical.kernel.error('final_closure_model_limits',e))
        entries=[] if trace is None else trace.entries;guards=[] if guard is None else guard.entries;blocked=[] if guard is None else guard.blocked
        complete=(not errors and not blocked and source_ok and before==after==p['expected_model_hash']
            and classes_before==classes_after and numeric_before==numeric_after==precision_contract()
            and peak is not None and all(v<=contract()[k] for v,k in zip(peak,('cuda_allocated_bytes','cuda_reserved_bytes'),strict=True)) and
            bridge.counts(entries)=={k:dict(attempted=v,completed=v) for k,v in p['expected_calls'].items()})
        # native seal precedes any analysis of the new encoded outputs.
        seal=dict(rows=rows,modules=entries,guards=guards,blocked=blocked,errors=errors)
        h=store.put('native_seal.json',seal,final=True)
        report=dict(schema=NAME+'.report',status='terminal_encoded_science_blocked' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).resolve()),prepare_sha256=expected,contract=contract(),**seal,
            counts=bridge.counts(entries),model_hash_before=before,model_hash_after=after,model_unchanged=before is not None and before==after,
            source_verified=source_ok,native_seal_sha256=h,artifact_sha256=deepcopy(store.hashes),event_count=store.index,
            elapsed_seconds=time.monotonic()-budget.started,precision_before=numeric_before,precision_after=numeric_after,
            model_inventory_before=classes_before,model_inventory_after=classes_after,CUDA_peak_bytes=peak,
            P1_admission=False,P2_admission=False,P2_mechanism=False,C3=False,physical_release=False)
        store.put('report.json',report,final=True)
        try:
            h=sha(store.path/'report.json');elapsed=time.monotonic()-budget.started
            require(0<=elapsed<WALL,'whole600s through report fsync/hash')
            store.put('completion.json',dict(schema=NAME+'.completion',report_sha256=h,elapsed_seconds=elapsed),final=True)
            require(time.monotonic()-budget.started<WALL,'completion publication within600s')
        except BaseException as e:
            complete=False
            store.put('terminal_failure.json',dict(stage='completion',error=repr(e)),final=True)
    require(complete,'partial retained/no retries '+str(errors))
    return dict(report=str(store.path/'report.json'),report_sha256=sha(store.path/'report.json'))


def analyze(spec,row,load):
    z=load(row['latent']) if row['latent'] is not None else None
    actual=bridge.mse(z,spec['goal192']) if z is not None else None
    anchor=spec['checkpoint']==0
    if z is not None and anchor:bridge.same(z,bridge.latent(spec['observed_anchor192']),'exact original observed cp0 re-encode')
    pred=spec['model_Q']
    return dict(index=spec['index'],plan_index=spec['plan_index'],root_slot=spec['root_slot'],condition='off',
        arm=spec['arm'],task_id=spec['task_id'],checkpoint=spec['checkpoint'],selection_digest=spec['selection_digest'],
        role='observed_anchor_reencode' if anchor else 'same_action_H6_prediction',
        model_Q=pred,actual_encoded_Q=actual,
        model_minus_actual_Q=pred-actual if pred is not None and actual is not None else None,
        absolute_Q_error=abs(pred-actual) if pred is not None and actual is not None else None,
        prediction_vs_encoded_MSE=bridge.mse(z,spec['predicted192']) if z is not None and not anchor else None,
        observed_anchor_reencode_MSE=bridge.mse(z,spec['observed_anchor192']) if z is not None and anchor else None,
        actual192=None if z is None else z.tolist(),predicted192=spec['predicted192'],goal192=spec['goal192'],
        latent_missing=z is None,source=spec['source'],native_metrics=spec['native_metrics'],response=spec['response'],
        causal_raw10_bits=spec['causal_raw10_bits'],physical_status=spec['physical_status'],
        physical_benefit_claim=False,P1_admission=False,P2_admission=False,C3=False)


def context_chains(p,results):
    chains=[]
    for i,plan in enumerate(p['plans']):
        context=deepcopy(plan['planner_plan']['root_context']);items=[]
        for j,step in enumerate(CP):
            row=results[7*i+j];spec=p['rows'][7*i+j];z=row['actual192']
            if z is None:context=None
            elif context is not None:
                if step==0:context['latent_history'][-1]=deepcopy(z)
                else:
                    context['latent_history']=deepcopy(context['latent_history'][1:])+[deepcopy(z)]
                    context['action_history']=deepcopy(spec['normalized_action_history'])
                K.context(context)
                require(context['action_history']==spec['normalized_action_history'],'causal actual history tokens once')
            items.append(dict(step=step,context=deepcopy(context),context_missing=context is None))
        chains.append(dict(plan_index=i,root_slot=plan['root_slot'],arm=plan['arm'],checkpoints=items,
            scope='derived observed contexts only, no new model prediction/control intervention'))
    return chains

def validate(p,r,events,load):
    require(r['contract']==p['contract']==contract() and len(p['rows'])==len(r['rows'])==28 and r['errors']==r['blocked']==[],'all28 clean checkpoint report')
    modules=[];guards=[];want=[];ordinal=0
    for spec,row in zip(p['rows'],r['rows'],strict=True):
        require({k:v for k,v in row.items() if k not in ('status','latent','pixels','preprocessed','modules')}==
            {k:v for k,v in spec.items() if k not in ('status','latent')},'original plan/new witness/goal/frame identity')
        if spec['source'] is None:require(row==spec,'missing not replaced or padded');continue
        require(row['status']=='encoded','available encoded once');pix=load(row['pixels'])
        bridge.same(pix,bridge.load_pixels(spec['source']),'actual new terminal PNG pixels')
        pre=load(row['preprocessed']);bridge.same(pre,bridge.pixels.preprocess_pixels(pix),'exact preprocessed input')
        z=load(row['latent']);bridge.latent(z);scope=dict(row=row['index'],step=row['checkpoint'],phase='encode')
        mods=row['modules'];require(len(mods)==2 and [m['resource'] for m in mods]==['encoder','projector']
            and all(m['scope']==scope and m['completed'] is True and len(m['inputs'])==1 for m in mods),'full scoped actual encoder/projector')
        en,proj=mods
        buffers=[pre,load(en['inputs'][0]),load(en['output']),load(proj['inputs'][0]),load(proj['output']),z]
        require(all(x.dtype==np.float32 and np.isfinite(x).all() for x in buffers),'literal full native float32 buffers')
        physical.saved.overlay.visual_bridge(*buffers,[1,257,192])
        want.append(dict(kind='image_attempt',scope=scope,guard_image=ordinal))
        for mod in mods:
            g=dict(index=len(guards),image=ordinal,resource=mod['resource'],completed=True);guards.append(g)
            attempt={k:mod[k] for k in ('index','scope','resource')}|dict(completed=False)
            want += [dict(kind='module_attempt',entry=attempt),dict(kind='guard_event',event=dict(kind='module_attempt',entry=dict(g,completed=False))),
                dict(kind='guard_event',event=dict(kind='module_return',entry=g)),dict(kind='module_return',entry=mod)]
        modules.extend(mods);want.append(dict(kind='image_return',scope=scope,row=row));ordinal+=1
    require(modules==r['modules'] and [m['index'] for m in modules]==list(range(len(modules))) and r['guards']==guards
        and events==want and r['counts']==bridge.counts(modules)=={k:dict(attempted=v,completed=v) for k,v in p['expected_calls'].items()},
        'entire native+forbidden-API guard chronology/counts, no unknown filtered')
    return [analyze(spec,row,load) for spec,row in zip(p['rows'],r['rows'],strict=True)]


def consume(output,expected):
    output=Path(output).resolve();require(sha(output/'report.json')==expected,'external actual encoder report SHA')
    r=read(output/'report.json');p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and r['status']=='terminal_encoded_science_blocked' and
        r['model_unchanged'] is r['source_verified'] is True and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash']
        and r['P1_admission'] is r['P2_admission'] is r['P2_mechanism'] is r['C3'] is r['physical_release'] is False and type(r['elapsed_seconds']) in (int,float) and np.isfinite(r['elapsed_seconds']) and 0<=r['elapsed_seconds']<WALL,'complete frozen CUDA only')
    require(r['precision_before']==r['precision_after']==precision_contract()
        and r['model_inventory_before']==r['model_inventory_after'] and r['model_inventory_before'],
        'actual numeric flags and presealed model inventory before/after')
    for entry in r['model_inventory_before']:
        require(p['bindings'][entry['path']]==entry['sha256'],'bound installed class identity')
    require(type(r['CUDA_peak_bytes']) is list and len(r['CUDA_peak_bytes'])==2
        and all(type(v) is int and 0<=v<=contract()[k] for v,k in zip(r['CUDA_peak_bytes'],
            ('cuda_allocated_bytes','cuda_reserved_bytes'),strict=True)),'actual CUDA peak resource axes')
    require(all(f.is_file() and not f.is_symlink() for f in output.iterdir()),'regular flat output artifacts')
    require({f.name for f in output.iterdir()}==set(r['artifact_sha256'])|{'report.json','completion.json'} and
        sum(f.stat().st_size for f in output.iterdir())<=CAP,'complete bounded output census')
    census=byte_census({f.name:f.read_bytes() for f in output.iterdir()})
    completion=read(output/'completion.json')
    require(set(completion)=={'schema','report_sha256','elapsed_seconds'} and completion['schema']==NAME+'.completion'
        and completion['report_sha256']==expected and type(completion['elapsed_seconds']) in (int,float)
        and np.isfinite(completion['elapsed_seconds']) and r['elapsed_seconds']<=completion['elapsed_seconds']<WALL,
        'whole stage completion report hash/wall')
    def blob(n):return bridge.rgb.flat(output,n).read_bytes()
    for n,h in r['artifact_sha256'].items():require(hashlib.sha256(blob(n)).hexdigest()==h,'all artifact file SHA')
    metadata_names={'binding.json','initial_roster.json','native_seal.json'}|{f'event_{i:04}.json' for i in range(r['event_count'])}
    require({n for n in r['artifact_sha256'] if not n.endswith('.npy.gz')}==metadata_names,
        'no unknown metadata artifacts declared by producer')
    require(read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']) and
        read(output/'initial_roster.json')==p['rows'],'original entire execution roster/binding')
    require(sha(output/'native_seal.json')==r['native_seal_sha256'] and read(output/'native_seal.json')==
        {k:r[k] for k in ('rows','modules','guards','blocked','errors')},'native output sealed before saved analysis')
    used=set()
    def load(ref):
        require(ref['file'] in r['artifact_sha256'],'referenced array in sealed inventory')
        used.add(ref['file']);return load_array(ref,blob)
    rows=validate(p,r,[read(output/f'event_{i:04}.json') for i in range(r['event_count'])],load)
    require(used=={n for n in r['artifact_sha256'] if n.endswith('.npy.gz')},'all saved arrays consumed, no extra blobs')
    bridge.verify_bindings(p['bindings']);require(sha(output/'report.json')==expected,'final report unchanged')
    return dict(schema=NAME+'.consume',status='all28_full30_RGB_verified_NOT_admitted',rows=rows,report_sha256=expected,
        counts=r['counts'],fixed_denominator=28,plan_denominator=4,known_roots=2,fresh_roots=0,total_FLOPs=None,actual_context_chains=context_chains(p,rows),
        physical_report_sha256=p['physical_report_sha256'],physical_receipt_sha256=p['physical_receipt_sha256'],
        inherited_physical_counts=p['inherited_physical_counts'],inherited_model_resources=p['inherited_model_resources'],
        inherited_model_wall_seconds=p['inherited_model_wall_seconds'],full_upstream_rebuilt_here=False,
        own_native_encoding_saved_reconstructed_here=True,prepare_sha256=r['prepare_sha256'],
        byte_census=census,array_budget=p['array_budget'],consume_new_neural_calls=0,consume_new_physics=0,
        P1_admission=False,P2_admission=False,P2_mechanism=False,C3=False,equal_total_compute=False,physical_release=False)



def main():
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('output','prepared','prepare-sha','report-sha','receipt'):q.add_argument('--'+k)
    for k,v in ACTUAL.items():q.add_argument('--'+k.replace('_','-'),default=v)
    q.add_argument('--authorize-host-cuda',action='store_true');a=q.parse_args()
    if a.command=='prepare':v=write(a.output,build(**{k:getattr(a,k) for k in ACTUAL}))
    elif a.command=='verify':p=verify(a.prepared,a.prepare_sha);v=dict(status='verified_no_execution',expected_calls=p['expected_calls'])
    elif a.command=='run':
        require(a.authorize_host_cuda,'MAIN explicit CUDA flag');v=run(a.prepared,a.prepare_sha,a.output,AUTH)
    else:
        require(a.receipt,'own saved receipt required');v=write(a.receipt,consume(a.output,a.report_sha))
    print(raw(v).decode(),end='',flush=True)


if __name__=='__main__':main()

