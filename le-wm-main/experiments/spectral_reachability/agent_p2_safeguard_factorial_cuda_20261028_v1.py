"""Prepare-bound fixed32 model forwards; MAIN HOST CUDA only, no physics.

Complete native trace validation precedes model-only factorial selection.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
import numpy as np
from experiments.spectral_reachability import agent_p2_safeguard_factorial_20261028_v1 as source
from experiments.spectral_reachability import agent_p2_midfeedback_raw_jacobian_fd_20261028_v1 as fd

NAME='agent_p2_safeguard_factorial_cuda_20261028_v1'
ROOT=source.ROOT
F,bridge,kernel,physical=fd.F,fd.bridge,fd.kernel,fd.physical
require,sha,raw=source.require,source.sha,source.raw
SOURCE_SHA='0af90ec7383c52f5061a395159dfeecd562a2c5a45be788abb97878d7bb97fc3'
FD_SHA='4329b05f6c8d5d11b5eb89d6a22cf7d019120f589fdba3bf7edc46de7cb4dc7f'
FACTORIAL_SHA='9ad5980d7943f11dc5e6d53c285f37dc4a86e69620d136036bcdd656d3e9d469'
AUTH='MAIN_FACTORIAL_FIXED32_HOST_CUDA_ONLY'
CAP,FINAL,FLOOR,WALL=64*1024**2,2*1024**2,200*1024**2,900
Modules,PredictAPI,NoAD=fd.Modules,fd.PredictAPI,fd.NoAD


def contract():
    return dict(requests=32,branches=8,cells=32,roots=2,alphas=[1.,.5,.25,.125],
        new_modules={k:32 for k in Modules.ORDER},predict_API_inclusive=32,
        AD=0,SVD=0,visual_encoder=0,visual_projector=0,physics=0,training=False,
        raw='literal supplied float32[5,2]; no cast repair, tanh, clipping or inverse at execution',
        CUDA_only=True,batch=1,TF32=False,autocast=False,SDP='MATH',
        selection='frozen source.choose, all64 proposal slots +8 incumbents validated before all32 cells',
        budget='same four-candidate pool shared by guard off/on; inherited anchor costs separate',
        total_FLOPs=None,equal_FLOPs=False,wall_seconds=WALL,output_cap_bytes=CAP,
        final_reserve_bytes=FINAL,disk_floor_bytes=FLOOR,retries=0,
        physical_release=False,P1_admission=False,P2_admission=False,fresh=False)


def own_files():return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def branch(q):
    return dict(observed_context=q['input_context'],action_mean=q['action_mean'],action_std=q['action_std'],
        goal=q['goal'],q=np.eye(192).tolist())


def schedule(q):return [dict(legal=True,label=f'request_{q["request_index"]:02d}')]


def material(p):
    """Whitelist: no native metrics/legacy physical references enter selector."""
    require(p['schema']==source.NAME+'.prepare' and p['contract']==source.contract()
        and p['status']=='blocked_missing_model_candidates' and len(p['rows'])==8 and len(p['requests'])==32,
        'fixed actual manifest ABI, all8/all32 required')
    expected={(i,j) for i in range(8) for j in range(1,5)};seen=set();rows=[]
    for i,q in enumerate(p['requests']):
        require(q['request_index']==i and q['branch']==i//4 and q['alpha_index']==i%4+1 and q['anchor']=='zero'
            and q['alpha']==contract()['alphas'][i%4] and q['domain']['legal'] is True,'fixed branch-major alpha order/domain')
        seen.add((q['branch'],q['alpha_index']));a=fd.packet(q['commands'])
        require(source.v2.suffix_bits(a)==q['raw_bits'] and np.array_equal(a,source.v2.chart(q['logits']))
            and q['context_sha256']==F.core.digest(q['input_context']),'exact chart raw/context identity')
        F.core.context(q['input_context'])
        require(np.asarray(q['raw_history10']).shape==(10,2) and np.asarray(q['goal']).shape==(192,)
            and q['native_input']==F.planner.incoming(branch(q),q['input_context'],q['commands']),
            'sealed original history/goal/normalizer and complete native input')
    require(seen==expected,'no dropped or repeated request')
    for i,row in enumerate(p['rows']):
        q=p['requests'][4*i]
        require(row['index']==i and row['context_sha256']==q['context_sha256']
            and row['goal_sha256']==F.core.digest(q['goal']) and [a['anchor'] for a in row['controlled_axes']]==['zero','incumbent'],
            'all8 causal selector rows')
        local=[p['requests'][4*i+j] for j in range(4)]
        require(all(all(x[k]==q[k] for k in ('input_context','raw_history10','goal','action_mean','action_std')) for x in local),
            'all four alphas use identical context/history/goal/normalizer')
        incumbent=deepcopy(row['incumbent']);require(incumbent['prediction'] is not None,'sealed old prediction')
        require(source.v2.suffix_bits(incumbent['commands'])==incumbent['raw_bits'],'literal old raw')
        axes=[]
        for a in row['controlled_axes']:
            require(len(a['candidates'])==4,'all four fixed-alpha proposals')
            items=deepcopy(a['candidates'])
            for j,x in enumerate(items):
                require(x['index']==j+1 and source.v2.suffix_bits(x['commands'])==x['raw_bits'],'candidate alpha/raw identity')
                if a['anchor']=='zero':require(x['prediction'] is None and x['commands']==local[j]['commands'],'only the missing zero-anchor slots')
                else:require(x['prediction'] is not None,'all incumbent-anchor predictions already sealed')
            axes.append(dict(anchor=a['anchor'],candidates=items))
        prefix=bytes.fromhex(row['prefix_raw_bits']);require(len(prefix)==40,'literal prefix5 bits')
        rows.append(dict(index=i,root_slot=row['root_slot'],condition=row['condition'],arm=row['arm'],task_id=row['task_id'],
            context_sha256=row['context_sha256'],goal_sha256=row['goal_sha256'],prefix_raw_bits=row['prefix_raw_bits'],
            incumbent=incumbent,axes=axes))
    return dict(requests=deepcopy(p['requests']),selector_rows=rows)


def build(factorial_prepared,factorial_sha):
    require(factorial_sha==FACTORIAL_SHA,'only pinned MAIN complete upstream prepare authority, reject before I/O')
    require(sha(source.__file__)==SOURCE_SHA and sha(fd.__file__)==FD_SHA,'frozen manifest and FD native hooks')
    path=Path(factorial_prepared).resolve();require(sha(path)==FACTORIAL_SHA,'actual immutable MAIN factorial prepare')
    p=physical.read(path);bridge.verify_bindings(p['bindings']);body=material(p)
    bindings=dict(p['bindings']);physical.merge(bindings,{str(path):factorial_sha,str(Path(fd.__file__).resolve()):FD_SHA})
    physical.merge(bindings,{str(f):sha(f) for f in own_files()});bridge.verify_bindings(bindings)
    return dict(schema=NAME+'.prepare',arguments=dict(factorial_prepared=str(path),factorial_sha=factorial_sha),
        contract=contract(),**body,bindings=bindings,expected_model_hash=p['expected_model_hash'],policy=p['policy'],
        inherited_compute=deepcopy(p['inherited_compute']),source_contract=deepcopy(p['contract']),
        full_upstream_rebuilt_here=False,authority='pinned MAIN actual complete factorial prepare and every upstream binding SHA',
        planned_new_compute=dict(predict=32,action_encoder=32,pred_proj=32,predict_API_inclusive=32,AD=0,SVD=0),
        actual_new_compute=None,physical_release=False,P1_admission=False,P2_admission=False)


def verify(prepared,expected):
    require(sha(prepared)==expected,'own prepare SHA');p=physical.read(prepared)
    require(raw(p)==raw(build(**p['arguments'])),'whole original manifest closure and exact runtime material');return p


def write(path,value):
    path=Path(path).absolute();payload=raw(value)
    require(path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink() and len(payload)<=32*1024**2
        and shutil.disk_usage(path.parent).free>=FLOOR+len(payload),'new own bounded metadata')
    with path.open('xb') as f:require(f.write(payload)==len(payload),'full write');f.flush();os.fsync(f.fileno())
    kernel.paired.fsync_dir(path.parent);return dict(path=str(path),sha256=sha(path),status=value.get('status','prepared'))


class Store(kernel.Store):
    def __init__(self,path,started):
        require(Path(path).name.startswith(NAME+'_'),'exclusive new output');super().__init__(path)
        self.started=started;self.request_index=None;self.native=[]
    @property
    def output(self):return self.path
    def check(self,size=4096,final=False):
        require(final or time.monotonic()-self.started<WALL,'whole-stage900s including verify, boundary watchdog only')
        used=sum(f.stat().st_size for f in self.path.iterdir())
        require(used+size<=CAP-(0 if final else FINAL) and shutil.disk_usage(self.path).free>=FLOOR+size,'64MiB /2MiB final /200MiB floor')
    def _write(self,n,b):return self.write(n,b)
    def emit(self,kind,**fields):
        e=dict(index=self.index,previous_sha256=self.previous,kind=kind,**deepcopy(fields))
        e['event_sha256']=kernel.source.digest(e)
        self.write('journal.jsonl',kernel.raw(e),append=True,final=kind=='run_finished')
        self.previous=e['event_sha256'];self.index+=1
    def event(self,e):
        self.emit('native_event',request_index=self.request_index,event=e);self.native.append(deepcopy(e))


class Forbidden:
    """No SVD or new visual forwards, scoped/restored even on failure."""
    def __init__(self,torch,model,sink):self.torch,self.model,self.sink=torch,model,sink;self.undo=[];self.handles=[];self.events=[]
    def deny(self,resource):
        def call(*a,**kw):
            e=dict(kind='forbidden_operation',resource=resource);self.events.append(e);self.sink(e)
            raise ValueError('zero budget: '+resource)
        return call
    def __enter__(self):
        try:
            for obj,keys in ((np.linalg,('svd',)),(self.torch,('svd',)),(self.torch.linalg,('svd','svdvals'))):
                for k in keys:self.undo.append((obj,k,getattr(obj,k)));setattr(obj,k,self.deny('SVD'))
            for name in ('encoder','projector'):
                self.handles.append(getattr(self.model,name).register_forward_pre_hook(self.deny(name)))
            return self
        except BaseException:self.__exit__(None,None,None);raise
    def __exit__(self,*a):
        for h in self.handles:h.remove()
        for obj,k,fn in reversed(self.undo):setattr(obj,k,fn)


def evaluate(q,backend,meter,store,row):
    row['status']='attempted';meter.stage=schedule(q)[0]['label'];meter.last_native=None
    meter.expected_input=F.planner.incoming(branch(q),q['input_context'],q['commands'])
    require(meter.expected_input==q['native_input'],'literal prepared native full3 input')
    store.event(dict(kind='evaluation_intent',request_index=q['request_index'],raw_bits=q['raw_bits'],expected_input=meter.expected_input))
    after=backend.step(deepcopy(q['input_context']),fd.packet(q['commands']))
    F.planner.advance(branch(q),q['input_context'],q['commands'],after);F.core.context(after)
    require(np.array_equal(meter.last_native,after['latent_history'][-1]),'actual native endpoint returned')
    stem=f'r{q["request_index"]:02d}'
    row.update(context_sha256=F.core.digest(after),model_Q=source.cost(branch(q),after),
        latent_array=F.save_array(store,stem+'.latents.npy',np.asarray(after['latent_history'],np.float32)),
        token_array=F.save_array(store,stem+'.tokens.npy',np.asarray(after['action_history'],np.float32)),status='recorded')
    store.event(dict(kind='evaluation_return',receipt={k:row[k] for k in ('index','status','context_sha256','model_Q','latent_array','token_array')}))
    return after


def counts(row):
    row['resources']={k:dict(attempted=sum(e['resource']==k for e in row['module_ledger']),
        completed=sum(e['resource']==k and e['completed'] for e in row['module_ledger'])) for k in Modules.ORDER}
    row['predict_API_resources']=dict(predict=dict(attempted=len(row['predict_API_ledger']),completed=sum(e['completed'] for e in row['predict_API_ledger'])))
    row['AD_API_resources']={k:dict(attempted=sum(e['resource']==k for e in row['AD_API_ledger']),
        completed=sum(e['resource']==k and e['completed'] for e in row['AD_API_ledger'])) for k in ('grad','jacobian')}


def check_request(q,row,events,load):
    require(row['index']==q['request_index'] and row['status']=='recorded' and row['AD_API_ledger']==[]
        and row['forbidden_ledger']==[],'exact completed request, no AD/SVD/visual calls')
    F.blind.exact_counts(row['resources'],{k:1 for k in Modules.ORDER})
    F.blind.exact_counts(row['predict_API_resources'],dict(predict=1));F.blind.exact_counts(row['AD_API_resources'],dict(grad=0,jacobian=0))
    label=schedule(q)[0]['label'];api=dict(index=0,stage=label,completed=True)
    require(row['predict_API_ledger']==[api] and len(row['module_ledger'])==3,'one complete native chain/API')
    incoming=F.planner.incoming(branch(q),q['input_context'],q['commands'])
    require(incoming==q['native_input'] and source.v2.suffix_bits(q['commands'])==q['raw_bits'],'prepared actual input/raw bits')
    want=[dict(kind='evaluation_intent',request_index=q['request_index'],raw_bits=q['raw_bits'],expected_input=incoming)]
    ins=[];outs=[]
    for i,(kind,m) in enumerate(zip(Modules.ORDER,row['module_ledger'],strict=True)):
        require(m['index']==i and m['resource']==kind and m['stage']==label and m['completed'] is True,'literal full module sequence')
        if kind=='predict':want.append(dict(kind='predict_API_attempt',entry=dict(api,completed=False)))
        want.extend([dict(kind='module_attempt',entry=dict(index=i,resource=kind,stage=label,completed=False,inputs=[])),
            dict(kind='module_inputs_saved',entry={k:v for k,v in m.items() if k!='output'}|dict(completed=False)),
            dict(kind='module_return',entry=m)])
        if kind=='pred_proj':want.append(dict(kind='predict_API_return',entry=api))
        ins.append([load(a) for a in m['inputs']]);outs.append(load(m['output']))
        require(all(a.dtype==np.float32 and np.isfinite(a).all() for a in ins[-1]+[outs[-1]]),'full finite native float32 buffers')
    require(len(ins[0])==1 and ins[0][0].shape==(1,3,10) and np.array_equal(ins[0][0][0],incoming['action_history'])
        and outs[0].ndim==3 and outs[0].shape[:2]==(1,3),'exact actual normalized action input')
    require(len(ins[1])==2 and ins[1][0].shape==(1,3,192) and np.array_equal(ins[1][0][0],incoming['latent_history'])
        and np.array_equal(ins[1][1],outs[0]) and outs[1].shape==(1,3,192),'actual context/embedding/predictor')
    require(len(ins[2])==1 and np.array_equal(ins[2][0],outs[1].reshape(3,192)) and outs[2].shape==(3,192),'all3 native projection rows')
    lat,tok=load(row['latent_array']),load(row['token_array'])
    require(lat.dtype==tok.dtype==np.float32 and lat.shape==(3,192) and tok.shape==(2,10),'complete returned shifted histories')
    after=dict(q['input_context'],latent_history=lat.astype(float).tolist(),action_history=tok.astype(float).tolist())
    F.planner.advance(branch(q),q['input_context'],q['commands'],after)
    require(np.array_equal(lat[-1],outs[2][-1]) and F.core.digest(after)==row['context_sha256']
        and row['model_Q']==source.cost(branch(q),after),'actual returned native endpoint and original Q')
    want.append(dict(kind='evaluation_return',receipt={k:row[k] for k in ('index','status','context_sha256','model_Q','latent_array','token_array')}))
    require(events==want,'complete exact event ABI: unknown, skipped, reordered or pending rejected')
    return after


def seal(p,contexts):
    require(len(contexts)==32 and len(p['selector_rows'])==8,'all32 required before ANY cell seal')
    rows=[]
    for i,row in enumerate(p['selector_rows']):
        q=p['requests'][4*i];b=branch(q);inc=deepcopy(row['incumbent'])
        require(inc['prediction']['cost']==source.cost(b,inc['prediction']['context']),'original incumbent Q')
        F.planner.advance(b,q['input_context'],inc['commands'],inc['prediction']['context'])
        for axis in row['axes']:
            items=deepcopy(axis['candidates'])
            for j,x in enumerate(items):
                if axis['anchor']=='zero':
                    after=contexts[4*i+j];F.planner.advance(b,q['input_context'],x['commands'],after)
                    x['prediction']=dict(context=deepcopy(after),cost=source.cost(b,after),authorities=[dict(request_index=4*i+j)])
                else:
                    F.planner.advance(b,q['input_context'],x['commands'],x['prediction']['context'])
                    require(x['prediction']['cost']==source.cost(b,x['prediction']['context']),'same original goal on saved candidate')
            for guarded in (False,True):
                selected=source.choose(inc,items,guarded);require(selected is not None,'no missing alpha allowed')
                e=inc if selected['selected_index']==0 else items[selected['selected_index']-1]
                full=row['prefix_raw_bits']+selected['raw_bits']
                body=dict(index=len(rows),branch=i,root_slot=row['root_slot'],condition=row['condition'],arm=row['arm'],task_id=row['task_id'],
                    anchor=axis['anchor'],guard=guarded,selection=selected,input_context_sha256=row['context_sha256'],goal_sha256=row['goal_sha256'],
                    prefix_raw_bits=row['prefix_raw_bits'],full10_raw_bits=full,
                    full10=np.frombuffer(bytes.fromhex(full),dtype='<f4').reshape(10,2).astype(float).tolist(),
                    new_suffix_context=deepcopy(e['prediction']['context']),
                    scope='actual-mid suffix5 Q only; NOT root-to-terminal imagined prediction',
                    P1_admission=False,P2_admission=False,physical_release=False)
                body['selection_digest']=F.core.digest(body);rows.append(body)
    require(len(rows)==32,'all32 cells, no favorable subset');return rows


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN explicit HOST CUDA authorization before I/O')
    started=time.monotonic();p=verify(prepared,expected);store=Store(output,started);store.check()
    rows=[dict(index=i,status='not_started',module_ledger=[],predict_API_ledger=[],AD_API_ledger=[],forbidden_ledger=[]) for i in range(32)]
    errors=[];model=None;before=after=None;flags=None;contexts=[];selection=None
    store.put('binding.json',dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected))
    store.emit('run_intent',requests=32,contract=contract())
    try:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        store.check();require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        bridge.native.frozen(model);before=bridge.native.tensor_hash(model);require(before==p['expected_model_hash'],'original frozen model')
        flags=torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            for i,(q,row) in enumerate(zip(p['requests'],rows,strict=True)):
                store.check();store.request_index=i;store.native=[]
                store.emit('request_intent',request_index=i,request_sha256=F.core.digest(q),budget={k:1 for k in Modules.ORDER})
                meter=SimpleNamespace(stage=None,expected_input=None,last_native=None);sync=lambda:torch.cuda.synchronize(device)
                modules=Modules(model,store,f'r{i:02d}',meter,schedule(q),sync)
                api=PredictAPI(model,meter,store,sync,schedule(q));ad=NoAD(torch,F.blind.ARMS[1],store.event,sync)
                forbidden=Forbidden(torch,model,store.event)
                try:
                    with forbidden,ad,modules,api:evaluate(q,F.d2p.TorchBackend(model,device),meter,store,row)
                finally:
                    row.update(module_ledger=deepcopy(modules.entries),predict_API_ledger=deepcopy(api.entries),
                        AD_API_ledger=deepcopy(ad.events),forbidden_ledger=deepcopy(forbidden.events));counts(row)
                contexts.append(check_request(q,row,store.native,lambda e:source.v1.load_npy(output,e)))
                store.emit('request_return',request_index=i,status=row['status'])
    except BaseException as e:errors.append(kernel.error('fixed32_forward',e))
    finally:
        if flags is not None:torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=flags
        try:
            if model is not None:bridge.native.frozen(model);after=bridge.native.tensor_hash(model)
            bridge.verify_bindings(p['bindings']);require(sha(prepared)==expected,'prepare unchanged');store.check()
            if not errors and before==after==p['expected_model_hash'] and len(contexts)==32:
                cells=seal(p,contexts);selection=dict(file='selections.json',sha256=store.put('selections.json',cells))
        except BaseException as e:errors.append(kernel.error('closure_model_limits_selection',e))
        summaries=[]
        for i,row in enumerate(rows):
            counts(row);name=f'request_{i:02d}.json';h=store.put(name,row,final=True)
            summaries.append(dict(index=i,status=row['status'],file=name,sha256=h,resources=row['resources'],predict_API_resources=row['predict_API_resources']))
        if time.monotonic()-started>=WALL:errors.append(dict(scope='final_wall',error='whole-stage wall budget exceeded'))
        complete=not errors and selection is not None and all(r['status']=='recorded' for r in rows)
        status='all32_forward_and_cells_sealed_science_blocked' if complete else 'partial_or_failed'
        store.emit('run_finished',status=status)
        report=dict(schema=NAME+'.report',status=status,contract=contract(),rows=summaries,errors=errors,selection=selection,
            prepared=str(Path(prepared).resolve()),prepare_sha256=expected,model_before=before,model_after=after,
            elapsed_seconds=time.monotonic()-started,journal_chain_tail=store.previous,
            artifact_sha256={f.name:sha(f) for f in store.path.iterdir()},fixed_request_denominator=32,fixed_cell_denominator=32,
            AD=0,SVD=0,new_visual=0,new_physics=0,full_upstream_rebuilt_here=False,authority=p['authority'],
            P1_admission=False,P2_admission=False,physical_release=False)
        store.put('report.json',report,final=True)
    require(complete,'partial32 retained with actual attempted costs, no retry: '+str(errors))
    return dict(report=str(store.path/'report.json'),report_sha256=sha(store.path/'report.json'))


def consume(output,expected):
    output=Path(output).resolve();require(sha(output/'report.json')==expected,'external actual report SHA')
    r=physical.read(output/'report.json');p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and r['status']=='all32_forward_and_cells_sealed_science_blocked'
        and r['contract']==contract()==p['contract'] and r['errors']==[] and len(r['rows'])==32
        and r['model_before']==r['model_after']==p['expected_model_hash'] and 0<=r['elapsed_seconds']<WALL
        and r['fixed_request_denominator']==r['fixed_cell_denominator']==32 and r['AD']==r['SVD']==r['new_visual']==r['new_physics']==0
        and r['full_upstream_rebuilt_here'] is False and r['authority']==p['authority']
        and r['P1_admission'] is r['P2_admission'] is r['physical_release'] is False,'whole completed fixed32 only')
    require({f.name for f in output.iterdir()}==set(r['artifact_sha256'])|{'report.json'}
        and sum(f.stat().st_size for f in output.iterdir())<=CAP,'full output census/cap')
    for n,h in r['artifact_sha256'].items():require(sha(bridge.rgb.flat(output,n))==h,'all artifacts including native arrays')
    require(physical.read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'actual execution binding')
    events=physical.journal(output/'journal.jsonl',r['journal_chain_tail'])
    want=[dict(kind='run_intent',requests=32,contract=contract())]
    contexts=[];resources={k:dict(attempted=0,completed=0) for k in Modules.ORDER}
    for i,(q,summary) in enumerate(zip(p['requests'],r['rows'],strict=True)):
        require(summary['index']==i and summary['status']=='recorded' and summary['file']==f'request_{i:02d}.json'
            and sha(output/summary['file'])==summary['sha256'],'fixed32 request files/status/SHA')
        row=physical.read(output/summary['file'])
        require(all(summary[k]==row[k] for k in ('resources','predict_API_resources')),'top-level resources agree')
        native=[e['event'] for e in events if e['kind']=='native_event' and e['request_index']==i]
        contexts.append(check_request(q,row,native,lambda e:source.v1.load_npy(output,e)))
        want.append(dict(kind='request_intent',request_index=i,request_sha256=F.core.digest(q),budget={k:1 for k in Modules.ORDER}))
        want.extend(dict(kind='native_event',request_index=i,event=e) for e in native)
        want.append(dict(kind='request_return',request_index=i,status='recorded'))
        for k in resources:
            for state in resources[k]:resources[k][state]+=row['resources'][k][state]
    want.append(dict(kind='run_finished',status=r['status']))
    previous=None;whole=[]
    for i,payload in enumerate(want):
        e=dict(index=i,previous_sha256=previous,**payload);e['event_sha256']=kernel.source.digest(e)
        whole.append(e);previous=e['event_sha256']
    require(events==whole,'entire native canonical hashchain and payload ABI, no unknown/skipped fields or events')
    cells=seal(p,contexts);require(r['selection']['file']=='selections.json' and sha(output/'selections.json')==r['selection']['sha256']
        and physical.read(output/'selections.json')==cells,'independent all32 factorial selections and raw full10 seals')
    bridge.verify_bindings(p['bindings']);require(sha(output/'report.json')==expected,'report unchanged after full consume')
    return dict(schema=NAME+'.consume',status='all32_engineering_verified_science_blocked',report_sha256=expected,
        prepare_sha256=r['prepare_sha256'],factorial_prepare_sha256=p['arguments']['factorial_sha'],cells=cells,
        full_upstream_rebuilt_here=False,authority=p['authority'],
        fixed_request_denominator=32,fixed_cell_denominator=32,known_roots=2,actual_new_modules=resources,
        actual_predict_API_inclusive=dict(attempted=32,completed=32),planned_new_compute=p['planned_new_compute'],
        inherited_compute=p['inherited_compute'],fair_budget=dict(per_branch_per_anchor_candidates=4,
            guard_off_on_share_same_evaluated_pool=True,zero_anchor_new_predict=32,incumbent_anchor_new_predict=0,
            guard_extra_predict=0,inherited_costs_separate=True,total_FLOPs=None,equal_FLOPs=False),
        AD=0,SVD=0,new_visual=0,new_physics=0,consume_model_calls=0,physical_metrics=None,
        P1_admission=False,P2_admission=False,physical_release=False,fresh=False)


def main():
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('factorial-prepared','factorial-sha','prepared','prepare-sha','output','report-sha','receipt'):q.add_argument('--'+k)
    q.add_argument('--authorize-host-cuda',action='store_true');a=q.parse_args()
    if a.command=='prepare':v=write(a.output,build(a.factorial_prepared,a.factorial_sha))
    elif a.command=='verify':p=verify(a.prepared,a.prepare_sha);v=dict(status='verified_no_execution',requests=len(p['requests']))
    elif a.command=='run':
        require(a.authorize_host_cuda,'MAIN CUDA flag');v=run(a.prepared,a.prepare_sha,a.output,AUTH)
    else:
        require(a.receipt,'own saved receipt');v=write(a.receipt,consume(a.output,a.report_sha))
    print(raw(v).decode(),end='',flush=True)


if __name__=='__main__':main()
