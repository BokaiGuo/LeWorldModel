"""Own-schema native spectral8 R tree. Only explicit MAIN run loads CUDA."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from experiments.spectral_reachability import agent_p3_parent_native_cuda_20261029_v1 as old
from experiments.spectral_reachability import agent_p3_parent_r_native_components_20261029_v1 as components
from experiments.spectral_reachability import agent_p3_r_spectral8_bank_20261029_v1 as spectral

parent,core,signal,live = old.parent,old.core,old.signal,old.live
kernel=spectral.kernel
require,canonical,digest,sha=core.require,core.canonical,core.digest,core.sha
ROOT=core.ROOT
NAME='agent_p3_r_native_orchestrator_20261029_v1'
FLAGS=dict(spectral.FLAGS,scientific_admission=False,full_policy_executed=False)


def sum_axes(*values):return {k:sum(v[k] for v in values) for k in core.AXES}
def empty_axes():return dict.fromkeys(core.AXES,0)


def reserve(node,h):
    v=live.predict_axes(2*(len(node['actions'])//5+h));v['replay_calls']=2
    return v


def group_cost(h):
    return sum_axes(spectral.bank_cost(),{k:kernel.minimum(h)[k] for k in core.AXES})


def profile():
    return dict(old.profile(),stop_contract='cap/unsupported stops tree; no_witness stops branch; runtime error blocks later slots',
        R_group_replay_calls=18,final_replay_calls=2,production_caps_unchanged=True)


def rules():
    return dict(protocol_sha256=spectral.PROTOCOL_SHA,bank=spectral.contract(),
        all_eligible_to_modelQ_beam=True,no_witness='stop branch, no fallback',
        original_thresholds_width_depth=True,queue_denominator=144,
        R_suffix_only_lookahead=True,zero_control_separately_paid=True,
        production_R_group_fits_original_replay_cap=False,**FLAGS)


def validate_node(root,node):
    core.keys(node,('context','depth','indices','actions','path','edges'))
    core.integer(node['depth']);require(node['depth']<=core.policy(root).max_depth,'original depth3')
    require(len(node['indices'])==len(node['edges'])==node['depth'],'full edge/depth/index chain')
    c=root['root_context'];raw=[];path=[]
    for i,e in zip(node['indices'],node['edges'],strict=True):
        core.integer(i);require(i<8,'original candidate index')
        core.keys(e,('kind','input_context_sha256','output_context_sha256','actions_sha256','producer_sha256','actions','path'))
        require(e['kind'] in ('progress','reconfiguration') and e['input_context_sha256']==digest(c)
            and e['actions_sha256']==digest(e['actions']),'real typed parent edge')
        core.hash_value(e['producer_sha256']);c=core.checked_path(root,c,e['actions'],e['path'])
        require(e['output_context_sha256']==digest(c),'exact postcontext identity')
        raw+=deepcopy(e['actions']);path+=deepcopy(e['path'])
    require(node['context']==c and node['actions']==raw and node['path']==path,'whole full3/between2/raw chain')
    return node


class CallSink:
    def __init__(self,session):self.session=session
    def event(self,value):self.session.event(value)


class Session:
    """One shared ledger, instance-local native callbacks, complete transcripts."""
    def __init__(self,root,factory,sink,ledger,kind,guard,capsule,saved=None):
        self.root,self.factory,self.sink,self.ledger=root,factory,sink,ledger
        self.kind,self.guard,self.capsule=kind,guard,capsule
        self.call=None;self.records=[];self.active=[];self.saved=saved
    def event(self,value):
        if self.call is not None:self.active.append(deepcopy(value))
        self.sink.event(dict(kind='io',call=self.call,payload=deepcopy(value)))
    def invoke(self,task,method,req):
        require(self.call is None,'no recursive native invocation')
        self.guard();self.call=len(self.records);self.active=[]
        outer=len(self.ledger.events)-1
        try:
            if self.saved is not None:
                require(self.call<len(self.saved),'no unsaved callback')
                record=self.saved[self.call];call=record['transcript']
                require(call[0]==dict(kind='native_call_attempt',method=method,task=task,request=req), 'saved exact native task/request/order')
                try:old.replay_resources(call[1:-1],self.ledger)
                finally:self.active=deepcopy(call) # nested failure also retains its exact original transcript
                if call[-1]['kind']=='native_call_failure':raise old.exception_from(call[-1]['error'])
                out=deepcopy(call[-1]['result'])
            else:
                self.event(dict(kind='native_call_attempt',method=method,task=deepcopy(task),request=deepcopy(req)))
                backend=self.factory(deepcopy(task),CallSink(self))
                require(backend.kind==self.kind,'no RAM/HOST switch')
                try:
                    out=backend.signal(deepcopy(req),self.ledger) if method=='signal' else backend.replay(
                        deepcopy(req['parent_context']),deepcopy(req['actions']),req['segmented'],self.ledger)
                    self.event(dict(kind='native_call_return',result=deepcopy(out)))
                except Exception as exc:
                    self.event(dict(kind='native_call_failure',error=dict(type=type(exc).__name__,message=str(exc))));raise
            require(old.audit_native(self.active) is True,'full native IO/AD/trajectory closure')
            self.guard();return out
        finally:
            self.records.append(dict(call_id=self.call,outer_index=outer,transcript=deepcopy(self.active)))
            self.call=None;self.active=[]
    def signal(self,task,h):
        req=core.request(task,dict(context=task['root_context']),operation='compile_finite_horizon_J',horizon=h)
        return self.invoke(task,'signal',req)
    def replay(self,task,raw,segmented=False):
        return self.invoke(task,'replay',dict(parent_context=task['root_context'],actions=deepcopy(raw),segmented=segmented))


class SignalBackend:
    def __init__(self,session,task):self.session,self.task,self.kind=session,task,session.kind
    def signal(self,req,ledger):
        require(ledger is self.session.ledger,'one actual shared ledger')
        require(req==core.request(self.task,dict(context=self.task['root_context']),operation='compile_finite_horizon_J',horizon=req['horizon']), 'exact parent signal')
        return self.session.invoke(self.task,'signal',req)


def source_evidence(session,mapping):
    records=[r for r in session.records if r['transcript'][0]==dict(kind='native_call_attempt',method='signal',
        task=mapping['parent_task'],request=mapping['native_request'])]
    require(records,'same parent paid source call')
    record=records[-1];call=record['transcript']
    pointer=session.capsule(record['call_id'],call)
    def index(kind):
        ids=[i for i,x in enumerate(call) if x['kind']==kind and (kind!='origin' or x['phase']=='nominal')]
        require(len(ids)==1,'one saved source '+kind);return ids[0]
    return dict(kind=session.kind,artifact_path=pointer['path'],artifact_sha256=pointer['sha256'],call_id=record['call_id'],
        native_call=deepcopy(call),origin_index=index('origin'),nominal_index=index('nominal'),jacobian_index=index('jacobian'),
        signal_compile_entry=deepcopy(session.ledger.events[record['outer_index']]))


def nominal_path(record,task,out):
    values=[(i,x) for i,x in enumerate(record['transcript']) if x['kind']=='boundary' and x['phase']=='nominal']
    require(len(values)==1 and values[0][1]['step']==0,'unique H1 nominal boundary, no extra predict')
    i,b=values[0];latent=signal.unblob(b['arrays']['latent']);hist=signal.unblob(b['arrays']['history'])
    require(latent.shape==(1,3,192) and hist.shape==(1,2,10),'full nominal context')
    c=dict(task['root_context'],latent_history=latent[0].tolist(),action_history=hist[0].tolist())
    core.checked_path(task,task['root_context'],[[0.,0.]]*5,[c])
    require(c['latent_history'][-1]==out['nominal'],'original nominal buffer exact')
    return [c],dict(call_id=record['call_id'],index=i,sha256=digest(b))


def evaluate_R(root,node,decision,mapping,session,verification_reserve,emit,on_start=lambda result:None):
    """Requires the entire bank+9group+final reserve admitted BEFORE bank SVD."""
    ledger=session.ledger;h=decision['horizon'];before=deepcopy(ledger.attempted)
    require(decision['extension_type']=='reconfiguration' and decision['candidates']==8,'original R8')
    require(all(before[k]+group_cost(h)[k]+verification_reserve[k]<=ledger.caps[k] for k in core.AXES),'whole R group preflight')
    bank=spectral.build(root=root,prefix_actions=node['actions'],prefix_path=node['path'],mapping=mapping,
        evidence=source_evidence(session,mapping),ledger=ledger,verification_reserve=verification_reserve,
        sink=lambda e:emit(dict(kind='bank_event',event=e)),permit_ram=session.kind=='RAM_FIXTURE')
    require(bank['status']=='complete','complete actual spectral8 bank')
    result=dict(bank=bank,control=dict(index='zero',status='not_requested'),
        rows=[dict(index=i,status='not_requested',eligible=None) for i in range(8)],eligible_indices=[],status='running')
    on_start(result) # retain all nine slots even if a native callback fails
    emit(dict(kind='R_attempt',parent_context_sha256=digest(node['context']),bank_sha256=digest(bank['actions'])))
    local=parent.parent_task(root,node['context'])
    for index,raw in [('zero',bank['zero'])]+list(enumerate(bank['actions'])):
        row=result['control'] if index=='zero' else result['rows'][index]
        row['status']='attempted';row['actions']=deepcopy(raw)
        emit(dict(kind='R_item_attempt',index=index,parent_context_sha256=digest(node['context'])))
        def compile_edge():return ledger.call('replay',lambda:session.replay(local,raw))
        path=ledger.call('program_compile',compile_edge)
        post=core.checked_path(root,node['context'],raw,path)
        row.update(path=path,postcontext=post,immediate=ledger.call('metric',lambda:core.model_cost(root,post)))
        posttask=parent.parent_task(root,post)
        response=ledger.call('signal_compile',lambda:session.signal(posttask,1))
        nominal,pointer=nominal_path(session.records[-1],posttask,response)
        residual=np.asarray(root['goal'],np.float64)-np.asarray(response['nominal'],np.float64)
        solve=ledger.call('solve',lambda:ledger.call('array_svd',lambda:kernel.original.bounded_ridge(response['jacobian'],residual)))
        suffix=components.suffix_raw(solve['params'])
        suffix_path=ledger.call('replay',lambda:session.replay(posttask,suffix))
        terminal=core.checked_path(root,post,suffix,suffix_path)
        row.update(status='complete',signal=response,nominal_path=nominal,nominal_pointer=pointer,solve=solve,
            reach=solve['own_reach_value'],suffix_actions=suffix,suffix_path=suffix_path,
            bounded_suffix_cost=ledger.call('metric',lambda:core.model_cost(root,terminal)))
        emit(dict(kind='R_item_return',row=deepcopy(row)))
    z=result['control']
    for row in result['rows']:
        row['eligible']=row['immediate']>z['immediate'] and row['reach']<z['reach']
    result['eligible_indices']=[x['index'] for x in result['rows'] if x['eligible']]
    result['diagnostic_reach_winner']=min(result['eligible_indices'],key=lambda i:(result['rows'][i]['reach'],i)) if result['eligible_indices'] else None
    result['status']='witness_found' if result['eligible_indices'] else 'no_witness'
    result['resources']={k:ledger.attempted[k]-before[k] for k in core.AXES}
    require(result['resources']==group_cost(h),'full9 plus bank exact resource census')
    emit(dict(kind='R_return',result=deepcopy(result)))
    return result


def child(root,node,req,raw,path,kind):
    terminal=core.checked_path(root,node['context'],raw,path)
    edge=dict(kind=kind,input_context_sha256=digest(node['context']),output_context_sha256=digest(terminal),
        actions_sha256=digest(raw),producer_sha256=sha(__file__),actions=deepcopy(raw),path=deepcopy(path))
    out=dict(context=terminal,depth=node['depth']+1,indices=node['indices']+[req['candidate_index']],
        actions=node['actions']+deepcopy(raw),path=node['path']+deepcopy(path),edges=node['edges']+[edge])
    validate_node(root,out);return out


def run_tree(task,factory,sink,*,mode,caps=None,permit_ram=False,kind='HOST_CUDA',guard=lambda:None,
             capsule=None,saved_calls=None,clock=time.monotonic_ns):
    task=core.validate_task(task);components.post_task(task,[],[])
    require(mode in ('fixed','adaptive') and (kind=='HOST_CUDA' or (permit_ram is True and kind=='RAM_FIXTURE')),'explicit fixed/adaptive HOST or RAM')
    require(callable(capsule),'explicit immutable signal capsule store')
    ledger=core.Ledger(parent.default_caps(mode) if caps is None else caps,clock=clock)
    session=Session(task,factory,sink,ledger,kind,guard,capsule,saved_calls)
    ledger.sink=session.event
    adapter=parent.ParentNativeAdapter(task,lambda t,s:SignalBackend(session,t),type('Sink',(),{'event':lambda self,e:session.event(e)})(),permit_ram=permit_ram)
    jobs=[dict(slot=i,status='not_requested',request=None,payload=None,node=None,error=None) for i in range(144)]
    r=dict(schema=NAME+'.tree',mode=mode,backend_kind=kind,root_task_sha256=digest(task),jobs=jobs,signals=[],queues=[],
        R_groups=[],replays=[dict(mode=x,status='not_requested',path=None) for x in ('whole','segmented')],
        status='running',errors=[],model_plan=None,diagnostic_best=None,fixed_denominator=dict(arms=1,requests=144),**FLAGS)
    policy=core.policy(task);frontier=[parent.root_node(task)];best=None;used=0;stopped=False
    def emit(e):session.event(e)
    def rank(n):return n['cost'],n['depth'],n['indices']
    try:
        for _ in range(policy.max_depth):
            children=[]
            for node in frontier:
                validate_node(task,node)
                ledger.scope=dict(mode=mode,phase='signal',parent_context_sha256=digest(node['context']))
                signals,receipt=core.local_signal(task,node,adapter,ledger,policy);r['signals'].append(receipt)
                ledger.scope=dict(mode=mode,phase='schedule',parent_context_sha256=digest(node['context']))
                decision=ledger.call('schedule',lambda:core.choose(task,node,signals,mode))
                reqs=[core.request(task,node,operation='compile_program_extension',**decision,candidate_index=i,
                    prefix_edges=deepcopy(node['edges']),prefix_actions=deepcopy(node['actions']),
                    signal_sha256=digest(receipt),local_response=deepcopy(receipt['responses'])) for i in range(decision['candidates'])] if decision['continue_expanding'] else []
                q=dict(parent=deepcopy(node),decision=deepcopy(decision),requests=reqs,status='stopped_by_original_child_policy',minimum=None,verification_reserve=None)
                r['queues'].append(q)
                if not reqs:emit(dict(kind='queue',queue=deepcopy(q)));continue
                n,h=decision['candidates'],decision['horizon'];require(n in (4,8) and h in (1,3),'original request width/horizon')
                kind_=decision['extension_type']
                cost=group_cost(h) if kind_=='reconfiguration' else parent.generation_minimum(h,n)
                rr=reserve(node,h);q.update(minimum=cost,verification_reserve=rr)
                if used+n>144:q['status']='expansion_cap_before_node_queue'
                elif kind_ not in ('progress','reconfiguration') or decision['program_type'] not in ('progress','reconfiguration','composition'):
                    q['status']='refused_unsupported_original_type'
                elif any(ledger.attempted[k]+cost[k]+rr[k]>ledger.caps[k] for k in core.AXES):q['status']='resource_cap_before_node_queue'
                else:q['status']='queued'
                emit(dict(kind='queue',queue=deepcopy(q)))
                if q['status']!='queued':r['status']=q['status'];stopped=True;break
                assigned=jobs[used:used+n]
                for row,req in zip(assigned,reqs,strict=True):row.update(status='queued',request=deepcopy(req))
                used+=n;ledger.scope=dict(mode=mode,phase='program',parent_context_sha256=digest(node['context']))
                mapping=next(m for m in reversed(adapter.maps) if m['outer_request']['parent_context_sha256']==digest(node['context']) and m['native_response']['horizon']==h)
                local=mapping['parent_task']
                if kind_=='reconfiguration':
                    group=evaluate_R(task,node,decision,mapping,session,rr,emit,on_start=r['R_groups'].append)
                    proposals=group['rows']
                else:
                    bank=None;proposals=[]
                    for req in reqs:
                        def generate():
                            nonlocal bank
                            if bank is None:bank=live.generate(local,mapping['native_response'],n,ledger)
                            raw=bank['actions'][req['candidate_index']]
                            return raw,session.replay(local,raw)
                        raw,path=ledger.call('program_compile',generate)
                        proposals.append(dict(index=req['candidate_index'],actions=raw,path=path,eligible=True,
                            immediate=ledger.call('metric',lambda path=path:core.model_cost(task,path[-1]))))
                    emit(dict(kind='progress_bank',bank=bank,parent_context_sha256=digest(node['context'])))
                for row,req,v in zip(assigned,reqs,proposals,strict=True):
                    c=child(task,node,req,v['actions'],v['path'],kind_);c['cost']=v['immediate']
                    row.update(status='returned_eligible' if v['eligible'] else 'returned_ineligible',payload=deepcopy(v),node=deepcopy(c))
                    if v['eligible']:
                        children.append(c)
                        if best is None or rank(c)<rank(best):best=deepcopy(c)
            if stopped or not children:break
            children.sort(key=rank);frontier=children[:policy.base_beam*(policy.expansion_factor if mode=='adaptive' else 1)]
            for c in frontier:c.pop('cost')
        if not stopped:
            if best is None:r['status']='no_witness'
            else:
                for slot in r['replays']:
                    slot['status']='attempted';ledger.scope=dict(mode=mode,phase='verification',replay_mode=slot['mode'])
                    path=ledger.call('replay',lambda slot=slot:session.replay(task,best['actions'],slot['mode']=='segmented'))
                    require(path==best['path'],'whole/segmented full root raw-chain exact');slot.update(status='returned',path=path)
                r.update(status='tree_complete_not_admitted',model_plan=deepcopy(best))
    except Exception as exc:
        r['status']='failed_or_budget_exhausted';r['errors'].append(dict(type=type(exc).__name__,message=str(exc)))
        for group in r['R_groups']:
            if group['status']=='running':
                group['status']='partial_runtime_failure'
                for item in [group['control']]+group['rows']:
                    if item['status']=='attempted':item['status']='failed'
        for row in jobs:
            if row['status']=='queued':row.update(status='blocked_after_failure',error=str(exc))
        for slot in r['replays']:
            if slot['status']=='attempted':slot['status']='failed'
    r.update(diagnostic_best=best,resource_receipt=ledger.receipt(),signal_mappings=deepcopy(adapter.maps),
        native_calls=deepcopy(session.records),realized_widths=[len(q['requests']) for q in r['queues'] if q['status']=='queued'])
    core.audit_ledger(r['resource_receipt'])
    return r


def execute(p,factory,store,*,permit_ram=False,check=lambda:None):
    rows=[dict(**x,status='not_requested',result=None) for x in old.roster()];failed=False
    store.event(dict(kind='denominator',rows=deepcopy(rows)))
    for row in rows:
        if failed:row['status']='blocked_after_runtime_failure';continue
        slot=row['slot'];store.event(dict(kind='arm_attempt',identity=old.roster()[slot]))
        sink=type('Sink',(),{'event':lambda self,e:store.event(dict(kind='arm_event',slot=slot,payload=deepcopy(e)))})()
        def capsule(cid,call):return store.capsule(slot,cid,call)
        try:
            check();r=run_tree(p['tasks'][row['root']],factory,sink,mode=row['mode'],
                caps=p['profile']['resource_caps'][row['mode']],permit_ram=permit_ram,
                kind='RAM_FIXTURE' if permit_ram else 'HOST_CUDA',guard=check,capsule=capsule)
            row.update(status=r['status'],result=r);failed=bool(r['errors']);check()
        except Exception as exc:
            row.update(status='outer_runtime_failure',error=dict(type=type(exc).__name__,message=str(exc)));failed=True
        store.event(dict(kind='arm_return',row=deepcopy(row)))
    return rows


class Store(signal.Store):
    def __init__(self,output,limits):
        super().__init__(output,limits);self.capsule_hashes={}
    def capsule(self,slot,cid,call):
        name=f'signal_s{slot}_{cid:04}.json';raw=canonical(call);self.check(len(raw));path=self.output/name
        require(not path.exists() and not path.is_symlink(),'immutable exclusive source capsule')
        with path.open('xb') as f:require(f.write(raw)==len(raw),'complete source capsule');f.flush();os.fsync(f.fileno())
        signal.fsync_dir(self.output);self.bytes+=len(raw);self.capsule_hashes[name]=sha(path)
        return dict(path=str(path),sha256=self.capsule_hashes[name])


def own_files():return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def build():
    components.check_sources();spectral.check_sources()
    require(sha(components.__file__)=='ce48a8c665ad5434fe65f5238b303570ffc18183956d24a0c830864fb5f12c7a','frozen native components')
    require(sha(spectral.__file__)=='e254f08db87b1e8b6985b6afd6ba3f3ed8f98e8f1af0d87f9edcd14021f2786d','frozen spectral bank')
    # Reuse source/task discovery only; no old schema or old verify is spoofed.
    prior=old.build();files=set(map(Path,prior['sources']))|set(own_files())|set(spectral.PINS)
    for stem in (components.NAME,spectral.NAME,kernel.NAME):
        files.update([ROOT/'experiments/spectral_reachability'/(stem+'.py'),ROOT/'tests'/('test_'+stem+'.py'),ROOT/'innerdocs'/(stem+'.md')])
    return dict(schema=NAME+'.prepare',tasks=prior['tasks'],policy=prior['policy'],expected_model_hash=prior['expected_model_hash'],
        authority=prior['authority'],sources={str(p.resolve()):sha(p) for p in sorted(files)},
        roster=old.roster(),profile=profile(),precision=old.precision_contract(),rules=rules(),**FLAGS)


def exclusive(path):
    p=Path(path).absolute()
    require(p.parent==ROOT/'results' and p.name.startswith(NAME) and p.resolve()==p and not p.exists() and not p.is_symlink(),'new exclusive own result')
    return p


def write_new(path,value,cap=16*1024**2):
    p=exclusive(path);raw=canonical(value)
    require(len(raw)<=cap and shutil.disk_usage(p.parent).free>=profile()['disk_floor_bytes']+len(raw),'output cap/floor')
    with p.open('xb') as f:require(f.write(raw)==len(raw),'complete own metadata');f.flush();os.fsync(f.fileno())
    signal.fsync_dir(p.parent);return dict(path=str(p),sha256=sha(p))


def prepare(output):return write_new(output,build())


def verify(prepared,expected):
    core.hash_value(expected);path=old.regular(prepared,16*1024**2)
    require(sha(path)==expected,'external own prepare SHA');p=signal.read(path)
    require(p==build() and sha(path)==expected,'own complete task/source/protocol/caps reconstruction')
    return p


def run(prepared,expected,output,*,authorization):
    require(authorization=='MAIN_APPROVED_HOST_CUDA','MAIN source audit and serial HOST CUDA only')
    started=time.monotonic();p=verify(prepared,expected);output=exclusive(output)
    require(shutil.disk_usage(output.parent).free>=p['profile']['output_bytes']+p['profile']['disk_floor_bytes'],'whole output allowance/floor')
    store=Store(output,p['profile']);store.started=started
    rows=[dict(**x,status='blocked_before_model',result=None) for x in old.roster()]
    errors=[];model=None;before=after=classes=classes_after=host=None;guards=[];source_ok=False
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        store.event(dict(kind='source_before',sha256=digest(p['sources'])))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        store.memory=lambda:(torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device))
        store.check();model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        before=signal.model_hash(model);require(before['original_state_dict']==p['expected_model_hash'],'original model bytes/tensors')
        classes=old.model_sources(model,p);store.event(dict(kind='model_before',hashes=before,classes=classes))
        forwards=[(m,'forward' in m.__dict__,m.__dict__.get('forward')) for m in model.modules()]
        with torch.autocast(device_type='cuda',enabled=False),torch.autocast(device_type='cpu',enabled=False),sdpa_kernel(SDPBackend.MATH):
            def guard():store.check();old.checked_precision(torch)
            guards.append(old.checked_precision(torch));store.event(dict(kind='precision_before',value=guards[-1]))
            rows=execute(p,lambda t,s:old.NativeTree(model,device,t,s),store,check=guard)
            guards.append(old.checked_precision(torch));store.event(dict(kind='precision_after',value=guards[-1]))
        torch.cuda.synchronize(device)
        require(all(('forward' in m.__dict__)==had and m.__dict__.get('forward') is value for m,had,value in forwards),'instance forwards restored')
    except BaseException as exc:errors.append(dict(stage='run',type=type(exc).__name__,message=str(exc)))
    finally:
        if model is not None:
            try:
                after=signal.model_hash(model);classes_after=old.model_sources(model,p)
                require(before==after and classes==classes_after and all(not m.training for m in model.modules())
                    and all(not v.requires_grad for v in model.parameters()),'model/buffer/classes unchanged')
                store.event(dict(kind='model_after',hashes=after,classes=classes_after))
            except BaseException as exc:errors.append(dict(stage='model_final',message=str(exc)))
        try:
            verify(prepared,expected);store.check();source_ok=True;store.event(dict(kind='source_after',sha256=digest(p['sources'])))
        except BaseException as exc:errors.append(dict(stage='source_final',message=str(exc)))
        store.close();settled=all(x['result'] is not None and not x['result']['errors'] for x in rows)
        report=dict(schema=NAME+'.report',status='four_slots_settled_not_admitted' if settled and not errors else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,rows=rows,errors=errors,model_before=before,model_after=after,
            classes_before=classes,classes_after=classes_after,host=host,precision=guards,sources_verified=source_ok,
            source_manifest_sha256=digest(p['sources']),wall_seconds=store.clock()-store.started,
            CUDA_peak_bytes=store.memory() if store.memory else None,journal_tail=store.tail,
            artifact_sha256={n:sha(output/n) for n in ('binding.json','events.jsonl') if (output/n).exists()}|store.capsule_hashes,**FLAGS)
        store.put('report.json',report)
    require(report['status']=='four_slots_settled_not_admitted','partial preserved; no retry: '+repr(errors))
    return dict(output=str(output),report_sha256=sha(output/'report.json'))


def replay_clock(receipt,events):
    """Use original measured times, so time-bearing hashes replay exactly."""
    ticks=[0]
    for e in events:
        if e['kind'] in old.RESOURCE:
            ticks.append(e['entry']['started_ns'] if e['kind'] in ('resource_attempt','resource_blocked') else e['entry']['ended_ns'])
    ticks.append(receipt['wall_ns']);it=iter(ticks)
    return lambda:next(it)


def validate_tree(task,result,events,caps,*,permit_fixture=False,capsule_reader):
    require(result['backend_kind']=='HOST_CUDA' or (permit_fixture and result['backend_kind']=='RAM_FIXTURE'),'no implicit RAM production')
    require(all(e['kind']=='io' for e in events),'only known session envelope')
    payloads=[e['payload'] for e in events]
    allowed=old.RESOURCE|old.CALL_ARRAYS|{'direction','native_call_attempt','native_call_return','native_call_failure',
        parent.NAME+'.signal_map','queue','bank_event','R_attempt','R_item_attempt','R_item_return','R_return','progress_bank'}
    require(all(e['kind'] in allowed for e in payloads),'unknown native/orchestrator event')
    require(result['resource_receipt']['caps']==caps,'fixed prepared resource caps')
    old.resource_census(result['resource_receipt'],payloads)
    records=result['native_calls'];require([r['call_id'] for r in records]==list(range(len(records))),'all native call IDs')
    for record in records:
        call=[e['payload'] for e in events if e['call']==record['call_id']]
        require(call==record['transcript'],'full call transcript equals journal')
        old.audit_native(call)
    require(all(e['call'] is None or type(e['call']) is int and 0<=e['call']<len(records) for e in events),'no orphan invocation')
    # Independently rerun inert scheduler/banks/R/solver/eligibility and every
    # saved native operation tree, never a model. No producer/fullstrict rerun.
    clock=replay_clock(result['resource_receipt'],payloads)
    sink=type('Sink',(),{'event':lambda self,e:None})()
    replay=run_tree(task,None,sink,mode=result['mode'],caps=caps,permit_ram=permit_fixture,
        kind=result['backend_kind'],capsule=capsule_reader,saved_calls=records,clock=clock)
    require(replay==result,'complete exact saved tree/resource/bank/eligibility/beam algebra')
    return dict(mode=result['mode'],status=result['status'],native_calls=len(records),
        R_groups=len(result['R_groups']),eligible_counts=[len(g['eligible_indices']) for g in result['R_groups']],
        plan_available=result['model_plan'] is not None,resources=result['resource_receipt'],
        AD_primal_predict_API=sum(c['transcript'][0]['request']['horizon'] for c in records if c['transcript'][0]['method']=='signal' and c['transcript'][-1]['kind']=='native_call_return'),
        Jacobian_elements=sum(192*10*c['transcript'][0]['request']['horizon'] for c in records if c['transcript'][0]['method']=='signal' and c['transcript'][-1]['kind']=='native_call_return'))


def validate_rows(p,rows,events,*,permit_fixture=False,capsule_reader):
    require(len(rows)==4 and [(r['slot'],r['root'],r['mode']) for r in rows]==[(x['slot'],x['root'],x['mode']) for x in old.roster()],'all fixed4')
    top={'denominator','arm_attempt','arm_return','source_before','source_after','model_before','model_after','precision_before','precision_after'}
    require(all(e['kind'] in top|{'arm_event'} for e in events),'unknown top event')
    require(all(type(e['slot']) is int and 0<=e['slot']<4 for e in events if e['kind']=='arm_event'),'no orphan arm events')
    initial=[dict(**x,status='not_requested',result=None) for x in old.roster()]
    require([e['rows'] for e in events if e['kind']=='denominator']==[initial],'denominator before all work')
    position=-1;failed=False;summary=[]
    for row,identity in zip(rows,old.roster(),strict=True):
        own=[e['payload'] for e in events if e['kind']=='arm_event' and e['slot']==row['slot']]
        starts=[i for i,e in enumerate(events) if e['kind']=='arm_attempt' and e['identity']==identity]
        ends=[i for i,e in enumerate(events) if e['kind']=='arm_return' and e['row']==row]
        if row['result'] is None:
            require(failed and row['status']=='blocked_after_runtime_failure' and not starts and not ends and not own,'later slots only blocked by runtime failure');continue
        require(not failed and len(starts)==len(ends)==1 and position<starts[0]<ends[0],'ordered settled arm')
        require(all(starts[0]<i<ends[0] for i,e in enumerate(events) if e['kind']=='arm_event' and e['slot']==row['slot']),'all arm work inside dispatch')
        position=ends[0];require(row['status']==row['result']['status'],'exact outer/inner disposition')
        summary.append(validate_tree(p['tasks'][row['root']],row['result'],own,p['profile']['resource_caps'][row['mode']],
            permit_fixture=permit_fixture,capsule_reader=lambda cid,call,slot=row['slot']:capsule_reader(slot,cid,call)))
        failed=bool(row['result']['errors'])
    require(len([e for e in events if e['kind']=='arm_attempt'])==len(summary) and
        len([e for e in events if e['kind']=='arm_return'])==len(summary),'no extra arm attempts/returns')
    return summary


def consume(output,report_sha,prepared,prepare_sha):
    output=Path(output).absolute();require(output.resolve()==output and not output.is_symlink(),'exact saved directory')
    core.hash_value(report_sha);rp=old.regular(output/'report.json',profile()['final_reserve_bytes'])
    require(sha(rp)==report_sha,'actual report SHA');r=signal.read(rp,profile()['final_reserve_bytes']);p=verify(prepared,prepare_sha)
    require(r['schema']==NAME+'.report' and r['prepared']==str(Path(prepared).absolute()) and r['prepare_sha256']==prepare_sha,'own authority schema')
    require(all(r[k] is False for k in FLAGS) and not r['errors'] and r['sources_verified'] is True
        and r['source_manifest_sha256']==digest(p['sources']),'frozen engineering closure')
    require(r['model_before']==r['model_after'] and r['model_before']['original_state_dict']==p['expected_model_hash'],'frozen model')
    require(r['classes_before']==r['classes_after'] and r['classes_before'],'module classes before/after')
    for c in r['classes_before']:require(p['sources'].get(c['path'])==c['sha256'],'installed source class bytes')
    require(r['precision']==[old.precision_contract()]*2 and r['host']['device']=='cuda:0' and r['host']['allocation_sum']==4.
        and r['host']['visible_nvidia_nodes'],'actual HOST/precision')
    names=set(r['artifact_sha256']);require({'binding.json','events.jsonl'}<=names and
        all(n in ('binding.json','events.jsonl') or n.startswith('signal_s') and n.endswith('.json') and Path(n).name==n for n in names),'only explicit immutable artifacts')
    require({x.name for x in output.iterdir()}==names|{'report.json'},'exact artifact census')
    for n,h in r['artifact_sha256'].items():require(sha(old.regular(output/n,profile()['output_bytes']))==h,'all artifact bytes')
    require(signal.read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=prepare_sha),'persisted binding')
    events=old.load_events(output,r)
    for phase in ('before','after'):
        require([e for e in events if e['kind']=='model_'+phase]==[dict(kind='model_'+phase,hashes=r['model_'+phase],classes=r['classes_'+phase])],'model journal')
        require([e for e in events if e['kind']=='source_'+phase]==[dict(kind='source_'+phase,sha256=digest(p['sources']))],'source journal')
        require([e['value'] for e in events if e['kind']=='precision_'+phase]==[old.precision_contract()],'precision journal')
    seen=set()
    def capsule_reader(slot,cid,call):
        n=f'signal_s{slot}_{cid:04}.json';require(n in names,'actual immutable source capsule')
        require(signal.read(old.regular(output/n,64*1024**2),64*1024**2)==call and sha(output/n)==digest(call),'saved source call bytes')
        seen.add(n);return dict(path=str(output/n),sha256=sha(output/n))
    summaries=validate_rows(p,r['rows'],events,capsule_reader=capsule_reader)
    require(seen==names-{'binding.json','events.jsonl'},'all source capsules consumed, no dangling evidence')
    settled=all(x['result'] is not None and not x['result']['errors'] for x in r['rows'])
    require(r['status']==('four_slots_settled_not_admitted' if settled else 'partial_or_failed'),'settled status exact')
    require(0<=r['wall_seconds']<=p['profile']['wall_seconds'] and sum(x.stat().st_size for x in output.iterdir())<=p['profile']['output_bytes'],'wall/disk limits')
    require(type(r['CUDA_peak_bytes']) is list and len(r['CUDA_peak_bytes'])==2 and all(type(v) is int and 0<=v<=p['profile'][k]
        for v,k in zip(r['CUDA_peak_bytes'],('cuda_allocated_bytes','cuda_reserved_bytes'))),'memory limits')
    require(sha(rp)==report_sha and all(sha(f)==h for f,h in p['sources'].items()),'unchanged final closure')
    return dict(schema=NAME+'.consume',status='saved_tree_engineering_verified',prepared=r['prepared'],prepare_sha256=prepare_sha,
        report_sha256=report_sha,denominator=4,request_slots_per_arm=144,arms=summaries,
        bank_protocol_sha256=spectral.PROTOCOL_SHA,total_FLOPs=None,**FLAGS)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('prepare');q.add_argument('--output',required=True)
    for cmd in ('verify','run','consume'):
        q=sub.add_parser(cmd);q.add_argument('--prepared',required=True);q.add_argument('--prepare-sha',required=True)
        if cmd=='run':q.add_argument('--output',required=True);q.add_argument('--authorization',required=True)
        if cmd=='consume':q.add_argument('--output',required=True);q.add_argument('--report-sha',required=True);q.add_argument('--receipt',required=True)
    a=p.parse_args(argv)
    if a.cmd=='prepare':out=prepare(a.output)
    elif a.cmd=='verify':out=dict(verified=True,schema=verify(a.prepared,a.prepare_sha)['schema'])
    elif a.cmd=='run':out=run(a.prepared,a.prepare_sha,a.output,authorization=a.authorization)
    else:out=write_new(a.receipt,consume(a.output,a.report_sha,a.prepared,a.prepare_sha),128*1024**2)
    print(canonical(out).decode())


if __name__=='__main__':main()
