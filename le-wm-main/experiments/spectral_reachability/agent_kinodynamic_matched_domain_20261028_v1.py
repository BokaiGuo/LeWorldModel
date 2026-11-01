"""Fixed action-witness beam20/6, same packet logit ball, 144 native calls/root.

MAIN-only CUDA; implementation/tests never sample physics or fit a model.
Not RRT, official reproduction, equal FLOPs, independent roots, or P1 admission.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import moving_prediction_matched_sampling_cuda_20261027_v1 as sampling
from experiments.spectral_reachability import d2p_kinodynamic_program_search as contexts

NAME='agent_kinodynamic_matched_domain_20261028_v1'
ARM='matched_domain_H2_beam20x6'
ROOT,core,enc,base=sampling.ROOT,sampling.core,sampling.enc,sampling.base
require,sha,read,canonical=sampling.require,sampling.sha,sampling.read,sampling.canonical
SEEDS=(20281028,20281129)
CAP=float(.75*np.sqrt(5))
SAMPLING_SHA='72195df2a7f1d10948c2bf5c494dd26e66400eaa2e269737332cbd876e599d40'
CONTEXT_SHA='c92a945ca884d503b918c7f7357be7ee55fa36377d9de60e2dddf68b42208b7d'
RESOURCES=sampling.RESOURCES
budget=sampling.budget
Store=sampling.Store
Meter=sampling.Meter


def bank(slot):
    require(type(slot) is int and slot in (0,1),'fixed two slots')
    rng=np.random.Generator(np.random.PCG64(SEEDS[slot]))
    directions=rng.normal(size=(19,10));norm=np.linalg.norm(directions,axis=1)
    require(np.all(norm>0),'nonzero seeded directions; no retry')
    radii=CAP*rng.uniform(size=19)**.1
    logits=np.vstack((np.zeros((1,10)),directions/norm[:,None]*radii[:,None]))
    raw=np.tanh(logits.astype(np.float32)).reshape(20,5,2)
    require(np.isfinite(logits).all() and np.all(np.linalg.norm(logits,axis=1)<=CAP) and
        len({x.tobytes() for x in raw})==20,'fixed unique legal packet bank')
    return dict(seed=SEEDS[slot],logits=logits.tolist(),commands=raw.astype(float).tolist())


def contract():
    return dict(roots=2,arms=[ARM],groups=2,history='original moving observed0/5/10 and two actual nonzero incoming tokens',
        horizon=2,packet_steps=5,raw_steps=10,packet_logit_cap=CAP,
        bank=dict(size=20,zero_index=0,seeds=list(SEEDS),rng='NumPy PCG64',
            draw_order='normal19x10 then uniform19; solid10-ball radius=cap*U**(1/10)',
            reuse='same fixed20 packet bank at both depths; arbitrary per-time5x2, not constant packets',
            bank_sha256=[core.digest(bank(s)) for s in (0,1)]),
        schedule=dict(first20=20,beam_width=6,second120=120,planning_predict=140,
            fresh_whole_predict=2,fresh_segmented_predict=2,total_predict=144,high_level_dispatches=143),
        per_root=budget(),total=budget(288),predictor_samples_per_root=144,total_predictor_samples=288,
        selection='stable (H1meanQ,path) retain6; stable terminalmeanQ/path over ALL120 H2; no H1 winner',
        replays='freeze selected witness, fresh whole2 and fresh segmented1+1 from original root; all contexts exact',
        action_map='float32 tanh of float64 ball logits; original float32 normalization ONCE; no clip or /100',
        Q='original identity192 terminal mean squared error; no fit',
        full_action_witness=True,latent_nearest_neighbor_path=False,algorithm='finite action-space beam, NOT RRT',
        support_claim='all candidate packets inside same D2P logit ball; finite library not equal search coverage',
        matched_claim='144 predict API AND batch1 predictor samples/root, same axis as matched sampling',
        equal_total_compute=False,equal_FLOPs=False,equal_search_coverage=False,official_reproduction=False,
        inherited_encoding_once=sampling.contract()['inherited_encoding_once'],
        with_target=False,training=False,Qfit=False,AD=0,J=0,VJP=0,SVD=0,
        new_encoder=0,new_projector=0,new_physics=0,MATH=True,TF32=False,autocast=False,
        no_CPU_neural_fallback=True,retries=0,padding=0,independent_confirmation=False,P1_admission=False,
        output_cap_bytes=sampling.CAP,floor_bytes=sampling.FLOOR,wall_seconds=sampling.WALL,
        wall_scope='Store dispatch/write checks from output creation; cannot preempt native blocked calls')


def build():
    require(sha(sampling.__file__)==SAMPLING_SHA and sha(contexts.__file__)==CONTEXT_SHA,'frozen transition/meter/context helpers')
    p=deepcopy(sampling.build())
    p.update(schema=NAME+'.prepare',arguments={},contract=contract(),banks=[bank(s) for s in (0,1)])
    for f in (Path(__file__),Path(contexts.__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):
        p['bindings_sha256'][str(f.resolve())]=sha(f)
    require(len(p['slots'])==2 and [s['source_slot'] for s in p['slots']]==[0,1],'all two tasks')
    for path,h in p['bindings_sha256'].items():require(sha(path)==h,'final source/model/task closure')
    return p


def prepare(output):
    p=build();raw=canonical(p);path=Path(output).absolute()
    require(len(raw)<=256*1024 and not path.exists() and not path.is_symlink() and
        shutil.disk_usage(path.parent).free>=sampling.FLOOR+len(raw),'NEW small prepare/floor')
    with path.open('xb') as f:require(f.write(raw)==len(raw),'short prepare');f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(raw))


def verify(path,expected):
    core.hash_value(expected);require(sha(path)==expected,'external prepare SHA')
    p=read(path);require(p==build(),'entire own source/model/task/bank closure');return p


def checked_context(value,root=None):
    c=contexts._validated_context(value,root)
    require(c['context_length']==3 and c['latent_dim']==192 and c['action_dim']==10,'native fullH3 layout')
    return c


def commands(value):
    a=np.asarray(value)
    require(a.shape in ((5,2),(10,2)) and a.dtype.kind in 'fi' and np.isfinite(a).all() and
        np.array_equal(a,a.astype(np.float32)) and np.all(np.abs(a)<1),'literal native float32 raw5/raw10')
    # Tanh roundoff is not inverted to redefine the ball. Source logits are
    # separately bound/reconstructed before these exact raw commands are used.
    return a.astype(np.float32)


def rolled(task,before,raw,last):
    c=checked_context(before);z=np.asarray(last)
    require(z.shape==(192,) and np.isfinite(z).all() and np.array_equal(z,z.astype(np.float32)), 'native192 finite successor')
    return dict(c,latent_history=c['latent_history'][1:]+[z.tolist()],
        action_history=[c['action_history'][-1],core.normalized_tokens(task,raw)[0].astype(float).tolist()])


def cost(task,c):
    a=np.asarray(c['latent_history'][-1],float);g=np.asarray(task['goal'],float)
    require(a.shape==g.shape==(192,) and np.isfinite(a).all() and np.isfinite(g).all(),'native goal/Q')
    return contexts._cost(np.mean((a-g)**2))


def search(task,slot,dispatch,sink):
    require(task['packet_logit_cap']==CAP,'same original ball cap')
    root=checked_context(task['root_context']);b=bank(slot)
    sink(dict(kind='bank_sealed_before_evaluation',bank=b))
    first=[];second=[]
    def node(parent,path):
        before=root if parent is None else parent['context']
        raw=b['commands'][path[-1]];cs=dispatch(before,raw,'first' if len(path)==1 else 'second',path)
        require(len(cs)==1,'one packet per tree edge')
        c=checked_context(cs[-1],root);full=([] if parent is None else parent['commands'])+deepcopy(raw)
        n=dict(node_id='node_'+'_'.join(map(str,path)),parent_id='root' if parent is None else parent['node_id'],
            path=path,edge_logits=b['logits'][path[-1]],edge_commands=raw,commands=full,
            context=c,cost=cost(task,c))
        sink(dict(kind='node_saved',node=n));return n
    for i in range(20):first.append(node(None,[i]))
    retained=sorted(first,key=lambda n:(n['cost'],n['path']))[:6]
    sink(dict(kind='frontier_sealed',paths=[n['path'] for n in retained]))
    for parent in retained:
        for j in range(20):second.append(node(parent,parent['path']+[j]))
    winner=min(second,key=lambda n:(n['cost'],n['path']))
    sink(dict(kind='selection_sealed_before_replays',path=winner['path'],commands=winner['commands'],cost=winner['cost']))
    whole=dispatch(root,winner['commands'],'fresh_whole',winner['path'])
    seg1=dispatch(root,winner['commands'][:5],'fresh_segmented',winner['path'][:1])
    seg2=dispatch(seg1[-1],winner['commands'][5:],'fresh_segmented',winner['path'])
    parent=next(n for n in first if n['path']==winner['path'][:1])
    require(whole==seg1+seg2 and whole==[parent['context'],winner['context']], 'whole/segmented/tree ALL contexts exact')
    sink(dict(kind='independent_replays_verified',path=winner['path']))
    return dict(task_id=core.task_id(task),source_slot=slot,arm=ARM,seed=SEEDS[slot],bank_sha256=core.digest(b),
        first=first,retained_paths=[n['path'] for n in retained],second=second,
        selected_path=winner['path'],commands=winner['commands'],model_terminal_cost=winner['cost'],
        prediction=winner['context'],whole_contexts=whole,segmented_contexts=seg1+seg2)


def validate_packets(task,before,raw,cs,tickets):
    raw=commands(raw);need=len(raw)//5
    require(len(cs)==need and len(tickets)==4*need,'actual packet/module denominator')
    previous=checked_context(before)
    for k in range(need):
        ops=tickets[k*4:(k+1)*4]
        require([o['resource'] for o in ops]==['action_encoder','predict_API','predictor','pred_proj'] and
            all(o['status']=='returned' and o['packet']==k and o['batch_size']==o['samples']==1 and o['context_rows']==3 for o in ops),
            'one API and modules returned per packet')
        ae,api,pred,proj=ops
        outgo=np.concatenate((np.asarray(previous['action_history'],np.float32),core.normalized_tokens(task,raw[k*5:k*5+5])))[None]
        require(len(ae['inputs'])==1 and np.array_equal(sampling.literal(ae['inputs'][0],(1,3,10)),outgo),'actual AE normalized complete history')
        a=sampling.literal(ae['output']);require(a.ndim==3 and a.shape[:2]==(1,3),'AE actual output')
        z=np.asarray(previous['latent_history'])[None]
        require(len(api['inputs'])==len(pred['inputs'])==2 and api['inputs']==pred['inputs'] and
            np.array_equal(sampling.literal(api['inputs'][0],(1,3,192)),z) and
            np.array_equal(sampling.literal(api['inputs'][1]),a),'actual model.predict/predictor bridge')
        p=sampling.literal(pred['output']);require(p.ndim==3 and p.shape[:2]==(1,3),'predictor native context')
        require(len(proj['inputs'])==1 and np.array_equal(sampling.literal(proj['inputs'][0]),p.reshape(3,-1)), 'predictor to pred_proj')
        out=sampling.literal(proj['output'],(3,192))
        require(np.array_equal(sampling.literal(api['output'],(1,3,192)),out[None]),'API/projector output exact')
        previous=rolled(task,previous,raw[k*5:k*5+5],out[-1])
        require(cs[k]==previous,'native successor/full history exact')
    return deepcopy(cs)


class NativeEdge(sampling.NativeCuda):
    def advance(self,task,before,raw,meter):
        require(core.task_id(task)==self.bound_id and meter is self._active_meter,'bound original task/meter')
        raw=commands(raw);c=checked_context(before);z=self._tensor(c['latent_history'])[None];h=self._tensor(c['action_history'])[None]
        a=self._tensor(raw);cs=[]
        for k in range(len(raw)//5):
            z,h=self._step(task,z,h,a[k*5:k*5+5]);c=self._saved(z,h);cs.append(c)
            meter.emit('native_packet',packet=k,context=c)
        return cs


class Dispatch:
    def __init__(self,task,backend,meter,sink):
        self.task,self.backend,self.meter,self.sink=task,backend,meter,sink;self.index=0;self.failed=False
    def __call__(self,before,raw,phase,path):
        require(not self.failed and self.index<143,'no retry or extra edge')
        raw=commands(raw).astype(float).tolist();before=checked_context(before)
        e=dict(index=self.index,phase=phase,path=path,before=before,commands=raw)
        self.index+=1;self.meter.group='edge_'+str(e['index']);self.meter.packet=0
        self.sink(dict(kind='edge_attempt',edge=e));start=len(self.meter.ops)
        try:
            cs=self.backend.advance(self.task,deepcopy(before),raw,self.meter)
            cs=validate_packets(self.task,before,raw,cs,self.meter.ops[start:])
            self.sink(dict(kind='edge_return',index=e['index'],contexts=cs));return cs
        except BaseException as exc:
            self.failed=True;self.meter.fault=True
            try:self.sink(dict(kind='edge_failed',index=e['index'],error=enc.error('edge',exc)))
            except BaseException as secondary:exc.add_note('edge failure receipt '+repr(secondary))
            raise


class Cursor:
    def __init__(self,events):self.events=events;self.index=0
    def take(self):
        require(self.index<len(self.events),'missing/partial journal');e=self.events[self.index];self.index+=1;return e
    def expect(self,e):require(self.take()==e,'exact causal schedule/bank/node/selection journal')


class SavedDispatch:
    def __init__(self,task,cursor):self.task,self.cursor=task,cursor;self.index=0;self.operations=[]
    def __call__(self,before,raw,phase,path):
        require(self.index<143,'saved extra dispatch');i=self.index;self.index+=1
        self.cursor.expect(dict(kind='edge_attempt',edge=dict(index=i,phase=phase,path=path,before=before,commands=raw)))
        tickets=[];live={};cs=[];packet=0
        while True:
            e=self.cursor.take()
            if e['kind']=='edge_return':
                require(not live and e==dict(kind='edge_return',index=i,contexts=cs),'complete edge return')
                break
            require(e.get('evaluation_id')=='edge_'+str(i),'actual edge scope')
            if e['kind']=='native_packet':
                require(not live and e['packet']==packet and len(tickets)==4*(packet+1),'durable packet after all actual returns')
                cs.append(e['context']);packet+=1;continue
            require(e['kind'] in ('dispatch_attempted','dispatch_returned'),'successful actual ticket events')
            r=e['receipt'];j=r['index'];res=r['resource']
            require(r['candidate']=='edge_'+str(i) and r['packet']==packet and res in RESOURCES,'ticket edge/packet scope')
            if e['kind']=='dispatch_attempted':
                require(j==len(self.operations) and r['status']=='pending','unique sequential predispatch ticket')
                require(not live if res in ('action_encoder','predict_API') else
                    len(live)==1 and next(iter(live.values()))['resource']=='predict_API','API sequential children')
                require(sum(x['resource']==res for x in self.operations)<144,'actual144 cap')
                self.operations.append(r);tickets.append(r);live[j]=r
            else:
                require(j in live and r==dict(live[j],status='returned',output=r['output']),'same returned ticket')
                require(res!='predict_API' or len(live)==1,'API after all children')
                self.operations[j]=r
                tickets[next(k for k,x in enumerate(tickets) if x['index']==j)]=r;del live[j]
        return validate_packets(self.task,before,raw,cs,tickets)
    def done(self):
        require(self.index==143 and len(self.operations)==576 and
            all(sum(o['resource']==r and o['status']=='returned' for o in self.operations)==144 for r in RESOURCES),
            'all143 edges/144 packets/API+modules complete')


def journal_check(task,slot,result,events):
    c=Cursor(events);d=SavedDispatch(task,c);rebuilt=search(task,slot,d,c.expect);d.done()
    require(c.index==len(events) and rebuilt==result,'entire result/journal no extras')
    return rebuilt


def model_plan(slot,result):
    return dict(schema=NAME+'.model_plan',source_slot=slot['source_slot'],arm=ARM,
        task_id=slot['task_id'],task_export=slot['task_export'],task_sha256=slot['task_sha256'],
        blind_sha256=slot['blind_sha256'],commands=result['commands'],selected_path=result['selected_path'],
        prediction=result['prediction'],model_terminal_cost=result['model_terminal_cost'],
        bank_sha256=result['bank_sha256'],with_target=False,selection_uses_physics=False,
        P1_admission=False,official_reproduction=False,equal_total_compute=False)


def initial_rows(p):
    return [dict(source_slot=s['source_slot'],arm=ARM,task_id=s['task_id'],blind_sha256=s['blind_sha256'],
        status='not_started',commands=None,resources={},predictor_samples=dict(attempted=0,completed=0)) for s in p['slots']]


def execute(task,slot,model,device,torch,sink,row,check):
    m=Meter(sink,lambda:torch.cuda.synchronize(device),check)
    try:
        require(not torch.is_autocast_enabled('cuda'),'autocast forbidden')
        b=NativeEdge(model,device,task)
        with torch.inference_mode(),b.guard(m):
            d=Dispatch(task,b,m,sink);result=search(task,slot,d,sink)
        require(d.index==143 and m.counts()==budget() and m.predictor_samples()==dict(attempted=144,completed=144), 'actual144 each axis, no padding')
        return result
    finally:row['resources']=m.counts();row['predictor_samples']=m.predictor_samples()


def plan_all(p,store,runner,rows):
    for slot,row in zip(p['slots'],rows,strict=True):
        row['status']='attempted';s=slot['source_slot']
        task=sampling.old.load_export(slot['task_export'],slot['task_sha256'])['gn_task']
        def sink(e):store.event(dict(source_slot=s,arm=ARM,task_id=slot['task_id'],event=e))
        sink(dict(kind='root_started',seed=SEEDS[s],task_sha256=slot['task_sha256']))
        result=runner(task,s,sink,row)
        require(row['resources']==budget() and row['predictor_samples']==dict(attempted=144,completed=144),'no missing/extra root budget')
        stem=f'root_{s}';store.put(stem+'.result.json.gz',result)
        plan=model_plan(slot,result);plan['selection_sha256']=core.digest(plan)
        store.put(stem+'.plan.json',plan)
        sink(dict(kind='plan_sealed',selection_sha256=plan['selection_sha256'],commands=plan['commands']))
        row.update(status='model_plan_sealed',commands=plan['commands'],selection_sha256=plan['selection_sha256'],
            result_file=stem+'.result.json.gz',plan_file=stem+'.plan.json')


def run(prepared,expected,output):
    p=verify(prepared,expected);store=Store(output);rows=initial_rows(p)
    errors=[];model=None;before=after=host=None;source_ok=False;complete=False
    try:
        print('Fixed known moving2; same packet ball, 144 API/samples each; not RRT/FLOPs/P1.',flush=True)
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
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
            plan_all(p,store,lambda t,s,k,r:execute(t,s,model,device,torch,k,r,store.check_time),rows)
    except BaseException as exc:
        errors.append(enc.error('execution',exc))
        try:store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as secondary:errors.append(enc.error('error_journal',secondary))
    finally:
        for r in rows:
            if r['status']=='attempted':r['status']='failed_partial'
        if model is not None:
            try:enc.frozen(model);after=enc.tensor_hash(model)
            except BaseException as exc:errors.append(enc.error('final_model',exc))
        try:verify(prepared,expected);source_ok=True
        except BaseException as exc:errors.append(enc.error('final_source',exc))
        try:store.close()
        except BaseException as exc:errors.append(enc.error('journal_finalize',exc))
        hashes={}
        try:files=list(store.output.iterdir())
        except BaseException as exc:errors.append(enc.error('artifact_listing',exc));files=[]
        for f in files:
            try:
                require(f.is_file() and not f.is_symlink(),'flat regular artifact');hashes[f.name]=sha(f)
            except BaseException as exc:errors.append(enc.error('artifact_hash',exc))
        resources={k:{phase:sum(r['resources'].get(k,{}).get(phase,0) for r in rows) for phase in ('attempted','completed')} for k in RESOURCES}
        samples={phase:sum(r['predictor_samples'][phase] for r in rows) for phase in ('attempted','completed')}
        complete=not errors and source_ok and before is not None and before==after and all(r['status']=='model_plan_sealed' for r in rows) and resources==budget(288) and samples==dict(attempted=288,completed=288)
        report=dict(schema=NAME+'.report',status='matched_domain_beam_plans_sealed_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,contract=p['contract'],rows=rows,
            resources=resources,predictor_samples=samples,errors=errors,first_error=errors[0] if errors else None,
            artifact_sha256=hashes,source_verified=source_ok,model_hash_before=before,model_hash_after=after,
            model_unchanged=before==after if before else None,host=host,commands_released=complete,
            inherited_encoding_once=p['contract']['inherited_encoding_once'],elapsed_seconds=time.monotonic()-store.started,
            training=False,new_physics=0,P1_admission=False,equal_total_compute=False)
        try:store.put('report.json',report,final=True)
        except BaseException as exc:complete=False;print(canonical(dict(report_write_error=enc.error('final_report',exc),errors=errors)).decode(),flush=True)
    require(complete,'partial evidence preserved, no retry: '+str(errors));return report


def consume(output,expected):
    core.hash_value(expected);output=Path(output).absolute();require(sha(output/'report.json')==expected,'external report SHA')
    r=read(output/'report.json');p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and r['status']=='matched_domain_beam_plans_sealed_not_admitted' and
        r['errors']==[] and r['first_error'] is None and r['commands_released'] is True and
        r['source_verified'] is True and r['model_unchanged'] is True and
        r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'] and
        r['contract']==p['contract'] and r['resources']==budget(288) and
        r['predictor_samples']==dict(attempted=288,completed=288) and
        r['inherited_encoding_once']==p['contract']['inherited_encoding_once'] and
        r['P1_admission'] is False and r['training'] is False and r['new_physics']==0,'complete own model/source/budget contract')
    names={'binding.json','events.jsonl.gz'}|{f'root_{s}.{x}' for s in (0,1) for x in ('result.json.gz','plan.json')}
    require(set(r['artifact_sha256'])==names and {f.name for f in output.iterdir()}==names|{'report.json'},'exact all artifact census')
    for name,h in r['artifact_sha256'].items():require(sha(output/name)==h,'artifact SHA')
    require(read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'execution binding')
    raw,eof=base.inflate((output/'events.jsonl.gz').read_bytes(),sampling.EXPANDED)
    require(eof and raw.endswith(b'\n'),'complete single bounded gzip journal')
    events=[base.decode(line) for line in raw.splitlines()];groups=[]
    for e in events:
        require(set(e)=={'source_slot','arm','task_id','event'} and e['arm']==ARM,'own scoped successful event')
        key=e['source_slot']
        if not groups or groups[-1]!=key:groups.append(key)
    require(groups==[0,1] and len(r['rows'])==2,'fixed both root denominators/order')
    plans=[]
    for slot,row in zip(p['slots'],r['rows'],strict=True):
        s=slot['source_slot'];stem=f'root_{s}'
        require(row['source_slot']==s and row['arm']==ARM and row['task_id']==slot['task_id'] and row['blind_sha256']==slot['blind_sha256'] and
            row['status']=='model_plan_sealed' and row['resources']==budget() and row['predictor_samples']==dict(attempted=144,completed=144) and
            row['result_file']==stem+'.result.json.gz' and row['plan_file']==stem+'.plan.json','all root/arm/actual counts')
        task=sampling.old.load_export(slot['task_export'],slot['task_sha256'])['gn_task'];result=read(output/row['result_file'])
        group=[e for e in events if e['source_slot']==s];require(all(e['task_id']==slot['task_id'] for e in group),'journal task identity')
        scope=[e['event'] for e in group]
        require(scope[0]==dict(kind='root_started',seed=SEEDS[s],task_sha256=slot['task_sha256']),'root start identity')
        journal_check(task,s,result,scope[1:-1]);plan=model_plan(slot,result);ident=core.digest(plan);plan['selection_sha256']=ident
        require(read(output/row['plan_file'])==plan and row['commands']==plan['commands'] and row['selection_sha256']==ident and
            scope[-1]==dict(kind='plan_sealed',selection_sha256=ident,commands=plan['commands']),'exact winner/replay plan seal')
        plans.append(dict(plan,whole_run_report_sha256=expected))
    for path,h in p['bindings_sha256'].items():require(sha(path)==h,'final source/model/task SHA')
    for name,h in r['artifact_sha256'].items():require(sha(output/name)==h,'final artifact SHA')
    require(sha(output/'report.json')==expected,'final report SHA')
    return dict(schema=NAME+'.consume',status='matched_domain_witnesses_verified_not_admitted',plans=plans,
        resources=r['resources'],predictor_samples=r['predictor_samples'],P1_admission=False,
        equal_total_compute=False,official_reproduction=False,errors=[])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('prepared','prepare-sha','output','report-sha'):p.add_argument('--'+k)
    a=p.parse_args()
    if a.command=='prepare':r=prepare(a.output)
    elif a.command=='verify':r=dict(status='verified_no_execution',roots=len(verify(a.prepared,a.prepare_sha)['slots']))
    elif a.command=='run':r=run(a.prepared,a.prepare_sha,a.output)
    else:r=consume(a.output,a.report_sha)
    print(canonical(r).decode(),flush=True)


if __name__=='__main__':main()
