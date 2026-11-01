"""Fixed H2 16/4 beam on two approved moving tasks; MAIN CUDA only.

Local bounded baseline, not official reproduction, equal-domain/compute or P1.
Prepare/verify/consume are saved-data only; no sourcepack/witness/physics input.
"""
from __future__ import annotations
import argparse
import ast
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import moving_history_planners_cuda_20261027_v1 as moving_plan
from experiments.spectral_reachability import p2_matched_composition_20261027 as matched
from experiments.spectral_reachability.d2p_kinodynamic_program_search import _bank, _cost, _validated_context

NAME='p1_moving_kinodynamic_baseline_cuda_20261027_v1'
ARM='fixed_H2_kinodynamic_beam16x4'
ROOT=moving_plan.ROOT
blind,moving,core=moving_plan.blind,moving_plan.moving,moving_plan.core
require,sha,read,canonical=blind.require,blind.sha,blind.read,blind.canonical
CAP,FLOOR=8*1024**2,200*1024**2
ENCODER_REPORT=ROOT/'results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/report.json'
ENCODER_SHA='e7fb554327b3bb5a96190f6291f8450b2b0723cd981817715cfb7e50aa266bf6'
SOURCEPACK_SHA='c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6'
BANK_SOURCE=ROOT/'results/pusht_p2_matched_composition4_v1_20261027_cuda/p2_matched_protocol.json'
BANK_SOURCE_SHA='38679b9a7339f822eb4d8cb9af2d5cf794c61badee6213021ac14434a0234720'
PINS={str(Path(moving_plan.__file__).resolve()):'f8f1d5507e2474fbbbfa56cae98f7bfa99f5929829946a2d7760c2c30ccc03c1',
      str(Path(matched.__file__).resolve()):'1b15e2c3f85e7459bef595655815c2a924bc2f5bd287a4d75b808b3af659ee9e',
      str(Path(blind.d2p.__file__).resolve()):'1dfa678a2a89856d557d50fe79a962235efc25d2e36b6114bf98e0a98d133f61',
      str(ROOT/'experiments/spectral_reachability/d2p_kinodynamic_program_search.py'):'c92a945ca884d503b918c7f7357be7ee55fa36377d9de60e2dddf68b42208b7d'}
# Literal old /prepared/{0,1}/bank/*/0, repeated five times WITHOUT recasting,
# resampling, clipping or action-domain restriction. Positional mapping frozen.
BANK_POINTS=(
 ((0.,0.),(-.35194534063339233,-.11348646134138107),(.2360766977071762,.9777212142944336),
  (-.16674906015396118,-.8386646509170532),(-.5749293565750122,.08259107172489166),
  (-.5478236675262451,.7505489587783813),(.41587963700294495,-.2373443841934204),
  (-.8634048104286194,-.012631894089281559),(-.9054667949676514,-.4206124246120453),
  (-.527147114276886,.820938229560852),(.30741173028945923,-.18261536955833435),
  (-.44456350803375244,-.8861082196235657),(.6568102836608887,.42583218216896057),
  (.03234505653381348,-.07335805892944336),(-.16670259833335876,.6744140386581421),
  (-.7490493655204773,-.7083526849746704)),
 ((0.,0.),(-.7196894884109497,.885351300239563),(-.29923003911972046,.1287318468093872),
  (-.2984808087348938,-.5939342975616455),(.6409215927124023,-.8484303951263428),
  (.8716726303100586,-.5530638098716736),(-.4635747969150543,.05819321051239967),
  (-.23930783569812775,.877028226852417),(.7703804969787598,.565495491027832),
  (-.27080631256103516,-.1460030972957611),(-.6189441680908203,.8764880895614624),
  (-.6901019811630249,.6159371137619019),(.8836219310760498,.9481450319290161),
  (-.7817689180374146,.25631341338157654),(-.6915401220321655,-.941928505897522),
  (.2273648977279663,-.8530343770980835)))


def bank(slot):
    require(type(slot) is int and slot in (0,1),'fixed positional bank slot')
    value=np.repeat(np.asarray(BANK_POINTS[slot])[:,None,:],5,axis=1).tolist()
    checked=_bank(value)
    require(len(checked)==16 and all(x.shape==(5,2) for x in checked),'exact sixteen raw5x2 bank')
    return value


def budget():
    s=matched.schedule(16,4)
    require(s['planning_predict_calls']==80 and s['replay_predict_calls']==4,'sealed fixed H2 schedule')
    return dict(planning_predict=80,H1_candidates=16,H2_candidates=64,beam_width=4,
        fresh_segmented_predict=2,fresh_whole_predict=2,predict=84,action_encoder=84,pred_projector=84,
        latent_input_tokens=252,action_input_tokens=252,pred_projector_rows=252,
        encoder=0,projector=0,AD=0,training=0,physics=0)


def contract():
    domains=[]
    for s in (0,1):
        b=np.asarray(bank(s));norms=np.linalg.norm(np.arctanh(b).reshape(16,10),axis=1)
        domains.append(dict(source_slot=s,bank_sha256=core.digest(bank(s)),bank_pointer=f'/prepared/{s}/bank',
            max_abs_raw=float(np.abs(b).max()),packet_logit_norms=norms.tolist(),
            outside_D2P_packet_ball=int(np.sum(norms>.75*np.sqrt(5)))))
    return dict(roots=2,arm=ARM,horizon_packets=2,packet_steps=5,bank_size=16,beam_width=4,
        per_root_budget=budget(),total_budget={k:2*v for k,v in budget().items() if k not in ('beam_width',)},
        selection='all16 H1 sorted(cost,path) retain4; all64 H2 sorted(terminalcost,path); NO H1 winner or padding',
        replay='search terminal == fresh segmented2 == fresh whole2, exact full contexts; no hidden composition predictions',
        bank=domains,bank_source_sha256=BANK_SOURCE_SHA,bank_mapping='old positions0,1 to moving opaque slots0,1; no outcome/native-identity lookup',
        action_support='sixteen repeated raw5x2 packets; not D2P independent per-time logits or its L2 ball; no clipping/rescaling',
        model_cost='mean squared native192 difference to own approved goal; identity Q, no fit',
        history='original nonzero normalized two packets; append new outgoing then roll states3/actions2 every packet',
        numerical=dict(SDP='MATH',matmul_TF32=False,cudnn_TF32=False),
        input_firewall='approved moving task exports only; bank literals; sourcepack SHA provenance-only never opened; no physical/witness input',
        inherited_encoder_report_sha256=ENCODER_SHA,sourcepack_provenance_only_sha256=SOURCEPACK_SHA,
        official_reproduction=False,equal_search_domain=False,equal_total_compute=False,
        independent_confirmation=False,P1_admission=False,P2_admission=False,
        inherited_encoding_cost='separate actual8 encoder/projector, not zero total pipeline cost',
        no_cpu_neural_fallback=True,prepare_cap_bytes=256*1024,output_cap_bytes=CAP,reserve_bytes=FLOOR,
        attempts='fixed order slots0,1; no retries/replacement; any error stops with partial/no commands release')


def local_closure(files):
    files={Path(f).resolve() for f in files};todo=list(files);seen=set()
    while todo:
        p=todo.pop()
        if p in seen or not p.is_relative_to(ROOT/'experiments') or p.suffix!='.py':continue
        seen.add(p)
        for node in ast.walk(ast.parse(p.read_text())):
            names=[]
            if isinstance(node,ast.Import):names=[x.name for x in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module and node.module.startswith('experiments'):
                names=[node.module]+[node.module+'.'+x.name for x in node.names]
            for name in names:
                if not name.startswith('experiments.'):continue
                child=ROOT/Path(*name.split('.')).with_suffix('.py')
                if child.is_file() and child not in files:files.add(child);todo.append(child)
    return sorted(files)


def build():
    for p,h in PINS.items():require(sha(p)==h,'frozen reusable source SHA')
    require(sha(moving.__file__)==moving_plan.LOADER_SHA,'approved moving loader SHA')
    require(sha(ENCODER_REPORT)==ENCODER_SHA and sha(BANK_SOURCE)==BANK_SOURCE_SHA,'encoder/bank provenance SHA')
    slots=[]
    for s,(path,h) in enumerate(zip(moving_plan.EXPORTS,moving_plan.EXPORT_SHAS,strict=True)):
        i=moving_plan.load_export(path,h);e=read(path)
        require(e['source_slot']==s and type(e['source_slot']) is int and
            e['task_contract']['prepare_sha256']==moving_plan.ENCODER_PREPARE_SHA,'approved ordered moving slots')
        slots.append(dict(source_slot=s,task_export=path,task_sha256=h,task_id=i['task_id'],blind_sha256=e['blind_sha256']))
    for k in ('task_export','task_sha256','task_id','blind_sha256'):
        require(len({x[k] for x in slots})==2,'unique two tasks')
    files=set(blind.source_files()+moving.source_files()+[Path(moving_plan.__file__),Path(matched.__file__),Path(__file__),
        ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')])
    files.update(Path(p) for p in PINS)
    files.add(ROOT/'experiments/spectral_reachability/pusht_composition_cuda_replay_v2.py')
    bindings={str(p):sha(p) for p in local_closure(files)}
    bindings.update(moving.MODELS);bindings.update({s['task_export']:s['task_sha256'] for s in slots})
    bindings.update({str(ENCODER_REPORT):ENCODER_SHA,str(BANK_SOURCE):BANK_SOURCE_SHA})
    for p,h in bindings.items():require(sha(p)==h,'final source/input/model SHA')
    return dict(schema=NAME+'.prepare',slots=slots,banks=[bank(s) for s in (0,1)],contract=contract(),
        bindings_sha256=bindings,policy=str(moving.POLICY),expected_model_hash=blind.encode.MODEL_TENSOR_SHA)


def prepare(output):
    p=build();raw=canonical(p);path=Path(output).absolute()
    require(not path.exists() and len(raw)<=256*1024 and shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'NEW <=256KiB prepare +200MiB reserve')
    with path.open('xb') as f:require(f.write(raw)==len(raw),'short prepare write');f.flush();os.fsync(f.fileno())
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(raw),contract=p['contract'])


def verify(path,expected):
    core.hash_value(expected);require(sha(path)==expected,'external own prepare SHA')
    p=read(path,256*1024);require(p==build(),'complete own prepare/source/task/model reconstruction');return p


def roll(t,before,raw,last):
    incoming=moving_plan.incoming(t,before,raw)
    return dict(before,latent_history=deepcopy(before['latent_history'][1:])+[deepcopy(last)],
        action_history=deepcopy(incoming['action_history'][-2:]))


def metric(t,context):
    z=np.asarray(context['latent_history'][-1],float);g=np.asarray(t['goal'],float)
    require(z.shape==g.shape==(192,) and np.isfinite(z).all() and np.isfinite(g).all(),'native192 goal/terminal')
    return _cost(np.mean((z-g)**2))


def beam(t,primitive_bank,dispatch):
    """Pure fixed-H2 schedule; dispatch is explicit CUDA or saved/RAM callback."""
    initial=_validated_context(t['root_context']);b=_bank(primitive_bank)
    require(len(b)==16 and all(x.shape==(5,2) for x in b),'fixed16 one-packet bank')
    first=[]
    for j in range(16):
        after=dispatch(initial,b[j].tolist(),'planning_H1',[j]);first.append((metric(t,after),[j],after))
    retained=sorted(first,key=lambda x:(x[0],x[1]))[:4];terminal=[]
    for _,path,parent in retained:
        for j in range(16):
            after=dispatch(parent,b[j].tolist(),'planning_H2',path+[j])
            terminal.append((metric(t,after),path+[j],after))
    winner=min(terminal,key=lambda x:(x[0],x[1]));path=winner[1]
    commands=np.concatenate([b[j] for j in path]).tolist()
    mid=dispatch(initial,commands[:5],'fresh_segmented',[path[0]])
    segmented=dispatch(mid,commands[5:],'fresh_segmented',path)
    whole=dispatch(initial,commands,'fresh_whole',path)
    require(segmented==whole==winner[2],'all THREE full-context replay paths exact')
    return dict(path=path,commands=commands,terminal_cost=winner[0],search_terminal=winner[2],
        fresh_segmented_mid=mid,fresh_segmented_terminal=segmented,fresh_whole_terminal=whole,
        first_costs=[x[0] for x in first],retained_paths=[x[1] for x in retained],
        terminal_costs=[dict(path=x[1],cost=x[0]) for x in terminal],H1_denominator=16,H2_denominator=64)


def validate_trace(t,before,raw,trace,after):
    """Saved actual module inputs/outputs, not a guessed dispatch resource bill."""
    c=_validated_context(before);raw=np.asarray(raw)
    require(raw.shape in ((5,2),(10,2)) and np.isfinite(raw).all() and np.abs(raw).max()<=1,'raw5 or raw10')
    require(len(trace)==6*(len(raw)//5),'every actual module attempt/return denominator')
    for j in range(len(raw)//5):
        events=trace[j*6:(j+1)*6]
        require([(x['kind'],x['resource']) for x in events]==[(k,r) for r in ('action_encoder','predict','pred_projector') for k in ('attempt','return')], 'module order/complete returns')
        want=moving_plan.incoming(t,c,raw[j*5:j*5+5])
        require(events[0]['value']==want['action_history'] and events[2]['value']==want['latent_history'],'actual normalized actions3/latent3 chronological inputs')
        last=events[5]['value'];require(np.asarray(last).shape==(192,) and np.isfinite(last).all(),'actual projected native successor')
        c=roll(t,c,raw[j*5:j*5+5],last)
    require(c==after,'saved actual module successor/returned context equality');return c


class NativeMeter:
    """Hooks count actual attempts BEFORE module execution, including failures."""
    def __init__(self,t,sink,sync,no_grad):
        self.t,self.sink,self.sync,self.no_grad=t,sink,sync,no_grad
        self.entries=[];self.active=False;self.failed=False;self.scope=None
        self.counts={r:dict(attempted=0,completed=0) for r in ('action_encoder','predict','pred_projector')}
    def start(self,before,raw):
        require(not self.active and not self.failed,'no overlapping/retried module scope')
        self.context=deepcopy(before);self.raw=np.asarray(raw);self.cursor=0;self.stage=0;self.entries=[];self.active=True
    def event(self,e):self.entries.append(deepcopy(e));self.sink(dict(kind='native_module',entry=e))
    def begin(self,resource,args):
        require(self.active and not self.failed and self.no_grad(),'only scoped no-grad CUDA calls')
        require(self.cursor<len(self.raw)//5 and resource==('action_encoder','predict','pred_projector')[self.stage//2] and self.stage%2==0,'exact native call order')
        count=self.counts[resource];require(count['attempted']<84,'actual84 per-root predispatch cap');count['attempted']+=1
        e=dict(kind='attempt',resource=resource)
        if resource in ('action_encoder','predict'):
            a=args[0];shape=(1,3,10 if resource=='action_encoder' else 192)
            e['value']=self.values(a,shape)[0]
            want=moving_plan.incoming(self.t,self.context,self.raw[self.cursor*5:self.cursor*5+5])
            require(e['value']==want['action_history' if resource=='action_encoder' else 'latent_history'],'actual CUDA chronological input')
        else:self.values(args[0],(3,192))
        self.event(e);self.stage+=1
    def end(self,resource,output):
        require(self.active and self.stage%2==1 and resource==('action_encoder','predict','pred_projector')[self.stage//2],'native unique return')
        self.sync();e=dict(kind='return',resource=resource)
        if resource=='pred_projector':
            e['value']=self.values(output,(3,192))[-1]
            self.context=roll(self.t,self.context,self.raw[self.cursor*5:self.cursor*5+5],e['value'])
        self.counts[resource]['completed']+=1;self.event(e);self.stage+=1
        if self.stage==6:self.cursor+=1;self.stage=0
    @staticmethod
    def values(tensor,shape):
        require(getattr(getattr(tensor,'device',None),'type',None)=='cuda' and tuple(tensor.shape)==shape,'CUDA-only actual module shape')
        a=tensor.detach().cpu().numpy();require(a.dtype==np.float32 and np.isfinite(a).all(),'finite native float32');return a.astype(float).tolist()
    def finish(self,after):
        require(self.cursor==len(self.raw)//5 and self.stage==0,'all native packet calls returned')
        require(after==self.context,'actual native returned full context');self.active=False
    @contextmanager
    def guard(self,model):
        hooks=[]
        try:
            for resource,module in (('action_encoder',model.action_encoder),('predict',model.predictor),('pred_projector',model.pred_proj)):
                hooks.append(module.register_forward_pre_hook(lambda m,a,r=resource:self.begin(r,a)))
                hooks.append(module.register_forward_hook(lambda m,a,o,r=resource:self.end(r,o)))
            def forbidden(m,a):raise ValueError('no image encoder/projector calls permitted')
            hooks.extend(m.register_forward_pre_hook(forbidden) for m in (model.encoder,model.projector))
            yield
        finally:
            for h in reversed(hooks):h.remove()


class Dispatch:
    def __init__(self,t,backend,meter,sink):self.t,self.backend,self.meter,self.sink=t,backend,meter,sink;self.index=0;self.failed=False
    def __call__(self,before,raw,phase,path):
        require(not self.failed and self.index<83,'no retry or extra dispatch')
        e=dict(index=self.index,phase=phase,path=path,before=deepcopy(before),raw=deepcopy(raw));self.index+=1
        self.sink(dict(kind='dispatch_attempt',call=e));self.meter.start(before,raw)
        try:
            after=_validated_context(self.backend.replay(deepcopy(before),raw) if phase=='fresh_whole' else self.backend.step(deepcopy(before),raw),before)
            self.meter.finish(after);validate_trace(self.t,before,raw,self.meter.entries,after)
            self.sink(dict(kind='dispatch_return',index=e['index'],after=after));return after
        except BaseException as exc:
            self.failed=True;self.meter.failed=True
            self.sink(dict(kind='dispatch_error',index=e['index'],error=blind.encode.error('dispatch',exc)));raise


class SavedDispatch:
    """Own consumer replays algorithm over saved data, never calls a model."""
    def __init__(self,t,events):self.t,self.events=t,deepcopy(events);self.offset=0;self.index=0
    def __call__(self,before,raw,phase,path):
        require(self.offset<len(self.events),'missing saved dispatch');event=self.events[self.offset];self.offset+=1
        want=dict(index=self.index,phase=phase,path=path,before=before,raw=raw);self.index+=1
        require(event==dict(kind='dispatch_attempt',call=want),'exact independent schedule/path/context/commands')
        trace=[]
        while self.offset<len(self.events) and self.events[self.offset]['kind']=='native_module':
            trace.append(self.events[self.offset]['entry']);self.offset+=1
        require(self.offset<len(self.events),'missing return/partial cannot release');event=self.events[self.offset];self.offset+=1
        require(event['kind']=='dispatch_return' and event['index']==want['index'],'exact unique dispatch return')
        after=_validated_context(event['after'],before);validate_trace(self.t,before,raw,trace,after);return after
    def done(self):require(self.offset==len(self.events) and self.index==83,'all83 dispatches/84 packets no extras')


def counts_ok(counts):return counts=={r:dict(attempted=84,completed=84) for r in ('action_encoder','predict','pred_projector')}


def total_counts(rows):
    return {r:{k:sum((row['resources'] or {}).get(r,{}).get(k,0) for row in rows)
        for k in ('attempted','completed')} for r in ('action_encoder','predict','pred_projector')}


def run(prepared,expected,output):
    p=verify(prepared,expected);store=blind.Store(output);errors=[];model=None;before=after=host=None;source_ok=False
    rows=[dict(**s,arm=ARM,status='blocked_not_attempted',resources=None) for s in p['slots']]
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        blind.encode.frozen(model);before=blind.encode.tensor_hash(model);require(before==p['expected_model_hash'],'same frozen model hash')
        backend=blind.d2p.TorchBackend(model,device)
        with sdpa_kernel(SDPBackend.MATH):
            for slot,row in zip(p['slots'],rows,strict=True):
                t=moving_plan.load_export(slot['task_export'],slot['task_sha256'])['gn_task'];row['status']='attempted'
                def sink(e):store.event(dict(source_slot=slot['source_slot'],task_id=slot['task_id'],event=e))
                meter=NativeMeter(t,sink,lambda:torch.cuda.synchronize(device),lambda:not torch.is_grad_enabled())
                try:
                    with meter.guard(model):result=beam(t,p['banks'][slot['source_slot']],Dispatch(t,backend,meter,sink))
                    require(counts_ok(meter.counts),'actual complete native84 ledger')
                    name=f"root_{slot['source_slot']}.result.json.gz";store.put(name,result)
                    row.update(status='model_plan_sealed',result_file=name,path=result['path'],terminal_cost=result['terminal_cost'],commands=result['commands'])
                finally:row['resources']=deepcopy(meter.counts)
    except BaseException as exc:
        errors.append(blind.encode.error('execution',exc))
        try:store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as exc:errors.append(blind.encode.error('journal_error',exc))
    finally:
        for row in rows:
            if row['status']=='attempted':row['status']='failed_partial'
        if model is not None:
            try:blind.encode.frozen(model);after=blind.encode.tensor_hash(model)
            except BaseException as exc:errors.append(blind.encode.error('final_model',exc))
        try:verify(prepared,expected);source_ok=True
        except BaseException as exc:errors.append(blind.encode.error('final_source',exc))
        try:store.close()
        except BaseException as exc:errors.append(blind.encode.error('journal_finalize',exc))
        complete=not errors and source_ok and before is not None and before==after and all(r['status']=='model_plan_sealed' and counts_ok(r['resources']) for r in rows)
        hashes={}
        try:
            for f in store.output.iterdir():require(f.is_file() and not f.is_symlink(),'flat artifacts');hashes[f.name]=sha(f)
        except BaseException as exc:errors.append(blind.encode.error('artifact_census',exc));complete=False
        report=dict(schema=NAME+'.report',status='fixed_H2_beam_plans_sealed_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,contract=p['contract'],rows=rows,errors=errors,
            first_error=errors[0] if errors else None,source_verified=source_ok,model_hash_before=before,model_hash_after=after,
            model_unchanged=before==after if before is not None else None,artifact_sha256=hashes,host=host,commands_released=complete,
            resources=total_counts(rows),AD=0,new_encoder=0,new_projector=0,
            P1_admission=False,P2_admission=False,training=False,new_physics=0,new_images=0)
        try:store.put('report.json',report,final=True)
        except BaseException as exc:complete=False;print(canonical(dict(error=blind.encode.error('final_report',exc))).decode(),flush=True)
    require(complete,'partial artifacts preserved; no retry: '+str(errors));return report


def consume(output,report_sha):
    core.hash_value(report_sha);output=Path(output).absolute();require(sha(output/'report.json')==report_sha,'external actual report SHA')
    r=read(output/'report.json');require(r['schema']==NAME+'.report' and r['status']=='fixed_H2_beam_plans_sealed_not_admitted' and
        r['errors']==[] and r['first_error'] is None and r['commands_released'] is True and r['source_verified'] is True and r['model_unchanged'] is True and
        r['P1_admission'] is False and r['P2_admission'] is False and r['training'] is False and
        r['AD']==r['new_encoder']==r['new_projector']==r['new_physics']==r['new_images']==0,'complete actual report only')
    p=verify(r['prepared'],r['prepare_sha256']);require(r['contract']==p['contract'] and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'],'model/protocol exact')
    names={'binding.json','events.jsonl.gz','root_0.result.json.gz','root_1.result.json.gz'}
    require(set(r['artifact_sha256'])==names and {f.name for f in output.iterdir()}==names|{'report.json'},'exact own artifact census')
    for name,h in r['artifact_sha256'].items():require(sha(output/name)==h,'actual artifact SHA')
    require(read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'own binding')
    plain,eof=blind.inflate((output/'events.jsonl.gz').read_bytes());require(eof and plain.endswith(b'\n'),'complete single gzip journal')
    groups=[]
    for line in plain.splitlines():
        e=blind.decode(line);core.fields(e,'source_slot task_id event','own scoped event')
        if not groups or groups[-1][0]!=(e['source_slot'],e['task_id']):groups.append(((e['source_slot'],e['task_id']),[]))
        groups[-1][1].append(e['event'])
    require([key for key,_ in groups]==[(s['source_slot'],s['task_id']) for s in p['slots']] and len(r['rows'])==2 and
        r['resources']==total_counts(r['rows']),'all two roots/denominator ordered/resource totals')
    plans=[]
    for slot,row,(_,events) in zip(p['slots'],r['rows'],groups,strict=True):
        require(all(row[k]==v for k,v in slot.items()) and row['arm']==ARM and row['status']=='model_plan_sealed' and counts_ok(row['resources']),'actual whole root row')
        t=moving_plan.load_export(slot['task_export'],slot['task_sha256'])['gn_task'];saved=SavedDispatch(t,events)
        result=beam(t,p['banks'][slot['source_slot']],saved);saved.done();name=f"root_{slot['source_slot']}.result.json.gz"
        require(row['result_file']==name and result==read(output/name) and all(row[k]==result[k] for k in ('path','commands','terminal_cost')),'independent saved arithmetic/model-only selection/three paths')
        plans.append(dict(**slot,arm=ARM,commands=result['commands'],path=result['path'],terminal_cost=result['terminal_cost']))
    require(sha(output/'report.json')==report_sha,'report final TOCTOU')
    return dict(schema=NAME+'.consume',status='fixed_H2_commands_verified_not_admitted',report=str(output/'report.json'),report_sha256=report_sha,
        plans=plans,errors=[],P1_admission=False,P2_admission=False,official_reproduction=False,equal_total_compute=False,equal_search_domain=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('prepared','prepare-sha','output','report-sha'):parser.add_argument('--'+k)
    a=parser.parse_args()
    if a.command=='prepare':result=prepare(a.output)
    elif a.command=='verify':result=dict(status='verified_no_execution',roots=len(verify(a.prepared,a.prepare_sha)['slots']))
    elif a.command=='run':result=run(a.prepared,a.prepare_sha,a.output)
    else:result=consume(a.output,a.report_sha)
    print(canonical(result).decode(),flush=True)


if __name__=='__main__':main()
