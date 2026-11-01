"""All10 root-prefix modes, 42 fixed proposals; known-root calibration, MAIN CUDA only.

Adapt compile_arm without global patching. No model/Q fit, physics or admission.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from io import BytesIO
import os
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
from experiments.spectral_reachability import p3_goal_patch_consistent_compiler_20261027_v2 as kernel
from experiments.spectral_reachability import pusht_full_packet_compiler_20261027 as compiler
from experiments.spectral_reachability import p1_moving_overlay_recompile_cuda_20261027_v1 as overlay
from experiments.spectral_reachability import agent_baselines_resume_20261027_next as capture

NAME='agent_allmode_prefix_20261028_v1'
ARM='allmode_full10_conditional_D2P'
AUTH='MAIN_ALLMODE42_PREFIX_CUDA_ONLY'
ROOT=overlay.ROOT
ORDER=[(s,c) for s in (0,1) for c in ('off','on')]
PARENT=ROOT/'results/p1_moving_overlay_recompile_cuda_20261027_v1_cuda_MAIN01'
PARENT_SHA='2b2e5064f74f8aca712ae0d0415fcbfb64a2c8e6325bdffe24ae4fdc3b373381'
PARENT_PREP_SHA='8d622196815f90476d02d7ca6750817116e2100e81945f192a46f740156a9a20'
CAP,FLOOR,FINAL,PREP_CAP,WALL=128*1024**2,200*1024**2,64*1024**2,2*1024**2,1800
require,sha,canonical,core=overlay.require,overlay.sha,overlay.canonical,overlay.core
io=kernel.io
validate_q,validate_context,validate_advance,cost,finish_event=kernel.validate_q,kernel.validate_context,kernel.validate_advance,kernel.cost,kernel.finish_event
# kernel's digest canonicalization is preserved, not replaced by another receipt schema.
digest=kernel.digest


def contract():
    return dict(roots=2,root_conditions=4,order=[list(k) for k in ORDER],arm=ARM,
        proposals=42,modes=list(range(10)),radii=list(compiler.RADII),signs=[-1,1],
        ordering='zero,DLS; radius outer, mode0..9, sign(-1,+1)',
        damping=compiler.DAMPING,packet_logit_cap=compiler.CAP,beta=1.,Q='original identity192; no fit',
        selection='stable(index) argmin(stage_cost + own_suffix_ridge_reach_value), NOT nonlinear terminal argmin',
        suffix='one original bounded_ridge per each42 prefix, own conditional nominal/J',
        per_root_condition=dict(predict=258,action_encoder=258,pred_proj=258,predictor_samples=258,
            jacobians=43,VJP_directions=8256,SVD_successful_frozen_helper_calls=44),
        total=dict(predict=1032,action_encoder=1032,pred_proj=1032,predictor_samples=1032,
            jacobians=172,VJP_directions=33024,SVD_successful_frozen_helper_calls=176),
        replay='42 independent H2 candidate replays + winner H2 whole + two H1 segments',
        raw_authority='once NumPy tanh(float32 params) -> literal float32; same model/replay/dispatch buffer',
        root_J_coordinates='10 packet logits at zero; CUDA tanh only for zero-point derivative, actual raw saved',
        math_accounting='root_bank frozen spectral_bank contains2 SVD; each suffix ridge contains1; failed internal SVD attempts unknown',
        saved_validation='independent arithmetic bank+42ridge recomputation per result, not neural calls; count separately',
        new_encoder=0,new_projector=0,new_physics=0,training=False,Qfit=False,retries=0,padding=0,
        inherited='four root-condition tasks share historical16 image encodes; original source/planner costs paid, not zero',
        equal_total_compute=False,equal_FLOPs=False,P1_admission=False,P3_admission=False,independent_confirmation=False,
        interpretation='rank3 omission only .56-.88percent J energy per MAIN; allmode is implementation-span ablation, not theory repair',
        CUDA_only=True,MATH=True,TF32=False,autocast=False,wall_seconds=WALL,
        output_cap_bytes=CAP,final_reserve_bytes=FINAL,disk_floor_bytes=FLOOR,prepare_cap_bytes=PREP_CAP)


def all_mode_bank(jacobian,nominal,goal,q):
    # Reuse sealed spectral/DLS math including full signed/canonicalized right basis.
    j=compiler.finite(jacobian,(192,10));n=compiler.finite(nominal,(192,));g=compiler.finite(goal,(192,))
    original=compiler.spectral_bank(j,n,g,validate_q(q))
    right=compiler.finite(original['right_vectors'],(10,10))
    values=[deepcopy(original['params'][0]),deepcopy(original['params'][1])]
    kinds=['zero','root_dls_progress']
    for radius in compiler.RADII:
        for mode in range(10):
            for sign in (-1,1):
                values.append((sign*radius*right[mode]).tolist())
                kinds.append(f'spectral_mode_{mode}_radius_{radius:.8f}_sign_{sign}')
    require(len(values)==42 and len(kinds)==42,'all10 modes including zero singular modes; no pruning')
    return dict(params=values,kinds=kinds,singular_values=original['singular_values'],right_vectors=original['right_vectors'])


def compile_arm(packet,q,backend,journal):
    """Backend-independent actual full10 H2 compiler; no inherited candidates.

    Production backend must implement separate nominal/jacobian/step/replay.
    Analytic test backends validate contracts only, never scientific evidence.
    """
    q=validate_q(q);goal=compiler.finite(packet['goal'],(192,))
    qsha=digest(q.tolist());initial=deepcopy(packet['root_context']);operations=[]
    validate_context(initial)
    def operation(kind,callback):
        event=dict(kind=kind,index=len(operations),completed=False,q_sha256=qsha)
        if kind=='jacobian':event.update(shape=[192,10],mathematical_vjp_directions=192)
        else:event.update(SVD_complete_if_returned=2 if kind=='root_bank' else 1, SVD_attempts_on_failure='unknown_inside_frozen_helper')
        operations.append(event);journal(deepcopy(event))
        try:
            result=callback();event['completed']=True
            return result
        except BaseException as exc:
            event['error']=type(exc).__name__+': '+str(exc);raise
        finally:finish_event(journal,event)
    def replay(context,commands,phase):
        event=dict(kind='replay',phase=phase,commands=np.asarray(commands).tolist(),q_sha256=qsha,completed=False)
        journal(deepcopy(event))
        try:
            context=backend.replay(context,commands);event.update(completed=True,context=context)
            return context
        except BaseException as exc:event['error']=type(exc).__name__+': '+str(exc);raise
        finally:finish_event(journal,event)
    def fresh(context,phase):
        n=compiler.finite(backend.nominal(deepcopy(context)),(192,))
        journal(dict(kind='nominal',phase=phase,q_sha256=qsha,completed=True,nominal=n.tolist()))
        j=compiler.finite(backend.jacobian(deepcopy(context),operation),(192,10))
        journal(dict(kind='derivative',phase=phase,q_sha256=qsha,completed=True,jacobian=j.tolist()))
        return n,j
    nominal,jacobian=fresh(initial,'root')
    bank=operation('root_bank',lambda:all_mode_bank(jacobian,nominal,goal,q))
    journal(dict(kind='bank',q_sha256=qsha,bank=bank,root_nominal=nominal.tolist(),root_jacobian=jacobian.tolist()))
    candidates=[]
    for index,params in enumerate(bank['params']):
        prefix=np.tanh(np.asarray(params,dtype=np.float32)).reshape(5,2)
        event=dict(kind='candidate',index=index,q_sha256=qsha,completed=False,prefix_commands=prefix.tolist())
        journal(deepcopy(event))
        try:
            after=backend.step(deepcopy(initial),prefix)
            journal(dict(kind='prefix',index=index,q_sha256=qsha,completed=True,context=after))
            validate_advance(initial,prefix,after)
            n,j=fresh(after,'suffix_'+str(index))
            event.update(predicted_prefix=after['latent_history'][-1],nominal_terminal=n.tolist(),jacobian=j.tolist())
            journal(deepcopy(event))
            solve=operation('suffix_ridge',lambda:compiler.bounded_ridge(q@j,q@(goal-n)))
            suffix=np.tanh(np.asarray(solve['params'],dtype=np.float32)).reshape(5,2)
            commands=np.concatenate((prefix,suffix))
            event.update(suffix_solve=solve,suffix_params=solve['params'],suffix_commands=suffix.tolist(),commands=commands.tolist())
            journal(deepcopy(event))
            predicted=backend.step(deepcopy(after),suffix)
            event.update(predicted_terminal=predicted['latent_history'][-1]);journal(deepcopy(event))
            validate_advance(initial,commands,predicted)
            verified=replay(deepcopy(initial),commands,'candidate_'+str(index))
            require(io.canonical(predicted)==io.canonical(verified),'independent H2 full-context nonlinear replay differs')
            stage=cost(after['latent_history'][-1],goal,q)
            event.update(completed=True,candidate_id=f'prefix_{index}',proposal_kind=bank['kinds'][index],
                prefix_params=params,suffix_params=solve['params'],suffix_commands=suffix.tolist(),
                commands=commands.tolist(),predicted_prefix=after['latent_history'][-1],
                predicted_terminal=predicted['latent_history'][-1],nominal_terminal=n.tolist(),
                jacobian=j.tolist(),stage_cost=stage,suffix_solve=solve,own_reach_value=solve['own_reach_value'],
                objective=stage+solve['own_reach_value'],nonlinear_terminal_cost=cost(predicted['latent_history'][-1],goal,q),
                context=verified,independent_model_replay_exact=True)
            candidates.append(deepcopy(event))
        except BaseException as exc:
            event['error']=type(exc).__name__+': '+str(exc);raise
        finally:finish_event(journal,event)
    winner=deepcopy(min(candidates,key=lambda c:(c['objective'],c['index'])))
    journal(dict(kind='selection_frozen_before_winner_replays',index=winner['index'],commands=deepcopy(winner['commands']),q_sha256=qsha))
    whole=replay(deepcopy(initial),winner['commands'],'winner_whole')
    segmented=deepcopy(initial)
    for k,packet_actions in enumerate(np.asarray(winner['commands']).reshape(2,5,2)):
        segmented=replay(segmented,packet_actions,'winner_segment_'+str(k))
    require(io.canonical(whole)==io.canonical(segmented)==io.canonical(winner['context']),'winner double replay mismatch')
    return dict(status='completed',q_sha256=qsha,bank=bank,root_jacobian=jacobian.tolist(),root_nominal=nominal.tolist(),
        candidates=candidates,winner=winner,whole_replay_context=whole,segmented_replay_context=segmented,
        whole_and_segmented_exact=True,operations=operations)


def validate_compilation(packet,q,result):
    """Independent saved arithmetic validator; never calls a backend/model."""
    q=validate_q(q);goal=compiler.finite(packet['goal'],(192,));initial=packet['root_context']
    validate_context(initial)
    require(result['status']=='completed' and result['q_sha256']==digest(q.tolist()),'original Q receipt')
    bank=all_mode_bank(result['root_jacobian'],result['root_nominal'],goal,q)
    require(result['bank']==bank and len(result['candidates'])==42,'all42 original proposals, including null directions')
    for index,c in enumerate(result['candidates']):
        prefix=np.tanh(np.asarray(bank['params'][index],np.float32)).reshape(5,2)
        solve=compiler.bounded_ridge(q@compiler.finite(c['jacobian'],(192,10)),q@(goal-compiler.finite(c['nominal_terminal'],(192,))))
        suffix=np.tanh(np.asarray(solve['params'],np.float32)).reshape(5,2)
        commands=np.concatenate((prefix,suffix)).tolist()
        require(c['index']==index and c['completed'] is True and c['prefix_params']==bank['params'][index] and
            c['proposal_kind']==bank['kinds'][index] and c['suffix_solve']==solve and c['suffix_params']==solve['params'] and
            c['prefix_commands']==prefix.tolist() and c['suffix_commands']==suffix.tolist() and c['commands']==commands,
            'fixed proposal/support, own suffix ridge and literal raw authority')
        validate_advance(initial,commands,c['context'])
        require(c['context']['latent_history']==[initial['latent_history'][-1],c['predicted_prefix'],c['predicted_terminal']],
            'whole H2 chronological full context, not repeated terminal receipt')
        stage=cost(c['predicted_prefix'],goal,q)
        require(c['stage_cost']==stage and c['own_reach_value']==solve['own_reach_value'] and
            c['objective']==stage+solve['own_reach_value'] and
            c['nonlinear_terminal_cost']==cost(c['predicted_terminal'],goal,q) and
            c['independent_model_replay_exact'] is True,'unchanged beta1 stage+reach objective, not endpoint oracle')
    winner=min(result['candidates'],key=lambda c:(c['objective'],c['index']))
    require(result['winner']==winner and result['whole_and_segmented_exact'] is True and
        result['whole_replay_context']==result['segmented_replay_context']==winner['context'],'stable winner and fresh whole/segmented replays')
    kinds=['jacobian','root_bank']+[k for _ in range(42) for k in ('jacobian','suffix_ridge')]
    ops=result['operations']
    require(len(ops)==86 and [e['kind'] for e in ops]==kinds and
        all(e['index']==i and e['completed'] is True and e['q_sha256']==result['q_sha256'] for i,e in enumerate(ops)),
        'all43 derivatives and43 frozen bank/solve helpers')
    for e in ops:
        if e['kind']=='jacobian':require(e['shape']==[192,10] and e['mathematical_vjp_directions']==192,'full derivative directions')
        else:require(e['SVD_complete_if_returned']==(2 if e['kind']=='root_bank' else 1) and
            e['SVD_attempts_on_failure']=='unknown_inside_frozen_helper','honest frozen-helper SVD accounting')
    return deepcopy(winner)


def validate_compiler_events(result,events):
    require([e for e in events if e['kind']=='candidate' and e['completed']]==result['candidates'],'all42 durable candidates')
    ns=[e for e in events if e['kind']=='nominal'];js=[e for e in events if e['kind']=='derivative']
    phases=['root']+['suffix_'+str(i) for i in range(42)]
    require([e['phase'] for e in ns]==[e['phase'] for e in js]==phases,'all43 nominal/J preserved before solves')
    require([e['nominal'] for e in ns]==[result['root_nominal']]+[c['nominal_terminal'] for c in result['candidates']] and
        [e['jacobian'] for e in js]==[result['root_jacobian']]+[c['jacobian'] for c in result['candidates']],'durable numerical arrays')
    banks=[e for e in events if e['kind']=='bank'];require(len(banks)==1 and banks[0]['bank']==result['bank'],'durable complete bank')
    replays=[e for e in events if e['kind']=='replay' and e['completed']]
    require([e['phase'] for e in replays]==['candidate_'+str(i) for i in range(42)]+['winner_whole','winner_segment_0','winner_segment_1'],
        '42 independent candidate and3 winner replay receipts')
    require([e['context'] for e in replays[:42]]==[c['context'] for c in result['candidates']] and
        replays[-3]['context']==result['whole_replay_context'] and replays[-1]['context']==result['segmented_replay_context'],'actual replay context receipts')
    selected=[(i,e) for i,e in enumerate(events) if e['kind']=='selection_frozen_before_winner_replays']
    require(len(selected)==1 and selected[0][1]['index']==result['winner']['index'] and selected[0][1]['commands']==result['winner']['commands'] and
        sum(e['kind']=='candidate' and e['completed'] for e in events[:selected[0][0]])==42 and
        events[selected[0][0]+1]['kind']=='replay' and events[selected[0][0]+1]['phase']=='winner_whole','selection frozen before independent winner replay')


def build():
    # Exact original four encoded tasks; no new encoder, root or outcome selection.
    overlay.consume(PARENT,PARENT_SHA)
    r=overlay.read(PARENT/'report.json',4*1024**2)
    require(r['prepare_sha256']==PARENT_PREP_SHA,'original paired blind task preparation')
    parent=overlay.verify(r['prepared'],PARENT_PREP_SHA)
    slots=[];bindings={str(PARENT/'report.json'):PARENT_SHA,str(Path(r['prepared'])):PARENT_PREP_SHA}
    for index,key in enumerate(ORDER):
        path=PARENT/f'task_{index:02}.json.gz'
        require(sha(path)==r['artifact_sha256'][path.name],'original task artifact hash')
        e=overlay.read(path,PREP_CAP);interface=overlay.task_interface(e)
        require((e['task_contract']['root_slot'],e['task_contract']['condition'])==key and
            np.array_equal(interface['q'],np.eye(192)),'four original root-condition tasks and identity Q')
        slots.append(dict(index=index,root_slot=key[0],condition=key[1],task_id=e['task_id'],
            blind_sha256=e['task_contract']['blind_sha256'],packet=dict(root_context=deepcopy(e['task']['root_context']),goal=deepcopy(e['task']['goal'])),
            q=np.asarray(interface['q'],np.float64).tolist(),task_file=str(path),task_sha256=sha(path)))
        bindings[str(path)]=sha(path)
    # Parent verify transitively retains its entire source/model/blind closure.
    for f in overlay.source_files()+[Path(__file__),Path(kernel.__file__),Path(compiler.__file__),Path(capture.__file__),
            Path(capture.prior.__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]:
        bindings[str(Path(f).resolve())]=sha(f)
    for path,h in bindings.items():require(sha(path)==h,'final input/source/model closure')
    return dict(schema=NAME+'.prepare',contract=contract(),slots=slots,bindings=bindings,
        policy=parent['policy'],expected_model_hash=parent['expected_model_hash'],source_report_sha256=parent['source_report_sha256'])


def write_new(output,raw):
    path=Path(output)
    require(path.name.startswith(NAME+'_') and path.name.endswith('.json') and
        isinstance(raw,bytes) and len(raw)<=PREP_CAP,'own allmode prefix JSON and bounded canonical bytes')
    return capture.write_new(output,raw,PREP_CAP)


def prepare(output):
    p=build();return write_new(output,canonical(p))


def verify(path,expected):
    core.hash_value(expected);require(sha(path)==expected,'external own prepare SHA')
    p=overlay.read(path,PREP_CAP);require(p==build(),'entire own4 task/source reconstruction');return p


class Store(capture.Store):
    def __init__(self,output):
        import shutil,zlib
        self.output=Path(output).absolute()
        require(not self.output.exists() and shutil.disk_usage(self.output.parent).free>=FLOOR+CAP,'NEW128MiB output +200MiB floor')
        self.output.mkdir();self.comp=zlib.compressobj(6,zlib.DEFLATED,31);self.closed=self.broken=False
        self._write('events.jsonl.gz',b'')
    def _write(self,name,raw,append=False,final=False):
        import shutil
        require(Path(name).name==name and name not in ('','.','..') and not (self.output/name).is_symlink(),'flat output')
        used=sum(p.stat().st_size for p in self.output.iterdir() if p.is_file())
        require(used+len(raw)<=CAP-(0 if final else FINAL) and shutil.disk_usage(self.output).free>=FLOOR+len(raw),'output cap/final reserve/floor')
        with (self.output/name).open('ab' if append else 'xb') as f:
            require(f.write(raw)==len(raw),'short write');f.flush();os.fsync(f.fileno())
        fd=os.open(self.output,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)


class Modules(capture.Modules):
    def pre(self,kind):
        def enter(module,args):
            require(not self.failed and not self.pending and len(self.entries)<774 and
                kind==self.ORDER[len(self.entries)%3] and self.meter.stage==len(self.entries)//3,
                '258 single-sample raw forwards, no extra/retry')
            e=dict(index=len(self.entries),resource=kind,stage=self.meter.stage,completed=False,inputs=[])
            self.entries.append(e);self.pending[module]=e
            self.store.event(dict(kind='module_attempt',entry=deepcopy(e)))
            for j,x in enumerate(args):
                a=self.array(x)
                if kind=='action_encoder':require(a.shape==(1,3,10) and np.array_equal(a[0],self.meter.expected_input['action_history']),'literal raw normalization')
                elif kind=='predict':
                    if j==0:require(a.shape==(1,3,192) and np.array_equal(a[0],self.meter.expected_input['latent_history']),'actual incoming context')
                    else:require(np.array_equal(a,self.action_embedding),'AE->predictor full bridge')
                else:require(a.shape==(3,192) and np.array_equal(a,self.predictor_output.reshape(3,192)),'predictor->projector bridge')
                e['inputs'].append(capture.prior.save_array(self.store,f'{self.stem}.module_{e["index"]:03}.input_{j}.npy',a))
            self.store.event(dict(kind='module_inputs_saved',entry=deepcopy(e)))
        return enter


class Backend:
    """Explicit CUDA calls, no global replacements; saved literal raw used by transition."""
    def __init__(self,model,device,torch,store,row,meter,check):
        require(device.type=='cuda','no neural CPU fallback')
        self.torch,self.device,self.store,self.row,self.meter,self.check=torch,device,store,row,meter,check
        self.transition=compiler.make_transition(model,device);row['forwards']=[];row['AD_API']=[]
    def forward(self,c,raw,phase):
        t=self.torch;self.check();n=len(self.row['forwards']);require(n<258,'predict API cap')
        a=raw.detach().cpu().numpy().copy()
        require(a.dtype==np.float32 and a.shape==(5,2) and np.isfinite(a).all() and np.max(abs(a))<=1,'actual literal CUDA raw')
        e=dict(index=n,phase=phase,input=deepcopy(c),raw=a.tolist(),completed=False)
        self.row['forwards'].append(e);self.meter.stage=n;self.meter.last_native=None
        self.meter.expected_input=capture.prior.planner.incoming(
            dict(action_mean=kernel.ACTION_MEAN.tolist(),action_std=kernel.ACTION_STD.tolist()),c,a)
        self.store.event(dict(kind='forward_attempt',root=self.row['index'],entry=deepcopy(e)))
        e['raw_array']=capture.prior.save_array(self.store,f'root_{self.row["index"]}.raw_{n:03}.npy',a)
        emb=t.tensor(c['latent_history'],dtype=t.float32,device=self.device)[None]
        history=t.tensor(c['action_history'],dtype=t.float32,device=self.device)[None]
        after,between=self.transition(emb,history,raw)
        self.check();out=dict(c,latent_history=after[0].detach().cpu().tolist(),action_history=between[0].detach().cpu().tolist())
        require(np.array_equal(out['latent_history'][-1],self.meter.last_native),'returned successor actual native module output')
        e.update(completed=True,output=out)
        self.store.event(dict(kind='forward_return',root=self.row['index'],entry=deepcopy(e)))
        return after,out
    def step(self,c,raw):
        a=np.asarray(raw,np.float32)
        with self.torch.no_grad():return self.forward(c,self.torch.tensor(a,device=self.device,dtype=self.torch.float32),'step')[1]
    def nominal(self,c):
        with self.torch.no_grad():return np.asarray(self.forward(c,self.torch.zeros((5,2),device=self.device,dtype=self.torch.float32),'nominal')[1]['latent_history'][-1])
    def jacobian(self,c,operation):
        t=self.torch
        def calculate():
            e=dict(index=len(self.row['AD_API']),completed=False,shape=[192,10],VJP_directions=192,
                functional_jacobian_calls=1,vectorize=True,engine_internal_calls='not independently intercepted')
            self.row['AD_API'].append(e);self.store.event(dict(kind='AD_attempt',root=self.row['index'],entry=deepcopy(e)))
            def fn(u):return self.forward(c,t.tanh(u).reshape(5,2),'jacobian_zero_forward')[0][0,-1]
            with t.enable_grad():j=t.autograd.functional.jacobian(fn,t.zeros(10,device=self.device,dtype=t.float32,requires_grad=True),vectorize=True)
            a=j.detach().cpu().numpy().copy();e['jacobian']=capture.prior.save_array(self.store,f'root_{self.row["index"]}.J_{e["index"]:02}.npy',a)
            e['completed']=True;self.store.event(dict(kind='AD_return',root=self.row['index'],entry=deepcopy(e)))
            return a
        return operation('jacobian',calculate)
    def replay(self,c,commands):
        for raw in np.asarray(commands,np.float32).reshape(-1,5,2):c=self.step(c,raw)
        return c


def expected_forward_inputs(packet,result):
    """Pure full schedule reconstruction, including each candidate and all replays."""
    initial=deepcopy(packet['root_context']);zero=np.zeros((5,2),np.float32).tolist();want=[]
    def add(c,raw):want.append((deepcopy(c),deepcopy(raw)))
    def prefix_context(c):
        after=deepcopy(initial);after['latent_history']=initial['latent_history'][1:]+[c['predicted_prefix']]
        after['action_history']=np.concatenate([initial['action_history'],kernel.normalized_history(c['prefix_commands'])]).tolist()[-2:]
        return after
    add(initial,zero);add(initial,zero)
    for c in result['candidates']:
        after=prefix_context(c)
        for ctx,raw in ((initial,c['prefix_commands']),(after,zero),(after,zero),(after,c['suffix_commands']),
                (initial,c['prefix_commands']),(after,c['suffix_commands'])):add(ctx,raw)
    c=result['winner'];after=prefix_context(c)
    for _ in range(2):add(initial,c['prefix_commands']);add(after,c['suffix_commands'])
    return want


def validate_execution(slot,row,load):
    result=row['result'];winner=validate_compilation(slot['packet'],slot['q'],result)
    validate_compiler_events(result,row['events'])
    fs=row['forwards'];want=expected_forward_inputs(slot['packet'],result)
    require(len(fs)==len(want)==258 and len(row['module_ledger'])==774,'all258 native forwards including43 AD forwards')
    require([f['phase'] for f in fs]==['nominal','jacobian_zero_forward']+
        [v for _ in range(42) for v in ('step','nominal','jacobian_zero_forward','step','step','step')]+['step']*4,'actual forward/AD phase ordering')
    for index,(f,(ctx,raw)) in enumerate(zip(fs,want,strict=True)):
        require(f['index']==index and f['completed'] is True and f['input']==ctx and np.array_equal(f['raw'],raw) and
            np.array_equal(load(f['raw_array']),np.asarray(raw,np.float32)),'actual raw buffer/input schedule, no second tanh')
        validate_advance(ctx,raw,f['output'])
        es=row['module_ledger'][3*index:3*index+3]
        require([e['resource'] for e in es]==list(Modules.ORDER) and all(e['index']==3*index+j and e['stage']==index and e['completed'] for j,e in enumerate(es)),'actual complete native ledger')
        ins=[[load(a) for a in e['inputs']] for e in es];outs=[load(e['output']) for e in es]
        tokens=np.concatenate([ctx['action_history'],kernel.normalized_history(raw)])
        require(len(ins[0])==1 and ins[0][0].shape==(1,3,10) and np.array_equal(ins[0][0][0],tokens) and
            len(ins[1])==2 and ins[1][0].shape==(1,3,192) and np.array_equal(ins[1][0][0],ctx['latent_history']) and
            np.array_equal(ins[1][1],outs[0]) and outs[1].shape==(1,3,192) and len(ins[2])==1 and
            np.array_equal(ins[2][0],outs[1].reshape(3,192)) and outs[2].shape==(3,192) and
            np.array_equal(outs[2][-1],f['output']['latent_history'][-1]),'full actual raw/AE/predictor/projector arrays')
    # Validate saved outputs against nominal/J-prefix/suffix result, not only module consistency.
    require(fs[0]['output']['latent_history'][-1]==result['root_nominal'],'root nominal actual')
    for i,c in enumerate(result['candidates']):
        entries=fs[2+6*i:8+6*i]
        require(entries[0]['output']==entries[4]['output'] and entries[3]['output']==entries[5]['output']==c['context'] and
            entries[0]['output']['latent_history'][-1]==c['predicted_prefix'] and
            entries[1]['output']['latent_history'][-1]==c['nominal_terminal'],'all candidates actual prefix/conditional nominal/nonlinear/replay')
    require(fs[-3]['output']==fs[-1]['output']==winner['context'],'winner whole/segment actual outputs')
    ads=row['AD_API'];require(len(ads)==43 and all(e['completed'] and e['index']==i and e['VJP_directions']==192 for i,e in enumerate(ads)),'all43 AD return receipts')
    for e,j in zip(ads,[result['root_jacobian']]+[c['jacobian'] for c in result['candidates']],strict=True):
        require(np.array_equal(load(e['jacobian']),np.asarray(j,np.float32)),'saved original full J')
    capture.prior.blind.exact_counts(row['resources'],dict(predict=258,action_encoder=258,pred_proj=258))
    return winner


def model_plan(slot,result):
    w=result['winner']
    p=dict(schema=NAME+'.model_plan',root_slot=slot['root_slot'],source_slot=slot['root_slot'],condition=slot['condition'],
        arm=ARM,task_id=slot['task_id'],blind_sha256=slot['blind_sha256'],task_sha256=slot['task_sha256'],
        source_report_sha256=overlay.SOURCE_SHA,commands=w['commands'],prefix_commands=w['prefix_commands'],suffix_commands=w['suffix_commands'],
        selected_index=w['index'],objective=w['objective'],model_Q=w['nonlinear_terminal_cost'],horizon=2,raw_steps=10,
        selection_uses_physics=False,P1_admission=False,P3_admission=False,equal_total_compute=False)
    p['selection_digest']=core.digest(p);return p


def math_resources(rows):
    ads=[e for r in rows for e in r.get('AD_API',[])]
    helper_attempts=[e for r in rows for e in r.get('events',[]) if e['kind'] in ('root_bank','suffix_ridge') and
        not e['completed'] and 'error' not in e]
    helpers=[e for r in rows for e in r.get('events',[]) if e['kind'] in ('root_bank','suffix_ridge') and e['completed']]
    return dict(functional_jacobian_attempted=len(ads),functional_jacobian_completed=sum(e['completed'] for e in ads),
        VJP_directions_requested=192*len(ads),VJP_directions_completed=192*sum(e['completed'] for e in ads),
        engine_internal_backward_calls=None,helper_attempted=len(helper_attempts),helper_completed=len(helpers),
        SVD_completed_from_successful_frozen_helpers=sum(e['SVD_complete_if_returned'] for e in helpers),
        SVD_attempts_in_failed_helpers_unknown=len(helper_attempts)!=len(helpers))


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN explicit CUDA authorization')
    start=time.monotonic();p=verify(prepared,expected);store=Store(output);errors=[];model=None;before=after=None;source_ok=False;host=None
    rows=[dict(index=i,root_slot=s['root_slot'],condition=s['condition'],status='not_started') for i,s in enumerate(p['slots'])]
    def check():require(time.monotonic()-start<=WALL,'whole job1800s predispatch; no retry')
    def err(phase,e):errors.append(overlay.moving.low.error(phase,e))
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        overlay.moving.low.frozen(model);before=overlay.moving.low.tensor_hash(model);require(before==p['expected_model_hash'],'original frozen weights')
        with sdpa_kernel(SDPBackend.MATH),torch.autocast(device_type='cuda',enabled=False):
            for slot,row in zip(p['slots'],rows,strict=True):
                row.update(status='attempted',events=[])
                store.event(dict(kind='root_started',slot=deepcopy(slot)))
                def sink(e):row['events'].append(deepcopy(e));store.event(dict(kind='compiler_event',root=row['index'],entry=e))
                meter=SimpleNamespace(stage=None,expected_input=None,last_native=None)
                modules=Modules(model,store,f'root_{row["index"]}',meter,lambda:torch.cuda.synchronize(device))
                backend=Backend(model,device,torch,store,row,meter,check)
                try:
                    with modules:row['result']=compile_arm(slot['packet'],slot['q'],backend,sink)
                finally:row['module_ledger']=deepcopy(modules.entries);row['resources']=modules.counts()
                validate_compilation(slot['packet'],slot['q'],row['result'])
                capture.prior.blind.exact_counts(row['resources'],dict(predict=258,action_encoder=258,pred_proj=258))
                row.update(status='plan_sealed',plan=model_plan(slot,row['result']))
                store.put(f'root_{row["index"]}.json.gz',row);store.event(dict(kind='root_sealed',index=row['index'],selection_digest=row['plan']['selection_digest']))
    except BaseException as e:
        err('execution',e)
        try:store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as e:err('error_journal',e)
    finally:
        for row in rows:
            if row['status']=='attempted':row['status']='failed_partial'
        if model is not None:
            try:overlay.moving.low.frozen(model);after=overlay.moving.low.tensor_hash(model)
            except BaseException as e:err('final_model',e)
        try:verify(prepared,expected);check();source_ok=True
        except BaseException as e:err('final_source_or_wall',e)
        try:store.close()
        except BaseException as e:err('final_journal',e)
        hashes={}
        try:hashes={f.name:sha(f) for f in store.output.iterdir()}
        except BaseException as e:err('final_hashes',e)
        good=not errors and source_ok and before==after==p['expected_model_hash'] and all(r['status']=='plan_sealed' for r in rows)
        report=dict(schema=NAME+'.report',status='all4_allmode_plans_sealed_not_admitted' if good else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,rows=rows,contract=contract(),errors=errors,first_error=errors[0] if errors else None,
            model_hash_before=before,model_hash_after=after,model_unchanged=before is not None and before==after,
            source_verified=source_ok,commands_released=good,artifact_sha256=hashes,host=host,wall_seconds=time.monotonic()-start,
            P1_admission=False,P3_admission=False,new_physics=0,new_encoder=0,new_projector=0,
            math_resources=math_resources(rows),
            resources={k:dict(attempted=sum(r.get('resources',{}).get(k,{}).get('attempted',0) for r in rows),
                completed=sum(r.get('resources',{}).get(k,{}).get('completed',0) for r in rows)) for k in Modules.ORDER})
        try:store.put('report.json',report,final=True)
        except BaseException as e:err('final_report',e);good=False;print(canonical(dict(errors=errors)).decode(),flush=True)
    require(good,'partial evidence retained; no retry '+str(errors))
    return dict(status=report['status'],report_sha256=sha(store.output/'report.json'),resources=report['resources'])


def consume(output,expected):
    output=Path(output).absolute();core.hash_value(expected);require(sha(output/'report.json')==expected,'external actual own report SHA')
    r=overlay.read(output/'report.json',CAP)
    require(r['schema']==NAME+'.report' and r['status']=='all4_allmode_plans_sealed_not_admitted' and r['errors']==[] and
        r['first_error'] is None and r['commands_released'] is r['model_unchanged'] is r['source_verified'] is True,'no partial command release')
    p=verify(r['prepared'],r['prepare_sha256'])
    require(r['contract']==contract() and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'],'frozen exact model/contract')
    require(r['math_resources']==math_resources(r['rows'])==dict(functional_jacobian_attempted=172,functional_jacobian_completed=172,
        VJP_directions_requested=33024,VJP_directions_completed=33024,engine_internal_backward_calls=None,
        helper_attempted=172,helper_completed=172,SVD_completed_from_successful_frozen_helpers=176,
        SVD_attempts_in_failed_helpers_unknown=False),'complete derivative/helper accounting, no invented internal API count')
    require({f.name for f in output.iterdir()}==set(r['artifact_sha256'])|{'report.json'} and
        sum(f.stat().st_size for f in output.iterdir())<=CAP,'all actual artifacts and cap')
    for name,h in r['artifact_sha256'].items():require(Path(name).name==name and not (output/name).is_symlink() and sha(output/name)==h,'artifact seal')
    require(overlay.read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'execution binding')
    plain,eof=capture.prior.blind.inflate((output/'events.jsonl.gz').read_bytes(),512*1024**2)
    require(eof and plain.endswith(b'\n'),'complete durable journal')
    events=[capture.prior.blind.decode(line) for line in plain.splitlines()]
    require([e['slot'] for e in events if e['kind']=='root_started']==p['slots'] and len(r['rows'])==4,'fixed full4 denominator')
    def load(a):
        require(a['file'] in r['artifact_sha256'] and a['sha256']==r['artifact_sha256'][a['file']],'array closure')
        raw=(output/a['file']).read_bytes();stream=BytesIO(raw);v=np.load(stream,allow_pickle=False)
        require(stream.tell()==len(raw) and str(v.dtype)==a['dtype'] and list(v.shape)==a['shape'] and v.dtype==np.float32 and np.isfinite(v).all(),'original finite float32 array')
        return v
    plans=[]
    for slot,row in zip(p['slots'],r['rows'],strict=True):
        index=slot['index'];require(row['index']==index and (row['root_slot'],row['condition'])==ORDER[index] and row['status']=='plan_sealed','stable own root identity')
        validate_execution(slot,row,load)
        require(overlay.read(output/f'root_{index}.json.gz',CAP)==row and row['plan']==model_plan(slot,row['result']),'result/dispatch seal')
        scoped=[e for e in events if e.get('root')==index]
        require([e['entry'] for e in scoped if e['kind']=='compiler_event']==row['events'] and
            [e['entry'] for e in scoped if e['kind']=='forward_return']==row['forwards'] and
            [e['entry'] for e in scoped if e['kind']=='forward_attempt']==[
                {k:v for k,v in dict(f,completed=False).items() if k not in ('raw_array','output')} for f in row['forwards']],
            'all actual predict attempts/returns and compiler journal')
        require([e['entry'] for e in scoped if e['kind']=='AD_return']==row['AD_API'] and
            [e['entry'] for e in scoped if e['kind']=='AD_attempt']==[
                {k:v for k,v in dict(e,completed=False).items() if k!='jacobian'} for e in row['AD_API']],'all AD attempts and returns')
        left=next(i for i,e in enumerate(events) if e['kind']=='root_started' and e['slot']['index']==index)
        right=next(i for i,e in enumerate(events) if e['kind']=='root_sealed' and e['index']==index)
        native=events[left:right+1];ledger=row['module_ledger']
        require([e['entry'] for e in native if e['kind']=='module_return']==ledger and
            [e['entry'] for e in native if e['kind']=='module_attempt']==[
                {k:v for k,v in dict(e,completed=False,inputs=[]).items() if k!='output'} for e in ledger],'all actual native attempts/returns')
        plans.append(dict(row['plan'],whole_run_report_sha256=expected))
    require(r['resources']=={k:dict(attempted=1032,completed=1032) for k in Modules.ORDER} and sha(output/'report.json')==expected,'1032 actual calls and final seal')
    return dict(schema=NAME+'.consume',status='all4_allmode_commands_verified_not_admitted',errors=[],plans=plans,
        P1_admission=False,P3_admission=False,equal_total_compute=False,new_neural_calls=0,new_physics_calls=0)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('output','prepared','prepare-sha','report-sha','authorization'):parser.add_argument('--'+k)
    a=parser.parse_args(argv)
    if a.command=='prepare':out=prepare(a.output)
    elif a.command=='verify':out=dict(status='verified_no_execution',slots=len(verify(a.prepared,a.prepare_sha)['slots']))
    elif a.command=='run':out=run(a.prepared,a.prepare_sha,a.output,a.authorization)
    else:out=consume(a.output,a.report_sha)
    print(canonical(out).decode(),end='');return 0


if __name__=='__main__':raise SystemExit(main())

