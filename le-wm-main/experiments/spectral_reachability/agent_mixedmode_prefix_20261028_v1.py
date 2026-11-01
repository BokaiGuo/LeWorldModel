"""Frozen222 mixed-prefix calibration. Only MAIN may dispatch CUDA; never physics.

Reuses sealed numerical kernels, transition, hook lifecycle and saved schedule
builder. Own small orchestration is explicit; no module/global replacement.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from io import BytesIO
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
import zlib
import numpy as np
from experiments.spectral_reachability import agent_allmode_prefix_20261028_v2 as base

NAME='agent_mixedmode_prefix_20261028_v1'
ARM='mixedmode_full10_conditional_D2P'
AUTH='MAIN_MIXEDMODE222_PREFIX_CUDA_ONLY'
N=222
FORWARDS=6*N+6
CAP,FINAL,FLOOR,PREP_CAP,WALL=2*1024**3,1024**3,200*1024**2,2*1024**2,7200
JOURNAL_CAP=2*1024**3  # Independent UNCOMPRESSED cumulative event bytes.
ROOT,ORDER=base.ROOT,base.ORDER
PREREG=ROOT.parent/'docs/D2P_P1_NEXT_CANDIDATE_POINCARE_20261028_v1.md'
PREREG_SHA='9a48dc236024971c2aa65368dc504ea716b9aa5bd6672b8385b0c8a623c6c68f'
require,sha,canonical,core=base.require,base.sha,base.canonical,base.core
compiler,kernel,capture,overlay=base.compiler,base.kernel,base.capture,base.overlay
digest=base.digest


def contract():
    return dict(schema=NAME+'.contract',roots=2,root_conditions=4,order=[list(k) for k in ORDER],arm=ARM,
        proposals=N,old_proposals=42,mixed_proposals=180,beta=1.,damping=compiler.DAMPING,
        packet_logit_cap=compiler.CAP,radii=list(compiler.RADII),Q='unchanged original identity192',
        formula='R*(si*vi+sj*vj)/sqrt(2); R=original outer radius; i<j; si,sj=-1,+1',
        ordering='unchanged42; i0..8,j(i+1)..9,si(-1,+1),sj(-1,+1)',
        selection='stable(objective,index) argmin(stage_cost+own_reach_value), beta1; no outcome',
        per_root_condition=dict(predict=FORWARDS,action_encoder=FORWARDS,pred_proj=FORWARDS,
            predictor_samples=FORWARDS,AD_forward=N+1,jacobians=N+1,VJP_directions=(N+1)*192,
            SVD_successful_helpers=N+2,suffix_solves=N),
        total=dict(predict=4*FORWARDS,action_encoder=4*FORWARDS,pred_proj=4*FORWARDS,
            predictor_samples=4*FORWARDS,AD_forward=4*(N+1),jacobians=4*(N+1),
            VJP_directions=4*(N+1)*192,SVD_successful_helpers=4*(N+2),suffix_solves=4*N),
        raw_authority='one NumPy float32 tanh per packet; same literal buffer for model/replay/dispatch; no /100',
        new_encoder=0,new_projector=0,new_physics=0,training=False,Qfit=False,retries=0,padding=0,
        inherited='same four original encoded tasks; historical16 encodes/source/planning/failed allmode v1 paid separately',
        equal_total_compute=False,equal_FLOPs=False,P1_admission=False,P3_admission=False,
        independent_confirmation=False,CUDA_only=True,MATH=True,TF32=False,autocast=False,
        output_cap_bytes=CAP,final_reserve_bytes=FINAL,disk_floor_bytes=FLOOR,prepare_cap_bytes=PREP_CAP,
        journal_uncompressed_cap_bytes=JOURNAL_CAP,
        saved_validation_per_successful_row=dict(root_bank=1,suffix_solves=N,SVD_successful_helpers=N+2,neural_calls=0),
        saved_validation_scope='run validates4rows; consume independently validates4rows; parent build validation cost unprofiled, not free',
        wall_seconds=WALL,engine_backward_calls=None,internal_solve_iterations=None,
        failed_helper_internal_SVD='unknown',prereg_sha256=PREREG_SHA)


def mixed_bank(jacobian,nominal,goal,q):
    old=base.all_mode_bank(jacobian,nominal,goal,q)
    result=deepcopy(old);right=np.asarray(old['right_vectors'],np.float64)
    receipts=[dict(kind='inherited',old_index=i,params_before=deepcopy(u),params_after=deepcopy(u),
        norm_before=float(np.linalg.norm(u)),norm_after=float(np.linalg.norm(u))) for i,u in enumerate(old['params'])]
    # capture.project is sealed float64 L2 screen; batching16 changes no pointwise math.
    for i in range(9):
        for j in range(i+1,10):
            for si in (-1,1):
                for sj in (-1,1):
                    before=compiler.CAP*(si*right[i]+sj*right[j])/np.sqrt(2.)
                    after=capture.project(np.tile(before,(16,1)),compiler.CAP)[0]
                    result['params'].append(after.tolist())
                    result['kinds'].append(f'mixed_{i}_{j}_{si}_{sj}')
                    receipts.append(dict(kind='mixed',i=i,j=j,si=si,sj=sj,radius=compiler.CAP,
                        params_before=before.tolist(),params_after=after.tolist(),
                        norm_before=float(np.linalg.norm(before)),norm_after=float(np.linalg.norm(after))))
    require(len(result['params'])==N and result['params'][:42]==old['params'],'all original42 plus fixed180')
    result.update(parameter_witnesses=receipts,right_vectors_sha256=digest(old['right_vectors']))
    return result


def diagnostics(packet,q,result):
    cs=result['candidates'];g=min(cs,key=lambda c:(c['stage_cost'],c['index']))
    old=min(cs[:42],key=lambda c:(c['objective'],c['index']))
    frontier=[c['index'] for c in cs if not any(
        d['stage_cost']<=c['stage_cost'] and d['own_reach_value']<=c['own_reach_value'] and
        (d['stage_cost']<c['stage_cost'] or d['own_reach_value']<c['own_reach_value']) for d in cs)]
    return dict(initial_Q=base.cost(packet['root_context']['latent_history'][-1],packet['goal'],q),
        zero_index=0,greedy_index=g['index'],old42_selected_index=old['index'],selected_index=result['winner']['index'],
        Pareto_indices=frontier,new_Pareto_indices=[i for i in frontier if i>=42],
        model_only=True,actual_success=None,P1_admission=False)


def compile_arm(packet,q,backend,journal):
    """Same allmode schedule with explicit222 bank; backend may be RAM stand-in."""
    q=base.validate_q(q);goal=compiler.finite(packet['goal'],(192,));initial=deepcopy(packet['root_context'])
    base.validate_context(initial);qsha=digest(q.tolist());operations=[]
    def op(kind,fn):
        e=dict(index=len(operations),kind=kind,completed=False,q_sha256=qsha)
        if kind=='jacobian':e.update(shape=[192,10],mathematical_vjp_directions=192)
        else:e.update(SVD_complete_if_returned=2 if kind=='root_bank' else 1,
            SVD_attempts_on_failure='unknown_inside_frozen_helper')
        operations.append(e);journal(deepcopy(e))
        try:out=fn();e['completed']=True;return out
        except BaseException as exc:e['error']=type(exc).__name__+': '+str(exc);raise
        finally:base.finish_event(journal,e)
    def fresh(c,phase):
        n=compiler.finite(backend.nominal(deepcopy(c)),(192,))
        journal(dict(kind='nominal',phase=phase,q_sha256=qsha,completed=True,nominal=n.tolist()))
        j=compiler.finite(backend.jacobian(deepcopy(c),op),(192,10))
        journal(dict(kind='derivative',phase=phase,q_sha256=qsha,completed=True,jacobian=j.tolist()))
        return n,j
    def replay(c,raw,phase):
        e=dict(kind='replay',phase=phase,commands=np.asarray(raw).tolist(),q_sha256=qsha,completed=False)
        journal(deepcopy(e))
        try:out=backend.replay(c,raw);e.update(completed=True,context=out);return out
        except BaseException as exc:e['error']=type(exc).__name__+': '+str(exc);raise
        finally:base.finish_event(journal,e)
    n,j=fresh(initial,'root');bank=op('root_bank',lambda:mixed_bank(j,n,goal,q))
    journal(dict(kind='bank',q_sha256=qsha,bank=bank,root_nominal=n.tolist(),root_jacobian=j.tolist()))
    cs=[]
    for index,params in enumerate(bank['params']):
        prefix=np.tanh(np.asarray(params,np.float32)).reshape(5,2)
        e=dict(kind='candidate',index=index,q_sha256=qsha,completed=False,prefix_commands=prefix.tolist())
        journal(deepcopy(e))
        try:
            after=backend.step(deepcopy(initial),prefix);base.validate_advance(initial,prefix,after)
            journal(dict(kind='prefix',index=index,q_sha256=qsha,completed=True,context=after))
            sn,sj=fresh(after,'suffix_'+str(index))
            e.update(predicted_prefix=after['latent_history'][-1],nominal_terminal=sn.tolist(),jacobian=sj.tolist())
            journal(deepcopy(e))
            solve=op('suffix_ridge',lambda:compiler.bounded_ridge(q@sj,q@(goal-sn)))
            suffix=np.tanh(np.asarray(solve['params'],np.float32)).reshape(5,2);raw=np.concatenate((prefix,suffix))
            e.update(suffix_solve=solve,suffix_params=solve['params'],suffix_commands=suffix.tolist(),commands=raw.tolist())
            journal(deepcopy(e))
            predicted=backend.step(deepcopy(after),suffix);base.validate_advance(initial,raw,predicted)
            e['predicted_terminal']=predicted['latent_history'][-1];journal(deepcopy(e))
            verified=replay(deepcopy(initial),raw,'candidate_'+str(index))
            require(base.io.canonical(predicted)==base.io.canonical(verified),'fresh H2 replay exact, no fallback')
            stage=base.cost(after['latent_history'][-1],goal,q)
            e.update(completed=True,candidate_id=f'prefix_{index}',proposal_kind=bank['kinds'][index],prefix_params=params,
                stage_cost=stage,own_reach_value=solve['own_reach_value'],objective=stage+solve['own_reach_value'],
                nonlinear_terminal_cost=base.cost(predicted['latent_history'][-1],goal,q),context=verified,
                independent_model_replay_exact=True)
            cs.append(deepcopy(e))
        except BaseException as exc:e['error']=type(exc).__name__+': '+str(exc);raise
        finally:base.finish_event(journal,e)
    winner=deepcopy(min(cs,key=lambda c:(c['objective'],c['index'])))
    journal(dict(kind='selection_frozen_before_winner_replays',index=winner['index'],commands=winner['commands'],q_sha256=qsha))
    whole=replay(deepcopy(initial),winner['commands'],'winner_whole');segmented=deepcopy(initial)
    for k,raw in enumerate(np.asarray(winner['commands']).reshape(2,5,2)):
        segmented=replay(segmented,raw,'winner_segment_'+str(k))
    require(whole==segmented==winner['context'],'whole+segmented winner exact')
    result=dict(status='completed',q_sha256=qsha,bank=bank,root_jacobian=j.tolist(),root_nominal=n.tolist(),
        candidates=cs,winner=winner,whole_replay_context=whole,segmented_replay_context=segmented,
        whole_and_segmented_exact=True,operations=operations)
    result['diagnostics']=diagnostics(packet,q,result)
    return result


def validate_compilation(packet,q,r):
    q=base.validate_q(q);goal=np.asarray(packet['goal']);initial=packet['root_context'];base.validate_context(initial)
    require(r['status']=='completed' and r['q_sha256']==digest(q.tolist()),'same Q and completed compilation')
    bank=mixed_bank(r['root_jacobian'],r['root_nominal'],goal,q)
    require(r['bank']==bank and len(r['candidates'])==N,'fixed222 bank and denominator')
    for i,c in enumerate(r['candidates']):
        solve=compiler.bounded_ridge(q@compiler.finite(c['jacobian'],(192,10)),q@(goal-compiler.finite(c['nominal_terminal'],(192,))))
        prefix=np.tanh(np.asarray(bank['params'][i],np.float32)).reshape(5,2)
        suffix=np.tanh(np.asarray(solve['params'],np.float32)).reshape(5,2);raw=np.concatenate((prefix,suffix)).tolist()
        require(c['index']==i and c['completed'] is True and c['proposal_kind']==bank['kinds'][i] and
            c['prefix_params']==bank['params'][i] and c['suffix_solve']==solve and c['suffix_params']==solve['params'] and
            c['prefix_commands']==prefix.tolist() and c['suffix_commands']==suffix.tolist() and c['commands']==raw,
            'all params/own ridge/literal raw witnesses')
        base.validate_advance(initial,raw,c['context'])
        require(c['context']['latent_history']==[initial['latent_history'][-1],c['predicted_prefix'],c['predicted_terminal']],
            'actual full H2 chronological history')
        stage=base.cost(c['predicted_prefix'],goal,q)
        require(c['stage_cost']==stage and c['own_reach_value']==solve['own_reach_value'] and c['objective']==stage+solve['own_reach_value'] and
            c['nonlinear_terminal_cost']==base.cost(c['predicted_terminal'],goal,q) and c['independent_model_replay_exact'] is True,
            'fixed beta1 score, nonlinear terminal not selector')
    w=min(r['candidates'],key=lambda c:(c['objective'],c['index']))
    require(r['winner']==w and r['whole_and_segmented_exact'] is True and
        r['whole_replay_context']==r['segmented_replay_context']==w['context'],'stable argmin/fresh winner replay')
    kinds=['jacobian','root_bank']+[k for _ in range(N) for k in ('jacobian','suffix_ridge')]
    require([e['kind'] for e in r['operations']]==kinds and all(e['index']==i and e['completed'] is True and
        e['q_sha256']==r['q_sha256'] for i,e in enumerate(r['operations'])),'all helper attempts/returns')
    for e in r['operations']:
        if e['kind']=='jacobian':require(e['shape']==[192,10] and e['mathematical_vjp_directions']==192,'J dimensions')
        else:require(e['SVD_complete_if_returned']==(2 if e['kind']=='root_bank' else 1) and
            e['SVD_attempts_on_failure']=='unknown_inside_frozen_helper','SVD bookkeeping')
    require(r['diagnostics']==diagnostics(packet,q,r),'all fixed-support descriptive diagnostics')
    return deepcopy(w)


class Modules(base.Modules):
    # Reuse sealed __enter__/post/__exit__/array/counts; only stage cap differs.
    def pre(self,kind):
        def enter(module,args):
            require(not self.failed and not self.pending and len(self.entries)<3*FORWARDS and
                kind==self.ORDER[len(self.entries)%3] and self.meter.stage==len(self.entries)//3,'1338 native forwards, no retries')
            e=dict(index=len(self.entries),resource=kind,stage=self.meter.stage,completed=False,inputs=[])
            self.entries.append(e);self.pending[module]=e;self.store.event(dict(kind='module_attempt',entry=deepcopy(e)))
            for k,x in enumerate(args):
                a=self.array(x)
                if kind=='action_encoder':require(a.shape==(1,3,10) and np.array_equal(a[0],self.meter.expected_input['action_history']),'actual incoming tokens')
                elif kind=='predict':
                    if k==0:require(a.shape==(1,3,192) and np.array_equal(a[0],self.meter.expected_input['latent_history']),'actual context')
                    else:require(np.array_equal(a,self.action_embedding),'AE predictor bridge')
                else:require(a.shape==(3,192) and np.array_equal(a,self.predictor_output.reshape(3,192)),'projector bridge')
                e['inputs'].append(capture.prior.save_array(self.store,f'{self.stem}.module_{e["index"]:04}.input_{k}.npy',a))
            self.store.event(dict(kind='module_inputs_saved',entry=deepcopy(e)))
        return enter


class Backend(base.Backend):
    # nominal/J/step/replay and actual frozen transition inherited, not copied.
    def forward(self,c,raw,phase):
        t=self.torch;self.check();n=len(self.row['forwards']);require(n<FORWARDS,'1338 predict cap')
        a=raw.detach().cpu().numpy().copy()
        require(a.dtype==np.float32 and a.shape==(5,2) and np.isfinite(a).all() and np.max(abs(a))<=1,'literal float32 raw')
        e=dict(index=n,phase=phase,input=deepcopy(c),raw=a.tolist(),completed=False)
        self.row['forwards'].append(e);self.meter.stage=n;self.meter.last_native=None
        self.meter.expected_input=capture.prior.planner.incoming(
            dict(action_mean=kernel.ACTION_MEAN.tolist(),action_std=kernel.ACTION_STD.tolist()),c,a)
        self.store.event(dict(kind='forward_attempt',root=self.row['index'],entry=deepcopy(e)))
        e['raw_array']=capture.prior.save_array(self.store,f'root_{self.row["index"]}.raw_{n:04}.npy',a)
        emb=t.tensor(c['latent_history'],dtype=t.float32,device=self.device)[None]
        hist=t.tensor(c['action_history'],dtype=t.float32,device=self.device)[None]
        after,between=self.transition(emb,hist,raw);self.check()
        out=dict(c,latent_history=after[0].detach().cpu().tolist(),action_history=between[0].detach().cpu().tolist())
        require(np.array_equal(out['latent_history'][-1],self.meter.last_native),'successor native output bridge')
        e.update(completed=True,output=out);self.store.event(dict(kind='forward_return',root=self.row['index'],entry=deepcopy(e)))
        return after,out


def validate_execution(slot,row,load):
    r=row['result'];winner=validate_compilation(slot['packet'],slot['q'],r)
    fs=row['forwards'];want=base.expected_forward_inputs(slot['packet'],r)
    phases=['nominal','jacobian_zero_forward']+[v for _ in range(N) for v in ('step','nominal','jacobian_zero_forward','step','step','step')]+['step']*4
    require(len(fs)==len(want)==FORWARDS and len(row['module_ledger'])==3*FORWARDS and [f['phase'] for f in fs]==phases,'full1338 schedule')
    for i,(f,(ctx,raw)) in enumerate(zip(fs,want,strict=True)):
        require(f['index']==i and f['completed'] is True and f['input']==ctx and np.array_equal(f['raw'],raw) and
            np.array_equal(load(f['raw_array']),np.asarray(raw,np.float32)),'raw array/actual incoming history')
        base.validate_advance(ctx,raw,f['output']);es=row['module_ledger'][3*i:3*i+3]
        require([e['resource'] for e in es]==list(Modules.ORDER) and all(e['index']==3*i+k and e['stage']==i and e['completed'] is True for k,e in enumerate(es)),'native order/attempt/return')
        ins=[[load(a) for a in e['inputs']] for e in es];outs=[load(e['output']) for e in es]
        tokens=np.concatenate([ctx['action_history'],kernel.normalized_history(raw)])
        require(len(ins[0])==1 and ins[0][0].shape==(1,3,10) and np.array_equal(ins[0][0][0],tokens) and
            len(ins[1])==2 and ins[1][0].shape==(1,3,192) and np.array_equal(ins[1][0][0],ctx['latent_history']) and
            np.array_equal(ins[1][1],outs[0]) and outs[1].shape==(1,3,192) and len(ins[2])==1 and
            np.array_equal(ins[2][0],outs[1].reshape(3,192)) and outs[2].shape==(3,192) and
            np.array_equal(outs[2][-1],f['output']['latent_history'][-1]),'full native array bridges')
    require(fs[0]['output']['latent_history'][-1]==r['root_nominal'],'actual root nominal')
    for i,c in enumerate(r['candidates']):
        e=fs[2+6*i:8+6*i]
        require(e[0]['output']==e[4]['output'] and e[3]['output']==e[5]['output']==c['context'] and
            e[0]['output']['latent_history'][-1]==c['predicted_prefix'] and e[1]['output']['latent_history'][-1]==c['nominal_terminal'],'all222 actual candidate outputs')
    require(fs[-3]['output']==fs[-1]['output']==winner['context'],'actual winner replay')
    ads=row['AD_API'];require(len(ads)==N+1 and all(e['completed'] is True and e['index']==i and
        e['shape']==[192,10] and e['VJP_directions']==192 and e['functional_jacobian_calls']==1 and e['vectorize'] is True for i,e in enumerate(ads)),'all223 actual J')
    for e,j in zip(ads,[r['root_jacobian']]+[c['jacobian'] for c in r['candidates']],strict=True):
        require(np.array_equal(load(e['jacobian']),np.asarray(j,np.float32)),'original J buffer')
    capture.prior.blind.exact_counts(row['resources'],{k:FORWARDS for k in Modules.ORDER})
    require(row['saved_validation']==dict(attempted=True,completed=True,SVD_successful_helpers=N+2,
        internal_SVD_on_failure='unknown',neural_calls=0),'run saved-arithmetic validation receipt, separate CPU fee')
    return winner


def validate_events(row,events):
    r=row['result'];ce=[e['entry'] for e in events if e['kind']=='compiler_event']
    require(ce==row['events'],'all compiler attempts/returns durable')
    returned=[e for e in ce if e['kind'] in ('jacobian','root_bank','suffix_ridge') and e['completed']]
    attempted=[e for e in ce if e['kind'] in ('jacobian','root_bank','suffix_ridge') and not e['completed']]
    require(returned==r['operations'] and attempted==[dict(e,completed=False) for e in r['operations']],
        'every helper/J attempt and returned operation, not only total count')
    require([e for e in ce if e['kind']=='candidate' and e['completed']]==r['candidates'],'all222 durable candidates')
    for kind,field,key in [('nominal','nominal','nominal_terminal'),('derivative','jacobian','jacobian')]:
        es=[e for e in ce if e['kind']==kind]
        require([e['phase'] for e in es]==['root']+['suffix_'+str(i) for i in range(N)] and
            [e[field] for e in es]==[r['root_nominal'] if kind=='nominal' else r['root_jacobian']]+[c[key] for c in r['candidates']],'all conditional nominal/J journal arrays')
    require([e['bank'] for e in ce if e['kind']=='bank']==[r['bank']],'complete bank journal')
    prefixes=[e for e in ce if e['kind']=='prefix']
    require([e['index'] for e in prefixes]==list(range(N)) and all(e['completed'] is True for e in prefixes) and
        [e['context'] for e in prefixes]==[row['forwards'][2+6*i]['output'] for i in range(N)],
        'all222 prefix contexts durable before suffix derivative')
    rr=[e for e in ce if e['kind']=='replay' and e['completed']]
    require([e['phase'] for e in rr]==['candidate_'+str(i) for i in range(N)]+['winner_whole','winner_segment_0','winner_segment_1'] and
        [e['context'] for e in rr[:N]]==[c['context'] for c in r['candidates']] and
        rr[-3]['context']==r['whole_replay_context'] and rr[-1]['context']==r['segmented_replay_context'],'independent replay arrays')
    selected=[i for i,e in enumerate(ce) if e['kind']=='selection_frozen_before_winner_replays']
    require(len(selected)==1,'one frozen selection');i=selected[0]
    require(ce[i]['index']==r['winner']['index'] and ce[i]['commands']==r['winner']['commands'] and
        sum(e['kind']=='candidate' and e['completed'] for e in ce[:i])==N and
        ce[i+1]['kind']=='replay' and ce[i+1]['phase']=='winner_whole','selection before winner replays')
    for prefix,items,removed in [('forward',row['forwards'],('raw_array','output')),('AD',row['AD_API'],('jacobian',)),('module',row['module_ledger'],('output',))]:
        require([e['entry'] for e in events if e['kind']==prefix+'_return']==items,'all '+prefix+' returns')
        attempts=[]
        for item in items:
            e=dict(item,completed=False)
            if prefix=='module':e['inputs']=[]
            attempts.append({k:v for k,v in e.items() if k not in removed})
        require([e['entry'] for e in events if e['kind']==prefix+'_attempt']==attempts,'all '+prefix+' attempts')
    require([e['entry'] for e in events if e['kind']=='module_inputs_saved']==[
        {k:v for k,v in dict(e,completed=False).items() if k!='output'} for e in row['module_ledger']],
        'every original module input saved before return')
    require([e['entry'] for e in events if e['kind']=='saved_validation_attempt']==[
        dict(row['saved_validation'],completed=False,SVD_successful_helpers=0)] and
        [e['entry'] for e in events if e['kind']=='saved_validation_return']==[row['saved_validation']],
        'saved arithmetic validation attempt and return')


def model_plan(slot,r):
    p=base.model_plan(slot,r);p.update(schema=NAME+'.model_plan',arm=ARM)
    p.pop('selection_digest');p['selection_digest']=core.digest(p);return p


def build():
    require(sha(PREREG)==PREREG_SHA,'frozen preregistration')
    p=base.build();p.update(schema=NAME+'.prepare',contract=contract())
    # Keep all inherited model/task/source paths, including sealed base tests/doc.
    for path in [PREREG,Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]:
        p['bindings'][str(path.resolve())]=sha(path)
    require(len(canonical(p))<=PREP_CAP,'canonical prepare2MiB')
    return p


def write_new(output,raw):
    path=Path(output)
    require(path.name.startswith(NAME+'_') and path.name.endswith('.json') and isinstance(raw,bytes) and len(raw)<=PREP_CAP,'own mixedmode JSON prefix/type/cap')
    return capture.write_new(output,raw,PREP_CAP)


def prepare(output):return write_new(output,canonical(build()))


def verify(path,expected):
    core.hash_value(expected);require(sha(path)==expected,'external own prepare SHA')
    p=overlay.read(path,PREP_CAP);require(p==build(),'full task/source/model/prereg closure');return p


class Store(base.Store):
    def __init__(self,output):
        self.output=Path(output).absolute()
        require(self.output.name.startswith(NAME+'_') and not self.output.exists() and
            shutil.disk_usage(self.output.parent).free>=FLOOR+CAP,'NEW own output2GiB plus floor200MiB')
        self.output.mkdir();self.comp=zlib.compressobj(6,zlib.DEFLATED,31);self.closed=self.broken=False
        self.journal_plain_bytes=0;self.used_bytes=0;self.file_sizes={};self.file_ids={};self.accounting_failed=False;self.drifted=False
        self._write('events.jsonl.gz',b'')
    def event(self,e):
        size=len(canonical(e))
        require(not self.closed and not self.broken,'no retry on failed/final journal')
        if self.journal_plain_bytes+size>JOURNAL_CAP:
            self.broken=True
            raise ValueError('uncompressed cumulative journal cap')
        # Count attempted bytes before durable write; failed writes are not
        # retried, and the count is an upper bound in a partial report.
        self.journal_plain_bytes+=size
        return super().event(e)
    def _write(self,name,raw,append=False,final=False):
        require(Path(name).name==name and name not in ('','.','..') and not (self.output/name).is_symlink(),'flat output')
        require(isinstance(raw,bytes),'literal artifact bytes')
        require(not self.accounting_failed and not self.drifted,'unknown byte accounting or observed inode/size drift')
        if final:self.audit_files()
        require(self.used_bytes+len(raw)<=CAP-(0 if final else FINAL) and
            shutil.disk_usage(self.output).free>=FLOOR+len(raw),'output cap/final reserve/floor')
        require((name in self.file_sizes)==append,'exclusive new file or owned append only')
        flags=os.O_WRONLY|os.O_NOFOLLOW|(os.O_APPEND if append else os.O_CREAT|os.O_EXCL)
        fd=os.open(self.output/name,flags,0o600)
        try:
            stat=os.fstat(fd);identity=(stat.st_dev,stat.st_ino)
            if append:
                self.drifted=identity!=self.file_ids[name] or stat.st_size!=self.file_sizes[name]
                require(not self.drifted,'owned append inode/size drift')
            else:
                require(stat.st_size==0,'new exclusive file must be empty')
                self.file_sizes[name]=0;self.file_ids[name]=identity
            require(os.write(fd,raw)==len(raw),'short write; actual partial bytes retained, no retry')
            os.fsync(fd)
        finally:
            active=sys.exception();cleanup=[]
            # Even write()/fsync() exceptions can follow a partial write. fstat
            # records actual file growth, not requested length or returned count.
            try:
                stat=os.fstat(fd)
                if name in self.file_sizes:
                    self.used_bytes+=stat.st_size-self.file_sizes[name];self.file_sizes[name]=stat.st_size
            except BaseException as e:self.accounting_failed=True;cleanup.append(e)
            try:os.close(fd)
            except BaseException as e:cleanup.append(e)
            if cleanup:
                if active is not None:active.add_note('I/O accounting/close failures: '+repr(cleanup))
                else:raise cleanup[0]
        fd=os.open(self.output,os.O_RDONLY)
        try:os.fsync(fd)
        finally:
            active=sys.exception()
            try:os.close(fd)
            except BaseException as e:
                if active is None:raise
                active.add_note('directory close failed: '+repr(e))
    def audit_files(self):
        """One terminal O(files) census; no per-write directory rescan."""
        require(not self.accounting_failed and not self.drifted,'unresolved I/O accounting/drift')
        files=list(self.output.iterdir())
        require({f.name for f in files}==set(self.file_sizes),'unknown/missing output file at terminal census')
        for f in files:
            require(f.is_file() and not f.is_symlink(),'flat real file census')
            stat=f.stat()
            require(stat.st_size==self.file_sizes[f.name] and (stat.st_dev,stat.st_ino)==self.file_ids[f.name],
                'terminal output inode/size drift')
        total=sum(self.file_sizes.values())
        require(total==self.used_bytes<=CAP,'terminal actual byte census/cap')
        return total


def census(p):
    return [dict(index=i,root_slot=s['root_slot'],condition=s['condition'],status='not_started',
        candidate_denominator=N) for i,s in enumerate(p['slots'])]


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN explicit CUDA authorization')
    start=time.monotonic();p=verify(prepared,expected);store=Store(output);rows=census(p)
    errors=[];model=None;before=after=host=None;source_ok=False
    def err(phase,e):
        errors.append(overlay.moving.low.error(phase,e))
        for secondary in getattr(e,'compiler_secondary_errors',[]):
            errors.append(overlay.moving.low.error(phase+'_compiler_secondary',secondary))
    def check():require(time.monotonic()-start<=WALL,'whole7200s predispatch bound, no retry')
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
        overlay.moving.low.frozen(model);before=overlay.moving.low.tensor_hash(model)
        require(before==p['expected_model_hash'],'frozen original model')
        with sdpa_kernel(SDPBackend.MATH),torch.autocast(device_type='cuda',enabled=False):
            for slot,row in zip(p['slots'],rows,strict=True):
                row.update(status='attempted',events=[]);store.event(dict(kind='root_started',slot=deepcopy(slot)))
                def sink(e):row['events'].append(deepcopy(e));store.event(dict(kind='compiler_event',root=row['index'],entry=e))
                meter=SimpleNamespace(stage=None,expected_input=None,last_native=None)
                modules=Modules(model,store,f'root_{row["index"]}',meter,lambda:torch.cuda.synchronize(device))
                backend=Backend(model,device,torch,store,row,meter,check)
                try:
                    with modules:row['result']=compile_arm(slot['packet'],slot['q'],backend,sink)
                finally:row['module_ledger']=deepcopy(modules.entries);row['resources']=modules.counts()
                v=dict(attempted=True,completed=False,SVD_successful_helpers=0,internal_SVD_on_failure='unknown',neural_calls=0)
                row['saved_validation']=v;store.event(dict(kind='saved_validation_attempt',root=row['index'],entry=deepcopy(v)))
                validate_compilation(slot['packet'],slot['q'],row['result'])
                v.update(completed=True,SVD_successful_helpers=N+2)
                store.event(dict(kind='saved_validation_return',root=row['index'],entry=deepcopy(v)))
                capture.prior.blind.exact_counts(row['resources'],{k:FORWARDS for k in Modules.ORDER})
                row.update(status='plan_sealed',plan=model_plan(slot,row['result']))
                store.put(f'root_{row["index"]}.json.gz',row)
                store.event(dict(kind='root_sealed',index=row['index'],selection_digest=row['plan']['selection_digest']))
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
        try:
            store.audit_files()
            hashes={f.name:sha(f) for f in store.output.iterdir()}
        except BaseException as e:err('final_hashes',e)
        good=not errors and source_ok and before==after==p['expected_model_hash'] and all(r['status']=='plan_sealed' for r in rows)
        report=dict(schema=NAME+'.report',status='all4_mixedmode_plans_sealed_not_admitted' if good else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,rows=rows,contract=contract(),errors=errors,
            first_error=errors[0] if errors else None,model_hash_before=before,model_hash_after=after,
            model_unchanged=before is not None and before==after,source_verified=source_ok,commands_released=good,
            artifact_sha256=hashes,host=host,wall_seconds=time.monotonic()-start,P1_admission=False,P3_admission=False,
            journal_plain_bytes=getattr(store,'journal_plain_bytes',None),
            artifact_bytes_before_report=getattr(store,'used_bytes',None),
            math_resources=base.math_resources(rows),resources={k:dict(
                attempted=sum(r.get('resources',{}).get(k,{}).get('attempted',0) for r in rows),
                completed=sum(r.get('resources',{}).get(k,{}).get('completed',0) for r in rows)) for k in Modules.ORDER})
        try:store.put('report.json',report,final=True)
        except BaseException as e:
            err('final_report',e);good=False
            try:store._write('failure.json',canonical(dict(schema=NAME+'.failure',commands_released=False,
                fixed_denominator=4,candidate_denominator=4*N,errors=errors,first_error=errors[0],
                census=[dict(index=r['index'],status=r['status'],candidate_denominator=N) for r in rows],
                model_hash_before=before,model_hash_after=after,source_verified=source_ok,P1_admission=False)),final=True)
            except BaseException as extra:err('failure_metadata',extra)
            print(canonical(dict(errors=errors)).decode(),flush=True)
    require(good,'partial evidence retained, no retry: '+str(errors))
    return dict(status=report['status'],report_sha256=sha(store.output/'report.json'),resources=report['resources'])


def consume(output,expected):
    output=Path(output).absolute();core.hash_value(expected);require(sha(output/'report.json')==expected,'external actual report SHA')
    r=overlay.read(output/'report.json',CAP)
    require(r['schema']==NAME+'.report' and r['status']=='all4_mixedmode_plans_sealed_not_admitted' and
        r['errors']==[] and r['first_error'] is None and r['commands_released'] is r['model_unchanged'] is r['source_verified'] is True,'no partial release')
    p=verify(r['prepared'],r['prepare_sha256'])
    require(r['contract']==contract() and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'],'exact contract/model')
    require({f.name for f in output.iterdir()}==set(r['artifact_sha256'])|{'report.json'} and
        sum(f.stat().st_size for f in output.iterdir())<=CAP and
        sum((output/n).stat().st_size for n in r['artifact_sha256'])==r['artifact_bytes_before_report'],
        'full file census/cap and actual successful bytes')
    for name,h in r['artifact_sha256'].items():require(Path(name).name==name and not (output/name).is_symlink() and sha(output/name)==h,'flat actual artifact hashes')
    require(overlay.read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'execution binding')
    plain,eof=capture.prior.blind.inflate((output/'events.jsonl.gz').read_bytes(),JOURNAL_CAP)
    require(eof and plain.endswith(b'\n') and len(plain)==r['journal_plain_bytes']<=JOURNAL_CAP,'complete journal and exact uncompressed byte census')
    events=[capture.prior.blind.decode(line) for line in plain.splitlines()]
    require([e['slot'] for e in events if e['kind']=='root_started']==p['slots'] and len(r['rows'])==4,'full four task denominator')
    def load(a):
        require(a['file'] in r['artifact_sha256'] and a['sha256']==r['artifact_sha256'][a['file']],'array file closure')
        raw=(output/a['file']).read_bytes();stream=BytesIO(raw);v=np.load(stream,allow_pickle=False)
        require(stream.tell()==len(raw) and str(v.dtype)==a['dtype'] and list(v.shape)==a['shape'] and v.dtype==np.float32 and np.isfinite(v).all(),'original finite float32 buffer')
        return v
    plans=[]
    for slot,row in zip(p['slots'],r['rows'],strict=True):
        i=slot['index'];require(row['index']==i and (row['root_slot'],row['condition'])==ORDER[i] and row['status']=='plan_sealed' and row['candidate_denominator']==N,'stable task identity')
        validate_execution(slot,row,load)
        require(overlay.read(output/f'root_{i}.json.gz',CAP)==row and row['plan']==model_plan(slot,row['result']),'durable result/dispatch seal')
        left=next(k for k,e in enumerate(events) if e['kind']=='root_started' and e['slot']['index']==i)
        right=next(k for k,e in enumerate(events) if e['kind']=='root_sealed' and e['index']==i)
        require(events[right]['selection_digest']==row['plan']['selection_digest'],'root seal')
        scoped=[e for e in events[left:right+1] if e.get('root',i)==i]
        validate_events(row,scoped)
        plans.append(dict(row['plan'],whole_run_report_sha256=expected))
    math=base.math_resources(r['rows']);n=4*(N+1)
    require(r['math_resources']==math==dict(functional_jacobian_attempted=n,functional_jacobian_completed=n,
        VJP_directions_requested=192*n,VJP_directions_completed=192*n,engine_internal_backward_calls=None,
        helper_attempted=n,helper_completed=n,SVD_completed_from_successful_frozen_helpers=4*(N+2),
        SVD_attempts_in_failed_helpers_unknown=False),'complete mathematical ledger')
    require(r['resources']=={k:dict(attempted=4*FORWARDS,completed=4*FORWARDS) for k in Modules.ORDER} and sha(output/'report.json')==expected,'all5352 native API returns and final seal')
    return dict(schema=NAME+'.consume',status='all4_mixedmode_commands_verified_not_admitted',errors=[],plans=plans,
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
