"""Safeguarded known-context suffix intervention. MAIN HOST CUDA/physics only.

Fixed model-only search; exact old logits/raw retained, never repaired. All8
denominators survive failures; native outcomes are NOT compiler inputs.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import agent_p2_contextual_midfeedback_recompile_20261028_v1 as old
from experiments.spectral_reachability import main_suffix_incumbent_guard_20261028_v1 as guard

NAME='agent_p2_contextual_midfeedback_recompile_20261028_v2'
ROOT=old.ROOT
bridge,physical,kernel,F,suffix=old.bridge,old.physical,old.kernel,old.F,old.suffix
require,sha,raw=old.require,old.sha,old.raw
AUTH='MAIN_SAFEGUARDED_MIDFEEDBACK_V2_HOST_ONLY'
OLD_SHA='e195749c68747f3d1d820cb62e53d0b109865c4d3c87ead9f040610c953d94ac'
GUARD_SHA='1a33cd589efe44acf71a316ceef5670f3ea0b551f4a154c8b589414ca9b2fbc0'
ALPHAS=(1.,.5,.25,.125)
STAGES=('incumbent','jacobian','candidate_1','candidate_2','candidate_3','candidate_4','selected_replay')
CAP=float(.75*np.sqrt(10))


def budget():return dict(predict=7,action_encoder=7,pred_proj=7,jacobian=1,svd=0)


def contract():
    return dict(plans=8,known_roots=2,scientific_slots=4,stages=list(STAGES),alphas=list(ALPHAS),
        chart='NumPy tanh(float64 logits).astype(float32), exact original frontier chart',
        derivative='native rawfloat32 Jacobian at literal incumbent; continuous float64 tanh chain proposal surrogate, NOT derivative of quantization',
        direction='toward half-radius negative-gradient point; only if strict negative directional derivative, else zero',
        selection='stable argmin(model native192 identity terminal MSE,index), incumbent index0; replay exact; no modeled worsening',
        guard_source_sha256=GUARD_SHA,
        legality='original logits, full joint20 norm and suffix remaining radius, no tolerance/inverse/clipping/repair',
        per_compile=budget(),total_compile={k:8*v for k,v in budget().items()},
        AD_API_per_compile=dict(jacobian=1,grad=1),mathematical_VJP_directions=1536,
        predict_API_per_compile=7,candidates_per_compile=5,new_encoder=0,new_projector=0,
        execution='saved actual-mid compile then same-prefix exact native mid replay gate; NOT online latency',
        horizon=10,prefix=5,suffix=5,warmup=10,physical_caps=kernel.contract()['caps'],
        wall_seconds=900,output_cap_bytes=kernel.CAP,final_reserve_bytes=kernel.FINAL,disk_floor_bytes=kernel.FLOOR,
        Q='unchanged identity192',goal='original',total_FLOPs=None,equal_FLOPs=False,
        inherited_compute=None,CPU_neural=False,training=False,Qfit=False,retries=0,
        P1_admission=False,P2_admission=False,fresh=False,physical_release=False,scientific_pass=False)


def own_files():return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def write(path,value):
    path=Path(path).absolute();payload=raw(value)
    require(path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink()
        and len(payload)<=8*1024**2 and shutil.disk_usage(path.parent).free>=kernel.FLOOR+len(payload),'new own metadata')
    with path.open('xb') as f:require(f.write(payload)==len(payload),'full metadata');f.flush();os.fsync(f.fileno())
    kernel.paired.fsync_dir(path.parent)
    return dict(path=str(path),sha256=sha(path),status=value.get('status','prepared'))


def chart(u):
    u=np.asarray(u,np.float64)
    require(u.shape==(10,) and np.isfinite(u).all(),'finite original suffix logits10')
    return np.tanh(u).astype(np.float32).reshape(5,2)


def suffix_bits(commands):
    a=np.asarray(commands)
    require(a.shape==(5,2) and a.dtype.kind in 'fi' and np.isfinite(a).all() and
        (np.abs(a)<1).all() and np.array_equal(a,a.astype(np.float32)),'literal exact float32 suffix5, no rounding')
    return a.astype('<f4').tobytes().hex()


def legality(b,u):
    u=np.asarray(u,np.float64);prefix=np.asarray(b['original_prefix_logits'],np.float64)
    require(u.shape==prefix.shape==(10,) and np.isfinite(u).all() and np.isfinite(prefix).all(),'finite original logits')
    radius=b['packet_logit_cap'];joint=np.concatenate((prefix,u))
    return dict(suffix_norm=float(np.linalg.norm(u)),remaining_radius=radius,joint_norm=float(np.linalg.norm(joint)),
        joint_cap=CAP,raw_strictly_inside=bool((np.abs(chart(u))<1).all()),
        legal=bool(np.linalg.norm(u)<=radius and np.linalg.norm(joint)<=CAP and (np.abs(chart(u))<1).all()))


def with_incumbent(b,original_logits):
    b=deepcopy(b);u=np.asarray(original_logits,np.float64)
    require(u.shape==(20,) and np.isfinite(u).all(),'original20, never inverse rounded raw')
    require(np.tanh(u).astype(np.float32).reshape(10,2).tobytes().hex()==
        physical.bits(b['prefix_commands']+b['old_suffix_commands']),'original full10 chart bits')
    require(b['packet_logit_cap']==float(np.sqrt(CAP**2-float(u[:10]@u[:10]))),'unchanged exact remaining radius')
    b.update(original_prefix_logits=u[:10].tolist(),incumbent_logits=u[10:].tolist())
    b['incumbent_domain']=legality(b,u[10:])
    b['branch_id']=F.core.digest({k:v for k,v in b.items() if k!='branch_id'})
    return b


def build(rgb_output,rgb_receipt):
    require(sha(old.__file__)==OLD_SHA,'frozen v1 reused, not edited')
    require(sha(guard.__file__)==GUARD_SHA,'MAIN frozen model-only guard')
    p=old.build(rgb_output,rgb_receipt)
    pp,_=physical.verify(p['physical_prepare'],p['physical_prepare_sha256'])
    results=[physical.saved.overlay.read(Path(pp['planner_output'])/f'slot{i}_result.json.gz',physical.saved.EXPANDED) for i in range(4)]
    branches=[]
    for i,b in enumerate(p['branches']):
        winner=results[i//2]['results'][i%2]
        selected=[a for a in winner['attempts'] if a['candidate_id']==winner['winner']['candidate_id']]
        require(len(selected)==1,'exact original winner logits authority')
        branches.append(with_incumbent(b,selected[0]['logits']))
    p.update(schema=NAME+'.prepare',contract=contract(),branches=branches,
        status='prepared' if all(b['incumbent_domain']['legal'] for b in branches) else 'blocked_illegal_incumbent')
    physical.merge(p['bindings'],{str(f):sha(f) for f in own_files()})
    physical.merge(p['bindings'],{str(Path(guard.__file__).resolve()):GUARD_SHA})
    bridge.verify_bindings(p['bindings']);return p


def verify(prepared,expected):
    require(sha(prepared)==expected,'v2 prepare SHA');p=physical.read(prepared)
    require(raw(p)==raw(build(**p['arguments'])),'whole v2 prepare reconstruction');return p


class Store(old.Store):
    def __init__(self,path):
        require(Path(path).name.startswith(NAME+'_'),'new v2 output only')
        kernel.Store.__init__(self,path);self.branch=None;self.started=time.monotonic()


class Modules(F.Modules):
    """Same original module ABI, now seven stages, no old globals patched."""
    def pre(self,kind):
        def enter(module,args):
            n=len(self.entries)
            require(not self.failed and not self.pending and n<3*len(STAGES) and
                kind==self.ORDER[n%3] and self.meter.stage==STAGES[n//3],'fixed7 module order/stage')
            e=dict(index=n,resource=kind,stage=self.meter.stage,completed=False,inputs=[])
            self.entries.append(e);self.pending[module]=e;self.store.event(dict(kind='module_attempt',entry=deepcopy(e)))
            require(len(args)==(2 if kind=='predict' else 1),'exact native argument arity')
            for j,x in enumerate(args):
                a=self.array(x)
                if kind=='action_encoder':
                    require(a.shape==(1,3,10) and np.array_equal(a[0],self.meter.expected_input['action_history']),'literal actual raw normalized3')
                elif kind=='predict':
                    require((j==0 and a.shape==(1,3,192) and np.array_equal(a[0],self.meter.expected_input['latent_history'])) or
                        (j==1 and np.array_equal(a,self.action_embedding)),'literal actual context/embedding')
                else:require(a.shape==(3,192) and np.array_equal(a,self.predictor_output.reshape(3,192)),'full native projection')
                e['inputs'].append(F.save_array(self.store,f'{self.stem}.module_{n:02d}.input_{j}.npy',a))
            self.store.event(dict(kind='module_inputs_saved',entry=deepcopy(e)))
        return enter


class PredictAPI(suffix.PredictAPI):
    def __enter__(self):
        self.had='predict' in self.model.__dict__;self.original_local=self.model.__dict__.get('predict');self.original=self.model.predict
        def call(*args,**kwargs):
            require(len(self.entries)<len(STAGES) and all(e['completed'] for e in self.entries)
                and self.meter.stage==STAGES[len(self.entries)],'seven nonretry predict API entries')
            e=dict(index=len(self.entries),stage=self.meter.stage,completed=False);self.entries.append(e)
            self.sink.event(dict(kind='predict_API_attempt',entry=deepcopy(e)))
            value=self.original(*args,**kwargs);self.sync();e['completed']=True
            self.sink.event(dict(kind='predict_API_return',entry=deepcopy(e)));return value
        self.model.predict=call;return self


class Backend(F.d2p.TorchBackend):
    def jacobian_raw(self,context,incumbent,operation):
        import torch
        emb,history=self.tensors(context)
        x=torch.tensor(incumbent,device=self.device,dtype=torch.float32).reshape(10).requires_grad_(True)
        def terminal(raw10):return self.transition(emb,history,raw10.reshape(5,2))[0][0,-1]
        j=operation('jacobian',lambda:torch.autograd.functional.jacobian(terminal,x,vectorize=True))
        return j.detach().cpu().numpy().astype(float)


def proposals(b,z,jac):
    """Pure deterministic proposal, no native labels or terminal actual latents."""
    u=np.asarray(b['incumbent_logits'],np.float64);z=np.asarray(z,np.float64);jac=np.asarray(jac,np.float64)
    require(legality(b,u)['legal'] and z.shape==(192,) and jac.shape==(192,10)
        and np.isfinite(z).all() and np.isfinite(jac).all(),'legal incumbent and finite saved derivative')
    q=np.asarray(b['q']);require(np.array_equal(q,np.eye(192)),'unchanged identity192')
    gradient=2*(jac*(1-np.tanh(u)**2)[None,:]).T@(z-np.asarray(b['goal']))/192
    norm=float(np.linalg.norm(gradient));require(np.isfinite(norm),'finite gradient norm')
    anchor=u.copy() if norm==0 else -.5*b['packet_logit_cap']*gradient/norm
    direction=anchor-u
    slope=float(gradient@direction)
    if slope>=0:direction=np.zeros(10,np.float64)
    logits=[u]+[u+alpha*direction for alpha in ALPHAS]
    require(all(legality(b,x)['legal'] for x in logits),'all fixed candidates legal; never repair/project incumbent')
    return dict(gradient=gradient.tolist(),direction=direction.tolist(),slope=slope,
        descent_used=slope<0,logits=[x.tolist() for x in logits])


def choose(b,evaluations):
    """Caller independently validates domain/native predictions; guard does not."""
    require(len(evaluations)==5 and all(e['status']=='completed' and np.isfinite(e['cost']) for e in evaluations),'all5 costs, no subset')
    def packet(i):
        e=evaluations[i]
        return dict(context_sha256=F.core.digest(b['observed_context']),goal192=b['goal'],commands=e['commands'],
            predicted192=e['context']['latent_history'][-1],domain_verified=True,prediction_verified=True)
    selected=0;receipts=[]
    for i in range(1,5):
        decision=guard.choose(packet(selected),packet(i))
        require(decision['incumbent_Q']==evaluations[selected]['cost'] and decision['proposal_Q']==evaluations[i]['cost'],'guard exact native Q')
        receipts.append({k:decision[k] for k in ('selected','selected_Q','incumbent_Q','proposal_Q','selection_sha256')})
        if decision['selected']=='proposal':selected=i
    return dict(selected_index=selected,guard_receipts=receipts)


def evaluate(b,backend,meter,store,row):
    require(legality(b,b['incumbent_logits'])==b['incumbent_domain'] and b['incumbent_domain']['legal'],'legal exact incumbent before any call')
    require(suffix_bits(chart(b['incumbent_logits']))==suffix_bits(b['old_suffix_commands']),'literal old raw chart before model entry')
    context=b['observed_context'];stem=f'branch_{row["index"]:02d}'
    row.update(status='attempted',arrays={},evaluations=[])
    def save(name,a):return F.save_array(store,stem+'.'+name+'.npy',np.asarray(a))
    for k,a in (('observed_latents',context['latent_history']),('observed_tokens',context['action_history']),
        ('observed_raw_history',b['observed_raw_history']),('goal',b['goal']),('prefix_commands',b['prefix_commands']),('old_suffix',b['old_suffix_commands'])):
        row['arrays'][k]=save(k,np.asarray(a,np.float32))
    row['arrays']['incumbent_logits']=save('incumbent_logits',np.asarray(b['incumbent_logits'],np.float64))
    def stage(name,commands):
        meter.stage=name;meter.last_native=None;meter.expected_input=F.planner.incoming(b,context,commands)
        store.event(dict(kind='stage_started',branch_id=b['branch_id'],stage=name,
            commands=np.asarray(commands).astype(float).tolist(),expected_input=meter.expected_input))
    def forward(name,commands,index):
        stage(name,commands);e=dict(index=index,stage=name,commands=np.asarray(commands).astype(float).tolist(),status='attempted')
        store.event(dict(kind='suffix_evaluation_attempt',receipt=deepcopy(e)))
        after=backend.step(deepcopy(context),commands);F.planner.advance(b,context,commands,after);F.core.context(after)
        z=np.asarray(after['latent_history'][-1],np.float64)
        if meter.require_native:require(np.array_equal(z,meter.last_native),'actual native endpoint')
        e.update(status='completed',context=after,cost=float(np.mean((z-np.asarray(b['goal']))**2)),
            latent_array=save(name+'.context_latents',np.asarray(after['latent_history'],np.float32)),
            token_array=save(name+'.context_tokens',np.asarray(after['action_history'],np.float32)))
        store.event(dict(kind='suffix_evaluation_return',receipt=deepcopy(e)));return e
    incumbent=np.asarray(b['old_suffix_commands'],np.float32)
    row['evaluations'].append(forward('incumbent',incumbent,0))
    stage('jacobian',incumbent);jac=np.asarray(backend.jacobian_raw(deepcopy(context),incumbent,meter.call),np.float64)
    require(jac.shape==(192,10) and np.isfinite(jac).all(),'raw incumbent192x10 AD')
    if meter.require_native:require(np.array_equal(meter.last_native,row['evaluations'][0]['context']['latent_history'][-1]),'AD forward identical incumbent raw')
    row['arrays']['jacobian']=save('jacobian',jac)
    row['proposal']=proposals(b,row['evaluations'][0]['context']['latent_history'][-1],jac)
    store.event(dict(kind='candidates_frozen',proposal=deepcopy(row['proposal'])))
    for i,u in enumerate(row['proposal']['logits'][1:],1):row['evaluations'].append(forward(STAGES[i+1],chart(u),i))
    row.update(choose(b,row['evaluations']));selected=row['selected_index']
    store.event(dict(kind='selection_frozen',index=selected,cost=row['evaluations'][selected]['cost']))
    row['replay']=forward('selected_replay',chart(row['proposal']['logits'][selected]),selected)
    require(row['replay']['context']==row['evaluations'][selected]['context'] and
        row['replay']['cost']==row['evaluations'][selected]['cost']<=row['evaluations'][0]['cost'],'exact replay and strict no modeled worsening')
    row.update(status='suffix_compiled',new_suffix_commands=row['evaluations'][selected]['commands'],
        prefix_commands=deepcopy(b['prefix_commands']),observed_context_sha256=F.core.digest(context),
        dispatch='safeguarded_model_only_stable_argmin',equal_new_replay=True)
    return row


def check_compile(b,r,load):
    require(r['branch_id']==b['branch_id'] and r['status']=='suffix_compiled' and
        r['dispatch']=='safeguarded_model_only_stable_argmin' and r['equal_new_replay'] is True,'complete v2 only')
    require(legality(b,b['incumbent_logits'])==b['incumbent_domain'] and b['incumbent_domain']['legal'],'legal old domain, no repair')
    require(suffix_bits(chart(b['incumbent_logits']))==suffix_bits(b['old_suffix_commands']),'exact original incumbent chart bits')
    require(r['prefix_commands']==b['prefix_commands'] and r['observed_context_sha256']==F.core.digest(b['observed_context']),'literal same prefix/context')
    for k,a in (('observed_latents',b['observed_context']['latent_history']),('observed_tokens',b['observed_context']['action_history']),
        ('observed_raw_history',b['observed_raw_history']),('goal',b['goal']),('prefix_commands',b['prefix_commands']),
        ('old_suffix',b['old_suffix_commands']),('incumbent_logits',b['incumbent_logits'])):
        value=load(r['arrays'][k]);require(np.array_equal(value,a),'saved compiler input '+k)
        require(value.dtype==np.dtype('float64' if k=='incumbent_logits' else 'float32'),'original input dtype')
    jac=load(r['arrays']['jacobian']);require(jac.dtype==np.float64,'saved raw derivative float64')
    proposal=proposals(b,r['evaluations'][0]['context']['latent_history'][-1],jac)
    require(r['proposal']==proposal and type(r['selected_index']) is int and r['selected_index'] in range(5),'independent fixed proposal/index')
    selected=r['evaluations'][r['selected_index']]
    require(r['replay']['context']==selected['context'] and r['replay']['cost']==selected['cost']<=r['evaluations'][0]['cost']
        and suffix_bits(r['new_suffix_commands'])==suffix_bits(selected['commands'])
        and r['replay']['index']==r['selected_index'],'no modeled worsening and exact independent replay')
    F.blind.exact_counts(r['resources'],budget());F.blind.exact_counts(r['AD_API_resources'],dict(jacobian=1,grad=1))
    F.blind.exact_counts(r['predict_API_resources'],dict(predict=7))
    require(r['predict_API_ledger']==[dict(index=i,stage=s,completed=True) for i,s in enumerate(STAGES)],'all7 API calls')
    require(r['math_ledger']==[dict(index=0,resource='jacobian',completed=True,stage='jacobian',shape=[192,10],mathematical_VJP=192)] and
        r['AD_API_ledger']==[dict(index=0,resource='jacobian',completed=True,is_grads_batched=False),
            dict(index=1,resource='grad',completed=True,is_grads_batched=True)],'raw AD exact API/math axes, no SVD')
    ev=r['evaluations'];require([e['index'] for e in ev]==list(range(5)),'all5 incumbent/candidates')
    raws=[b['old_suffix_commands']]*2+[chart(u).tolist() for u in proposal['logits'][1:]]+[r['new_suffix_commands']]
    contexts=[ev[0]['context']]*2+[e['context'] for e in ev[1:]]+[r['replay']['context']]
    evaluations=[ev[0],None]+ev[1:]+[r['replay']]
    entries=r['module_ledger'];require(len(entries)==21,'all21 full native module calls')
    for n,(stage,commands,ctx,evaluation) in enumerate(zip(STAGES,raws,contexts,evaluations,strict=True)):
        incoming=F.planner.incoming(b,b['observed_context'],commands);ins=[];outs=[]
        for j,(entry,kind) in enumerate(zip(entries[3*n:3*n+3],Modules.ORDER,strict=True)):
            require(entry['index']==3*n+j and entry['resource']==kind and entry['stage']==stage and entry['completed'] is True,'complete native stage/order')
            ins.append([load(x) for x in entry['inputs']]);outs.append(load(entry['output']))
            require(all(x.dtype==np.float32 and np.isfinite(x).all() for x in ins[-1]+[outs[-1]]),'native float32 buffers')
        require(len(ins[0])==1 and ins[0][0].shape==(1,3,10) and np.array_equal(ins[0][0][0],incoming['action_history'])
            and outs[0].ndim==3 and outs[0].shape[:2]==(1,3),'all3 normalized raw tokens')
        require(len(ins[1])==2 and ins[1][0].shape==(1,3,192) and np.array_equal(ins[1][0][0],incoming['latent_history'])
            and np.array_equal(ins[1][1],outs[0]) and outs[1].shape==(1,3,192),'actualmid -> predictor')
        require(len(ins[2])==1 and np.array_equal(ins[2][0],outs[1].reshape(3,192)) and outs[2].shape==(3,192)
            and np.array_equal(outs[2][-1],ctx['latent_history'][-1]),'full projector path, exact endpoint')
        if evaluation is not None:
            require(evaluation['stage']==stage and suffix_bits(evaluation['commands'])==suffix_bits(commands),'literal candidate raw bits evaluation')
            F.planner.advance(b,b['observed_context'],commands,ctx);F.core.context(ctx)
            require(evaluation['cost']==float(np.mean((np.asarray(ctx['latent_history'][-1])-np.asarray(b['goal']))**2)),'independent Q')
            require(np.array_equal(load(evaluation['latent_array']),ctx['latent_history']) and
                np.array_equal(load(evaluation['token_array']),ctx['action_history']),'whole shifted response history')
    selection=choose(b,r['evaluations'])
    require(all(r[k]==v for k,v in selection.items()),'native-verified stable MAIN guard decisions, no labels')


def check_events(b,r,events):
    keys=dict(module='module_ledger',math='math_ledger',AD_API='AD_API_ledger',predict_API='predict_API_ledger')
    terminal=dict(module='module_return',math='math_return',AD_API='AD_API_completed',predict_API='predict_API_return')
    allowed={'stage_started','candidates_frozen','selection_frozen','suffix_evaluation_attempt','suffix_evaluation_return','module_inputs_saved'}
    allowed|={k+'_attempt' for k in keys}|set(terminal.values())
    require(all(e['kind'] in allowed for e in events),'unknown event never filtered')
    raws=[b['old_suffix_commands']]*2+[chart(u).tolist() for u in r['proposal']['logits'][1:]]+[r['new_suffix_commands']]
    starts=[e for e in events if e['kind']=='stage_started']
    require(starts==[dict(kind='stage_started',branch_id=b['branch_id'],stage=s,commands=commands,
        expected_input=F.planner.incoming(b,b['observed_context'],commands)) for s,commands in zip(STAGES,raws,strict=True)],'all7 exact native input stages')
    for kind,key in keys.items():
        entries=r[key]
        attempted=[dict(e,completed=False) if kind!='module' else dict(index=e['index'],resource=e['resource'],stage=e['stage'],completed=False,inputs=[]) for e in entries]
        returns=[entries[1],entries[0]] if kind=='AD_API' else entries
        require([e['entry'] for e in events if e['kind']==kind+'_attempt']==attempted and
            [e['entry'] for e in events if e['kind']==terminal[kind]]==returns,'exact attempts/returns '+kind)
    require([e['entry'] for e in events if e['kind']=='module_inputs_saved']==[
        {k:v for k,v in e.items() if k!='output'}|{'completed':False} for e in r['module_ledger']],'all durable module inputs')
    evaluations=r['evaluations']+[r['replay']]
    require([e['receipt'] for e in events if e['kind']=='suffix_evaluation_return']==evaluations and
        [e['receipt'] for e in events if e['kind']=='suffix_evaluation_attempt']==[
            {k:e[k] for k in ('index','stage','commands')}|{'status':'attempted'} for e in evaluations],'every evaluation intent/return')
    require([e for e in events if e['kind']=='candidates_frozen']==[dict(kind='candidates_frozen',proposal=r['proposal'])] and
        [e for e in events if e['kind']=='selection_frozen']==[dict(kind='selection_frozen',index=r['selected_index'],cost=r['evaluations'][r['selected_index']]['cost'])],'own candidate/selection durable seals')
    stack=[];seen=set();active=None;done=[];frozen=False;selected=False
    for e in events:
        kind=e['kind']
        if kind=='stage_started':
            require(not stack,'no pending work between stages');active=e['stage']
            if active.startswith('candidate_'):require(frozen,'candidates frozen before evaluation')
            if active=='selected_replay':require(selected,'selection before replay')
        elif kind.endswith('_attempt') and kind[:-8] in keys:
            tag=kind[:-8];entry=e['entry'];require(entry.get('stage',active)==active,'operation own stage');stack.append((tag,entry['index']))
        elif kind=='module_inputs_saved':
            require(stack and stack[-1]==('module',e['entry']['index']),'durable input between entry/return');seen.add(e['entry']['index'])
        elif kind in terminal.values():
            tag=next(k for k,v in terminal.items() if v==kind)
            require(stack and stack.pop()==(tag,e['entry']['index']),'exact nested return')
            if tag=='module':require(e['entry']['index'] in seen,'return requires saved inputs')
        elif kind=='suffix_evaluation_return':require(not stack and e['receipt']['stage']==active,'evaluation after actual return');done.append(active)
        elif kind=='candidates_frozen':require(not stack and done==['incumbent'] and active=='jacobian','proposal only after incumbent AD');frozen=True
        elif kind=='selection_frozen':require(not stack and done==[STAGES[0]]+list(STAGES[2:6]),'all finite candidates before selection');selected=True
    require(not stack and done==[STAGES[0]]+list(STAGES[2:]),'no pending or omitted stages')


def compile_run(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN explicit HOST CUDA before I/O')
    started=time.monotonic();p=verify(prepared,expected)
    require(p['status']=='prepared' and len(p['branches'])==8 and all(b['incumbent_domain']['legal'] for b in p['branches']),
        'all8 legal original incumbents, blocked before model/output; no repair')
    store=Store(output);store.started=started;store.check()
    rows=[dict(index=i,branch_id=b['branch_id'],status='not_started') for i,b in enumerate(p['branches'])]
    errors=[];model=None;before=after=None
    store.put('binding.json',dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected))
    try:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        bridge.native.frozen(model);before=bridge.native.tensor_hash(model);require(before==p['expected_model_hash'],'original frozen checkpoint')
        mm,cn=torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            with torch.enable_grad(),torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
                for i,(b,row) in enumerate(zip(p['branches'],rows,strict=True)):
                    store.check();store.branch=i;row['status']='attempted'
                    store.emit('compile_intent',branch=i,branch_id=b['branch_id'],budget=budget())
                    sink=suffix.Sink(store,store.check);sync=lambda:torch.cuda.synchronize(device)
                    meter=F.Meter(sink.event,sync);meter.require_native=True
                    modules=Modules(model,sink,f'branch_{i:02d}',meter,sync)
                    ad=F.SuffixAD(torch,F.blind.ARMS[1],sink.event,sync);ad.stage=lambda:meter.stage
                    api=PredictAPI(model,meter,sink,sync)
                    try:
                        with ad,modules,api:evaluate(b,Backend(model,device),meter,sink,row)
                    finally:
                        row.update(resources={**modules.counts(),**meter.counts()},AD_API_resources=ad.counts(),predict_API_resources=api.counts(),
                            module_ledger=deepcopy(modules.entries),math_ledger=deepcopy(meter.entries),AD_API_ledger=deepcopy(ad.events),predict_API_ledger=deepcopy(api.entries))
                    check_compile(b,row,sink.load);store.emit('compile_sealed',branch=i,row=row)
        finally:torch.backends.cuda.matmul.allow_tf32=mm;torch.backends.cudnn.allow_tf32=cn
    except BaseException as e:errors.append(kernel.error('compile',e))
    finally:
        try:
            if model is not None:bridge.native.frozen(model);after=bridge.native.tensor_hash(model)
            bridge.verify_bindings(p['bindings']);store.check()
        except BaseException as e:errors.append(kernel.error('final_model_source_wall',e))
        complete=not errors and before==after==p['expected_model_hash'] and all(r['status']=='suffix_compiled' for r in rows)
        status='all8_safeguarded_science_blocked' if complete else 'partial_or_failed'
        store.emit('run_finished',status=status)
        report=dict(schema=NAME+'.compile',status=status,prepared=str(Path(prepared).resolve()),prepare_sha256=expected,
            contract=contract(),rows=rows,errors=errors,model_hash_before=before,model_hash_after=after,
            elapsed_seconds=time.monotonic()-started,artifact_sha256={f.name:sha(f) for f in store.path.iterdir()},journal_chain_tail=store.previous)
        store.put('report.json',report,final=True)
    require(complete,'partial preserved/no retries: '+str(errors))
    return dict(report=str(store.path/'report.json'),report_sha256=sha(store.path/'report.json'))


def seal_plan(b,r,original,expected):
    # Feed the selected pair to the frozen provenance constructor; no old claim
    # is carried as a prediction of changed actions.
    selected=r['evaluations'][r['selected_index']]
    adapted=deepcopy(r);adapted['evaluations']=[r['evaluations'][0],selected,r['replay']]
    plan=old.seal_plan(b,adapted,original,expected)
    plan.update(safeguard=dict(incumbent_Q=r['evaluations'][0]['cost'],selected_Q=selected['cost'],
        selected_index=r['selected_index'],no_modeled_worsening=True,chart=contract()['chart'],domain=b['incumbent_domain']))
    plan['new_suffix_witness']['compile_row_sha256']=physical.saved.digest(r)
    plan['compiler_selection_digest']=physical.saved.digest(dict(branch=b,result=r))
    plan['selection_digest']=physical.saved.digest({k:v for k,v in plan.items() if k!='selection_digest'})
    return plan


def consume_compile(output,expected):
    output=Path(output).resolve();require(sha(output/'report.json')==expected,'external compile SHA')
    r=physical.read(output/'report.json');p=verify(r['prepared'],r['prepare_sha256'])
    require(p['status']=='prepared' and r['schema']==NAME+'.compile' and r['status']=='all8_safeguarded_science_blocked'
        and r['errors']==[] and r['contract']==contract() and len(r['rows'])==len(p['branches'])==len(p['plans'])==8 and
        r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'] and 0<=r['elapsed_seconds']<900,'whole8 completed frozen compile only')
    require({f.name for f in output.iterdir()}==set(r['artifact_sha256'])|{'report.json'} and
        sum(f.stat().st_size for f in output.iterdir())<=kernel.CAP,'whole compile census/cap')
    for n,h in r['artifact_sha256'].items():require(sha(bridge.rgb.flat(output,n))==h,'actual artifact SHA')
    require(physical.read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'original execution binding')
    events=physical.journal(output/'journal.jsonl',r['journal_chain_tail'])
    require(events[-1]['kind']=='run_finished' and events[-1]['status']==r['status'] and
        all(e['kind'] in ('compile_intent','compiler_event','compile_sealed') and type(e['branch']) is int and e['branch'] in range(8) for e in events[:-1])
        and [e['branch'] for e in events[:-1]]==sorted(e['branch'] for e in events[:-1]),'entire ordered journal, unknown rejected')
    plans=[]
    for i,(b,row,original) in enumerate(zip(p['branches'],r['rows'],p['plans'],strict=True)):
        segment=[e for e in events[:-1] if e['branch']==i]
        require(segment[0]['kind']=='compile_intent' and segment[0]['budget']==budget() and segment[0]['branch_id']==b['branch_id']
            and segment[-1]['kind']=='compile_sealed' and segment[-1]['row']==row and all(e['kind']=='compiler_event' for e in segment[1:-1]),'all8 exact intent/work/seal')
        require(row['index']==i,'fixed8 index');check_compile(b,row,lambda x:old.load_npy(output,x))
        check_events(b,row,[e['event'] for e in segment[1:-1]])
        plans.append(seal_plan(b,row,original,expected))
    require(sha(output/'report.json')==expected,'compile unchanged after consume');return p,plans


class ReplayGateStore(old.ReplayGateStore,Store):
    def __init__(self,path,midpoints):Store.__init__(self,path);self.midpoints=midpoints;self.matched=set()


def physical_material(compile_output,compile_sha):
    p,plans=consume_compile(compile_output,compile_sha);pp,data=physical.verify(p['physical_prepare'],p['physical_prepare_sha256'])
    pp=deepcopy(pp);pp['plans']=plans;physical.merge(pp['bindings'],p['bindings']);physical.validate_plans(plans,data)
    return p,pp,data


def run_physical(compile_output,compile_sha,output,factory,authorization):
    require(authorization==AUTH,'MAIN explicit physics before I/O');started=time.monotonic()
    p,pp,data=physical_material(compile_output,compile_sha);r=physical.read(Path(compile_output)/'report.json')
    binding=dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256'],planner_report_sha256=pp['planner_report_sha256'],
        strict_receipt_sha256=pp['strict_receipt_sha256'],feedback_compile_output=str(Path(compile_output).resolve()),feedback_compile_sha256=compile_sha)
    store=ReplayGateStore(output,p['midpoints']);store.started=started
    return physical.execute(pp,data,store,factory,authorization=physical.AUTH,binding=binding,started=started)


def consume_physical(output,expected):
    output=Path(output).resolve();require(sha(output/'report.json')==expected,'external actual physical SHA')
    r=physical.read(output/'report.json');binding=r['execution_binding']
    p,pp,data=physical_material(binding['feedback_compile_output'],binding['feedback_compile_sha256'])
    events=physical.journal(output/'journal.jsonl',r['journal_chain_tail'])
    require(sha(output/'journal.jsonl')==r['journal_sha256'] and sha(output/'execution_binding.json')==r['execution_binding_sha256']
        and physical.read(output/'execution_binding.json')==binding,'full native binding/journal')
    require(binding==dict(prepared=r['execution_binding']['prepared'],prepare_sha256=r['execution_binding']['prepare_sha256'],
        planner_report_sha256=pp['planner_report_sha256'],strict_receipt_sha256=pp['strict_receipt_sha256'],
        feedback_compile_output=str(Path(binding['feedback_compile_output']).resolve()),feedback_compile_sha256=binding['feedback_compile_sha256'])
        and verify(binding['prepared'],binding['prepare_sha256'])==p,'native execution same whole prepare')
    def image(f):
        from PIL import Image
        path=bridge.rgb.flat(output,f['png']);require(sha(path)==f['file_sha256'],'real PNG SHA')
        with Image.open(path) as im:return np.asarray(im).copy()
    strict=physical.assemble(r,events,pp,data,image)
    require(strict['status']=='engineering_verified_science_blocked' and len(strict['rows'])==8,'all8 native strict')
    require({f.name for f in output.iterdir()}==set(strict['PNG_files'])|{'report.json','journal.jsonl','execution_binding.json'} and
        sum(f.stat().st_size for f in output.iterdir())<=kernel.CAP,'whole physical census/cap')
    for i,mid in enumerate(p['midpoints']):
        frame=[e['receipt'] for e in events if e['kind']=='fresh_observation' and e['branch']==i and e['receipt']['phase']=='plan'
            and e['receipt']['step']==5 and e['receipt']['condition']==mid['condition'] and e['receipt']['overlay_index']==(1 if mid['condition']=='on' else 0)]
        if strict['rows'][i]['own_returned']>5:
            require(len(frame)==1 and all(frame[0][k]==mid[k] for k in ('file_sha256','pixel_sha256','state','snapshot')),'exact causal realmid gate')
    rows=[]
    for i,(actual,control,parent,plan) in enumerate(zip(strict['rows'],p['control_rows'],p['plans'],pp['plans'],strict=True)):
        require((control['root_slot'],control['condition'],control['arm'],control['selection_digest'])==
            (parent['root_slot'],parent['condition'],parent['arm'],parent['selection_digest']),'all8 original control identity')
        a,c=actual['final_metrics'],control['final_metrics']
        delta={k:None if a is None or c is None or a[k] is None or c[k] is None else float(a[k])-float(c[k]) for k in
            ('native_reward','joint_xy_error','agent_xy_error','object_xy_error','agent_velocity_error','wrapped_angle_error','native_coverage')}
        rows.append(dict(index=i,control=control,intervention=actual,new_minus_control=delta,safeguard=plan['safeguard'],
            native_success_control=None if c is None else c['native_success'],native_success_new=None if a is None else a['native_success']))
    bridge.verify_bindings(p['bindings']);require(sha(output/'report.json')==expected,'physical unchanged')
    return dict(schema=NAME+'.physical_consume',status='paired_known_intervention_recorded_NOT_science',rows=rows,
        physical_report_sha256=expected,feedback_compile_sha256=binding['feedback_compile_sha256'],counts=strict['counts'],
        fixed_denominator=8,known_roots=2,total_FLOPs=None,P1_admission=False,P2_admission=False,physical_release=False)


def main():
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument('command',choices=('prepare','verify','compile','consume_compile','run_physical','consume_physical'))
    for k in ('output','prepared','prepare-sha','rgb-output','rgb-receipt','compile-output','compile-sha','report-sha','receipt'):q.add_argument('--'+k)
    q.add_argument('--authorize-host-cuda',action='store_true');q.add_argument('--authorize-physics',action='store_true');a=q.parse_args()
    if a.command=='prepare':v=write(a.output,build(a.rgb_output,a.rgb_receipt))
    elif a.command=='verify':verify(a.prepared,a.prepare_sha);v=dict(status='verified_no_execution')
    elif a.command=='compile':
        require(a.authorize_host_cuda,'MAIN HOST CUDA explicit');v=compile_run(a.prepared,a.prepare_sha,a.output,AUTH)
    elif a.command=='consume_compile':
        _,plans=consume_compile(a.compile_output,a.compile_sha)
        v=write(a.receipt,dict(schema=NAME+'.compile_consume',status='compiled_engineering_only',plans=plans,P2_admission=False,physical_release=False))
    elif a.command=='run_physical':
        require(a.authorize_physics,'MAIN physics explicit')
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False,render_action=False)
        r=run_physical(a.compile_output,a.compile_sha,a.output,factory,AUTH)
        v=dict(status=r['status'],report=str(Path(a.output)/'report.json'),report_sha256=sha(Path(a.output)/'report.json'))
    else:v=write(a.receipt,consume_physical(a.output,a.report_sha))
    print(raw(v).decode(),end='',flush=True)


if __name__=='__main__':main()
