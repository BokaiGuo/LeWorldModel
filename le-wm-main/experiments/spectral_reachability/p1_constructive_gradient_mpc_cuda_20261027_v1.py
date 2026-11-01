"""Frozen-model open-loop gradient MPC solver, MAIN-only CUDA.

Two exact approved blind tasks, 32 fixed projected scalar-gradient steps.
No outcome access, training, line search, full Jacobian, or resource-fairness claim.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p1_constructive_blind_planners_cuda_20261027_v1 as base

NAME='p1_constructive_gradient_mpc_cuda_20261027_v1'
ARM='gradient_MPC'
ROOT,gn,enc=base.ROOT,base.gn,base.encode
core,screen,io=enc.core,gn.screen,gn.io
require,sha,read,canonical=base.require,base.sha,base.read,base.canonical
UPDATES,STEP_SIZE=32,1.0
BASE_SHA='e56bee610f4eb412cc03fbe49b3df91054d0defe23d24b6fd9c77e6a7f1e9f27'
GN_SHA='825c148cf515ad42d59b187d0d8c83984826db67f58d66b2e2d4f289dcecf70a'
EXPORTS=[str(ROOT/f'results/p1_constructive_blind_encode_cuda_20261027_v1_cuda_main01/task_{i:02d}.json.gz') for i in range(2)]
EXPORT_SHAS=['714f471761278fa654542d4bb0878ce18e3c304df6158005af46d2129533265a',
            '70865d56f9b66f0072b2ad7de01cc09d9a9275047925e188104a13d5182ce1f2']
CAP,FLOOR,FINAL=base.CAP,base.FLOOR,base.FINAL


def budget():
    # initial H2=2; each gradient ADforward2 + proposal H2=2; winner triple=6.
    return dict(predict=2+4*UPDATES+6,action_encoder=2+4*UPDATES+6,
                backward=UPDATES,proposal=UPDATES,selection=1)


def contract():
    return dict(arm=ARM,roots=2,initial_logits=np.zeros((2,10)).tolist(),updates=UPDATES,step_size=STEP_SIZE,
        method='x_next=packet_ball_projection(x-step_size*gradient(meanQ)); accept every update, no line search',
        selection='stable minimum(cost,iteration) over zero and ALL32 iterates, then independent triple fullH2 replay',
        gradient='one scalar autograd.grad through complete H2 native state AND action-history graph; analytic tanh chain once',
        objective='identity native192 mean squared terminal goal residual; no action penalty or learned readout',
        logits_shape=[2,10],raw_commands_shape=[10,2],packet_logit_cap=float(.75*np.sqrt(5)),
        per_root_budget=budget(),total_budget={k:2*v for k,v in budget().items()},
        gradient_forward_predict_per_root=64,non_AD_predict_per_root=72,
        scalar_reverse_sweeps_per_root=32,AD_API_per_root=dict(grad=32),
        reverse_accounting='backward meter and grad API receipt describe SAME32 scalar reverse sweeps, not64',
        jacobian=0,svd=0,linear_solve=0,new_images=0,new_physics=0,training=False,new_Q=False,
        with_target=False,closed_loop=False,inherited_encoding='separate paid encoder/projector source cost, not zero pipeline cost',
        numerical=dict(SDP='MATH',matmul_TF32=False,cudnn_TF32=False),
        parameter_authority='fixed before this baseline run; no terminal outcome or prior planner score used to choose parameters',
        equal_forward_budget_claim=False,equal_total_resource_claim=False,P1_admission=False,
        new_independent_confirmation=False,output_cap_bytes=CAP,reserve_bytes=FLOOR,
        scope='two existing calibration tasks, open-loop differentiable MPC optimizer only; no physical efficacy claim')


def build():
    require(sha(base.__file__)==BASE_SHA and sha(gn.__file__)==GN_SHA,'approved sealed interface/native/replay helpers')
    p=base.build(EXPORTS,EXPORT_SHAS)  # reads approved exports/local models/sources only
    slots=[]
    for k,s in enumerate(p['slots']):
        e=read(s['task_export'])
        require(e['source_slot']==k and e['blind_source_sha256']==e['blind_sha256']==e['task_contract']['blind_source_sha256']==list(enc.APPROVED_INPUTS.values())[k],
                'ordered exact approved blind identity, no witness/native-root lookup')
        slots.append(dict(s,source_slot=k,blind_sha256=e['blind_sha256']))
    bindings=deepcopy(p['bindings_sha256'])
    for f in (Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):
        bindings[str(f.resolve())]=sha(f)
    for path,h in bindings.items():require(sha(path)==h,'prepare source/task/model TOCTOU')
    return dict(schema=NAME+'.prepare',slots=slots,policy=p['policy'],expected_model_hash=p['expected_model_hash'],
                contract=contract(),bindings_sha256=bindings,global_freshness='existing_two_calibration_tasks_only')


def prepare(output):
    p=build();raw=canonical(p);path=Path(output).absolute()
    require(len(raw)<256*1024 and not path.exists() and shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'NEW small prepare +200MiB')
    with path.open('xb') as f:
        require(f.write(raw)==len(raw),'short prepare');f.flush();os.fsync(f.fileno())
    io.fsync_directory(path.parent)
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(raw),contract=p['contract'])


def verify(path,expected):
    core.hash_value(expected);require(sha(path)==expected,'external prepare SHA')
    p=read(path);require(p==build(),'complete own source/export/model/protocol reconstruction');return p


class Meter(screen.Meter):
    def __init__(self,sink,sync=lambda:None):
        super().__init__(budget(),sink);self.sync=sync
    def emit(self,kind,**values):
        self.sink(dict(kind=kind,evaluation_id=self.group,**deepcopy(values)))
    def begin(self,resource):
        require(all(e['completed'] for e in self.extended),'no recursive/retried/failed work')
        return super().begin(resource)
    def end(self,token):
        require(token[0] is self.extended[-1] and not token[0]['completed'],'unique latest return')
        self.sync();super().end(token)
    def counts(self):
        return {k:dict(attempted=sum(e['resource']==k for e in self.extended),
                       completed=sum(e['resource']==k and e['completed'] for e in self.extended)) for k in self.caps}


class GradGate:
    """Permit only the32 explicitly metered scalar-grad entries, block other AD."""
    def __init__(self,torch,meter):
        self.torch,self.meter=torch,meter;self.events=[];self.hooks=io.Hooks();self.failed=False
    def dispatch(self,fn,args,kwargs):
        require(not self.failed and len(self.events)<UPDATES and self.meter.extended and
            self.meter.extended[-1]['resource']=='backward' and not self.meter.extended[-1]['completed'],
            'grad only within declared scalar backward attempt')
        require(len(args)==2 and args[0].ndim==0 and tuple(args[1].shape)==(10,2) and
            args[1].is_leaf and args[1].requires_grad and args[1].grad is None and
            kwargs==dict(retain_graph=False,create_graph=False), 'one scalar loss / one fresh raw leaf, no higher-order graph')
        e=dict(index=len(self.events),resource='grad',evaluation_id=self.meter.group,completed=False,
               scalar_outputs=1,input_shape=[10,2],retain_graph=False,create_graph=False)
        self.events.append(e);self.meter.emit('AD_API_attempt',entry=e)
        try:
            value=fn(*args,**kwargs);self.meter.sync();e['completed']=True
            self.meter.emit('AD_API_completed',entry=e);return value
        except BaseException:self.failed=True;raise
    def __enter__(self):
        def denied(*a,**kw):
            self.failed=True;self.meter.emit('forbidden_AD_API');raise ValueError('unbudgeted AD API')
        try:
            original=self.torch.autograd.grad
            self.hooks.install(self.torch.autograd,'grad',lambda *a,**kw:self.dispatch(original,a,kw))
            groups=[(self.torch.autograd,('backward',)),(self.torch.Tensor,('backward',)),
                (self.torch.autograd.functional,('jacobian','hessian','jvp','vjp','hvp','vhp')),
                (self.torch.func,('jvp','vjp','jacrev','jacfwd','hessian','grad','grad_and_value'))]
            for obj,keys in groups:
                for key in keys:
                    if hasattr(obj,key):self.hooks.install(obj,key,denied)
            return self
        except BaseException as exc:
            self.__exit__(type(exc),exc,exc.__traceback__);raise
    def __exit__(self,typ,exc,tb):
        failures=[];self.hooks.restore(failures)
        if failures:
            if exc is not None:exc.add_note('AD restore errors: '+repr(failures))
            else:raise RuntimeError('AD restore errors: '+repr(failures))
    def counts(self):
        return dict(grad=dict(attempted=len(self.events),completed=sum(e['completed'] for e in self.events)))


class NativeGradient(screen.NativeCuda):
    def gradient(self,task,incumbent,meter):
        torch=self.torch
        require(np.array_equal(task['projection'],np.eye(192)),'original identity meanQ only')
        with torch.enable_grad():
            # NumPy float32 tanh defines the exact evaluated forward point.
            # Differentiate raw once, then apply analytic logit tanh chain ONCE.
            raw=self._tensor(incumbent['commands']).requires_grad_(True)
            z,h=self._start(task);paths=[]
            for k in range(2):
                z,h=self._step(task,z,h,raw[5*k:5*k+5])
                paths.append(self._saved(z,h))
                meter.emit('gradient_forward_packet',packet=k,context=paths[-1])
            terminal=z[0,-1];loss=((terminal-self._tensor(task['goal']))**2).mean()
            before=dict(task_id=core.task_id(task),at_logits=deepcopy(incumbent['logits']),contexts=paths,
                at_terminal=terminal.detach().cpu().tolist(),native_float32_loss=float(loss.detach().cpu()),
                saved_float64_cost=core.model_cost(task,paths[-1]['latent_history'][-1]))
            meter.emit('gradient_forward_complete',forward=before)
            # Save forward before exact-point verification/backward: mismatches
            # and backward failures remain durable scientific observations.
            require(paths==incumbent['contexts'],'fresh scalar-gradient forward must match full incumbent path')
            derivative=meter.call('backward',lambda:torch.autograd.grad(loss,raw,retain_graph=False,create_graph=False)[0])
            raw_gradient=derivative.detach().cpu().numpy().astype(float)
        raw_gradient=core.array(raw_gradient,(10,2))
        chain=1-np.asarray(incumbent['commands'],float)**2
        g=(raw_gradient*chain).reshape(2,10)
        value=dict(**before,raw_control_gradient=raw_gradient.tolist(),objective_gradient=g.tolist(),
                   tanh_chain=chain.tolist(),scalar_reverse_sweeps=1)
        meter.emit('gradient_complete',gradient=value)
        return value


def proposal(logits,gradient,cap):
    x,g=core.array(logits,(2,10)),core.array(gradient,(2,10))
    return screen.projected_logits(x-STEP_SIZE*g,cap)


def compact(p):
    return dict(logits=p['logits'],commands=p['commands'],cost=p['cost'],terminal=p['contexts'][-1]['latent_history'][-1])


def check_gradient(task,incumbent,d):
    require(d['task_id']==core.task_id(task) and d['at_logits']==incumbent['logits'] and
        d['contexts']==incumbent['contexts'] and d['at_terminal']==incumbent['contexts'][-1]['latent_history'][-1] and
        d['saved_float64_cost']==incumbent['cost'] and d['scalar_reverse_sweeps']==1,'fresh same-task complete H2 gradient point')
    require(np.isfinite(d['native_float32_loss']) and d['native_float32_loss']>=0,'finite native mean loss')
    raw=core.array(d['raw_control_gradient'],(10,2));chain=1-np.asarray(incumbent['commands'],float)**2
    require(np.array_equal(d['tanh_chain'],chain) and
        np.array_equal(d['objective_gradient'],(raw*chain).reshape(2,10)),'analytic tanh chain exactly once, no192 multiplier')


def optimize(task,backend,meter):
    core.validate_task(task);require(np.array_equal(task['projection'],np.eye(192)),'same identity meanQ')
    meter.group='iterate_0';current=gn.single(task,np.zeros((2,10)),backend,meter,'whole_replay')
    candidates=[dict(iteration=0,**compact(current))];gradients=[]
    meter.emit('iterate_saved',candidate=candidates[0])
    for it in range(1,UPDATES+1):
        meter.group=f'gradient_{it}';meter.emit('gradient_request',iteration=it,logits=current['logits'],commands=current['commands'])
        d=backend.gradient(task,current,meter);check_gradient(task,current,d);gradients.append(d)
        x=meter.call('proposal',lambda:proposal(current['logits'],d['objective_gradient'],task['packet_logit_cap']))
        meter.emit('proposal_frozen',iteration=it,step_size=STEP_SIZE,logits=x.tolist(),parent_logits=current['logits'])
        meter.group=f'iterate_{it}';current=gn.single(task,x,backend,meter,'whole_replay')
        candidate=dict(iteration=it,**compact(current));candidates.append(candidate);meter.emit('iterate_saved',candidate=candidate)
    meter.group='selection'
    winner=meter.call('selection',lambda:min(range(UPDATES+1),key=lambda k:(candidates[k]['cost'],k)))
    meter.emit('winner_frozen_before_replay',iteration=winner,candidate=candidates[winner])
    meter.group='winner_triple_replay';fresh=gn.verified(task,candidates[winner]['logits'],backend,meter)
    require(compact(fresh)=={k:candidates[winner][k] for k in compact(fresh)},'winner fresh fullH2 triple exact reproduction')
    base.exact_counts(meter.counts(),budget())
    return dict(task_id=core.task_id(task),updates=UPDATES,step_size=STEP_SIZE,candidates=candidates,gradients=gradients,
                winner_iteration=winner,final_prediction=fresh,resources=meter.counts())


def choice(task,result):
    require(result['task_id']==core.task_id(task) and result['updates']==UPDATES and result['step_size']==STEP_SIZE,'own fixed gradient configuration')
    base.exact_counts(result['resources'],budget());bank=result['candidates'];ds=result['gradients']
    require(len(bank)==UPDATES+1 and len(ds)==UPDATES,'all33 iterates and32 scalar gradients')
    previous=None
    for it,c in enumerate(bank):
        x=core.array(c['logits'],(2,10));terminal=core.array(c['terminal'],(192,))
        require(c['iteration']==it and np.array_equal(x,x.astype(np.float32)) and
            (np.linalg.norm(x,axis=1)<=task['packet_logit_cap']).all() and
            c['commands']==np.tanh(x.astype(np.float32)).reshape(10,2).astype(float).tolist() and
            c['cost']==core.model_cost(task,terminal),'every own-Q iterate/logit/raw command')
        if it==0:require(not x.any(),'single zero initialization')
        else:
            d=ds[it-1]
            incumbent=dict(task_id=result['task_id'],**previous,contexts=d['contexts'])
            # Contexts are checked against actual per-packet journal at release.
            require(d['contexts'][-1]['latent_history'][-1]==previous['terminal'],'gradient incumbent terminal')
            check_gradient(task,incumbent,d)
            require(np.array_equal(x,proposal(previous['logits'],d['objective_gradient'],task['packet_logit_cap'])),'fixed projected descent recurrence')
        previous=c
    index=min(range(UPDATES+1),key=lambda k:(bank[k]['cost'],k));p=result['final_prediction'];core.check_prediction(task,p)
    require(result['winner_iteration']==index and compact(p)=={k:bank[index][k] for k in compact(p)},'stable own model winner exact replay')
    return dict(task_id=core.task_id(task),logits=p['logits'],commands=p['commands'],prediction=p['contexts'][-1],
                model_terminal_cost=p['cost'],winner_iteration=index)


def execute(task,model,device,torch,sink,row):
    meter=Meter(sink,lambda:torch.cuda.synchronize(device));gate=GradGate(torch,meter)
    try:
        backend=NativeGradient(model,device,task)
        with gate,backend.guard(meter):return optimize(task,backend,meter)
    finally:row['resources']=meter.counts();row['AD_API_resources']=gate.counts()


def model_plan(slot,task,selected):
    plan=dict(schema=NAME+'.model_plan',source_slot=slot['source_slot'],arm=ARM,blind_sha256=slot['blind_sha256'],
        task_export=slot['task_export'],task_sha256=slot['task_sha256'],task_contract_sha256=task['task_contract_sha256'],
        with_target=False,physical_execution=False,selection_uses_physics=False,P1_admission=False,**selected)
    return dict(plan,selection_sha256=core.digest(plan))


def plan_all(p,store,runner,rows):
    for slot,row in zip(p['slots'],rows,strict=True):
        t=base.load_export(slot['task_export'],slot['task_sha256'])['gn_task'];row['status']='attempted'
        def sink(e):store.event(dict(source_slot=slot['source_slot'],arm=ARM,task_id=slot['task_id'],event=e))
        sink(dict(kind='arm_started',task_sha256=slot['task_sha256'],updates=UPDATES,step_size=STEP_SIZE))
        result=runner(t,sink,row);selected=choice(t,result)
        base.exact_counts(row['resources'],budget());base.exact_counts(row['AD_API_resources'],dict(grad=UPDATES))
        stem=f'root_{slot["source_slot"]}_{ARM}';store.put(stem+'.result.json.gz',result)
        plan=model_plan(slot,t,selected);store.put(stem+'.plan.json',plan)
        sink(dict(kind='model_selection_sealed',selection_sha256=plan['selection_sha256'],commands=plan['commands']))
        row.update(status='model_plan_sealed',commands=plan['commands'],result_file=stem+'.result.json.gz',
                   plan_file=stem+'.plan.json',selection_sha256=plan['selection_sha256'])


def run(prepared,expected,output):
    p=verify(prepared,expected);store=base.Store(output);issues=[];model=None;before=after=host=None;source_ok=False
    rows=[dict(source_slot=s['source_slot'],arm=ARM,blind_sha256=s['blind_sha256'],task_id=s['task_id'],
        status='not_started',commands=None,resources={},AD_API_resources={}) for s in p['slots']]
    try:
        print('Fixed zero/32-step eta1 projected gradient, scalar meanQ backward; NOT full-resource fair or closed-loop efficacy.',flush=True)
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        enc.frozen(model);before=enc.tensor_hash(model);require(before==p['expected_model_hash'],'original model tensor SHA')
        with sdpa_kernel(SDPBackend.MATH):plan_all(p,store,lambda t,s,r:execute(t,model,device,torch,s,r),rows)
    except BaseException as exc:
        issues.append(enc.error('execution',exc))
        try:store.event(dict(kind='execution_error',error=issues[0]))
        except BaseException as e:issues.append(enc.error('error_journal',e))
    finally:
        for row in rows:
            if row['status']=='attempted':row['status']='failed_partial'
        if model is not None:
            try:enc.frozen(model);after=enc.tensor_hash(model)
            except BaseException as e:issues.append(enc.error('final_model',e))
        try:verify(prepared,expected);source_ok=True
        except BaseException as e:issues.append(enc.error('final_source',e))
        try:store.close()
        except BaseException as e:issues.append(enc.error('journal_finalize',e))
        complete=not issues and source_ok and before is not None and before==after and all(r['status']=='model_plan_sealed' for r in rows)
        hashes={}
        try:files=list(store.output.iterdir())
        except BaseException as e:issues.append(enc.error('artifact_listing',e));complete=False;files=[]
        for f in files:
            try:
                if f.is_file():hashes[f.name]=sha(f)
            except BaseException as e:issues.append(enc.error('artifact_hash',e));complete=False
        resources={k:dict(attempted=sum(r['resources'].get(k,{}).get('attempted',0) for r in rows),
                          completed=sum(r['resources'].get(k,{}).get('completed',0) for r in rows)) for k in budget()}
        report=dict(schema=NAME+'.report',status='gradient_mpc_model_plans_sealed_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,rows=rows,resources=resources,
            AD_API_resources=dict(grad=dict(attempted=sum(r['AD_API_resources'].get('grad',{}).get('attempted',0) for r in rows),
                completed=sum(r['AD_API_resources'].get('grad',{}).get('completed',0) for r in rows))),
            errors=issues,first_error=issues[0] if issues else None,artifact_sha256=hashes,source_verified=source_ok,
            model_hash_before=before,model_hash_after=after,model_unchanged=before==after if before is not None else None,
            host=host,contract=p['contract'],commands_released=complete,new_images=0,new_physics=0,training=False,new_Q=False,
            P1_admission=False,equal_total_resource_claim=False,global_freshness=p['global_freshness'])
        try:store.put('report.json',report,final=True)
        except BaseException as e:
            issues.append(enc.error('final_report',e));complete=False;print(canonical(dict(first_error=issues[0],errors=issues)).decode(),flush=True)
    require(complete and not issues,'partial gradient MPC saved; no retry: '+str(issues));return report


def journal_check(task,result,events):
    """Own command-release gate; no separate audit output or neural execution."""
    pending=None;attempts=[];returns=[];active=None;paths=[];evaluations=[];ad_forward=[];grads=[];api=[];api_done=[]
    forward_packets=[];forward=None;frozen=[];winner=None
    for e in events:
        kind=e['kind'];require('forbidden' not in kind and 'error' not in e,'no failed/forbidden journal')
        if kind=='operation_attempted':
            v=e['event'];require(pending is None and v['index']==len(attempts) and v['completed'] is False and
                v['evaluation_id']==e['evaluation_id'] and v['samples']==1 and v['phase']=='direct','ordered nonrefundable attempt')
            attempts.append(v);pending=v
        elif kind=='operation_completed':
            v=e['event'];require(pending is not None and v==dict(pending,completed=True),'unique native/math return')
            returns.append(v);pending=None
        elif kind=='AD_API_attempt':
            v=e['entry'];require(pending is not None and pending['resource']=='backward' and v['index']==len(api) and
                v==dict(index=len(api),resource='grad',evaluation_id=pending['evaluation_id'],completed=False,
                    scalar_outputs=1,input_shape=[10,2],retain_graph=False,create_graph=False),'actual scalar grad inside backward receipt')
            api.append(v)
        elif kind=='AD_API_completed':
            v=e['entry'];require(len(api)>len(api_done) and v==dict(api[-1],completed=True),'actual grad return');api_done.append(v)
        elif kind=='evaluation_request':
            require(active is None,'no overlapping fullH2 evaluation');active=e;paths=[]
        elif kind=='native_packet':
            require(active is not None and e['evaluation_id']==active['evaluation_id'] and e['mode']==active['mode'] and
                e['packet']==len(paths),'native path scope/order');paths.append(e['context'])
        elif kind=='evaluation_complete':
            require(active is not None and len(paths)==2 and e['evaluation_id']==active['evaluation_id'] and e['mode']==active['mode'],'fullH2 before cost')
            gn.saved.check_path(task,np.asarray(active['commands']),paths)
            p=dict(task_id=core.task_id(task),logits=active['logits'],commands=active['commands'],contexts=deepcopy(paths),
                   cost=core.model_cost(task,paths[-1]['latent_history'][-1]))
            require(e['prediction']==p,'raw paths/cost complete content');evaluations.append(dict(request=active,prediction=p));active=None
        elif kind=='gradient_request':
            require(forward is None and len(evaluations)==len(grads)+1 and e['iteration']==len(grads)+1,'fresh gradient after own iterate')
            incumbent=evaluations[-1]['prediction'];require(e['logits']==incumbent['logits'] and e['commands']==incumbent['commands'],'gradient request at actual last iterate')
            forward_packets=[]
        elif kind=='gradient_forward_packet':
            require(e['packet']==len(forward_packets),'both AD packets ordered');forward_packets.append(e['context'])
        elif kind=='gradient_forward_complete':
            forward=e['forward'];require(len(forward_packets)==2 and forward['contexts']==forward_packets==evaluations[-1]['prediction']['contexts'],'AD path equals real incumbent path')
            ad_forward.append(forward)
        elif kind=='gradient_complete':
            d=e['gradient'];require(forward is not None and {k:d[k] for k in forward}==forward and pending is None and
                len(api)==len(api_done)==len(grads)+1 and api[-1]['evaluation_id']==e['evaluation_id'],
                'saved gradient binds fresh native forward and completed backward')
            check_gradient(task,evaluations[-1]['prediction'],d);grads.append(d);forward=None
        elif kind=='proposal_frozen':
            require(len(grads)==len(frozen)+1 and e['iteration']==len(grads) and e['step_size']==STEP_SIZE and
                e['parent_logits']==evaluations[-1]['prediction']['logits'] and
                e['logits']==proposal(e['parent_logits'],grads[-1]['objective_gradient'],task['packet_logit_cap']).tolist(),'projected step before new evaluation')
            frozen.append(e)
        elif kind=='winner_frozen_before_replay':
            require(winner is None and len(evaluations)==UPDATES+1 and len(frozen)==UPDATES,'all fixed updates before selection')
            winner=e
    require(active is None and pending is None and forward is None and len(grads)==len(ad_forward)==len(frozen)==UPDATES and
        len(evaluations)==UPDATES+4 and len(api)==len(api_done)==UPDATES and len(attempts)==len(returns)==sum(budget().values()),'complete unfiltered work/path census')
    base.exact_counts({k:dict(attempted=sum(v['resource']==k for v in attempts),completed=sum(v['resource']==k for v in returns)) for k in budget()},budget())
    expected_groups={f'iterate_{it}':dict(predict=2,action_encoder=2) for it in range(UPDATES+1)}
    expected_groups.update({f'gradient_{it}':dict(predict=2,action_encoder=2,backward=1,proposal=1) for it in range(1,UPDATES+1)})
    expected_groups.update(selection=dict(selection=1),winner_triple_replay=dict(predict=6,action_encoder=6))
    require(set(v['evaluation_id'] for v in attempts)==set(expected_groups),'no unscoped/reused evaluation operations')
    for group,wanted in expected_groups.items():
        group_counts={k:sum(v['resource']==k and v['evaluation_id']==group for v in attempts) for k in budget()}
        require(group_counts=={k:wanted.get(k,0) for k in budget()},'exact per-evaluation native and reverse work')
    require(grads==result['gradients'],'all raw gradients preserved')
    bank=result['candidates']
    require([e['candidate'] for e in events if e['kind']=='iterate_saved']==bank,'all durable iterates')
    for it,e in enumerate(evaluations[:UPDATES+1]):
        require(e['request']['evaluation_id']==f'iterate_{it}' and e['request']['mode']=='whole_replay' and
            compact(e['prediction'])=={k:bank[it][k] for k in compact(e['prediction'])},'all actual iterate scores')
        if it:require(e['request']['logits']==frozen[it-1]['logits'],'dispatch equals prior frozen proposal')
    index=result['winner_iteration']
    require(winner is not None and winner['iteration']==index and winner['candidate']==bank[index],'model-only selected iterate seal')
    for e,mode in zip(evaluations[-3:],('nominal','whole_replay','segmented_replay'),strict=True):
        require(e['request']['evaluation_id']=='winner_triple_replay' and e['request']['mode']==mode and
            e['prediction']['contexts']==result['final_prediction']['contexts'] and compact(e['prediction'])==compact(result['final_prediction']),'all three independent fullH2 winner paths')


def consume(output,expected):
    output=Path(output).absolute();core.hash_value(expected);require(sha(output/'report.json')==expected,'external actual report SHA')
    r=read(output/'report.json')
    require(r['schema']==NAME+'.report' and r['status']=='gradient_mpc_model_plans_sealed_not_admitted' and
        r['commands_released'] is True and r['errors']==[] and r['first_error'] is None and r['source_verified'] is True and
        r['model_unchanged'] is True and r['model_hash_before']==r['model_hash_after']==enc.MODEL_TENSOR_SHA,'whole own CUDA closure only')
    require(all(r[k] is False for k in ('P1_admission','new_Q','training','equal_total_resource_claim')) and
        r['new_images']==r['new_physics']==0,'no scope or cost upgrade')
    p=verify(r['prepared'],r['prepare_sha256']);require(r['contract']==p['contract'] and len(r['rows'])==2,'two own fixed branches')
    names={'binding.json','events.jsonl.gz'}|{f'root_{s}_{ARM}.{ext}' for s in (0,1) for ext in ('result.json.gz','plan.json')}
    require(set(r['artifact_sha256'])==names and {f.name for f in output.iterdir()}==names|{'report.json'},'exact own artifact census')
    for name,h in r['artifact_sha256'].items():require(sha(output/name)==h,'saved artifact hash')
    require(sum(f.stat().st_size for f in output.iterdir())<=CAP,'whole-output bound')
    require(read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'binding/prepare/report identity')
    raw,eof=base.inflate((output/'events.jsonl.gz').read_bytes());require(eof and raw.endswith(b'\n'),'complete single gzip journal')
    events=[base.decode(line) for line in raw.splitlines()]
    require(all(set(e)=={'source_slot','arm','task_id','event'} and e['source_slot'] in (0,1) and e['arm']==ARM for e in events),'only own branch scopes')
    require([e['source_slot'] for k,e in enumerate(events) if k==0 or e['source_slot']!=events[k-1]['source_slot']]==[0,1],'causal ordered two-source census')
    plans=[]
    for slot,row in zip(p['slots'],r['rows'],strict=True):
        s=slot['source_slot'];stem=f'root_{s}_{ARM}'
        require(row['source_slot']==s and row['arm']==ARM and row['task_id']==slot['task_id'] and row['blind_sha256']==slot['blind_sha256'] and
            row['status']=='model_plan_sealed' and row['result_file']==stem+'.result.json.gz' and row['plan_file']==stem+'.plan.json','own slot/arm/task row')
        base.exact_counts(row['resources'],budget());base.exact_counts(row['AD_API_resources'],dict(grad=UPDATES))
        t=base.load_export(slot['task_export'],slot['task_sha256'])['gn_task'];result=read(output/row['result_file']);selected=choice(t,result)
        plan=read(output/row['plan_file']);want=model_plan(slot,t,selected)
        require(plan==want and plan['selection_sha256']==row['selection_sha256'] and row['commands']==selected['commands'],'own commands/plan/model selection hash')
        group=[e['event'] for e in events if e['source_slot']==s]
        require(all(e['task_id']==slot['task_id'] for e in events if e['source_slot']==s) and
            group[0]==dict(kind='arm_started',task_sha256=slot['task_sha256'],updates=UPDATES,step_size=STEP_SIZE) and
            group[-1]==dict(kind='model_selection_sealed',selection_sha256=plan['selection_sha256'],commands=plan['commands']),'durable own lifecycle')
        journal_check(t,result,group);plans.append(dict(plan,whole_run_report_sha256=expected))
    base.exact_counts(r['resources'],p['contract']['total_budget']);base.exact_counts(r['AD_API_resources'],dict(grad=2*UPDATES))
    require(sha(output/'report.json')==expected,'final report TOCTOU')
    for name,h in {**p['bindings_sha256'],**{str(output/n):h for n,h in r['artifact_sha256'].items()}}.items():require(sha(name)==h,'final source/artifact drift')
    return dict(schema=NAME+'.consume',status='own_gradient_model_commands_verified_not_admitted',rows=plans,new_physics=0,P1_admission=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','verify','run','consume'))
    for key in ('prepared','prepare-sha','output','report-sha'):p.add_argument('--'+key)
    a=p.parse_args()
    if a.command=='prepare':result=prepare(a.output)
    elif a.command=='verify':result=dict(status='verified_no_execution',roots=len(verify(a.prepared,a.prepare_sha)['slots']))
    elif a.command=='consume':result=consume(a.output,a.report_sha)
    else:result=run(a.prepared,a.prepare_sha,a.output)
    print(canonical(result).decode().strip())


if __name__=='__main__':main()
