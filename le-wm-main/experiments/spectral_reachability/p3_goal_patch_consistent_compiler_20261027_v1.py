"""Q-consistent, fresh H2 spectral/ridge compiler prerequisite; MAIN-only CUDA.

Not v4 random search, not P3 dynamic scheduling, not a physical experiment.
Prepare/verify import neither torch nor a simulator. No CPU neural fallback.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import os
import numpy as np
from experiments.spectral_reachability import p3_horizon_residual_budget_20261027_v4 as old
from experiments.spectral_reachability import p3p4_goal_patch_metric_20261027_v1 as metric
from experiments.spectral_reachability import pusht_full_packet_compiler_20261027 as compiler
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal

NAME='p3_goal_patch_consistent_compiler_20261027_v1'
ARMS=('old_Q','goal_patch_Q')
PROJECT=Path(__file__).resolve().parents[2]
METRIC_PREPARED='results/p3p4_goal_patch_metric_20261027_v1_prepared01'
METRIC_PREPARE_SHA='c1447c9b0170a38af180498cfea2d0ad31cabcf720d9da23dc1dcc1068aa6de8'
METRIC_REPORT='results/p3p4_goal_patch_metric_20261027_v1_cuda_main01/report.json'
METRIC_REPORT_SHA='7b38b91e18497b625f36e7c52d59c07e53204e6047af083c10b4c4631aa2fee6'
ARRAY_REPORT='results/p3p4_goal_patch_metric_array_audit_20261027_v1_main_v1/report.json'
ARRAY_REPORT_SHA='7349c84cf5f1f87c153dd5307bbb744093b200a888ade28e1e79847703c74a16'
io,require,write,digest=old.io,old.require,old.write,old.digest


def contract():
    return dict(component='H2 full10 prefix/suffix SRA prerequisite, not complete P3 dynamic',
        dynamic_scheduler_implemented=False,horizon=2,steps=10,candidates_per_arm=10,
        coordinates='each packet: 10 time-major logits; raw=tanh(logits)',
        prefix='fresh H1 root Jacobian; compiler.spectral_bank(QJ,Q(goal0-nominal))',
        suffix='fresh conditional H1 Jacobian for EVERY prefix; compiler.bounded_ridge(QJ,Q(goal0-nominal))',
        damping=compiler.DAMPING,cap_per_packet=compiler.CAP,radii=list(compiler.RADII),
        parameter_authority='unchanged full-packet compiler constants/bank/order/objective; no outcome-based tuning',
        selection='stage_cost + own_reach_value; stable bank index tie, as full-packet compile_arm',
        score='mean((Q@(z-goal0))**2); same Q for bank, ridge, stage, terminal and verification',
        nonlinear_validation='every prefix+optimized suffix independently replayed; winner whole+segmented replay',
        optimization='exact trust-ball linearized ridge solve, NOT iterative nonlinear optimization',
        arm_predict_cap=66,arm_action_encoder_cap=66,pair_predict_cap=132,pair_action_encoder_cap=132,
        four_root_predict_cap=528,four_root_action_encoder_cap=528,
        budget_detail='root nominal+ADforward=2; 10*(prefix+suffix nominal+ADforward+optimized suffix+H2 replay)=60; winner replay4',
        jacobians_per_arm=11,svds_per_arm=12,mathematical_vjp_directions_per_arm=2112,
        backward_engine_api_count=None,visual_encoder_samples=0,
        inherited_metric_cost='80 encoder + 80 existing-projector forwards + 8 metric SVDs; paid historically, not zero',
        inherited_context_cost='unknown; not full-pipeline compute equivalence',
        failed_attempts_charged=True,duplicate_actions_retained=True,shared_derivatives=False,
        no_padding_no_refunds=True,total_flops_equivalent=False,new_independent_roots=0,
        P1_admission=False,P3_admission=False,P4_admission=False,physical_quality_measured=False)


def validate_q(q):
    q=compiler.finite(q,(192,192))
    require(np.allclose(q,q.T,atol=1e-8,rtol=0) and np.allclose(q@q,q,atol=1e-8,rtol=0)
        and np.trace(q)>.5,'nonzero native orthogonal projector required; no fallback')
    return q


def cost(z,goal,q):
    residual=validate_q(q)@(compiler.finite(z,(192,))-compiler.finite(goal,(192,)))
    return float(np.mean(residual**2))


def make_bank(jacobian,nominal,goal,q):
    return compiler.spectral_bank(compiler.finite(jacobian,(192,10)),
        compiler.finite(nominal,(192,)),compiler.finite(goal,(192,)),validate_q(q))


def metric_id(packet,q,model_bindings,metric_source):
    return digest(dict(root_sha256=packet['root_sha256'],goal=packet['goal'],
        projection=validate_q(q).tolist(),model_bindings=model_bindings,metric_source=metric_source))


def with_svd_receipts(callback,operation):
    """Single-threaded scoped instrumentation of actual helper SVD invocations.

    No surrogate analytical count: failures between the bank's two SVDs retain
    exactly the calls attempted. Restored even on BaseException. No concurrency.
    """
    original=np.linalg.svd
    def counted(*args,**kwargs):
        return operation('svd',lambda:original(*args,**kwargs))
    np.linalg.svd=counted
    try:return callback()
    finally:np.linalg.svd=original


def compile_arm(packet,q,backend,journal):
    """Backend-independent actual full10 H2 compiler; no inherited candidates.

    Production backend must implement fresh nominal_jacobian/step/replay.
    Analytic test backends validate contracts only, never scientific evidence.
    """
    q=validate_q(q);goal=compiler.finite(packet['goal'],(192,))
    qsha=digest(q.tolist());initial=deepcopy(packet['root_context']);operations=[]
    def operation(kind,callback):
        event=dict(kind=kind,index=len(operations),completed=False,q_sha256=qsha)
        if kind=='jacobian':event.update(shape=[192,10],mathematical_vjp_directions=192)
        operations.append(event);journal(deepcopy(event))
        try:
            result=callback();event['completed']=True
            return result
        except BaseException as exc:
            event['error']=type(exc).__name__+': '+str(exc);raise
        finally:journal(deepcopy(event))
    def replay(context,commands,phase):
        event=dict(kind='replay',phase=phase,commands=np.asarray(commands).tolist(),q_sha256=qsha,completed=False)
        journal(deepcopy(event))
        try:
            context=backend.replay(context,commands);event.update(completed=True,context=context)
            return context
        except BaseException as exc:event['error']=type(exc).__name__+': '+str(exc);raise
        finally:journal(deepcopy(event))
    nominal,jacobian=backend.nominal_jacobian(deepcopy(initial),operation)
    bank=with_svd_receipts(lambda:make_bank(jacobian,nominal,goal,q),operation)
    journal(dict(kind='bank',q_sha256=qsha,bank=bank,root_nominal=nominal.tolist(),root_jacobian=jacobian.tolist()))
    candidates=[]
    for index,params in enumerate(bank['params']):
        prefix=np.tanh(np.asarray(params,dtype=np.float32)).reshape(5,2)
        event=dict(kind='candidate',index=index,q_sha256=qsha,completed=False,prefix_commands=prefix.tolist())
        journal(deepcopy(event))
        try:
            after=backend.step(deepcopy(initial),prefix)
            n,j=backend.nominal_jacobian(deepcopy(after),operation)
            event.update(predicted_prefix=after['latent_history'][-1],nominal_terminal=n.tolist(),jacobian=j.tolist())
            journal(deepcopy(event))
            solve=with_svd_receipts(lambda:compiler.bounded_ridge(q@j,q@(goal-n)),operation)
            suffix=np.tanh(np.asarray(solve['params'],dtype=np.float32)).reshape(5,2)
            commands=np.concatenate((prefix,suffix))
            event.update(suffix_solve=solve,suffix_params=solve['params'],suffix_commands=suffix.tolist(),commands=commands.tolist())
            journal(deepcopy(event))
            predicted=backend.step(deepcopy(after),suffix)
            event.update(predicted_terminal=predicted['latent_history'][-1]);journal(deepcopy(event))
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
        finally:journal(deepcopy(event))
    winner=deepcopy(min(candidates,key=lambda c:(c['objective'],c['index'])))
    whole=replay(deepcopy(initial),winner['commands'],'winner_whole')
    segmented=deepcopy(initial)
    for k,packet_actions in enumerate(np.asarray(winner['commands']).reshape(2,5,2)):
        segmented=replay(segmented,packet_actions,'winner_segment_'+str(k))
    require(io.canonical(whole)==io.canonical(segmented)==io.canonical(winner['context']),'winner double replay mismatch')
    return dict(status='completed',q_sha256=qsha,bank=bank,root_jacobian=jacobian.tolist(),root_nominal=nominal.tolist(),
        candidates=candidates,winner=winner,whole_replay_context=whole,segmented_replay_context=segmented,
        whole_and_segmented_exact=True,operations=operations)


class TorchBackend:
    """CUDA-only transition/AD backend; all contexts retain original 3/2 layout."""
    def __init__(self,model,device):
        require(getattr(device,'type',None)=='cuda','CUDA only, no neural CPU')
        from experiments.spectral_reachability.pusht_composition_cuda_replay_v2 import CudaReplay
        self.device=device;self.transition=compiler.make_transition(model,device);self._replay=CudaReplay(model,device)

    def tensors(self,context):
        import torch
        return (torch.tensor(context['latent_history'],device=self.device,dtype=torch.float32)[None],
            torch.tensor(context['action_history'],device=self.device,dtype=torch.float32)[None])

    def step(self,context,actions):
        import torch
        emb,history=self.tensors(context)
        with torch.no_grad():
            after,between=self.transition(emb,history,torch.tensor(actions,device=self.device,dtype=torch.float32))
        return dict(context,latent_history=after[0].cpu().tolist(),action_history=between[0].cpu().tolist())

    def replay(self,context,actions):return self._replay(context,actions)

    def nominal_jacobian(self,context,operation):
        import torch
        emb,history=self.tensors(context)
        def terminal(u):return self.transition(emb,history,torch.tanh(u).reshape(5,2))[0][0,-1]
        zero=emb.new_zeros(10)
        with torch.no_grad():nominal=terminal(zero)
        jac=operation('jacobian',lambda:torch.autograd.functional.jacobian(terminal,zero.requires_grad_(True),vectorize=True))
        return nominal.detach().cpu().numpy().astype(float),jac.detach().cpu().numpy().astype(float)


class NeuralCap(old.NeuralCap):
    """NEW H2 caps; inherit only source-sealed patch/restore/scope machinery."""
    def dispatch(self,kind,callback,args,kwargs):
        require(kind in ('predict','action_encoder') and self.role in ARMS,'unscoped/unknown neural entry')
        x=args[0]
        require(getattr(getattr(x,'device',None),'type',None)=='cuda' and tuple(x.shape)==
            ((1,3,192) if kind=='predict' else (1,3,10)),'single full-context CUDA input')
        count=lambda role:sum(e['kind']==kind and (role is None or e['role']==role) for e in self.events)
        if count(None)>=132 or count(self.role)>=66:
            self.blocked+=1;self.journal(dict(kind='blocked',entry=kind,role=self.role))
            raise ValueError('nonrefundable neural cap exceeded before dispatch')
        e=dict(kind=kind,role=self.role,context_tokens=3,completed=False,index=len(self.events))
        self.events.append(e);self.journal(deepcopy(e))
        try:result=callback(*args,**kwargs);e['completed']=True;return result
        except BaseException as exc:e['error']=type(exc).__name__+': '+str(exc);raise
        finally:self.journal(deepcopy(e))


def source_inputs(root):
    """Exact sealed goal-only metrics + old context; never extract old outcomes."""
    root=Path(root).resolve()
    mp=metric.validate_prepared(root/METRIC_PREPARED,METRIC_PREPARE_SHA)
    previous,_=old.validate_prepared(root/metric.PREPARED,metric.PREPARE_SHA)
    bindings={str(root/METRIC_PREPARED/'prepare.json'):METRIC_PREPARE_SHA,
        str(root/METRIC_REPORT):METRIC_REPORT_SHA,str(root/ARRAY_REPORT):ARRAY_REPORT_SHA}
    for path,h in bindings.items():require(io.sha(Path(path).read_bytes())==h,'sealed input bytes: '+path)
    report=io.decode((root/METRIC_REPORT).read_bytes());audit=io.decode((root/ARRAY_REPORT).read_bytes())
    require(report['schema']==metric.NAME+'.report' and report['status']=='model_only_metric_compiled' and
        report['model_unchanged'] is True and report['failure'] is None and report['preparation_sha256']==METRIC_PREPARE_SHA,'sealed metric run')
    require(audit['status']=='saved_goal_patch_metric_consistent' and audit['all_checks'] is True and
        audit['compiler_report_sha256']==METRIC_REPORT_SHA,'independent metric array audit')
    require(mp['packets']==previous['packets'] and mp['model_bindings']==previous['model_bindings'],'same model and original inputs')
    require(len(previous['roots'])==4 and len(set(previous['roots']))==4,'four distinct calibration roots')
    indexed=io.index(report['rows'],('root_sha256',))
    require(set(indexed)=={(r,) for r in previous['roots']},'exact four-root metric census')
    packets=[]
    for i,p in enumerate(previous['packets']):
        row=indexed[p['root_sha256'],]
        require(p['root_sha256']==previous['roots'][i] and row['root_index']==i and
            np.array_equal(row['goal_vectors'][0],p['goal']),'same root/order/unchanged original goal0')
        new={k:deepcopy(p[k]) for k in ('root_sha256','root_context','goal')}
        new['source_input_sha256']=p['input_sha256']
        new['projections']={'old_Q':p['projection'],'goal_patch_Q':row['metric']['projection']}
        for arm,label in zip(ARMS,('old_Q','new_Q'),strict=True):
            q=validate_q(new['projections'][arm])
            path=root/Path(ARRAY_REPORT).parent/f'root_{i}.{label}.npy'
            require(str(path) in audit['artifacts'],'array audit artifact binding')
            h=audit['artifacts'][str(path)];require(io.sha(path.read_bytes())==h,'audited Q bytes')
            require(np.allclose(np.load(path,allow_pickle=False),q,atol=1e-10,rtol=1e-8),'sealed Q audit alignment')
            bindings[str(path)]=h
        new['active_metric_ids']={arm:metric_id(new,new['projections'][arm],previous['model_bindings'],
            dict(metric_report_sha256=METRIC_REPORT_SHA,metric_prepare_sha256=METRIC_PREPARE_SHA,
                 catalog_sha256=digest(row['catalog']),array_audit_report_sha256=ARRAY_REPORT_SHA,arm=arm)) for arm in ARMS}
        new['input_sha256']=digest(new);packets.append(new)
    return dict(roots=previous['roots'],packets=packets,model_bindings=previous['model_bindings'],bindings=bindings)


def prepare(root,output):
    root=Path(root).resolve();output=Path(output).absolute();require(not output.exists(),'exclusive NEW prepare path')
    inputs=source_inputs(root)
    sources=old.prior.physical.source_closure(root,[Path(__file__).resolve(),root/'tests'/('test_'+NAME+'.py'),root/'innerdocs'/(NAME+'.md')])
    body=dict(schema=NAME+'.protocol',contract=contract(),project_root=str(root),**inputs,
        sources_sha256={str(p):io.sha(p.read_bytes()) for p in sources},
        arm_order=[list(ARMS if i%2==0 else ARMS[::-1]) for i in range(4)],P3_admission=False)
    identity=digest(body);output.mkdir(exist_ok=False)
    write(output/'protocol.json',dict(protocol=body,protocol_id=identity))
    seals=[create_seal(files=sources[i:i+40],args_json=io.canonical({'protocol_id':identity}),output_dir=output/f'source_seal_{i//40}')
        for i in range(0,len(sources),40)]
    receipt=dict(status='prepared_not_executed',protocol_id=identity,protocol_sha256=io.sha((output/'protocol.json').read_bytes()),
        source_seals=seals,new_neural_calls=0,new_physical_steps=0,P3_admission=False)
    write(output/'prepare.json',receipt)
    validate_prepared(output,io.sha((output/'prepare.json').read_bytes()))
    return receipt


def validate_prepared(prepared,expected_sha):
    prepared=Path(prepared).absolute();raw=(prepared/'prepare.json').read_bytes()
    require(io.sha(raw)==expected_sha,'prepare SHA');receipt=io.decode(raw)
    raw=(prepared/'protocol.json').read_bytes();require(io.sha(raw)==receipt['protocol_sha256'],'protocol bytes')
    envelope=io.decode(raw);p=envelope['protocol'];identity=envelope['protocol_id']
    require(digest(p)==identity==receipt['protocol_id'] and p['schema']==NAME+'.protocol','protocol identity')
    require(p['contract']==contract() and p['P3_admission'] is False,'frozen contract')
    require(p['arm_order']==[list(ARMS if i%2==0 else ARMS[::-1]) for i in range(4)],'fixed arm order')
    for key,value in source_inputs(p['project_root']).items():require(p[key]==value,'exact sealed source input: '+key)
    for path,h in p['sources_sha256'].items():require(io.sha(Path(path).read_bytes())==h,'source drift: '+path)
    for seal in receipt['source_seals']:verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
    return p,identity


def accounting_ok(result,receipt,arm):
    ops=result['operations'];events=[e for e in receipt['events'] if e['role']==arm]
    return (result['status']=='completed' and len(result['candidates'])==10 and result['whole_and_segmented_exact'] is True
        and all(e['completed'] for e in events+ops) and not receipt['blocked']
        and all(sum(e['kind']==k for e in events)==66 for k in ('predict','action_encoder'))
        and sum(e['kind']=='jacobian' for e in ops)==11 and sum(e['kind']=='svd' for e in ops)==12)


def validate_compilation(packet,q,result):
    """Saved-array verification only: 12 explicitly counted helper SVDs per arm.

    Validates algebra, all candidate actions/costs, recorded exact replay and
    selection; does not regenerate neural outputs or certify a physical result.
    """
    q=validate_q(q);goal=compiler.finite(packet['goal'],(192,));qsha=digest(q.tolist())
    require(result['status']=='completed' and result['q_sha256']==qsha,'complete same-Q compiler')
    bank=make_bank(result['root_jacobian'],result['root_nominal'],goal,q)
    require(bank==result['bank'] and len(result['candidates'])==10,'exact fresh spectral bank and full census')
    ops=result['operations'];expected=['jacobian','svd','svd']+['jacobian','svd']*10
    require(len(ops)==23 and all(e['index']==i and e['kind']==kind and e['completed'] is True and e['q_sha256']==qsha
        for i,(e,kind) in enumerate(zip(ops,expected,strict=True))),'completed fresh derivative/SVD ledger')
    require(all(e['shape']==[192,10] and e['mathematical_vjp_directions']==192 for e in ops if e['kind']=='jacobian'),'AD dimensions/directions')
    for index,c in enumerate(result['candidates']):
        require(c['index']==index and c['candidate_id']==f'prefix_{index}' and c['q_sha256']==qsha and
            c['completed'] is True and c['independent_model_replay_exact'] is True,'candidate identity/complete nonlinear replay')
        prefix=np.tanh(np.asarray(bank['params'][index],dtype=np.float32)).reshape(5,2)
        j=compiler.finite(c['jacobian'],(192,10));n=compiler.finite(c['nominal_terminal'],(192,))
        solve=compiler.bounded_ridge(q@j,q@(goal-n))
        suffix=np.tanh(np.asarray(solve['params'],dtype=np.float32)).reshape(5,2)
        require(c['prefix_params']==bank['params'][index] and c['proposal_kind']==bank['kinds'][index] and
            c['suffix_solve']==solve and c['suffix_params']==solve['params'],'Q-specific bank and conditional optimized solve')
        require(np.array_equal(c['prefix_commands'],prefix) and np.array_equal(c['suffix_commands'],suffix) and
            np.array_equal(c['commands'],np.concatenate((prefix,suffix))),'exact raw actions, no clipping/relabeling')
        stage=cost(c['predicted_prefix'],goal,q);terminal=cost(c['predicted_terminal'],goal,q)
        require(c['stage_cost']==stage and c['nonlinear_terminal_cost']==terminal and
            c['own_reach_value']==solve['own_reach_value'] and c['objective']==stage+solve['own_reach_value'],'same-Q stage/reach/terminal')
        z=compiler.finite(c['context']['latent_history'],(3,192))
        compiler.finite(c['context']['action_history'],(2,10))
        require(np.array_equal(z[0],packet['root_context']['latent_history'][-1]) and
            np.array_equal(z[1],c['predicted_prefix']) and np.array_equal(z[2],c['predicted_terminal']),'full nonlinear context chronology')
    winner=min(result['candidates'],key=lambda c:(c['objective'],c['index']))
    require(winner==result['winner'] and result['whole_and_segmented_exact'] is True and
        result['whole_replay_context']==result['segmented_replay_context']==winner['context'],'stable winner/full-context double replay')
    return winner


def execute_cuda(prepared,expected_sha,output):
    """Main host execution only. Preserve eight slots and fsync every attempt."""
    body,identity=validate_prepared(prepared,expected_sha);output=Path(output).absolute()
    require(not output.exists(),'exclusive NEW CUDA output')
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    import torch
    for path,h in body['model_bindings'].items():require(checked.digest(path)==h,'frozen model bytes')
    policy=Path(next(p for p in body['model_bindings'] if Path(p).name=='weights.pt')).parent
    output.mkdir(exist_ok=False)
    write(output/'execution_protocol.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected_sha,protocol_id=identity))
    rows=[dict(root_sha256=p['root_sha256'],root_index=i,arm=arm,input_sha256=p['input_sha256'],
        source_input_sha256=p['source_input_sha256'],active_metric_id=p['active_metric_ids'][arm],
        q_sha256=digest(p['projections'][arm]),status='not_started') for i,p in enumerate(body['packets']) for arm in ARMS]
    roots=[];before=after=None;host=None;failure=None;active=None;failures=[]
    try:
        host=require_cuda_allocation('cuda');device=torch.device('cuda:0')
        model=load_compatible_pretrained(str(policy),device=device).eval().requires_grad_(False)
        before=checked.tensor_state_hash(model);backend=TorchBackend(model,device)
        for i,p in enumerate(body['packets']):
            record=dict(root_sha256=p['root_sha256'],root_index=i);roots.append(record)
            path=output/f'root_{i}.events.jsonl';record['journal_path']=str(path)
            with path.open('x') as stream:
                active_arm=None
                def journal(event):
                    stream.write(io.canonical(dict(event,root_sha256=p['root_sha256'],arm=active_arm))+'\n')
                    stream.flush();os.fsync(stream.fileno())
                cap=NeuralCap(model,journal)
                try:
                    with cap:
                        for arm in body['arm_order'][i]:
                            active_arm=arm;active=next(r for r in rows if r['root_index']==i and r['arm']==arm)
                            active['status']='incomplete'
                            with cap.scope(arm):
                                active['compilation']=compile_arm(p,p['projections'][arm],backend,journal)
                            torch.cuda.synchronize(device)
                            require(accounting_ok(active['compilation'],cap.receipt(),arm),'actual complete arm accounting')
                            active['status']='completed';write(output/f'root_{i}_{arm}.json',active)
                finally:record['neural_receipt']=cap.receipt()
            record['journal_sha256']=checked.digest(path);write(output/f'root_{i}.json',record)
        after=checked.tensor_state_hash(model)
    except (Exception,KeyboardInterrupt) as exc:
        failure=dict(type=type(exc).__name__,message=str(exc))
        failures.append(failure)
        if active is not None and active['status']!='completed':active['failure']=failure
    for record in roots:
        path=Path(record['journal_path']);record['journal_sha256']=checked.digest(path)
        if not (output/f'root_{record["root_index"]}.json').exists():write(output/f'root_{record["root_index"]}.json',record)
    try:validate_prepared(prepared,expected_sha)
    except Exception as exc:
        drift=dict(type='SourceDrift',message=str(exc));failures.append(drift)
        if failure is None:failure=drift
    for row in rows:
        if row['status']=='not_started':row.update(status='blocked',failure=failure)
    complete=failure is None and before is not None and before==after and all(r['status']=='completed' for r in rows)
    resources=dict(attempted_predict_samples=sum(r['neural_receipt']['attempted']['predict'] for r in roots),
        attempted_action_encoder_samples=sum(r['neural_receipt']['attempted']['action_encoder'] for r in roots),
        completed_predict_samples=sum(r['neural_receipt']['completed']['predict'] for r in roots),
        completed_action_encoder_samples=sum(r['neural_receipt']['completed']['action_encoder'] for r in roots),
        attempted_jacobians=0,completed_jacobians=0,attempted_svds=0,completed_svds=0,
        backward_engine_api_calls=None,mathematical_vjp_directions_requested=0,
        mathematical_vjp_directions_completed=0,incomplete_direction_work_unknown=not complete)
    for record in roots:
        entries=[io.decode(line.encode()) for line in Path(record['journal_path']).read_text().splitlines()]
        for kind,key in (('jacobian','jacobians'),('svd','svds')):
            starts=[e for e in entries if e['kind']==kind and not e['completed'] and 'error' not in e]
            ends=[e for e in entries if e['kind']==kind and e['completed']]
            resources['attempted_'+key]+=len(starts);resources['completed_'+key]+=len(ends)
    resources['mathematical_vjp_directions_requested']=192*resources['attempted_jacobians']
    resources['mathematical_vjp_directions_completed']=192*resources['completed_jacobians']
    report=dict(schema=NAME+'.report',status='cuda_completed_h2_prerequisite' if complete else 'incomplete',
        protocol_id=identity,rows=rows,roots=roots,resources=resources,failure=failure,failures=failures,host=host,
        model_unchanged=before is not None and before==after,contract=contract(),new_physical_steps=0,
        P1_admission=False,P3_admission=False,P4_admission=False,physical_quality_measured=False)
    write(output/'report.json',report)
    if not complete:raise RuntimeError('incomplete: preserve all partials; no automatic retry')
    return report


def selected_actions(output):
    """Model-free, own-schema export of eight NEW H2 programs, never v4 plans.

    Rebuilds 96 saved-array SVDs, zero neural/physics; no output is overwritten.
    A new guarded physical consumer and its own source seal remain required.
    """
    output=Path(output).absolute();binding=io.decode((output/'execution_protocol.json').read_bytes())
    body,identity=validate_prepared(binding['prepared'],binding['prepare_sha256'])
    raw=(output/'report.json').read_bytes();report=io.decode(raw)
    require(binding['protocol_id']==identity==report['protocol_id'] and report['schema']==NAME+'.report','own source/schema identity')
    require(report['status']=='cuda_completed_h2_prerequisite' and report['model_unchanged'] is True and
        report['failure'] is None and report['failures']==[] and report['contract']==contract(),'complete unchanged-model compilation')
    rows=io.index(report['rows'],('root_sha256','arm'));roots=io.index(report['roots'],('root_sha256',))
    require(set(rows)=={(r,a) for r in body['roots'] for a in ARMS} and
        set(roots)=={(r,) for r in body['roots']},'exact four-root/eight-arm census')
    plans=[]
    for i,p in enumerate(body['packets']):
        record=roots[p['root_sha256'],];receipt=record['neural_receipt']
        require(record==io.decode((output/f'root_{i}.json').read_bytes()),'saved root record/report')
        path=Path(record['journal_path']);require(io.sha(path.read_bytes())==record['journal_sha256'],'durable journal hash')
        entries=[io.decode(line.encode()) for line in path.read_text().splitlines()]
        require(all(e['root_sha256']==p['root_sha256'] and e['arm'] in ARMS for e in entries),'journal root/arm identity')
        def strip(e):return {k:v for k,v in e.items() if k not in ('root_sha256','arm')}
        neural=[strip(e) for e in entries if e['kind'] in ('predict','action_encoder')]
        require(len(neural)==2*len(receipt['events']),'attempt/completion census')
        for index,e in enumerate(receipt['events']):
            pair=[n for n in neural if n['index']==index]
            require(len(pair)==2 and pair[0]==dict(e,completed=False) and pair[1]==e and e['completed'] is True,'neural journal/receipt binding')
        for arm in ARMS:
            row=rows[p['root_sha256'],arm];result=row['compilation'];q=p['projections'][arm]
            require(row==io.decode((output/f'root_{i}_{arm}.json').read_bytes()),'saved arm record/report')
            require(row['status']=='completed' and row['root_index']==i and row['input_sha256']==p['input_sha256'] and
                row['source_input_sha256']==p['source_input_sha256'] and row['active_metric_id']==p['active_metric_ids'][arm] and
                row['q_sha256']==digest(q),'separate source and active metric identity')
            require(accounting_ok(result,receipt,arm),'actual arm-specific accounting')
            actual=[strip(e) for e in entries if e['arm']==arm and e['kind'] in ('jacobian','svd')]
            require(len(actual)==2*len(result['operations']),'AD/SVD journal census')
            for op in result['operations']:
                pair=[e for e in actual if e['index']==op['index']]
                require(pair==[dict(op,completed=False),op],'AD/SVD journal exact receipt')
            candidates=[strip(e) for e in entries if e['arm']==arm and e['kind']=='candidate' and e['completed']]
            require(candidates==result['candidates'],'durable ALL candidates/solve/action binding')
            replays=[e for e in entries if e['arm']==arm and e['kind']=='replay' and e['completed']]
            require(len(replays)==13,'all ten candidate plus winner full/segmented replay receipts')
            winner=validate_compilation(p,q,result)
            require(all(replays[k]['phase']=='candidate_'+str(k) and replays[k]['commands']==c['commands'] and
                replays[k]['context']==c['context'] for k,c in enumerate(result['candidates'])),'candidate replay witnesses')
            require(replays[10]['phase']=='winner_whole' and replays[10]['context']==winner['context'] and
                replays[12]['phase']=='winner_segment_1' and replays[12]['context']==winner['context'],'winner replay witnesses')
            plan=dict(root_index=i,root_sha256=p['root_sha256'],arm=arm,horizon=2,actions=winner['commands'],
                root_context=p['root_context'],goal=p['goal'],projection=q,selected_predicted_context=winner['context'],
                active_metric_id=p['active_metric_ids'][arm],q_sha256=row['q_sha256'],input_sha256=p['input_sha256'],
                source_input_sha256=p['source_input_sha256'],protocol_id=identity,report_sha256=io.sha(raw),
                model_bindings=body['model_bindings'],predicted_Q_cost=winner['nonlinear_terminal_cost'],
                objective=winner['objective'],source_schema=NAME+'.report',export_saved_array_svds_total=96,
                new_neural_calls=0,P1_admission=False,P3_admission=False,physical_ready=False)
            plan['selection_sha256']=digest(plan);plans.append(plan)
    require(report['resources']['attempted_predict_samples']==report['resources']['completed_predict_samples']==528 and
        report['resources']['attempted_action_encoder_samples']==report['resources']['completed_action_encoder_samples']==528 and
        report['resources']['attempted_jacobians']==report['resources']['completed_jacobians']==88 and
        report['resources']['attempted_svds']==report['resources']['completed_svds']==96,'complete resource totals')
    return plans


def main():
    parser=argparse.ArgumentParser(description=__doc__);action=parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare','verify','execute-cuda'):action.add_argument('--'+name,action='store_true')
    parser.add_argument('--project-root',type=Path,default=PROJECT);parser.add_argument('--output',type=Path)
    parser.add_argument('--prepared',type=Path);parser.add_argument('--prepare-sha256');args=parser.parse_args()
    if args.prepare:
        require(args.output is not None,'exclusive prepare output');result=prepare(args.project_root,args.output)
    else:
        require(args.prepared is not None and args.prepare_sha256 is not None,'reviewed prepare path/SHA')
        if args.verify:
            _,identity=validate_prepared(args.prepared,args.prepare_sha256);result=dict(status='verified',protocol_id=identity)
        else:
            require(args.output is not None,'exclusive CUDA output');result=execute_cuda(args.prepared,args.prepare_sha256,args.output)
    print(io.canonical({k:v for k,v in result.items() if k not in ('rows','roots','source_seals')}))


if __name__=='__main__':main()
