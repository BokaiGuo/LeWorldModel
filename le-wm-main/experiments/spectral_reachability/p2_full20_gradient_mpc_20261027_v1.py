"""Internal projected-gradient H2 MPC baseline; not an official reproduction.

prepare/verify and array fixtures are model-free. Only explicit main-thread
cuda may evaluate the frozen neural model. No physics, outcome tuning or P2.
"""
import argparse
from contextlib import nullcontext
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from experiments.spectral_reachability import p2_no_composition_sra_20261027_v1 as historical
from experiments.spectral_reachability import p2_full20_sampling_20261027_v2 as full20
from experiments.spectral_reachability import pusht_full_packet_compiler_20261027 as compiler
from experiments.spectral_reachability.d2p_composition_witness_v2 import compose, digest
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal

TAG = 'p2_full20_gradient_mpc_20261027_v1'
ARM = 'internal_projected_gradient_mpc'
PROJECT = Path(__file__).resolve().parents[2]
CAP = float(np.sqrt(10)*.75)
SCHEDULE = dict(horizon_tokens=2,primitive_steps=10,proposal_dimensions=20,packet_shape=[2,5,2],
    candidates=40,optimizer='projected vanilla gradient descent',optimizer_steps=39,learning_rate=.1,
    initial_logits=0.,seed=20261027,random_draws=0,restarts=0,line_search=False,early_stop=False,
    joint_logit_l2_cap=CAP,parameter_dtype='numpy float64',raw_dtype='float32',
    action_mapping='numpy tanh(logits).astype(float32).reshape(2,5,2), single joint logit L2 ball',
    gradient='one CUDA scalar-objective VJP wrt raw20; analytic tanh chain in float64; float32 cast treated as differentiable identity',
    update='project_L2_CAP(logits - 0.1 * raw_gradient * (1 - tanh(logits)**2))',
    candidate_order='u0=zero, then u1..u39 after39 fixed updates; last candidate no backward',
    selection='minimum saved nonlinear original goal0/Q terminal MSE; stable first tie over all40',
    planning_predict_calls=80,planning_predict_samples=80,replay_predict_calls=4,replay_predict_samples=4,
    total_predict_calls=84,total_predict_samples=84,extra_derivative_forward_predict_samples=0,
    autograd_grad_calls=39,scalar_objective_vjp_directions=39,jacobian_calls=0,svd_calls=0,
    action_encoder_calls=84,processed_context_token_samples=252,selection_evaluations=40,batch_size=1,
    model_optimizer_steps=0,physical_env_steps=0,attention_backend='CUDA SDPA MATH for all candidates and winner replay',
    deterministic_algorithms=True,cublas_workspace_config=':4096:8')
BOUNDARY = dict(calibration_only=True,internal_baseline=True,official_paper_implementation=False,
    P1_admission=False,P2_admission=False,composition_admission=False,physical_efficacy_verified=False,
    fair_total_budget_comparison=False,same_proposal_distribution=False,
    unmeasured=['backward engine/kernel multiplicity and FLOPs','upstream RGB encoding/bank construction',
                'shared end-to-end wall-time cap'],
    support='same20 raw coordinates/box and Q; restricted tanh joint-logit-ball support like SRA, NOT CEM/random full-box proposal support',
    mpc_scope='single open-loop H2 planning from each fixed root, no receding-horizon feedback or physical control claim')


def require(value,message):
    if not value: raise ValueError(message)


def logits_array(value):
    result = compiler.finite(value,(20,))
    require(np.linalg.norm(result) <= CAP*(1+1e-12),'joint20 logit ball required')
    return result


def commands_for(logits):
    return np.tanh(logits_array(logits)).astype(np.float32).reshape(2,5,2)


def gradient_step(logits,raw_gradient):
    """Pure-array, frozen39-update kernel; no neural CPU operations."""
    u = logits_array(logits)
    gradient = compiler.finite(raw_gradient,(20,))*(1-np.tanh(u)**2)
    updated = u-SCHEDULE['learning_rate']*gradient
    norm = float(np.linalg.norm(updated))
    require(np.isfinite(norm),'finite logit update required')
    if norm > CAP: updated *= CAP/norm
    return logits_array(updated),gradient


class CudaObjective:
    """Two differentiable raw-packet forwards, no extra derivative forwards.

    Model parameters stay frozen. The two-token latent recurrence is NOT
    detached. Only export the completed candidate's context after its forward.
    """
    def __init__(self,model,device,ready,compiled,replay):
        require(getattr(device,'type',None)=='cuda','host CUDA only; no neural CPU fallback')
        require(type(replay) is full20.RootAnchoredCudaReplay,'root-anchored CUDA replay required')
        self.transition = compiler.make_transition(model,device)
        self.model,self.device,self.root,self.compiled,self.replay = model,device,ready['root'],compiled,replay

    def evaluate(self,logits,index,need_gradient,event,ledger,journal):
        import torch
        initial = self.root['context']
        emb = torch.tensor(initial['latent_history'],dtype=torch.float32,device=self.device)[None]
        history = torch.tensor(initial['action_history'],dtype=torch.float32,device=self.device)[None]
        raw = torch.tensor(commands_for(logits),dtype=torch.float32,device=self.device,requires_grad=need_gradient)
        target = torch.tensor(self.compiled['goal_vectors'][0],dtype=torch.float64,device=self.device)
        q = torch.tensor(self.compiled['scoring']['metric']['projection'],dtype=torch.float64,device=self.device)
        event['actions'] = full20.raw_actions(commands_for(logits))
        self.replay._p2_scope = dict(arm=ARM,phase='planning',path=[index])
        try:
            with torch.set_grad_enabled(need_gradient):
                for packet in range(2):
                    dispatch = dict(phase='planning',path=[index],packet=packet,predict_calls=1,
                        predict_samples=1,action_encoder_calls=1,completed=False)
                    ledger.append(dispatch)
                    started = time.perf_counter()
                    try:
                        emb,history = self.transition(emb,history,raw[packet])
                        dispatch['completed'] = True
                    except Exception as exc:
                        dispatch.update(error_type=type(exc).__name__,error=str(exc))
                        raise
                    finally:
                        dispatch['wall_time_s'] = time.perf_counter()-started
                        journal(dict(kind='candidate_packet',candidate_index=index,**deepcopy(dispatch)))
                residual = q@(emb[0,-1].to(torch.float64)-target)
                objective = (residual*residual).mean()
            terminal = dict(initial,latent_history=emb[0].detach().cpu().tolist(),
                action_history=history[0].detach().cpu().tolist())
            terminal = full20._validated_context(terminal,initial)
            event.update(forward_completed=True,terminal_context=deepcopy(terminal),cost=full20._cost(full20.metric_for(self.compiled)(terminal)),
                         differentiated_cost=float(objective.detach().cpu()))
            require(np.isclose(event['cost'],event['differentiated_cost'],atol=1e-10,rtol=1e-12),
                    'CUDA differentiated scalar differs from original saved Q arithmetic')
            journal(dict(kind='candidate_cost',candidate_index=index,cost=event['cost']))
            gradient = None
            if need_gradient:
                operation = dict(kind='autograd_grad',candidate_index=index,completed=False,
                    scalar_objective_vjp_directions_requested=1,extra_forward_requested=0)
                event['derivative'] = operation
                journal(deepcopy(operation))
                started = time.perf_counter()
                try:
                    gradient_tensor, = torch.autograd.grad(objective,raw,create_graph=False,retain_graph=False)
                    operation['completed'] = True
                    gradient = gradient_tensor.detach().cpu().numpy().astype(np.float64).reshape(20)
                    compiler.finite(gradient,(20,))
                except Exception as exc:
                    operation.update(error_type=type(exc).__name__,error=str(exc))
                    raise
                finally:
                    torch.cuda.synchronize(self.device)
                    operation['wall_time_s'] = time.perf_counter()-started
                    journal(deepcopy(operation))
            return terminal,gradient
        finally:
            self.replay._p2_scope = None


def run_trajectory(*,root,provenance,action_domain,objective,replay,target_metric,fixture=False,
                   phase=lambda name:nullcontext(),journal=lambda event:None):
    """One fixed40-slot trajectory; failures consume work, never refill/retry."""
    if not fixture:
        require(type(objective) is CudaObjective and type(replay) is full20.RootAnchoredCudaReplay,
                'production requires frozen CUDA objective/replay; fixture callbacks forbidden')
    initial = full20._validated_context(root['context'])
    require(action_domain['name']=='raw' and action_domain['low']==[-1.,-1.]
        and action_domain['high']==[1.,1.] and action_domain['action_block']==5,'raw official packet domain required')
    def edge(terminal,actions,identity):
        return dict(program_id=identity,kind='progress',root_id=root['root_id'],provenance=deepcopy(provenance),
            action_domain=deepcopy(action_domain),input_context=deepcopy(initial),
            output_context=deepcopy(terminal),actions=deepcopy(actions))
    compose(root=root,provenance=provenance,action_domain=action_domain,
        edges=[edge(initial,full20.raw_actions(commands_for(np.zeros(20))),'metadata')],
        replay=lambda c,a:c,max_edges=1,max_actions=10)
    ledger = full20._Ledger(replay,ARM)
    logits = np.zeros(20)
    attempts,best,witness,error,blocked = [],None,None,None,False
    with phase('planning'):
        for index in range(40):
            event = dict(candidate_id=f'{ARM}_{index:02d}',candidate_index=index,completed=False,
                forward_completed=False,blocked=blocked,needs_gradient=index<39)
            attempts.append(event)
            if blocked:
                event['blocked_reason'] = 'earlier trajectory failure; no replacement/retry/refund'
                journal(deepcopy(event))
                continue
            event.update(logits=logits.tolist(),commands=commands_for(logits).tolist(),
                         actions=full20.raw_actions(commands_for(logits)))
            try:
                terminal,gradient = objective.evaluate(logits,index,index<39,event,ledger.events,journal)
                cost = full20._cost(target_metric(terminal))
                require(event['forward_completed'] is True and event['cost']==cost,'original full-H2 metric required')
                if index<39:
                    next_logits,logit_gradient = gradient_step(logits,gradient)
                    event.update(raw_gradient=compiler.finite(gradient,(20,)).tolist(),
                        logit_gradient=logit_gradient.tolist(),next_logits=next_logits.tolist(),optimizer_step_completed=True)
                    logits = next_logits
                else:
                    require(gradient is None,'last candidate must not run unused backward')
                    event['optimizer_step_completed'] = False
                event['completed'] = True
                if best is None or cost<best['cost']:
                    best = dict(candidate_id=event['candidate_id'],cost=cost,
                        terminal=deepcopy(terminal),actions=deepcopy(event['actions']))
            except Exception as exc:
                event.update(error_type=type(exc).__name__,error=str(exc))
                blocked = True
            journal(deepcopy(event))
    if all(e['completed'] for e in attempts):
        with phase('evaluation'):
            try:
                mid = ledger.call(initial,best['actions'][:5],'replay',[40])
                end = ledger.call(mid,best['actions'][5:],'replay',[41])
                full = ledger.call(initial,best['actions'],'replay',[42])
                require(full==end==best['terminal'],'search/segmented/independent full H2 replay mismatch')
                require(full20._cost(target_metric(full))==best['cost'],'saved winner Q cost mismatch')
                receipt = compose(root=root,provenance=provenance,action_domain=action_domain,
                    edges=[edge(end,best['actions'],best['candidate_id'])],replay=lambda c,a:deepcopy(full),
                    max_edges=1,max_actions=10)
                witness = dict(receipt=receipt,edges=[edge(end,best['actions'],best['candidate_id'])],
                    independent_full_replay_exact=True)
            except Exception as exc:
                error = dict(error_type=type(exc).__name__,error=str(exc))
    counts = {p:sum(e['predict_samples'] for e in ledger.events if e['phase']==p) for p in ('planning','replay')}
    complete = witness is not None and counts==dict(planning=80,replay=4) and all(e['completed'] for e in ledger.events)
    return dict(arm=ARM,fixture_only=fixture,attempts=attempts,ledger=ledger.events,
        status='search_complete' if complete else 'incomplete_or_failed',candidate_and_replay_complete=complete,
        selected_candidate_id=best['candidate_id'] if complete else None,
        selected_model_cost=best['cost'] if complete else None,witness=witness if complete else None,replay_error=error,
        planning_predict_samples=counts['planning'],replay_predict_samples=counts['replay'],
        selection_evaluations=sum(e['forward_completed'] for e in attempts),
        attempted_candidate_evaluations=sum(not e['blocked'] for e in attempts),
        optimizer_steps=sum(e.get('optimizer_step_completed',False) for e in attempts),
        attempted_autograd_grad_calls=sum('derivative' in e for e in attempts),
        completed_autograd_grad_calls=sum(e.get('derivative',{}).get('completed',False) for e in attempts),
        **deepcopy(BOUNDARY))


def source_files():
    return sorted(set(historical.source_files()+[Path(__file__),PROJECT/'tests'/f'test_{TAG}.py',
        PROJECT/'innerdocs'/f'{TAG}.md',Path(full20.__file__),Path(compiler.__file__),
        PROJECT/'experiments/spectral_reachability/d2p_adaptive_budget_v2.py',
        PROJECT/'experiments/spectral_reachability/d2p_model_call_instrumentation.py']))


def reference_contract(reference,reader):
    """Inherit prepared contexts, not physical outcomes or even sampler scores."""
    reference = Path(reference).resolve()
    protocol = reader.load(reference/'protocol.json')
    prepared = reader.load(reference/'prepared.json')
    require(protocol['schema']==full20.TAG+'.protocol' and digest(protocol)==prepared['protocol_sha256'],
            'sealed full20 prepared protocol required')
    reader.seals(prepared['source_seals'])
    reader.bindings(protocol['bindings'])  # Hash-only provenance; no outcome parsing.
    compiled = reader.load(protocol['compilation'],historical.audit.COMPILATION_SHA)
    source = historical.audit.index(compiled['rows'],('root_sha256',))
    cases = protocol['prepared']
    require(len(cases)==4 and len(source)==4 and {r['root_index'] for r in cases}==set(range(4))
        and {(r['root_sha256'],) for r in cases}==set(source),'four fixed unique prepared roots required')
    require(protocol['schedule']==full20.SCHEDULE and protocol['arms']==['cem','random']
        and protocol['objective']=='mean square Q@(terminal-goal_vectors[0])','original full20 schedule/Q required')
    ready = []
    for case in sorted(cases,key=lambda r:r['root_index']):
        saved = source[case['root_sha256'],]['compiled']
        initial = saved['input_context']
        expected = dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,
            action_dim=10,latent_history=initial['emb'][0],action_history=initial['actions'][0][1:])
        domain = case['action_domain']
        require(case['root']['root_id']==case['root_sha256'] and case['root']['context']==expected
            and case['metric_sha256']==saved['scoring']['metric']['sha256']
            and case['seed']==historical.SEEDS[case['root_index']]
            and domain['name']=='raw' and domain['low']==[-1.,-1.] and domain['high']==[1.,1.]
            and domain['action_block']==5 and domain['normalization_sha256']==digest(protocol['normalization']),
            'fixed context/metric/raw-domain identity changed')
        full20.metric_for(saved)
        ready.append(dict(case,goal_vector_sha256=digest(saved['goal_vectors'][0]),
            projection_sha256=digest(saved['scoring']['metric']['projection'])))
    return dict(reference=str(reference),cases=ready,compilation=protocol['compilation'],policy=protocol['policy'],
        normalization=protocol['normalization'],objective=protocol['objective'],original_source_seals=prepared['source_seals'])


def prepare(reference,output):
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    require(TAG in output.name,'new exclusive tagged output required')
    reader = historical.audit.EvidenceReader()
    # Known full20 planning reference only. Never load SRA/MPPI physical outcomes.
    context = reference_contract(reference,reader)
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    source,_,config = checked.validate_inputs(context['compilation'],context['policy'])
    files = source_files()+checked.sources_before_allocate(source,config)
    files += [Path(m.__file__) for n,m in tuple(sys.modules.items())
              if n.startswith('experiments.') and getattr(m,'__file__','').endswith('.py')]
    files = sorted(set(p.resolve() for p in files))
    protocol = dict(schema=TAG+'.protocol',context=context,schedule=deepcopy(SCHEDULE),boundary=deepcopy(BOUNDARY),
        bindings=deepcopy(reader.hashes),source_paths=[str(p) for p in files],
        frozen_before_gradient_cuda=True,physical_outcomes_used=False,future_heldout_results_opened=False,
        expected_four_root_totals=dict(predict_calls=336,predict_samples=336,action_encoder_calls=336,
            processed_context_token_samples=1008,autograd_grad_calls=156,scalar_objective_vjp_directions=156,
            optimizer_steps=156,model_optimizer_steps=0))
    output.mkdir(exist_ok=False)
    full20.write(output/'protocol.json',protocol)
    seals = [create_seal(files=files[i:i+40],args_json=json.dumps(dict(protocol_sha256=digest(protocol))),
        output_dir=output/f'source_seal_{i//40}') for i in range(0,len(files),40)]
    full20.write(output/'prepared.json',dict(schema=TAG+'.prepared',status='prepared_not_executed',
        protocol_sha256=digest(protocol),source_seals=seals,new_neural_calls=0,new_physical_steps=0,P2_admission=False))
    reader.unchanged()
    load_prepared(output)
    return protocol


def load_prepared(output):
    output = Path(output).resolve()
    protocol = historical.audit.read_json(output/'protocol.json')
    prepared = historical.audit.read_json(output/'prepared.json')
    require(protocol['schema']==TAG+'.protocol' and prepared['schema']==TAG+'.prepared'
        and prepared['protocol_sha256']==digest(protocol) and protocol['schedule']==SCHEDULE
        and protocol['boundary']==BOUNDARY and protocol['frozen_before_gradient_cuda'] is True
        and protocol['physical_outcomes_used'] is False and protocol['future_heldout_results_opened'] is False,
        'frozen gradient protocol changed')
    paths = []
    require(bool(prepared['source_seals']),'source seals required')
    for seal in prepared['source_seals']:
        verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
        directory = Path(seal['output_dir'])
        require(historical.audit.read_json(directory/'args.json')==dict(protocol_sha256=digest(protocol)),
                'source/protocol binding mismatch')
        paths += [e['source'] for e in historical.audit.read_json(directory/'manifest.json')['files']]
    require(sorted(paths)==protocol['source_paths'] and set(map(str,source_files()))<=set(paths),'source closure missing')
    reader = historical.audit.EvidenceReader()
    reader.bindings(protocol['bindings'])
    context = protocol['context']
    require(context==reference_contract(context['reference'],reader),
            'fixed root/context/goal/Q identity changed')
    reader.unchanged()
    return output,protocol,prepared


def provenance_for(protocol,prepared,ready):
    return dict(checkpoint_sha256=historical.audit.CHECKPOINT_SHA,encoder_sha256=historical.audit.CHECKPOINT_SHA,
        source_sha256=prepared['source_seals'][0]['manifest_sha256'],protocol_sha256=digest(protocol),
        metric_sha256=ready['metric_sha256'],candidate_bank_sha256=digest(SCHEDULE))


def accounting_ok(entries,actual,result,provenance,root_id):
    return (result['candidate_and_replay_complete'] and result['optimizer_steps']==39
        and result['attempted_autograd_grad_calls']==result['completed_autograd_grad_calls']==39
        and full20.entries_match(entries,result['ledger'],ARM) and len(entries)==168
        and len(result['ledger'])==84
        and [e['phase'] for e in result['ledger']]==['planning']*80+['replay']*4
        and actual['binding']==dict(protocol_id=provenance['source_sha256'],task_id=TAG,
            root_id=root_id,frozen_model_id=provenance['checkpoint_sha256'])
        and actual['method']==ARM and actual['predict_sample_limit']==84
        and actual['attempted_predict_samples']==84 and actual['blocked_calls']==0
        and [(p['phase'],p['predict_calls'],p['predict_samples']) for p in actual['phases']]==
            [('planning',80,80),('evaluation',4,4)]
        and all(p['completed'] is True and p['failed_predict_calls']==0 for p in actual['phases']))


def cuda(output):
    """Main only, after review. Never invoked by prepare/verify or tests."""
    output,protocol,prepared = load_prepared(output)
    full20.write(output/'started.json',dict(status='cuda_requested',protocol_sha256=digest(protocol)))
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability.d2p_adaptive_budget_v2 import Binding,PredictionBudget
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = SCHEDULE['cublas_workspace_config']
    host = base.require_cuda_allocation('cuda')
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    device = torch.device('cuda:0')
    np.random.seed(SCHEDULE['seed'])
    torch.manual_seed(SCHEDULE['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    model = base.load_compatible_pretrained(Path(protocol['context']['policy']),device=device).eval().requires_grad_(False)
    before = checked.tensor_state_hash(model)
    torch.cuda.reset_peak_memory_stats(device)
    compiled = {r['root_sha256']:r['compiled'] for r in historical.audit.read_json(protocol['context']['compilation'])['rows']}
    rows = []
    for ready in protocol['context']['cases']:
        started = time.perf_counter()
        prov = provenance_for(protocol,prepared,ready)
        replay = full20.RootAnchoredCudaReplay(model,device)
        objective = CudaObjective(model,device,ready,compiled[ready['root_sha256']],replay)
        binding = Binding(prov['source_sha256'],TAG,ready['root_sha256'],checked.OFFICIAL_SHA)
        with (output/f'root_{ready["root_index"]}_journal.jsonl').open('x') as stream:
            def journal(event):
                stream.write(json.dumps(event,allow_nan=False)+'\n')
                stream.flush()
                os.fsync(stream.fileno())
            with sdpa_kernel(SDPBackend.MATH),full20.actual_entries(model,replay) as entries, PredictionBudget(
                    model,binding=binding,method=ARM,sample_limit=84) as budget:
                result = run_trajectory(root=ready['root'],provenance=prov,action_domain=ready['action_domain'],
                    objective=objective,replay=replay,target_metric=full20.metric_for(compiled[ready['root_sha256']]),
                    phase=lambda p:budget.phase(p,method=ARM,horizon_tokens=2),journal=journal)
                actual = budget.receipt()
        torch.cuda.synchronize(device)
        row = dict(root_index=ready['root_index'],root_sha256=ready['root_sha256'],root=ready['root'],
            action_domain=ready['action_domain'],result=result,actual_model_entries=entries,
            actual_prediction_budget=actual,complete=accounting_ok(entries,actual,result,prov,ready['root_sha256']),
            predict_calls=sum(e['kind']=='predict' for e in entries),
            action_encoder_calls=sum(e['kind']=='action_encoder' for e in entries),
            processed_context_token_samples=sum(e['context_tokens'] for e in entries if e['kind']=='predict'),
            actual_root_wall_time_s=time.perf_counter()-started,**deepcopy(BOUNDARY))
        full20.write(output/f'root_{ready["root_index"]}.json',row)
        rows.append(row)
    load_prepared(output)
    gates = dict(all_four_retained=len(rows)==4,all_complete=all(r['complete'] for r in rows),
        actual336=sum(r['predict_calls'] for r in rows)==336,
        actual_encoder336=sum(r['action_encoder_calls'] for r in rows)==336,
        actual_context1008=sum(r['processed_context_token_samples'] for r in rows)==1008,
        backward156=sum(r['result']['completed_autograd_grad_calls'] for r in rows)==156,
        updates156=sum(r['result']['optimizer_steps'] for r in rows)==156,
        model_unchanged=before==checked.tensor_state_hash(model),
        frozen_eval=all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters()))
    report = dict(schema=TAG+'.report',status='calibration_completed' if all(gates.values()) else 'incomplete_or_failed',
        protocol_sha256=digest(protocol),source_seals=prepared['source_seals'],rows=rows,gates=gates,host=host,
        actual_predict_calls=sum(r['predict_calls'] for r in rows),
        actual_action_encoder_calls=sum(r['action_encoder_calls'] for r in rows),
        actual_processed_context_token_samples=sum(r['processed_context_token_samples'] for r in rows),
        completed_autograd_grad_calls=sum(r['result']['completed_autograd_grad_calls'] for r in rows),
        completed_scalar_objective_vjp_directions=sum(r['result']['completed_autograd_grad_calls'] for r in rows),
        actual_optimizer_steps=sum(r['result']['optimizer_steps'] for r in rows),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
        numpy_version=np.__version__,torch_version=torch.__version__,**deepcopy(BOUNDARY))
    full20.write(output/'report.json',report)
    if not all(gates.values()): raise RuntimeError('incomplete gradient trajectory retained; no physical deployment')


def verify_result(row,protocol,prepared,compiled):
    """Array-only saved trajectory/winner/witness validation, no backward calls."""
    cases = {r['root_sha256']:r for r in protocol['context']['cases']}
    ready = cases[row['root_sha256']]
    prov = provenance_for(protocol,prepared,ready)
    result = row['result']
    require(row['root']==ready['root'] and row['root_index']==ready['root_index']
        and row['action_domain']==ready['action_domain'] and row['complete'] is True
        and result['fixture_only'] is False and result['arm']==ARM
        and result['status']=='search_complete' and result['replay_error'] is None,
        'real complete fixed-root trajectory required')
    require(accounting_ok(row['actual_model_entries'],row['actual_prediction_budget'],result,prov,ready['root_sha256']),
            'actual prediction/backward accounting mismatch')
    logits = np.zeros(20)
    attempts = result['attempts']
    require(len(attempts)==40,'all40 trajectory candidates required')
    for i,event in enumerate(attempts):
        require(event['candidate_id']==f'{ARM}_{i:02d}' and event['candidate_index']==i
            and event['completed'] is True and event['blocked'] is False
            and event['logits']==logits.tolist() and event['commands']==commands_for(logits).tolist()
            and event['actions']==full20.raw_actions(commands_for(logits)), 'saved logit/action trajectory changed')
        full20._cost(event['cost'])
        require(event['forward_completed'] is True and event['needs_gradient']==(i<39)
            and abs(full20.metric_for(compiled)(event['terminal_context'])-event['cost'])<=1e-12,
            'saved candidate terminal/cost mismatch')
        if i<39:
            next_logits,gradient = gradient_step(logits,event['raw_gradient'])
            require(event['derivative']['completed'] is True and event['derivative']['candidate_index']==i
                and event['optimizer_step_completed'] is True
                and event['logit_gradient']==gradient.tolist() and event['next_logits']==next_logits.tolist(),
                'frozen gradient update mismatch')
            logits = next_logits
        else:
            require('derivative' not in event and event['optimizer_step_completed'] is False,'unused last backward forbidden')
    best = min(attempts,key=lambda e:e['cost'])
    saved = result['witness']
    require(saved['independent_full_replay_exact'] is True and len(saved['edges'])==1,'single-edge independent H2 replay required')
    receipt = saved['receipt']
    rebound = compose(root=ready['root'],provenance=prov,action_domain=ready['action_domain'],edges=saved['edges'],
        replay=lambda c,a:deepcopy(receipt['terminal_context']),max_edges=1,max_actions=10)
    require(receipt==rebound and receipt['actions']==best['actions']
        and result['selected_candidate_id']==best['candidate_id'] and result['selected_model_cost']==best['cost']
        and abs(full20.metric_for(compiled)(receipt['terminal_context'])-best['cost'])<=1e-12,
        'saved stable argmin/witness/goal0Q mismatch')
    return np.asarray(receipt['actions'],dtype=np.float32)


def selected_actions(output):
    output,protocol,prepared = load_prepared(output)
    reader = historical.audit.EvidenceReader()
    report = reader.load(output/'report.json')
    require(report['schema']==TAG+'.report' and report['status']=='calibration_completed'
        and report['protocol_sha256']==digest(protocol) and report['source_seals']==prepared['source_seals']
        and bool(report['gates']) and all(v is True for v in report['gates'].values()),'real complete source-bound calibration required')
    rows = historical.audit.index(report['rows'],('root_sha256',))
    require(len(rows)==4 and set(rows)=={(r['root_sha256'],) for r in protocol['context']['cases']},'all four stable roots required')
    compiled = {r['root_sha256']:r['compiled'] for r in reader.load(protocol['context']['compilation'],historical.audit.COMPILATION_SHA)['rows']}
    actions = {}
    for (key,),row in rows.items():
        require(reader.load(output/f'root_{row["root_index"]}.json')==row,'persisted root report differs')
        reader.bind(output/f'root_{row["root_index"]}_journal.jsonl')
        actions[key,ARM] = verify_result(row,protocol,prepared,compiled[key])
    require(report['actual_predict_calls']==336 and report['completed_autograd_grad_calls']==156
        and report['actual_optimizer_steps']==156,'four-root prediction/backward totals mismatch')
    for key,value in BOUNDARY.items(): require(report[key]==value,'claim boundary changed')
    reader.unchanged()
    return dict(actions=actions,report_sha256=reader.hashes[str(output/'report.json')],new_neural_calls=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='phase',required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--reference',type=Path,default=historical.DEFAULT_REFERENCE)
    prep.add_argument('--output',type=Path,required=True)
    for phase in ('verify','cuda'): commands.add_parser(phase).add_argument('--output',type=Path,required=True)
    args = vars(parser.parse_args(argv))
    phase = args.pop('phase')
    if phase=='prepare': prepare(**args)
    elif phase=='verify': load_prepared(**args)
    else: cuda(**args)
    print(json.dumps(dict(status=phase+'_completed',output=str(args['output']),P2_admission=False)))


if __name__=='__main__': main()
