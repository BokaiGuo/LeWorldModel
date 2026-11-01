"""Predeclared conditional-suffix ablation under authorized implementation.

Fresh joint-H2 spectral slots supply identical prefixes to both arms. One arm
keeps the joint proposal; the other compiles a local full10 suffix at each
predicted prefix. This is not P1, scientific P2, or total-compute matching.
Prepare/verify are model-free; only explicit main-thread cuda loads a model.
"""
import argparse
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import numpy as np

from experiments.spectral_reachability import p2_no_composition_sra_20261027_v2 as sra
from experiments.spectral_reachability.d2p_composition_witness_v2 import compose, digest
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal

full20, compiler, historical = sra.full20, sra.compiler, sra.historical
TAG = 'p2_composition_paired_20261027_v2'
PROJECT = Path(__file__).resolve().parents[2]
ARMS = ('joint_h2_no_composition', 'conditional_suffix_composition')
SCHEDULE = dict(shared_bank=deepcopy(sra.SCHEDULE), candidates_per_arm=40,
    composition='same prefix logits; full10 suffix Jacobian at zero suffix in own complete predicted context; bounded ridge .002',
    suffix_cap='sqrt(max(0, joint_cap**2 - prefix_logits dot prefix_logits))',
    zero_cap='still nominal and Jacobian; exact singleton zero solution; no SVD/solve dispatched',
    prefix_policy='all40 retained, no greedy filtering, no deduplication or context cache',
    selection='original nonlinear Q terminal mean-square; stable first tie independently per arm',
    witness='every candidate: one edge for joint, two for composition; winner independent segmented + whole H2 replay',
    shared_predict=4, joint_exclusive_predict=84, composition_exclusive_predict=164,
    unique_predict_per_root=252, standalone_attributed_predict=[88,168],
    unique_jacobian_per_root=41, scalar_output_vjp_directions_per_root=7872,
    svd_per_root='2 shared + number of positive-radius suffixes (0..40)',
    replay_per_arm=4, encoder_calls_per_predict=1, context_tokens_per_entry=3,
    joint_logit_l2_cap=sra.CAP, damping=.002, candidate_steps=10,
    no_prefix_graph_gradient='prefix is a fixed action, local suffix differentiation only; not a joint Hessian')
BOUNDARY = dict(P1_admission=False, P2_admission=False, calibration_only=True,
    physical_efficacy_verified=False, fair_total_budget_comparison=False,
    same_proposal_distribution=False, independent_cohort=False,
    mechanism='minimal conditional suffix recompilation ablation, not non-greedy reconfiguration evidence',
    unmeasured=['FLOPs', 'upstream RGB/context/Q construction', 'backward engine kernel multiplicity'],
    shared_cost_policy='charge once in paired unique total; include in each standalone attribution; never sum attributions')


def suffix_solution(prefix_logits, jacobian, nominal, goal, projection):
    prefix = compiler.finite(prefix_logits, (10,))
    j = compiler.finite(jacobian, (192,10))
    nominal = compiler.finite(nominal, (192,))
    goal = compiler.finite(goal, (192,))
    q = compiler.finite(projection, (192,192))
    remaining = sra.CAP**2-float(prefix@prefix)
    if remaining < -1e-12:
        raise ValueError('prefix exceeds joint ball')
    cap = float(np.sqrt(max(0., remaining)))
    if cap == 0:
        solve = dict(params=[0.]*10, radius=0., norm=0., singleton_zero=True)
    else:
        solve = compiler.bounded_ridge(q@j, q@(goal-nominal), damping=.002, cap=cap)
    return dict(jacobian=j.tolist(),nominal=nominal.tolist(),prefix_logits=prefix.tolist(),
        remaining_radius=cap,solve=solve,svd_calls=int(cap>0),bounded_solve_calls=int(cap>0),
        suffix_logits=solve['params'])


def saved_witness(root, provenance, domain, contexts, chunks, identity):
    """Bind saved executed transitions, not extra neural replay or proof thereof."""
    edges = [dict(program_id=f'{identity}_edge_{i}',kind='progress',root_id=root['root_id'],
        provenance=deepcopy(provenance),action_domain=deepcopy(domain),
        input_context=deepcopy(contexts[i]),output_context=deepcopy(contexts[i+1]),
        actions=deepcopy(chunk)) for i,chunk in enumerate(chunks)]
    def saved(context, actions):
        for edge in edges:
            if context == edge['input_context'] and actions == edge['actions']:
                return deepcopy(edge['output_context'])
        raise ValueError('disconnected saved transition')
    receipt = compose(root=root,provenance=provenance,action_domain=domain,edges=edges,
        replay=saved,max_edges=len(chunks),max_actions=10)
    return dict(edges=edges,receipt=receipt,transition_validation='saved-context binding; not independent neural replay')


def run_arm(*, arm, bank, root, provenance, action_domain, target_metric, replay,
            suffix=None, fixture=False, phase=lambda name:nullcontext(), journal=lambda e:None):
    if arm not in ARMS or (not fixture and type(replay) is not full20.RootAnchoredCudaReplay):
        raise ValueError('supported arm and host CUDA replay required')
    if arm == ARMS[1] and not callable(suffix):
        raise ValueError('actual own-context suffix compiler required')
    logits = compiler.finite(bank['logits'], (40,20))
    commands = compiler.finite(bank['commands'], (40,2,5,2))
    if (np.max(np.linalg.norm(logits,axis=1)) > sra.CAP*(1+1e-12)
            or not np.array_equal(commands.astype(np.float32),np.tanh(logits).astype(np.float32).reshape(40,2,5,2))):
        raise ValueError('shared bank mapping/ball changed')
    initial = full20._validated_context(root['context'])
    saved_witness(root,provenance,action_domain,[initial,initial],
        [full20.raw_actions(commands[0])],'metadata')
    if (action_domain['low'] != [-1.,-1.] or action_domain['high'] != [1.,1.]
            or action_domain['action_block'] != 5):
        raise ValueError('original raw action contract required')
    ledger = full20._Ledger(replay,arm)
    attempts = []
    started = time.perf_counter()
    with phase('planning'):
        for index in range(40):
            event = dict(candidate_id=f'{arm}_{index:02d}',index=index,
                prefix_logits=logits[index,:10].tolist(),completed=False,metric_attempted=False)
            attempts.append(event)
            try:
                prefix = commands[index,0].astype(np.float32).tolist()
                mid = ledger.call(initial,prefix,'planning',[index,0])
                if arm == ARMS[1]:
                    local = suffix(deepcopy(mid),logits[index,:10].copy(),index)
                    event['suffix_compilation'] = deepcopy(local)
                    if local['input_context_sha256'] != digest(mid):
                        raise ValueError('suffix compiled at wrong full prefix context')
                    suffix_logits = compiler.finite(local['suffix_logits'],(10,))
                else:
                    suffix_logits = logits[index,10:]
                joint = np.concatenate((logits[index,:10],suffix_logits))
                if np.linalg.norm(joint) > sra.CAP*(1+1e-12):
                    raise ValueError('generated program exceeds joint ball')
                second = np.tanh(suffix_logits).astype(np.float32).reshape(5,2).tolist()
                end = ledger.call(mid,second,'planning',[index,1])
                event.update(logits=joint.tolist(),actions=prefix+second)
                contexts, chunks = ([initial,mid,end],[prefix,second]) if arm==ARMS[1] else ([initial,end],[prefix+second])
                event['witness'] = saved_witness(root,provenance,action_domain,contexts,chunks,event['candidate_id'])
                event['metric_attempted'] = True
                metric_started = time.perf_counter()
                try:
                    event['cost'] = full20._cost(target_metric(deepcopy(end)))
                finally:
                    event['selection_metric_wall_time_s'] = time.perf_counter()-metric_started
                event['completed'] = True
            except Exception as exc:
                event.update(error_type=type(exc).__name__,error=str(exc))
            journal(deepcopy(event))
    winner, replay_error = None, None
    if all(e['completed'] for e in attempts):
        best = min(attempts,key=lambda e:e['cost'])
        with phase('evaluation'):
            try:
                actions = best['actions']
                mid = ledger.call(initial,actions[:5],'replay',[40])
                end = ledger.call(mid,actions[5:],'replay',[41])
                whole = ledger.call(initial,actions,'replay',[42])
                if end != whole or end != best['witness']['receipt']['terminal_context']:
                    raise ValueError('winner full/segmented/search context mismatch')
                if arm == ARMS[1] and mid != best['witness']['edges'][0]['output_context']:
                    raise ValueError('winner prefix mismatch')
                winner = dict(candidate_id=best['candidate_id'],cost=best['cost'],
                    witness=deepcopy(best['witness']),independent_full_replay_exact=True)
            except Exception as exc:
                replay_error = dict(error_type=type(exc).__name__,error=str(exc))
    return dict(arm=arm,fixture_only=fixture,attempts=attempts,ledger=ledger.events,
        winner=winner,replay_error=replay_error,complete=winner is not None,
        selection_evaluations=sum(e['metric_attempted'] for e in attempts),
        actual_wall_time_s=time.perf_counter()-started)


def suffix_cuda(model, device, replay, context, prefix, compiled, index, journal):
    if getattr(device,'type',None) != 'cuda':
        raise ValueError('host CUDA only; no neural CPU fallback')
    import torch
    transition = compiler.make_transition(model,device)
    emb = torch.tensor(context['latent_history'],dtype=torch.float32,device=device)[None]
    history = torch.tensor(context['action_history'],dtype=torch.float32,device=device)[None]
    zero = emb.new_zeros(10)
    def terminal(u):
        state,_ = transition(emb,history,torch.tanh(u).reshape(5,2))
        return state[0,-1]
    try:
        replay._p2_scope = dict(arm=ARMS[1],phase='suffix_compilation',path=[index,'nominal'])
        with torch.no_grad():
            nominal = terminal(zero)
        replay._p2_scope = dict(arm=ARMS[1],phase='suffix_compilation',path=[index,'jacobian'])
        op = dict(kind='suffix_jacobian',index=index,completed=False,shape=[192,10],vectorize=True)
        journal(deepcopy(op))
        jac_started = time.perf_counter()
        jac = torch.autograd.functional.jacobian(terminal,zero.requires_grad_(True),vectorize=True)
        op['completed'] = True
        op['dispatch_wall_time_s'] = time.perf_counter()-jac_started
        journal(deepcopy(op))
        op = dict(kind='suffix_solve',index=index,completed=False,
            svd_calls_requested=int(sra.CAP**2-float(prefix@prefix)>0))
        journal(deepcopy(op))
        solve_started = time.perf_counter()
        arrays = lambda t:t.detach().cpu().numpy().astype(np.float64)
        result = suffix_solution(prefix,arrays(jac),arrays(nominal),compiled['goal_vectors'][0],
            compiled['scoring']['metric']['projection'])
        result['input_context_sha256'] = digest(context)
        op.update(completed=True,svd_calls=result['svd_calls'],bounded_solve_calls=result['bounded_solve_calls'])
        op['cpu_transfer_and_solve_wall_time_s'] = time.perf_counter()-solve_started
        journal(deepcopy(op))
        return result
    finally:
        replay._p2_scope = None


def source_files():
    return sorted(set(sra.source_files()+[Path(__file__),PROJECT/'tests'/f'test_{TAG}.py',
        PROJECT/'innerdocs'/f'{TAG}.md']))


def prepare(reference, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if TAG not in output.name:
        raise ValueError('new dedicated tagged path required')
    reader = historical.audit.EvidenceReader()
    context = historical.reference_contract(reference,historical.DEFAULT_ADMISSION,reader)
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    source,_,config = checked.validate_inputs(context['compilation'],context['policy'])
    files = source_files()+checked.sources_before_allocate(source,config)
    files += [Path(m.__file__) for n,m in tuple(sys.modules.items())
        if n.startswith('experiments.') and getattr(m,'__file__','').endswith('.py')]
    files = sorted(set(p.resolve() for p in files))
    protocol = dict(schema=TAG+'.protocol',context=context,schedule=deepcopy(SCHEDULE),
        boundary=deepcopy(BOUNDARY),bindings=deepcopy(reader.hashes),source_paths=list(map(str,files)),
        authorization='implementation-thread predeclared under authorized implementation, pending main review; not user-picked parameters',
        frozen_before_new_paired_outcomes=True,expected_four_root_predict=1008,
        input_selection='same four already-used roots; original goal0/Q; no outcome-driven selection')
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
    p = historical.audit.read_json(output/'protocol.json')
    s = historical.audit.read_json(output/'prepared.json')
    if (p['schema'] != TAG+'.protocol' or s['schema'] != TAG+'.prepared'
            or digest(p) != s['protocol_sha256'] or p['schedule'] != SCHEDULE
            or p['boundary'] != BOUNDARY or p['frozen_before_new_paired_outcomes'] is not True):
        raise ValueError('frozen paired protocol changed')
    paths = []
    for seal in s['source_seals']:
        verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
        directory = Path(seal['output_dir'])
        if historical.audit.read_json(directory/'args.json') != dict(protocol_sha256=digest(p)):
            raise ValueError('source/protocol binding changed')
        paths += [e['source'] for e in historical.audit.read_json(directory/'manifest.json')['files']]
    if sorted(paths) != p['source_paths'] or not set(map(str,source_files())) <= set(paths):
        raise ValueError('source closure missing')
    reader = historical.audit.EvidenceReader()
    reader.bindings(p['bindings'])
    c = p['context']
    if historical.reference_contract(c['reference'],c['admission'],reader) != c:
        raise ValueError('original roots/Q/goals/context changed')
    reader.unchanged()
    return output,p,s


def provenance_for(protocol, prepared, ready, bank, arm):
    prov = sra.provenance_for(protocol,prepared,ready)
    # One truthful bank-generation identity per arm, no fabricated global SHA.
    prov['candidate_bank_sha256'] = digest(dict(shared_bank=bank,arm=arm,schedule=SCHEDULE))
    return prov


def accounting_ok(entries, budget, results):
    if (len(results)!=2 or not all(r['complete'] for r in results)
            or [(p['phase'],p['predict_calls'],p['predict_samples']) for p in budget['phases']]
            != [('planning',4,4),('planning',80,80),('evaluation',4,4),('planning',160,160),('evaluation',4,4)]
            or not all(p['completed'] for p in budget['phases'])
            or not all(e['completed'] and e['samples']==1 and e['context_tokens']==3 and e['scope'] for e in entries)):
        return False
    shared = [e for e in entries if e['scope']['phase']=='compilation']
    suffix = [e for e in entries if e['scope']['phase']=='suffix_compilation']
    ordinary = [e for e in entries if e not in shared and e not in suffix]
    if ([(e['kind'],e['scope']) for e in shared] !=
        [(kind,dict(arm=sra.ARM,phase='compilation',path=[stage]))
         for stage in ('nominal','joint_jacobian') for _ in range(2) for kind in ('action_encoder','predict')]):
        return False
    if [(e['kind'],e['scope']) for e in suffix] != [
            (kind,dict(arm=ARMS[1],phase='suffix_compilation',path=[i,stage]))
            for i in range(40) for stage in ('nominal','jacobian') for kind in ('action_encoder','predict')]:
        return False
    return all(full20.entries_match([e for e in ordinary if e['scope']['arm']==arm],r['ledger'],arm)
        for arm,r in zip(ARMS,results,strict=True)) and len(entries)==504


def operation_counts(operations):
    done = [e for e in operations if e.get('completed')]
    jac = sum(e.get('kind') in ('jacobian','suffix_jacobian') for e in done)
    return dict(jacobian_calls=jac,scalar_output_vjp_directions=192*jac,
        svd_calls=sum(e.get('kind') in ('bounded_ridge_svd','mode_svd') for e in done)
            +sum(e.get('svd_calls',0) for e in done if e.get('kind')=='suffix_solve'),
        bounded_solve_calls=sum(e.get('kind')=='bounded_ridge_svd' for e in done)
            +sum(e.get('bounded_solve_calls',0) for e in done if e.get('kind')=='suffix_solve'))


def resource_receipt(entries, results, operations):
    """Actual completed/failed entry work, with shared attribution kept separate."""
    def entry_counts(events):
        return dict(predict_calls=sum(e['kind']=='predict' for e in events),
            predict_samples=sum(e['samples'] for e in events if e['kind']=='predict'),
            action_encoder_calls=sum(e['kind']=='action_encoder' for e in events),
            action_encoder_samples=sum(e['samples'] for e in events if e['kind']=='action_encoder'),
            predict_context_tokens=sum(e['context_tokens'] for e in events if e['kind']=='predict'),
            encoder_context_tokens=sum(e['context_tokens'] for e in events if e['kind']=='action_encoder'),
            failed_entries=sum(not e['completed'] for e in events))
    shared = [e for e in entries if e['scope'] and e['scope']['phase']=='compilation']
    exclusive = {arm:entry_counts([e for e in entries if e['scope'] and e['scope']['arm']==arm]) for arm in ARMS}
    shared_counts = entry_counts(shared)
    return dict(unique=entry_counts(entries),shared=shared_counts,exclusive=exclusive,
        standalone_attributed={arm:{k:v+shared_counts[k] for k,v in exclusive[arm].items()} for arm in ARMS},
        selection_evaluations={r['arm']:r['selection_evaluations'] for r in results},
        completed_operations=operation_counts(operations),
        operation_requests=sum(e.get('completed') is False for e in operations),
        vjp_scope='mathematical scalar-output directions, not measured backward API calls',
        wall_time_scope='root end-to-end CUDA-synchronized wall time includes compile/search/serialization; not a common time cap')


def calibration_gates(rows, model_unchanged, frozen_eval):
    return dict(all_four_roots=len(rows)==4,all_complete=all(r['complete'] for r in rows),
        actual_predict1008=sum(r['predict_calls'] for r in rows)==1008,
        actual_encoder1008=sum(r['action_encoder_calls'] for r in rows)==1008,
        predict_context3024=sum(r['predict_context_tokens'] for r in rows)==3024,
        encoder_context3024=sum(r['encoder_context_tokens'] for r in rows)==3024,
        jac164=sum(r['completed_operations']['jacobian_calls'] for r in rows)==164,
        selections320=sum(sum(r['resource_receipt']['selection_evaluations'].values()) for r in rows)==320,
        model_unchanged=model_unchanged,frozen_eval=frozen_eval)


def cuda(output):
    """Explicit main launch only; no physics, training or CPU fallback."""
    output,p,s = load_prepared(output)
    full20.write(output/'started.json',dict(protocol_sha256=digest(p),status='cuda_requested'))
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability.d2p_adaptive_budget_v2 import Binding,PredictionBudget
    host = base.require_cuda_allocation('cuda')
    import torch
    device = torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32 = False
    model = base.load_compatible_pretrained(Path(p['context']['policy']),device=device).eval().requires_grad_(False)
    before = checked.tensor_state_hash(model)
    torch.cuda.reset_peak_memory_stats(device)
    source = historical.audit.read_json(p['context']['compilation'])
    compiled_rows = {r['root_sha256']:r['compiled'] for r in source['rows']}
    rows = []
    for ready in p['context']['cases']:
        started = time.perf_counter()
        replay = full20.RootAnchoredCudaReplay(model,device)
        compiled = compiled_rows[ready['root_sha256']]
        results, operations, bank, error = [], [], None, None
        shared_compile_wall_time_s = None
        binding = Binding(s['source_seals'][0]['manifest_sha256'],TAG,ready['root_sha256'],checked.OFFICIAL_SHA)
        with (output/f'root_{ready["root_index"]}_journal.jsonl').open('x') as stream:
            def journal(event):
                if 'kind' in event:
                    operations.append(deepcopy(event))
                stream.write(json.dumps(event,allow_nan=False)+'\n')
                stream.flush()
            with full20.actual_entries(model,replay) as entries, PredictionBudget(
                    model,binding=binding,method=TAG,sample_limit=252) as budget:
                try:
                    with budget.phase('planning',method=TAG,horizon_tokens=2):
                        compile_started = time.perf_counter()
                        bank,_ = sra.compile_cuda(model,device,ready,compiled,replay,journal)
                        shared_compile_wall_time_s = time.perf_counter()-compile_started
                    full20.write(output/f'root_{ready["root_index"]}_bank.json',bank)
                    for arm in ARMS:
                        result = run_arm(arm=arm,bank=bank,root=ready['root'],
                            provenance=provenance_for(p,s,ready,bank,arm),action_domain=ready['action_domain'],
                            target_metric=full20.metric_for(compiled),replay=replay,
                            suffix=lambda c,u,i:suffix_cuda(model,device,replay,c,u,compiled,i,journal),
                            phase=lambda name:budget.phase(name,method=TAG,horizon_tokens=2),journal=journal)
                        results.append(result)
                except Exception as exc:
                    error = dict(error_type=type(exc).__name__,error=str(exc))
                actual = budget.receipt()
        counts = operation_counts(operations)
        expected_svd = 2+sum(a.get('suffix_compilation',{}).get('svd_calls',0)
            for r in results if r['arm']==ARMS[1] for a in r['attempts'])
        complete = (error is None and accounting_ok(entries,actual,results)
            and counts['jacobian_calls']==41 and counts['svd_calls']==expected_svd
            and len([e for e in operations if e.get('completed') and e['kind']=='suffix_solve'])==40)
        torch.cuda.synchronize(device)
        row = dict(root_sha256=ready['root_sha256'],root_index=ready['root_index'],results=results,
            actual_model_entries=entries,actual_prediction_budget=actual,operations=operations,
            completed_operations=counts,predict_calls=sum(e['kind']=='predict' for e in entries),
            action_encoder_calls=sum(e['kind']=='action_encoder' for e in entries),
            predict_context_tokens=sum(e['context_tokens'] for e in entries if e['kind']=='predict'),
            encoder_context_tokens=sum(e['context_tokens'] for e in entries if e['kind']=='action_encoder'),
            resource_receipt=resource_receipt(entries,results,operations),
            shared_compile_wall_time_s=shared_compile_wall_time_s,
            complete=complete,error=error,actual_root_wall_time_s=time.perf_counter()-started)
        full20.write(output/f'root_{ready["root_index"]}.json',row)
        rows.append(row)
    load_prepared(output)
    gates = calibration_gates(rows,before==checked.tensor_state_hash(model),
        all(not m.training for m in model.modules()) and all(not t.requires_grad for t in model.parameters()))
    report = dict(schema=TAG+'.report',status='calibration_completed' if all(gates.values()) else 'incomplete_or_failed',
        protocol_sha256=digest(p),source_seals=s['source_seals'],rows=rows,gates=gates,host=host,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
        actual_unique_predict_calls=sum(r['predict_calls'] for r in rows),
        actual_unique_action_encoder_calls=sum(r['action_encoder_calls'] for r in rows),
        actual_selection_evaluations=sum(sum(r['resource_receipt']['selection_evaluations'].values()) for r in rows),
        completed_jacobian_calls=sum(r['completed_operations']['jacobian_calls'] for r in rows),
        completed_svd_calls=sum(r['completed_operations']['svd_calls'] for r in rows),
        completed_scalar_output_vjp_directions=sum(r['completed_operations']['scalar_output_vjp_directions'] for r in rows),
        numpy_version=np.__version__,torch_version=torch.__version__,
        vjp_scope='mathematical scalar-output directions; not measured engine backward API calls',
        failed_work='all operation requests/entries retained; no refund, padding or rerun',**deepcopy(BOUNDARY))
    full20.write(output/'report.json',report)
    if not all(gates.values()):
        raise RuntimeError('incomplete calibration retained; no deployable actions')


def selected_actions(output):
    """Readonly source-bound export; array re-solves are separately disclosed."""
    output,p,s = load_prepared(output)
    reader = historical.audit.EvidenceReader()
    report = reader.load(output/'report.json')
    if (report['schema']!=TAG+'.report' or report['status']!='calibration_completed'
            or report['protocol_sha256']!=digest(p) or report['source_seals']!=s['source_seals']
            or not report['gates'] or not all(v is True for v in report['gates'].values())
            or report['gates']!=calibration_gates(report['rows'],True,True)
            or any(report[k]!=v for k,v in BOUNDARY.items())):
        raise ValueError('source-bound completed calibration required')
    rows = {r['root_sha256']:r for r in report['rows']}
    cases = {r['root_sha256']:r for r in p['context']['cases']}
    if len(report['rows'])!=4 or len(rows)!=4 or rows.keys()!=cases.keys():
        raise ValueError('four unique original roots required')
    source = reader.load(p['context']['compilation'],historical.audit.COMPILATION_SHA)
    compiled = {r['root_sha256']:r['compiled'] for r in source['rows']}
    actions, svds, solves = {}, 0, 0
    for key,row in rows.items():
        ready = cases[key]
        if (not row['complete'] or row['error'] is not None or row['root_index']!=ready['root_index']
                or not accounting_ok(row['actual_model_entries'],row['actual_prediction_budget'],row['results'])
                or operation_counts(row['operations'])!=row['completed_operations']
                or resource_receipt(row['actual_model_entries'],row['results'],row['operations'])!=row['resource_receipt']):
            raise ValueError('incomplete resource receipt')
        counts = row['resource_receipt']['unique']
        if (row['predict_calls']!=counts['predict_calls'] or row['action_encoder_calls']!=counts['action_encoder_calls']
                or row['predict_context_tokens']!=counts['predict_context_tokens']
                or row['encoder_context_tokens']!=counts['encoder_context_tokens']):
            raise ValueError('resource summary drift')
        bank = reader.load(output/f'root_{ready["root_index"]}_bank.json')
        c = compiled[key]
        if sra.spectral_bank(bank['jacobian'],bank['nominal'],c['goal_vectors'][0],c['scoring']['metric']['projection'])!=bank:
            raise ValueError('shared bank differs; same-runtime SVD reproduction required')
        svds += 2
        solves += 1
        for arm,result in zip(ARMS,row['results'],strict=True):
            attempts = result['attempts']
            if (result['arm']!=arm or result['fixture_only'] or len(attempts)!=40
                    or result['selection_evaluations']!=40 or not all(a['completed'] and a['metric_attempted'] for a in attempts)):
                raise ValueError('all real candidate slots required')
            prov = provenance_for(p,s,ready,bank,arm)
            for i,a in enumerate(attempts):
                if a['candidate_id']!=f'{arm}_{i:02d}' or a['index']!=i or a['prefix_logits']!=bank['logits'][i][:10]:
                    raise ValueError('shared prefix/candidate identity changed')
                joint = np.asarray(bank['logits'][i]).copy()
                if arm==ARMS[1]:
                    local = a['suffix_compilation']
                    rebuilt = suffix_solution(joint[:10],local['jacobian'],local['nominal'],c['goal_vectors'][0],c['scoring']['metric']['projection'])
                    rebuilt['input_context_sha256'] = digest(a['witness']['edges'][0]['output_context'])
                    if local!=rebuilt:
                        raise ValueError('own-context suffix solve changed')
                    svds += rebuilt['svd_calls']
                    solves += rebuilt['bounded_solve_calls']
                    joint[10:] = local['suffix_logits']
                expected_actions = np.tanh(joint).astype(np.float32).reshape(10,2).tolist()
                w = a['witness']
                n = 1 if arm==ARMS[0] else 2
                if len(w['edges'])!=n:
                    raise ValueError('wrong composition edge count')
                contexts = [e['input_context'] for e in w['edges']]+[w['edges'][-1]['output_context']]
                chunks = [e['actions'] for e in w['edges']]
                rebound = saved_witness(ready['root'],prov,ready['action_domain'],contexts,chunks,a['candidate_id'])
                if (w!=rebound or a['logits']!=joint.tolist() or a['actions']!=expected_actions
                        or w['receipt']['actions']!=expected_actions
                        or abs(full20._cost(a['cost'])-full20.metric_for(c)(w['receipt']['terminal_context']))>1e-12):
                    raise ValueError('all-candidate action/witness/original-Q binding changed')
            best = min(attempts,key=lambda a:a['cost'])
            winner = result['winner']
            if (winner is None or winner['candidate_id']!=best['candidate_id'] or winner['cost']!=best['cost']
                    or winner['witness']!=best['witness'] or winner['independent_full_replay_exact'] is not True
                    or result['replay_error'] is not None):
                raise ValueError('winner selection/replay differs')
            actions[key,arm] = np.asarray(best['actions'],dtype=np.float32)
    reader.unchanged()
    return dict(actions=actions,report_sha256=reader.hashes[str(output/'report.json')],
        verification_work=dict(neural_predict_calls=0,svd_calls=svds,bounded_solve_calls=solves,
            scope='additional array-only validation, not calibration ledger or independent neural proof'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='phase',required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--reference',type=Path,default=historical.DEFAULT_REFERENCE)
    prep.add_argument('--output',type=Path,required=True)
    for phase in ('verify','cuda'):
        sub.add_parser(phase).add_argument('--output',type=Path,required=True)
    args = vars(parser.parse_args(argv))
    phase = args.pop('phase')
    if phase=='prepare':
        prepare(**args)
    elif phase=='verify':
        load_prepared(**args)
    else:
        cuda(**args)
    print(json.dumps(dict(status=phase+'_completed',output=str(args['output']),P2_admission=False)))


if __name__=='__main__':
    main()
