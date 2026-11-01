"""Joint-H2 spectral shooting, main-thread predeclared v2; no P2 admission.

Prepare is model-free. Only the explicit cuda command may load a frozen neural
model, on host CUDA. No prefix branching or suffix Jacobian recompilation.
"""
import argparse
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import numpy as np

from experiments.spectral_reachability import p2_no_composition_sra_20261027_v1 as historical
from experiments.spectral_reachability import p2_full20_sampling_20261027_v2 as full20
from experiments.spectral_reachability import pusht_full_packet_compiler_20261027 as compiler
from experiments.spectral_reachability.d2p_composition_witness_v2 import compose, digest
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal

TAG = 'p2_no_composition_sra_20261027_v2'
ARM = 'joint_h2_no_composition_sra'
PROJECT = Path(__file__).resolve().parents[2]
CAP = float(np.sqrt(10)*.75)
RADII = (float(np.sqrt(10)*.25), CAP)
SCHEDULE = dict(horizon_tokens=2, primitive_steps=10, packet_shape=[2,5,2],
    proposal_dimensions=20, latent_dimension=192, nominal_logits=0., damping=.002,
    joint_logit_l2_cap=CAP, spectral_radii=list(RADII), dls_scales=[.25,.5,1.], top_modes=9,
    candidates=40, nominal_predict_samples=2, jacobian_forward_predict_samples=2,
    candidate_predict_samples=80, replay_predict_samples=4, total_predict_samples=88,
    total_predict_calls=88, jacobian_calls=1, svd_calls=2, scalar_output_vjp_directions=192,
    vectorized_autograd_jacobian_calls=1, bounded_solve_calls=1,
    action_encoder_calls=88, processed_context_token_samples=264,
    selection_evaluations=40, batch_size=1, optimizer_steps=0, physical_env_steps=0,
    candidate_order='zero; DLS scales .25/.5/1; mode 0..8, radius low/cap, sign -/+ ',
    selection='minimum nonlinear original Q terminal mean-square; stable first tie',
    spectral_sign='largest absolute coordinate positive; first argmax tie',
    degenerate_spectrum='retain NumPy LAPACK SVD basis including zero modes; no truncation/redraw; archive basis/runtime',
    action_mapping='raw float32 tanh(joint logits).reshape(2,5,2); one total 20D logit ball, not per-packet balls')
BOUNDARY = dict(calibration_only=True, P1_admission=False, P2_admission=False,
    physical_efficacy_verified=False, fair_total_budget_comparison=False,
    prediction_budget_matched=False, same_proposal_distribution=False,
    additional_predict_samples_over_reference=4,
    unmeasured=['FLOPs','upstream RGB encoding/bank construction','common wall-time cap',
        'backward engine internals/kernel multiplicity; VJP directions are not backward API calls'])


def spectral_bank(jacobian, nominal, goal, projection, journal=lambda e:None):
    """Pure array kernel; two SVDs total: bounded ridge and spectral modes."""
    j = compiler.finite(jacobian, (192,20))
    origin = compiler.finite(nominal, (192,))
    target = compiler.finite(goal, (192,))
    q = compiler.finite(projection, (192,192))
    response = q@j
    # Existing full-coordinate exact bounded ridge uses ONE SVD.
    def operation(name, callback):
        event = dict(kind=name,completed=False,svd_calls_requested=1)
        journal(deepcopy(event))
        started = time.perf_counter()
        try:
            value = callback()
            event['completed'] = True
            return value
        finally:
            event['wall_time_s'] = time.perf_counter()-started
            journal(deepcopy(event))
    dls = operation('bounded_ridge_svd',lambda:compiler.bounded_ridge(
        response, q@(target-origin), damping=.002, cap=CAP))
    _, singular, right = operation('mode_svd',lambda:np.linalg.svd(response, full_matrices=False))
    for vector in right:
        if vector[np.argmax(np.abs(vector))] < 0:
            vector *= -1
    logits = [np.zeros(20)] + [s*np.asarray(dls['params']) for s in (.25,.5,1.)]
    labels = ['zero','dls_scale_.25','dls_scale_.5','dls_scale_1']
    for mode in range(9):
        for radius_index, radius in enumerate(RADII):
            for sign in (-1,1):
                logits.append(sign*radius*right[mode])
                labels.append(f'mode_{mode}_radius_{radius_index}_sign_{sign}')
    logits = np.stack(logits)
    if logits.shape != (40,20) or np.max(np.linalg.norm(logits,axis=1)) > CAP*(1+1e-12):
        raise ValueError('joint ball/candidate schedule mismatch')
    commands = np.tanh(logits).astype(np.float32).reshape(40,2,5,2)
    return dict(logits=logits.tolist(), commands=commands.tolist(), labels=labels,
        dls=dls, singular_values=singular.tolist(), right_vectors=right.tolist(),
        jacobian=j.tolist(), nominal=origin.tolist(), projection_sha256=digest(q.tolist()),
        goal_sha256=digest(target.tolist()), jacobian_calls=1, svd_calls=2,
        scalar_output_vjp_directions=192, bounded_solve_calls=1)


def run_bank(*, bank, root, provenance, action_domain, target_metric, replay,
             fixture=False, phase=lambda name:nullcontext(), journal=lambda e:None):
    """Evaluate all forty complete programs; no compile/selection prefix scores."""
    if not fixture and type(replay) is not full20.RootAnchoredCudaReplay:
        raise ValueError('sealed host CUDA replay required, no neural CPU fallback')
    commands = compiler.finite(bank['commands'], (40,2,5,2))
    logits = compiler.finite(bank['logits'], (40,20))
    if (bank['labels'] != ['zero','dls_scale_.25','dls_scale_.5','dls_scale_1']+
            [f'mode_{i}_radius_{r}_sign_{s}' for i in range(9) for r in range(2) for s in (-1,1)]
            or not np.array_equal(commands.astype(np.float32),np.tanh(logits).astype(np.float32).reshape(40,2,5,2))
            or np.max(np.linalg.norm(logits,axis=1)) > CAP*(1+1e-12)):
        raise ValueError('frozen bank mapping changed')
    initial = full20._validated_context(root['context'])
    def edge(terminal, actions, identity):
        return dict(program_id=identity,kind='progress',root_id=root['root_id'],
            provenance=deepcopy(provenance),action_domain=deepcopy(action_domain),
            input_context=deepcopy(initial),output_context=deepcopy(terminal),actions=deepcopy(actions))
    compose(root=root,provenance=provenance,action_domain=action_domain,
        edges=[edge(initial,full20.raw_actions(commands[0]),'metadata')],
        replay=lambda c,a:c,max_edges=1,max_actions=10)
    if (action_domain['name'] != 'raw' or action_domain['low'] != [-1.,-1.]
            or action_domain['high'] != [1.,1.] or action_domain['action_block'] != 5):
        raise ValueError('official raw box/action block required')
    ledger = full20._Ledger(replay,ARM)
    attempts, best, witness, error = [], None, None, None
    started = time.perf_counter()
    with phase('planning'):
        for index, raw in enumerate(commands):
            event = dict(candidate_id=f'{ARM}_{index:02d}',label=bank['labels'][index],
                logits=logits[index].tolist(),commands=raw.tolist(),actions=full20.raw_actions(raw),
                metric_attempted=False,completed=False)
            attempts.append(event)
            try:
                terminal = ledger.call(initial,event['actions'],'planning',[index])
                event['metric_attempted'] = True
                cost = full20._cost(target_metric(deepcopy(terminal)))
                event.update(completed=True,cost=cost)
                if best is None or cost < best['cost']:
                    best = dict(candidate_id=event['candidate_id'],cost=cost,
                        actions=deepcopy(event['actions']),terminal=deepcopy(terminal))
            except Exception as exc:
                event.update(error_type=type(exc).__name__,error=str(exc))
            journal(deepcopy(event))
    # No failed-search winner is deployable; no refund/padding/retry.
    if best is not None and all(a['completed'] for a in attempts):
        with phase('evaluation'):
            try:
                actions = best['actions']
                mid = ledger.call(initial,actions[:5],'replay',[40])
                end = ledger.call(mid,actions[5:],'replay',[41])
                full = ledger.call(initial,actions,'replay',[42])
                if full != end or full != best['terminal']:
                    raise ValueError('search/segmented/full independent replay mismatch')
                edges = [edge(end,actions,best['candidate_id'])]
                receipt = compose(root=root,provenance=provenance,action_domain=action_domain,
                    edges=edges,replay=lambda c,a:deepcopy(full),max_edges=1,max_actions=10)
                witness = dict(receipt=receipt,edges=edges,independent_full_replay_exact=True)
            except Exception as exc:
                error = dict(error_type=type(exc).__name__,error=str(exc))
    counts = {name:sum(e['predict_samples'] for e in ledger.events if e['phase']==name)
              for name in ('planning','replay')}
    okay = witness is not None and counts == dict(planning=80,replay=4) and all(e['completed'] for e in ledger.events)
    return dict(arm=ARM,fixture_only=fixture,attempts=attempts,ledger=ledger.events,
        selected_candidate_id=best['candidate_id'] if okay else None,
        selected_model_cost=best['cost'] if okay else None,witness=witness if okay else None,
        replay_error=error,planning_predict_samples=counts['planning'],replay_predict_samples=counts['replay'],
        candidate_and_replay_complete=okay,status='search_complete' if okay else 'incomplete_or_failed',
        selection_evaluations=sum(e['metric_attempted'] for e in attempts),
        attempted_candidate_evaluations=len(attempts),
        actual_wall_time_s=time.perf_counter()-started,**deepcopy(BOUNDARY))


def compile_cuda(model, device, ready, compiled, replay, journal):
    """One 192x20 joint Jacobian, no intermediate detach or suffix compilation."""
    if getattr(device,'type',None) != 'cuda':
        raise ValueError('host CUDA only')
    import torch
    transition = compiler.make_transition(model,device)
    context = ready['root']['context']
    emb = torch.tensor(context['latent_history'],dtype=torch.float32,device=device)[None]
    history = torch.tensor(context['action_history'],dtype=torch.float32,device=device)[None]
    zero = emb.new_zeros(20)
    events = []
    def terminal(u):
        state, between = emb, history
        raw = torch.tanh(u).reshape(2,5,2)
        for packet in range(2):
            event = dict(phase='compilation',stage=stage,packet=packet,completed=False)
            events.append(event)
            try:
                state,between = transition(state,between,raw[packet])
                event['completed'] = True
            finally:
                journal(deepcopy(event))
        return state[0,-1]
    try:
        stage = 'nominal'
        replay._p2_scope = dict(arm=ARM,phase='compilation',path=[stage])
        with torch.no_grad():
            nominal = terminal(zero)
        stage = 'joint_jacobian'
        replay._p2_scope = dict(arm=ARM,phase='compilation',path=[stage])
        request = dict(kind='jacobian',shape=[192,20],vectorize=True,
            scalar_output_vjp_directions_requested=192,completed=False)
        journal(deepcopy(request))
        jac = torch.autograd.functional.jacobian(terminal,zero.requires_grad_(True),vectorize=True)
        request['completed'] = True
        journal(deepcopy(request))
        arrays = lambda t:t.detach().cpu().numpy().astype(np.float64)
        return spectral_bank(arrays(jac),arrays(nominal),compiled['goal_vectors'][0],
            compiled['scoring']['metric']['projection'],journal=journal), events
    finally:
        replay._p2_scope = None


def source_files():
    return sorted(set(historical.source_files()+[Path(__file__),
        PROJECT/'tests'/f'test_{TAG}.py',PROJECT/'innerdocs'/f'{TAG}.md',
        Path(compiler.__file__),Path(full20.__file__),
        PROJECT/'experiments/spectral_reachability/d2p_adaptive_budget_v2.py',
        PROJECT/'experiments/spectral_reachability/d2p_model_call_instrumentation.py']))


def prepare(reference, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    historical.audit.require(TAG in output.name,'new dedicated tagged output required')
    reader = historical.audit.EvidenceReader()
    context = historical.reference_contract(reference,historical.DEFAULT_ADMISSION,reader)
    # Bind and validate full runtime closure using existing model-free checks.
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    source,_,config = checked.validate_inputs(context['compilation'],context['policy'])
    files = source_files()+checked.sources_before_allocate(source,config)
    files += [Path(m.__file__) for n,m in tuple(sys.modules.items())
              if n.startswith('experiments.') and getattr(m,'__file__','').endswith('.py')]
    files = sorted(set(p.resolve() for p in files))
    protocol = dict(schema=TAG+'.protocol',context=context,schedule=deepcopy(SCHEDULE),
        boundary=deepcopy(BOUNDARY),bindings=deepcopy(reader.hashes),
        authorization='main-thread predeclared under authorized implementation; not inherited v1 refusal',
        frozen_before_new_v2_outcomes=True,source_paths=[str(p) for p in files],
        expected_four_root_totals=dict(predict_calls=352,predict_samples=352,jacobian_calls=4,
            svd_calls=8,scalar_output_vjp_directions=768,processed_context_token_samples=1056))
    output.mkdir(exist_ok=False)
    full20.write(output/'protocol.json',protocol)
    seals = [create_seal(files=files[i:i+40],args_json=json.dumps(dict(protocol_sha256=digest(protocol))),
                        output_dir=output/f'source_seal_{i//40}') for i in range(0,len(files),40)]
    full20.write(output/'prepared.json',dict(schema=TAG+'.prepared',status='prepared_not_executed',
        protocol_sha256=digest(protocol),source_seals=seals,new_neural_calls=0,new_physical_steps=0,
        P2_admission=False))
    reader.unchanged()
    load_prepared(output)
    return protocol


def load_prepared(output):
    output = Path(output).resolve()
    protocol = historical.audit.read_json(output/'protocol.json')
    prepared = historical.audit.read_json(output/'prepared.json')
    if (protocol['schema'] != TAG+'.protocol' or prepared['schema'] != TAG+'.prepared'
            or digest(protocol) != prepared['protocol_sha256'] or protocol['schedule'] != SCHEDULE
            or protocol['boundary'] != BOUNDARY or protocol['frozen_before_new_v2_outcomes'] is not True):
        raise ValueError('v2 frozen protocol changed')
    paths = []
    for seal in prepared['source_seals']:
        verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
        directory = Path(seal['output_dir'])
        if historical.audit.read_json(directory/'args.json') != dict(protocol_sha256=digest(protocol)):
            raise ValueError('source/protocol binding changed')
        paths += [e['source'] for e in historical.audit.read_json(directory/'manifest.json')['files']]
    if sorted(paths) != protocol['source_paths'] or not set(map(str,source_files())) <= set(paths):
        raise ValueError('source closure missing')
    reader = historical.audit.EvidenceReader()
    reader.bindings(protocol['bindings'])
    context = protocol['context']
    actual = historical.reference_contract(context['reference'],context['admission'],reader)
    if actual != context:
        raise ValueError('root/goal/context changed')
    reader.unchanged()
    return output,protocol,prepared


def provenance_for(protocol, prepared, ready):
    return dict(checkpoint_sha256=historical.audit.CHECKPOINT_SHA,
        encoder_sha256=historical.audit.CHECKPOINT_SHA,
        source_sha256=prepared['source_seals'][0]['manifest_sha256'],protocol_sha256=digest(protocol),
        metric_sha256=ready['metric_sha256'],candidate_bank_sha256=digest(SCHEDULE))


def accounting_ok(entries, actual, result):
    phases = [(p['phase'],p['predict_calls'],p['predict_samples']) for p in actual['phases']]
    compilation = [e for e in entries if e['scope'] and e['scope']['phase']=='compilation']
    rest = [e for e in entries if e not in compilation]
    # Existing guard admits planning/evaluation only. Keep a separate four-call
    # planning scope for compilation; actual entry metadata labels its role.
    return (phases == [('planning',4,4),('planning',80,80),('evaluation',4,4)]
        and all(p['completed'] for p in actual['phases'])
        and len(compilation)==8 and all(e['completed'] and e['samples']==1 and e['context_tokens']==3 for e in entries)
        and [(e['kind'],e['scope']['path']) for e in compilation] ==
            [(kind,[stage]) for stage in ('nominal','joint_jacobian') for _ in range(2)
             for kind in ('action_encoder','predict')]
        and full20.entries_match(rest,result['ledger'],ARM))


def cuda(output):
    """Main-thread explicit launch only. Not called by prepare or tests."""
    output,protocol,prepared = load_prepared(output)
    full20.write(output/'started.json',dict(protocol_sha256=digest(protocol),status='cuda_requested'))
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability.d2p_adaptive_budget_v2 import Binding,PredictionBudget
    host = base.require_cuda_allocation('cuda')
    import torch
    device = torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32 = False
    model = base.load_compatible_pretrained(Path(protocol['context']['policy']),device=device).eval().requires_grad_(False)
    before = checked.tensor_state_hash(model)
    torch.cuda.reset_peak_memory_stats(device)
    source = historical.audit.read_json(protocol['context']['compilation'])
    source_rows = {r['root_sha256']:r['compiled'] for r in source['rows']}
    rows = []
    for ready in protocol['context']['cases']:
        root_started = time.perf_counter()
        prov = provenance_for(protocol,prepared,ready)
        binding = Binding(prov['source_sha256'],TAG,ready['root_sha256'],checked.OFFICIAL_SHA)
        replay = full20.RootAnchoredCudaReplay(model,device)
        bank, result, error = None, None, None
        operations = []
        with (output/f'root_{ready["root_index"]}_journal.jsonl').open('x') as stream:
            def journal(event):
                if event.get('kind') in ('jacobian','bounded_ridge_svd','mode_svd'):
                    operations.append(deepcopy(event))
                stream.write(json.dumps(event,allow_nan=False)+'\n')
                stream.flush()
            with full20.actual_entries(model,replay) as entries, PredictionBudget(
                    model,binding=binding,method=ARM,sample_limit=88) as budget:
                try:
                    with budget.phase('planning',method=ARM,horizon_tokens=2):
                        bank, compile_events = compile_cuda(model,device,ready,
                            source_rows[ready['root_sha256']],replay,journal)
                    full20.write(output/f'root_{ready["root_index"]}_bank.json',bank)
                    prov['candidate_bank_sha256'] = digest(bank)
                    result = run_bank(bank=bank,root=ready['root'],provenance=prov,
                        action_domain=ready['action_domain'],target_metric=full20.metric_for(source_rows[ready['root_sha256']]),
                        replay=replay,phase=lambda name:budget.phase(name,method=ARM,horizon_tokens=2),journal=journal)
                except Exception as exc:
                    error = dict(error_type=type(exc).__name__,error=str(exc))
                actual = budget.receipt()
        okay = (error is None and result is not None and result['candidate_and_replay_complete']
                 and accounting_ok(entries,actual,result))
        torch.cuda.synchronize(device)
        completed_ops = [e for e in operations if e['completed']]
        okay = okay and [e['kind'] for e in completed_ops] == ['jacobian','bounded_ridge_svd','mode_svd']
        row = dict(root_index=ready['root_index'],root_sha256=ready['root_sha256'],
            root=ready['root'],action_domain=ready['action_domain'],result=result,error=error,
            complete=okay,actual_model_entries=entries,actual_prediction_budget=actual,
            predict_calls=sum(e['kind']=='predict' for e in entries),
            action_encoder_calls=sum(e['kind']=='action_encoder' for e in entries),
            action_encoder_processed_context_token_samples=sum(e['context_tokens'] for e in entries if e['kind']=='action_encoder'),
            processed_context_token_samples=sum(e['context_tokens'] for e in entries if e['kind']=='predict'),
            compilation_operations=operations,
            compile_receipt=dict(completed_jacobian_calls=sum(e['kind']=='jacobian' for e in completed_ops),
                completed_svd_calls=sum(e['kind'] in ('bounded_ridge_svd','mode_svd') for e in completed_ops),
                completed_scalar_output_vjp_directions=192*sum(e['kind']=='jacobian' for e in completed_ops),
                vjp_count_scope='mathematical scalar-output directions in completed vectorized Jacobian; not measured engine backward calls',
                failed_or_partial_work='operation journal and actual model entries retained; no refund'),
            actual_root_wall_time_s=time.perf_counter()-root_started,**deepcopy(BOUNDARY))
        full20.write(output/f'root_{ready["root_index"]}.json',row)
        rows.append(row)
    load_prepared(output)
    gates = dict(all_four_retained=len(rows)==4,all_complete=all(r['complete'] for r in rows),
        actual352=sum(r['predict_calls'] for r in rows)==352,
        actual_context1056=sum(r['processed_context_token_samples'] for r in rows)==1056,
        completed_jac4=sum(r['compile_receipt']['completed_jacobian_calls'] for r in rows)==4,
        completed_svd8=sum(r['compile_receipt']['completed_svd_calls'] for r in rows)==8,
        model_unchanged=before==checked.tensor_state_hash(model),
        frozen_eval=all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters()))
    report = dict(schema=TAG+'.report',status='calibration_completed' if all(gates.values()) else 'incomplete_or_failed',
        protocol_sha256=digest(protocol),source_seals=prepared['source_seals'],rows=rows,gates=gates,host=host,
        actual_predict_calls=sum(r['predict_calls'] for r in rows),
        actual_action_encoder_calls=sum(r['action_encoder_calls'] for r in rows),
        actual_action_encoder_processed_context_token_samples=sum(r['action_encoder_processed_context_token_samples'] for r in rows),
        actual_selection_evaluations=sum(r['result']['selection_evaluations'] if r['result'] else 0 for r in rows),
        actual_processed_context_token_samples=sum(r['processed_context_token_samples'] for r in rows),
        completed_jacobian_calls=sum(r['compile_receipt']['completed_jacobian_calls'] for r in rows),
        completed_svd_calls=sum(r['compile_receipt']['completed_svd_calls'] for r in rows),
        completed_scalar_output_vjp_directions=sum(r['compile_receipt']['completed_scalar_output_vjp_directions'] for r in rows),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
        numpy_version=np.__version__,torch_version=torch.__version__,**deepcopy(BOUNDARY))
    full20.write(output/'report.json',report)
    if not all(gates.values()):
        raise RuntimeError('retain failed branches; no physical deployment')


def selected_actions(output):
    """Read-only export after a real calibration; never execute physics.

    Revalidate source/input seals, bank/actions and every saved candidate before
    exporting. Hashes bind saved data, not independent neural execution proof.
    """
    output,protocol,prepared = load_prepared(output)
    reader = historical.audit.EvidenceReader()
    report = reader.load(output/'report.json')
    if (report['schema'] != TAG+'.report' or report['status'] != 'calibration_completed'
            or report['protocol_sha256'] != digest(protocol)
            or report['source_seals'] != prepared['source_seals'] or not report['gates']
            or not all(v is True for v in report['gates'].values())):
        raise ValueError('complete source-bound real calibration required')
    for key,value in BOUNDARY.items():
        if report[key] != value:
            raise ValueError('claim boundary changed')
    rows = {r['root_sha256']:r for r in report['rows']}
    ready = {r['root_sha256']:r for r in protocol['context']['cases']}
    if len(report['rows']) != 4 or len(rows) != 4 or rows.keys() != ready.keys():
        raise ValueError('all four unique roots required')
    compilation = reader.load(protocol['context']['compilation'],historical.audit.COMPILATION_SHA)
    compiled = {r['root_sha256']:r['compiled'] for r in compilation['rows']}
    actions = {}
    for key,row in rows.items():
        case = ready[key]
        result = row['result']
        if (row['complete'] is not True or row['error'] is not None
                or row['root'] != case['root'] or row['action_domain'] != case['action_domain']
                or result['fixture_only'] is not False or result['candidate_and_replay_complete'] is not True
                or result['status'] != 'search_complete'
                or not accounting_ok(row['actual_model_entries'],row['actual_prediction_budget'],result)):
            raise ValueError('incomplete/fixture or root/domain mismatch')
        bank = reader.load(output/f'root_{case["root_index"]}_bank.json')
        rebuild = spectral_bank(bank['jacobian'],bank['nominal'],compiled[key]['goal_vectors'][0],
            compiled[key]['scoring']['metric']['projection'])
        if bank != rebuild:
            raise ValueError('spectral bank changed; cross-runtime SVD basis reproduction is required for this exporter')
        attempts = result['attempts']
        if len(attempts) != 40 or not all(e['completed'] is True for e in attempts):
            raise ValueError('all forty candidates required')
        for index,event in enumerate(attempts):
            if (event['candidate_id'] != f'{ARM}_{index:02d}' or event['label'] != bank['labels'][index]
                    or event['logits'] != bank['logits'][index] or event['commands'] != bank['commands'][index]
                    or event['actions'] != full20.raw_actions(bank['commands'][index])):
                raise ValueError('candidate bank/action binding changed')
        best = min(attempts,key=lambda e:full20._cost(e['cost']))
        witness = result['witness']
        if (witness is None or witness['independent_full_replay_exact'] is not True
                or len(witness['edges']) != 1 or result['replay_error'] is not None):
            raise ValueError('one-edge independent H2 replay required')
        prov = provenance_for(protocol,prepared,case)
        prov['candidate_bank_sha256'] = digest(bank)
        receipt = witness['receipt']
        rebound = compose(root=case['root'],provenance=prov,action_domain=case['action_domain'],
            edges=witness['edges'],replay=lambda c,a:deepcopy(receipt['terminal_context']),max_edges=1,max_actions=10)
        if (rebound != receipt or receipt['actions'] != best['actions']
                or len(receipt['actions']) != 10 or best['candidate_id'] != result['selected_candidate_id']
                or best['cost'] != result['selected_model_cost']
                or abs(best['cost']-full20.metric_for(compiled[key])(receipt['terminal_context']))>1e-12):
            raise ValueError('winner/witness/original Q cost differs')
        actions[key,ARM] = np.asarray(receipt['actions'],dtype=np.float32)
    reader.unchanged()
    return dict(actions=actions,report_sha256=reader.hashes[str(output/'report.json')],
        verification_work=dict(svd_calls=8,bounded_solve_calls=4,neural_predict_calls=0,
            scope='additional read-only array verification, NOT included in the 352-call calibration ledger'))


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
    if phase == 'prepare':
        prepare(**args)
    elif phase == 'verify':
        load_prepared(**args)
    else:
        cuda(**args)
    print(json.dumps(dict(status=phase+'_completed',output=str(args['output']),P2_admission=False)))


if __name__ == '__main__':
    main()
