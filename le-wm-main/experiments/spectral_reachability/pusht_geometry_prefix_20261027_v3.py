"""Prefix synthesis using variation of future action response, host CUDA only.

Known-root, fixed-protocol generation ablation. No physical result selection,
training, stage admission or claim of physical controllability from model rank.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import pusht_full_packet_compiler_20261027 as packet

TAG = 'pusht_geometry_prefix_20261027_v3'
ARMS = ('geometry', 'stopK')
EPSILONS = (.01, .005)  # first is the fixed generator; second diagnostic only
ITERATIONS = 2
STEP = packet.CAP/4


def objective(measured, goal, q, fixed_jacobian=None):
    goal=packet.finite(goal)
    if goal.ndim!=1 or not len(goal):raise ValueError('nonempty goal vector required')
    q=packet.finite(q,(len(goal),len(goal)))
    prefix=packet.finite(measured['predicted_prefix'],goal.shape)
    stage = float(np.mean((q@(goal-prefix))**2))
    jac = np.asarray(measured['jacobian'] if fixed_jacobian is None else fixed_jacobian)
    own = packet.bounded_ridge(q@jac, q@(goal-np.asarray(measured['nominal_terminal'])))
    return dict(stage_cost=stage, own_reach_value=own['own_reach_value'],
                objective=stage+own['own_reach_value'], suffix_solve=own)


def gradients_from_records(records, goal, q, *, center=None, epsilons=EPSILONS):
    center = np.zeros(10,dtype=np.float32) if center is None else np.asarray(center,dtype=np.float32)
    if len(records) != 1+20*len(epsilons) or not np.array_equal(records[0]['prefix_params'],center):
        raise ValueError('center plus signed full-coordinate finite-difference grids required')
    fixed = np.asarray(records[0]['jacobian'])
    results = []
    for grid, epsilon in enumerate(epsilons):
        gradients = {arm: [] for arm in ARMS}
        for axis in range(10):
            minus, plus = records[1+grid*20+2*axis:3+grid*20+2*axis]
            for sign, r in ((-1, minus), (1, plus)):
                expected = center.copy(); expected[axis] += sign*epsilon
                if not np.array_equal(r['prefix_params'], expected):
                    raise ValueError('finite-difference grid identity/coordinate mismatch')
            for arm in ARMS:
                kwargs = {} if arm == 'geometry' else dict(fixed_jacobian=fixed)
                scores = [objective(r, goal, q, **kwargs)['objective'] for r in (minus, plus)]
                gradients[arm].append((scores[1]-scores[0])/(2*epsilon))
        if not all(np.isfinite(v).all() for v in gradients.values()):
            raise ValueError('nonfinite objective derivative')
        results.append(dict(epsilon=epsilon, gradients=gradients))
    return results


def update(center, gradient):
    center = packet.finite(center,(10,))
    gradient = packet.finite(gradient, (10,))
    norm = np.linalg.norm(gradient)
    direction = -gradient/norm if norm > 0 else np.zeros(10)
    result = center+STEP*direction
    if np.linalg.norm(result) > packet.CAP+1e-6:
        raise ValueError('fixed starts and two updates must remain in the declared prefix ball')
    return result.astype(np.float32).astype(float).tolist()


def starts(common):
    return [np.zeros(10,dtype=np.float32).tolist(),
            (np.asarray(common['params'][1])*.5).astype(np.float32).astype(float).tolist()]


def augmented_bank(common, runs, goal, q, arm):
    if arm not in ARMS or len(runs)!=2:
        raise ValueError('two fixed optimization starts per declared arm required')
    endpoints=[]
    for initial,run in zip(starts(common),runs,strict=True):
        center=initial
        if len(run)!=ITERATIONS:
            raise ValueError('two fixed iterations required')
        for trace in run:
            if center!=trace['center_params']:
                raise ValueError('optimizer center changed')
            gradients=gradients_from_records(trace['probe_records'],goal,q,center=center,epsilons=EPSILONS[:1])
            next_params=update(center,gradients[0]['gradients'][arm])
            if gradients!=trace['gradients'] or next_params!=trace['next_params']:
                raise ValueError('geometry/fixed-response update differs')
            center=next_params
        endpoints.append(center)
    return dict(params=list(common['params'])+endpoints,
                kinds=list(common['kinds'])+['future_objective_from_zero','future_objective_from_half_dls'])


def measure(prefix, *, emb, history, transition, counter, role, counts):
    import torch
    params = torch.tensor(prefix, dtype=emb.dtype, device=emb.device)
    with counter.phase('planning', method=role, horizon_tokens=2):
        with torch.no_grad():
            after, between = transition(emb, history, torch.tanh(params).reshape(5,2))
        after, between = after.detach(), between.detach()
        def terminal(u):
            return transition(after, between, torch.tanh(u).reshape(5,2))[0][0,-1]
        zero = emb.new_zeros(10)
        with torch.no_grad():
            nominal = terminal(zero)
        jac = torch.autograd.functional.jacobian(terminal, zero.requires_grad_(True), vectorize=True)
        counts['jacobian_backwards'] += 1
    array = lambda x: x.detach().cpu().numpy().astype(np.float64).tolist()
    result = dict(prefix_params=array(params), predicted_prefix=array(after[0,-1]),
                  nominal_terminal=array(nominal), jacobian=array(jac))
    return result, after, between, terminal


def reference_commands(prefix_params,suffix_params):
    """High-precision mathematical reference; threshold stays exactly 1e-7.

    NumPy float32 tanh is itself an approximate implementation, not a stable
    reference for CUDA float32 tanh; two approximate errors can add.
    """
    return np.concatenate([np.tanh(packet.finite(prefix_params,(10,))).reshape(5,2),
                           np.tanh(packet.finite(suffix_params,(10,))).reshape(5,2)])


def verify_plans(report, source):
    if report['status'] != 'plans_sealed' or not all(v is True for v in report['gates'].values()):
        raise ValueError('complete sealed geometry planning required')
    if report['protocol']['epsilons'] != list(EPSILONS) or report['protocol']['arms'] != list(ARMS):
        raise ValueError('fixed generation protocol differs')
    if any(report['protocol'][k]!=v for k,v in dict(damping=packet.DAMPING,cap=packet.CAP,
            radii=list(packet.RADII),iterations=ITERATIONS,step=STEP,candidates_per_arm=12).items()):
        raise ValueError('fixed objective/optimizer constraints differ')
    roots = {r['root_sha256']:r for r in source['rows']}
    if len(report['plans']) != 4 or len(roots) != 4 or {p['root_sha256'] for p in report['plans']} != set(roots):
        raise ValueError('original four unique roots required')
    plans = []
    for p in report['plans']:
        compiled = roots[p['root_sha256']]['compiled']
        goal, q = map(np.asarray, (compiled['goal_vectors'][0], compiled['scoring']['metric']['projection']))
        grids = gradients_from_records(p['probe_records'], goal, q)
        if grids != p['gradient_grids']:
            raise ValueError('saved finite-difference derivatives differ')
        common = packet.spectral_bank(np.asarray(p['root_jacobian']), np.asarray(p['root_nominal']), goal, q)
        if len(p['arms']) != 2 or {a['arm'] for a in p['arms']} != set(ARMS):
            raise ValueError('two unique arms required')
        for arm in p['arms']:
            bank = augmented_bank(common, arm['optimization'],goal,q,arm['arm'])
            if len(arm['candidates']) != 12 or bank != arm['bank']:
                raise ValueError('twelve fixed generation candidates required')
            for i,c in enumerate(arm['candidates']):
                expected = np.asarray(bank['params'][i], dtype=np.float32).astype(float)
                score = objective(c, goal, q)
                suffix = np.asarray(score['suffix_solve']['params'],dtype=np.float32)
                if (c['candidate_id'] != f'prefix_{i}' or not np.array_equal(c['prefix_params'],expected)
                        or not np.array_equal(c['suffix_params'],suffix.astype(float))
                        or not np.allclose(c['commands'],reference_commands(c['prefix_params'],suffix.astype(float)),rtol=0,atol=1e-7)
                        or c['independent_model_replay_exact'] is not True
                        or any(c[k] != score[k] for k in ('stage_cost','own_reach_value','objective'))):
                    raise ValueError('candidate/score/action/replay binding differs')
                plans.append(dict(root_index=p['root_index'],root_sha256=p['root_sha256'],
                    method=arm['arm']+'_'+c['candidate_id'],commands=c['commands'],
                    planned_horizon_tokens=2,planned_commands=10,
                    selected=c['candidate_id']==arm['selected_candidate_id'],
                    greedy=c['candidate_id']==arm['greedy_candidate_id'],candidate_id=c['candidate_id']))
            if (min(arm['candidates'],key=lambda c:c['objective'])['candidate_id'] != arm['selected_candidate_id']
                    or min(arm['candidates'],key=lambda c:c['stage_cost'])['candidate_id'] != arm['greedy_candidate_id']):
                raise ValueError('saved selection differs')
    if len(plans) != 96 or len({(p['root_sha256'],p['method']) for p in plans}) != 96:
        raise ValueError('complete unique 96-program census required')
    return plans


def run(compilation, policy, output):
    import torch
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability.pusht_composition_cuda_replay_v2 import CudaReplay
    from experiments.spectral_reachability.d2p_model_call_instrumentation import PredictInstrumentation
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal
    source, bindings, config = checked.validate_inputs(compilation,policy)
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    protocol = dict(schema=TAG,compilation=str(Path(compilation).resolve()),policy=str(Path(policy).resolve()),
        bindings=bindings,roots=source['protocol']['roots'],arms=list(ARMS),epsilons=list(EPSILONS),
        damping=packet.DAMPING,cap=packet.CAP,radii=list(packet.RADII),candidates_per_arm=12,
        generation='shared10 spectral/DLS prefixes +2 two-iteration optimized prefixes from zero and half rootDLS',
        contrast='full K(B) derivative vs center-fixed K derivative, refreshed each iteration; same starts/step/iterations; actual K used to score both banks',
        iterations=ITERATIONS,step=STEP,starts='zero and half current-root DLS',
        diagnostic_half_step_not_used_for_generation=True,root_outcomes_used=False,
        selection='unchanged stage plus bounded ridge value, beta1; stable bank ties',
        scope='known-root generation ablation, not independent P1 or equal total runtime efficacy',P1_admission=False)
    files = checked.sources_before_allocate(source,config)+[Path(__file__),Path(packet.__file__),
        Path(__file__).resolve().parents[2]/f'tests/test_{TAG}.py',
        Path(__file__).resolve().parents[3]/f'docs/{TAG}.md']
    output.mkdir(exist_ok=False)
    base.write_new(output/'protocol.json',protocol)
    files = sorted(set(p.resolve() for p in files))
    seals = [create_seal(files=files[i:i+40],args_json=json.dumps(protocol),output_dir=output/f'source_seal_{i//40}') for i in range(0,len(files),40)]
    print('Generate prefixes from future reachability derivative versus frozen-response derivative; same goal/Q and fixed action caps. No physical outcomes.',flush=True)
    host=base.require_cuda_allocation('cuda'); device=torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32=False
    model=base.load_compatible_pretrained(Path(policy),device=device).eval().requires_grad_(False)
    before=checked.tensor_state_hash(model); transition=packet.make_transition(model,device)
    counts=dict(jacobian_backwards=0); plans=[]
    with PredictInstrumentation(model) as counter:
        for row in source['rows']:
            compiled=row['compiled']; initial=compiled['input_context']
            emb=torch.tensor(initial['emb'],dtype=torch.float32,device=device)
            history=torch.tensor(initial['actions'],dtype=torch.float32,device=device)[:,1:]
            goal,q=map(np.asarray,(compiled['goal_vectors'][0],compiled['scoring']['metric']['projection']))
            zero=np.zeros(10)
            with counter.phase('planning',method='shared_current_response',horizon_tokens=1):
                def current(u):return transition(emb,history,torch.tanh(u).reshape(5,2))[0][0,-1]
                origin=current(emb.new_zeros(10)); jac=torch.autograd.functional.jacobian(current,emb.new_zeros(10).requires_grad_(True),vectorize=True)
                counts['jacobian_backwards']+=1
            jac_np=jac.detach().cpu().numpy().astype(float); origin_np=origin.detach().cpu().numpy().astype(float)
            records=[]
            def probe(center,epsilons,role):
                center=np.asarray(center,dtype=np.float32)
                probes=[center.copy()]
                for epsilon in epsilons:
                    for axis in range(10):
                        for sign in (-1,1):
                            delta=center.copy(); delta[axis]+=sign*epsilon; probes.append(delta)
                measured=[]
                for params in probes:
                    if np.linalg.norm(params)>packet.CAP+1e-6:
                        raise ValueError('finite-difference query outside prefix logit ball')
                    value,*_=measure(params,emb=emb,history=history,transition=transition,counter=counter,
                                      role=role,counts=counts)
                    measured.append(value)
                return measured
            records=probe(zero,EPSILONS,'shared_zero_fd_diagnostic')
            grids=gradients_from_records(records,goal,q)
            common=packet.spectral_bank(jac_np,origin_np,goal,q); arms=[]
            initial_replay=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,
                action_dim=10,latent_history=emb[0].cpu().tolist(),action_history=history[0].cpu().tolist())
            replay=CudaReplay(model,device)
            for arm in (ARMS if row['root_index']%2==0 else ARMS[::-1]):
                runs=[]
                for center in starts(common):
                    trace=[]
                    for iteration in range(ITERATIONS):
                        measurements=probe(center,EPSILONS[:1],'optimizer_'+arm)
                        gradients=gradients_from_records(measurements,goal,q,center=center,epsilons=EPSILONS[:1])
                        next_params=update(center,gradients[0]['gradients'][arm])
                        trace.append(dict(iteration=iteration,center_params=center,probe_records=measurements,
                                          gradients=gradients,next_params=next_params))
                        center=next_params
                    runs.append(trace)
                bank=augmented_bank(common,runs,goal,q,arm); candidates=[]
                for i,params in enumerate(bank['params']):
                    measured,after,between,terminal=measure(params,emb=emb,history=history,transition=transition,
                        counter=counter,role='candidate_'+arm,counts=counts)
                    score=objective(measured,goal,q); suffix=torch.tensor(score['suffix_solve']['params'],dtype=emb.dtype,device=device)
                    with counter.phase('planning',method='candidate_nonlinear_'+arm,horizon_tokens=1),torch.no_grad():predicted=terminal(suffix)
                    commands=torch.cat([torch.tanh(torch.tensor(params,dtype=emb.dtype,device=device)).reshape(5,2),torch.tanh(suffix).reshape(5,2)]).cpu().tolist()
                    with counter.phase('evaluation',method='independent_replay_'+arm,horizon_tokens=2): verified=replay(initial_replay,commands)
                    if verified['latent_history'][-1]!=predicted.cpu().tolist():raise ValueError('full replay differs')
                    candidates.append(dict(**measured,**score,candidate_id=f'prefix_{i}',kind=bank['kinds'][i],
                        commands=commands,suffix_params=suffix.cpu().tolist(),predicted_terminal=predicted.cpu().tolist(),
                        nonlinear_terminal_cost=float(np.mean((q@(goal-predicted.cpu().numpy()))**2)),independent_model_replay_exact=True))
                winner=min(candidates,key=lambda c:c['objective'])['candidate_id']; greedy=min(candidates,key=lambda c:c['stage_cost'])['candidate_id']
                arms.append(dict(arm=arm,bank=bank,optimization=runs,candidates=candidates,selected_candidate_id=winner,greedy_candidate_id=greedy))
                print(f'root {row["root_index"]} {arm}: selected={winner}, greedy={greedy}',flush=True)
            plan=dict(root_index=row['root_index'],root_sha256=row['root_sha256'],root_jacobian=jac_np.tolist(),
                root_nominal=origin_np.tolist(),probe_records=records,gradient_grids=grids,arms=arms)
            base.write_new(output/f'root_{row["root_index"]}.plan.json',plan); plans.append(plan)
    gates=dict(four_roots=len(plans)==4,all96_replayed=sum(len(a['candidates']) for p in plans for a in p['arms'])==96,
        model_unchanged=before==checked.tensor_state_hash(model),frozen_eval=all(not m.training for m in model.modules()),
        all_model_records_complete=all(r['completed'] for r in counter.records),
        inputs_unchanged=all(base.digest(Path(p))==h for p,h in bindings.items()))
    verified=[verify_seal(output_dir=s['output_dir'],expected_manifest_sha256=s['manifest_sha256']) for s in seals]
    report=dict(schema=TAG,status='plans_sealed' if all(gates.values()) else 'integrity_fail',protocol=protocol,
        plans=plans,predict=counter.records,host=host,source_seals=seals,source_verification=verified,gates=gates,
        actual_predict_samples=sum(r['predict_samples'] for r in counter.records),**counts,
        scalar_output_vjp_directions=counts['jacobian_backwards']*192,P1_admission=False,physical_outcomes_used=False)
    try:
        verify_plans(report,source)
    except Exception as exc:
        report.update(status='integrity_fail',verification_failure=dict(type=type(exc).__name__,message=str(exc)))
        base.write_new(output/'report.json',report)
        raise
    base.write_new(output/'report.json',report)
    with (output/'notes.txt').open('x') as stream:stream.write('Fixed-protocol model-only generation ablation; original score and stages unchanged. Actual physical capability/benefit not inferred.\n')
    if not all(gates.values()):raise RuntimeError('geometry planning integrity failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('compilation','policy','output'):parser.add_argument('--'+name,type=Path,required=True)
    run(**vars(parser.parse_args()))

