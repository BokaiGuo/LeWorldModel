"""Matched-candidate constant/full packet compiler, host CUDA only.

Finite calibration ablation of an implementation restriction, not P1 admission.
All predictions and actions are sealed before a separate physical experiment.
"""
import argparse
import inspect
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability.pusht_full_packet_chart_20261027 import chart_matrix

TAG = 'pusht_full_packet_compiler_20261027'
DAMPING = .002
CAP = float(np.sqrt(5)*.75)
RADII = (float(np.sqrt(5)*.25), CAP)


def finite(value, shape=None):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'fi' or not np.isfinite(raw).all() or (shape is not None and raw.shape != shape):
        raise ValueError('finite real aligned arrays required')
    return raw.astype(np.float64)


def bounded_ridge(response, residual, *, damping=DAMPING, cap=CAP):
    """Exact L2-ball ridge solution, not a clipped unbounded score/action pair."""
    k, r = finite(response), finite(residual)
    if k.ndim != 2 or r.shape != k.shape[:1] or min(k.shape) < 1:
        raise ValueError('aligned nonempty response/residual required')
    if not np.isfinite([damping, cap]).all() or damping <= 0 or cap <= 0:
        raise ValueError('positive ridge damping and radius required')
    left, s, right = np.linalg.svd(k, full_matrices=False)
    target = left.T @ r
    def solution(multiplier):
        return right.T @ (s*target/(s*s+damping+multiplier))
    multiplier = 0.
    step = solution(multiplier)
    if np.linalg.norm(step) > cap:
        low, high = 0., 1.
        while np.linalg.norm(solution(high)) > cap:
            high *= 2
        for _ in range(70):
            mid = (low+high)/2
            if np.linalg.norm(solution(mid)) > cap:
                low = mid
            else:
                high = mid
        multiplier, step = high, solution(high)
    remaining = r-k@step
    return dict(params=step.tolist(), singular_values=s.tolist(),
        linear_terminal_cost=float(remaining@remaining/len(r)),
        ridge_penalty=float(damping*(step@step)/len(r)),
        own_reach_value=float((remaining@remaining+damping*(step@step))/len(r)),
        radius=cap, norm=float(np.linalg.norm(step)), ball_multiplier=multiplier)


def spectral_bank(jacobian, nominal, goal, projection):
    jacobian, nominal, goal, projection = map(finite, (jacobian, nominal, goal, projection))
    if jacobian.ndim != 2 or jacobian.shape[1] not in (2,10):
        raise ValueError('constant2 or full10 Jacobian required')
    if nominal.shape != goal.shape or projection.shape != (len(goal),len(goal)) or jacobian.shape[0] != len(goal):
        raise ValueError('metric/goal/Jacobian dimensions differ')
    response = projection@jacobian
    _, singular, right = np.linalg.svd(response,full_matrices=False)
    for vector in right:
        if vector[np.argmax(np.abs(vector))] < 0:
            vector *= -1
    progress = bounded_ridge(response, projection@(goal-nominal))
    values = [np.zeros(jacobian.shape[1]), np.asarray(progress['params'])]
    kinds = ['zero','root_dls_progress']
    for radius in RADII:
        for mode in range(2):
            for sign in (-1,1):
                values.append(sign*radius*right[mode])
                kinds.append(f'spectral_mode_{mode}_radius_{radius:.8f}_sign_{sign}')
    return dict(params=np.stack(values).tolist(), kinds=kinds,
                singular_values=singular.tolist(), right_vectors=right.tolist())


def make_transition(model, device):
    if getattr(device, 'type', None) != 'cuda':
        raise ValueError('host CUDA only; neural CPU fallback forbidden')
    import torch
    from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
    if any(m.training for m in model.modules()) or any(p.requires_grad or p.device != device for p in model.parameters()):
        raise ValueError('entire model must be frozen eval on explicit CUDA')
    def transition(emb, history, raw):
        if emb.shape != (1,3,192) or history.shape != (1,2,10) or raw.shape != (5,2):
            raise ValueError('emb3/between2/time-major packet required')
        if emb.device != device or history.device != device or raw.device != device:
            raise ValueError('same explicit CUDA device required')
        token = p0.packets.normalize_live_packet(raw, mean=p0.PUBLIC_ACTION_MEAN, std=p0.PUBLIC_ACTION_STD)
        outgoing = torch.cat((history,token[None,None]),dim=1)
        successor = model.predict(emb,model.action_encoder(outgoing))[:,-1]
        if not torch.isfinite(successor).all():
            raise ValueError('nonfinite predicted successor')
        return torch.cat((emb[:,1:],successor[:,None]),dim=1),outgoing[:,1:]
    return transition


def compile_arm(*, arm, emb, history, goal, q, jac_full, origin, transition, model, device, counter):
    import torch
    from experiments.spectral_reachability.pusht_composition_cuda_replay_v2 import CudaReplay
    basis = chart_matrix(arm)
    matrix = torch.tensor(basis,dtype=emb.dtype,device=device)
    d = basis.shape[1]
    bank = spectral_bank(jac_full@basis,origin,goal,q)
    arrays = lambda x: x.detach().cpu().numpy().astype(np.float64)
    replay = CudaReplay(model,device)
    initial = dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,
        action_dim=10,latent_history=arrays(emb[0]).tolist(),action_history=arrays(history[0]).tolist())
    candidates = []
    for i, params in enumerate(bank['params']):
        prefix_params = torch.tensor(params,dtype=emb.dtype,device=device)
        raw_prefix = torch.tanh(matrix@prefix_params).reshape(5,2)
        with counter.phase('planning',method='packet_compile_'+arm,horizon_tokens=2):
            with torch.no_grad():
                after, between = transition(emb,history,raw_prefix)
            after, between = after.detach(), between.detach()
            def terminal(u):
                return transition(after,between,torch.tanh(matrix@u).reshape(5,2))[0][0,-1]
            zero = emb.new_zeros(d)
            with torch.no_grad():
                nominal = terminal(zero)
            jac = torch.autograd.functional.jacobian(terminal,zero.requires_grad_(True),vectorize=True)
            own = bounded_ridge(q@arrays(jac),q@(goal-arrays(nominal)))
            suffix_params = torch.tensor(own['params'],dtype=emb.dtype,device=device)
            raw_suffix = torch.tanh(matrix@suffix_params).reshape(5,2)
            with torch.no_grad():
                predicted = terminal(suffix_params)
        commands = torch.cat((raw_prefix,raw_suffix)).detach().cpu().tolist()
        with counter.phase('evaluation',method='packet_independent_replay_'+arm,horizon_tokens=2):
            verified = replay(initial,commands)
        difference = float(np.max(np.abs(np.asarray(verified['latent_history'][-1])-arrays(predicted))))
        if difference != 0:
            raise ValueError('independent full packet replay differs from segmented compile')
        stage_residual = q@(goal-arrays(after[0,-1]))
        residual = q@(goal-arrays(predicted))
        stage = float(stage_residual@stage_residual/192)
        candidates.append(dict(candidate_id=f'prefix_{i}',kind=bank['kinds'][i],
            prefix_params=arrays(prefix_params).tolist(),suffix_params=arrays(suffix_params).tolist(),
            prefix_commands=commands[:5],suffix_commands=commands[5:],commands=commands,
            predicted_prefix=arrays(after[0,-1]).tolist(),predicted_terminal=arrays(predicted).tolist(),
            nominal_terminal=arrays(nominal).tolist(),jacobian=arrays(jac).tolist(),
            stage_cost=stage,own_reach_value=own['own_reach_value'],objective=stage+own['own_reach_value'],
            suffix_solve=own,nonlinear_terminal_cost=float(residual@residual/192),
            independent_model_replay_exact=True))
    winner = min(candidates,key=lambda r:r['objective'])
    greedy = min(candidates,key=lambda r:r['stage_cost'])
    return dict(arm=arm,chart=basis.tolist(),bank=bank,candidates=candidates,
        selected_candidate_id=winner['candidate_id'],greedy_candidate_id=greedy['candidate_id'],
        jacobian_backwards=10,scalar_output_vjp_directions=1920,
        selection='stage plus own bounded linear ridge value; stable bank ties; no physical labels')


def run(compilation,policy,output):
    import torch
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import pusht_full_packet_chart_20261027 as charts
    from experiments.spectral_reachability import pusht_composition_cuda_replay_v2 as adapter
    from experiments.spectral_reachability.d2p_model_call_instrumentation import PredictInstrumentation
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal,verify_seal
    source,bindings,config = checked.validate_inputs(compilation,policy)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('new exclusive output required')
    protocol = dict(schema=TAG+'.protocol',bindings=bindings,compilation=str(Path(compilation).resolve()),
        policy=str(Path(policy).resolve()),roots=source['protocol']['roots'],arms=['constant','full'],
        candidates_per_arm=10,packets_per_program=2,steps_per_packet=5,damping=DAMPING,
        norm_cap_atanh_per_packet=CAP,spectral_radii=RADII,
        fixed_roles_metric='same original Q and goal0 at stage/reach/terminal',
        selection='stage plus own bounded linear ridge, nonlinear forecast diagnostic only',
        scope='known four calibration roots; two-dimensional repeated vs full10 within-packet controls',
        same_total_compute_claim=False,P1_admission=False,physical_outcomes_used_for_selection=False)
    files = checked.sources_before_allocate(source,config)+[Path(__file__),Path(charts.__file__),Path(adapter.__file__),
        Path(inspect.getfile(PredictInstrumentation)),Path(__file__).resolve().parents[2]/f'tests/test_{TAG}.py',
        Path(__file__).resolve().parents[3]/'docs/PUSHT_FULL_PACKET_COMPILER_20261027.md']
    files = sorted(set(p.resolve() for p in files))
    output.mkdir(exist_ok=False)
    base.write_new(output/'protocol.json',protocol)
    seals=[create_seal(files=files[i:i+40],args_json=json.dumps(protocol),output_dir=output/f'source_seal_{i//40}')
           for i in range(0,len(files),40)]
    print('Compare constant2/full10 packet controls with identical Q, roots, candidate count and norm cap; no physical outcomes in compilation.',flush=True)
    host=base.require_cuda_allocation('cuda');device=torch.device('cuda:0')
    model=base.load_compatible_pretrained(Path(policy),device=device).eval().requires_grad_(False)
    before=checked.tensor_state_hash(model)
    transition=make_transition(model,device)
    plans=[]
    with PredictInstrumentation(model) as counter:
        for row in source['rows']:
            compiled=row['compiled'];initial=compiled['input_context']
            emb=torch.tensor(initial['emb'],dtype=torch.float32,device=device)
            history=torch.tensor(initial['actions'],dtype=torch.float32,device=device)[:,1:]
            goal=finite(compiled['goal_vectors'][0]);q=finite(compiled['scoring']['metric']['projection'])
            with counter.phase('planning',method='shared_full_packet_root_jacobian_and_fd',horizon_tokens=1):
                def terminal(u):
                    return transition(emb,history,torch.tanh(u).reshape(5,2))[0][0,-1]
                zero=emb.new_zeros(10)
                with torch.no_grad(): origin=terminal(zero)
                jac=torch.autograd.functional.jacobian(terminal,zero.requires_grad_(True),vectorize=True)
                c=torch.tensor(chart_matrix('constant'),dtype=emb.dtype,device=device)
                direct=torch.autograd.functional.jacobian(lambda u:terminal(c@u),emb.new_zeros(2).requires_grad_(True),vectorize=True)
                chain_error=float((direct-jac@c).abs().max().detach())
                epsilon=.005;columns=[]
                with torch.no_grad():
                    for axis in range(10):
                        delta=emb.new_zeros(10);delta[axis]=epsilon
                        columns.append((terminal(delta)-terminal(-delta))/(2*epsilon))
                fd=torch.stack(columns,dim=1)
                relative=float((fd-jac).norm().detach()/jac.norm().detach().clamp_min(1e-20))
                if not np.isfinite([chain_error,relative]).all() or chain_error>5e-5 or relative>.1:
                    raise ValueError('chart chain-rule or finite-difference integrity failed')
            arms=[]
            for arm in (('constant','full') if row['root_index']%2==0 else ('full','constant')):
                result=compile_arm(arm=arm,emb=emb,history=history,goal=goal,q=q,
                    jac_full=jac.detach().cpu().numpy().astype(np.float64),origin=origin.cpu().numpy(),
                    transition=transition,model=model,device=device,counter=counter)
                arms.append(result)
                print(f'root {row["root_index"]} {arm}: selected {result["selected_candidate_id"]}, greedy {result["greedy_candidate_id"]}',flush=True)
            j=jac.detach().cpu().numpy().astype(np.float64);c=chart_matrix('constant')
            rank=lambda a:int(np.linalg.matrix_rank(a))
            plan=dict(root_index=row['root_index'],root_sha256=row['root_sha256'],arms=arms,
                root_jacobian_full=j.tolist(),root_nominal_terminal=origin.cpu().tolist(),
                chain_rule_max_error=chain_error,full_fd_relative_error=relative,
                q_response_rank_constant=rank(q@j@c),q_response_rank_full=rank(q@j),
                temporal_only_response_frobenius=float(np.linalg.norm(q@j@(np.eye(10)-c@c.T))))
            base.write_new(output/f'root_{row["root_index"]}.plan.json',plan);plans.append(plan)
        records=counter.records
    gates=dict(four_roots=len(plans)==4,all_eighty_candidates=sum(len(a['candidates']) for p in plans for a in p['arms'])==80,
        model_unchanged=before==checked.tensor_state_hash(model),frozen_eval=all(not m.training for m in model.modules()),
        all_model_records_complete=bool(records) and all(r['completed'] for r in records),
        independent_replay_exact=all(c['independent_model_replay_exact'] for p in plans for a in p['arms'] for c in a['candidates']),
        inputs_unchanged=all(base.digest(Path(p))==h for p,h in bindings.items()))
    verification=[verify_seal(output_dir=s['output_dir'],expected_manifest_sha256=s['manifest_sha256']) for s in seals]
    report=dict(schema=TAG+'.v1',status='plans_sealed' if all(gates.values()) else 'integrity_fail',protocol=protocol,
        host=host,plans=plans,gates=gates,predict=records,source_seals=seals,source_verification=verification,
        actual_predict_samples=sum(r['predict_samples'] for r in records),jacobian_backwards=88,
        scalar_output_vjp_directions=88*192,new_physical_steps=0,P1_admission=False,
        all_plans_before_physical_execution=True,training=False)
    base.write_new(output/'report.json',report)
    with (output/'notes.txt').open('x') as stream:
        stream.write('Compilation only: no physical benefit inferred. Root/full and suffix Jacobians expand actual packet coordinates; all old sealed sources remain unchanged.\n')
    if not all(gates.values()):
        raise RuntimeError('planning integrity failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('compilation','policy','output'):
        parser.add_argument('--'+field,type=Path,required=True)
    run(**vars(parser.parse_args()))
