"""Recompile suffixes in sealed imagined/observed contexts; main host CUDA only.

Known-root intervention, not independent P1 admission. No outcome enters action
selection; fixed original metric, spectral bank, logit radius and ridge damping.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import inspect
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import pusht_geometry_feedback_response_20261027_v1 as response
from experiments.spectral_reachability import pusht_full_packet_compiler_20261027 as packet

TAG = 'pusht_feedback_suffix_compile_20261027_v1'
REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO/'results/pusht_geometry_feedback_response_20261027_v1_prepare_v1'


def contract():
    return dict(roles=list(response.ROLES), arms=list(response.ARMS), candidates_per_context=10,
        chart='full10 tanh(logits), exactly five raw commands per suffix',
        damping=packet.DAMPING, logit_l2_cap=packet.CAP, radii=list(packet.RADII),
        selection='lowest nonlinear original Q goal MSE; stable bank index tie',
        generation='zero, bounded original ridge, signed top2 spectral modes at two fixed radii',
        context_intervention='same saved older latents/actions; only observed versus imagined last latent',
        predict_samples=78, jacobian_materializations=6, scalar_vjp_directions=1152,
        svd_calls=12, visual_encode_calls=0, new_physical_steps=0,
        scope='one previously used calibration root; new suffixes, no new independent roots',
        total_compute_matched=False, P1_admission=False)


def checked_candidate(candidate, prefix, q, goal):
    u = response.array(candidate['suffix_params'], (10,))
    commands = response.array(candidate['commands'], (10,2))
    p = response.array(prefix, (5,2))
    if np.linalg.norm(u) > packet.CAP+1e-6 or np.any(np.abs(commands) > 1):
        raise ValueError('original suffix action ball violated')
    if not np.array_equal(commands[:5], p):
        raise ValueError('fixed prefix changed')
    response.require_close(commands[5:], np.tanh(u).reshape(5,2), 'raw suffix/logit mapping', atol=1e-7)
    terminal = response.array(candidate['predicted_terminal'], (192,))
    cost = float(np.mean((np.asarray(q)@(np.asarray(goal)-terminal))**2))
    if not np.isfinite(candidate['nonlinear_terminal_cost']) or abs(cost-candidate['nonlinear_terminal_cost']) > 1e-12:
        raise ValueError('nonlinear original metric cost differs')
    return cost


def verify_plan(plan):
    if plan['root_sha256'] != response.ROOT or plan['root_index'] != 1:
        raise ValueError('original calibration root required')
    if plan['role'] not in response.ROLES or plan['arm'] not in response.ARMS:
        raise ValueError('unknown role/context arm')
    q, goal = response.array(plan['q'], (192,192)), response.array(plan['goal'], (192,))
    c = plan['context']
    response.array(c['latent_history'], (3,192)); response.array(c['action_history'], (2,10))
    nominal = response.array(plan['nominal_terminal'], (192,))
    jac = response.array(plan['jacobian'], (192,10))
    bank = packet.spectral_bank(jac, nominal, goal, q)
    if len(plan['candidates']) != 10:
        raise ValueError('exact full ten-candidate census required')
    for i, candidate in enumerate(plan['candidates']):
        if candidate['candidate_id'] != f'suffix_{i}' or candidate['kind'] != bank['kinds'][i]:
            raise ValueError('candidate bank order/identity differs')
        # Device float32 rounding is checked against the fixed float64 generator.
        response.require_close(candidate['suffix_params'], np.asarray(bank['params'][i], dtype=np.float32),
                               'fixed spectral suffix', atol=0)
        checked_candidate(candidate, plan['prefix'], q, goal)
    winner = min(plan['candidates'], key=lambda x:x['nonlinear_terminal_cost'])
    if winner['candidate_id'] != plan['selected_candidate_id']:
        raise ValueError('winner is not fixed nonlinear model-cost argmin')
    if not plan['independent_replay_exact']:
        raise ValueError('independent winner replay missing')
    return winner


def prepare(output, source=SOURCE):
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal
    from experiments.spectral_reachability.p3_matched_physical_20261027_v1 import write
    from experiments.spectral_reachability.d2p_model_call_instrumentation import PredictInstrumentation
    from experiments.spectral_reachability import pusht_composition_cuda_replay_v2 as adapter
    output, source = Path(output).resolve(), Path(source).resolve()
    if output.exists(): raise FileExistsError(output)
    f = response.verify_frozen(source)
    r = response.read(source/'report.json')
    if r['status'] != 'diagnostic_recorded' or r['failure'] is not None or len(r['contexts']) != 3:
        raise ValueError('complete sealed response contexts required')
    bindings = dict(f['bindings'])
    files = {Path(__file__), Path(packet.__file__), Path(response.__file__), Path(adapter.__file__),
        Path(inspect.getfile(PredictInstrumentation)),
        REPO/f'tests/test_{TAG}.py', REPO/f'innerdocs/{TAG}.md'}
    receipt = response.read(source/'prepare_receipt.json')
    for seal in receipt['source_seals']:
        verify_seal(output_dir=seal['output_dir'], expected_manifest_sha256=seal['manifest_sha256'])
        manifest = Path(seal['output_dir'])/'manifest.json'
        files.update(Path(e['source']) for e in response.read(manifest)['files'])
        bindings[str(manifest)] = response.sha(manifest)
    for name in ('protocol.json','prepare_receipt.json','report.json'):
        p = source/name; bindings[str(p)] = response.sha(p)
    cases = []
    for role in response.ROLES:
        path = source/f'{role}.contexts.json'
        value = response.read(path)
        if value != next(v for v in r['contexts'] if v['role'] == role):
            raise ValueError('context file/report mismatch')
        bindings[str(path)] = response.sha(path)
        original = next(v for v in f['cases'] if v['role'] == role)
        actual, delta = response.pair_contexts(value['imagined'], value['actual'])
        if actual != value['actual'] or delta != 0:
            raise ValueError('older history not exactly shared')
        for arm in response.ARMS:
            # Deliberately exclude physical terminal vectors and diagnostic scores.
            cases.append(dict(root_sha256=response.ROOT, root_index=1, role=role, arm=arm,
                prefix=original['prefix'], context=value[arm], goal=f['goal'], q=f['q']))
    protocol = dict(schema=TAG, source=str(source), contract=contract(), cases=cases,
        bindings=bindings, policy=f['policy'], training=False, outcome_selected=False,
        original_initial_context=f['initial_context'], original_root_rgb=f['root_rgb'])
    if any(response.sha(p) != h for p,h in bindings.items()):
        raise ValueError('upstream input changed')
    output.mkdir(exist_ok=False); write(output/'protocol.json', protocol)
    files = sorted({p.resolve() for p in files})
    seals = [create_seal(files=files[i:i+32], args_json=json.dumps(dict(protocol_sha256=response.sha(output/'protocol.json'))),
        output_dir=output/f'source_seal_{i//32}') for i in range(0,len(files),32)]
    prepared = dict(status='prepared_not_run', protocol_sha256=response.sha(output/'protocol.json'),
        source_seals=seals, P1_admission=False)
    write(output/'prepare_receipt.json',prepared)
    return prepared


def verify_prepared(output):
    from experiments.spectral_reachability.d2p_prerun_source_seal import verify_seal
    output=Path(output); r=response.read(output/'prepare_receipt.json'); f=response.read(output/'protocol.json')
    if r['status'] != 'prepared_not_run' or response.sha(output/'protocol.json') != r['protocol_sha256']:
        raise ValueError('prepared protocol byte mismatch')
    if f['contract'] != contract(): raise ValueError('fixed contract changed')
    if any(response.sha(p) != h for p,h in f['bindings'].items()): raise ValueError('input changed')
    for s in r['source_seals']:
        verify_seal(output_dir=s['output_dir'], expected_manifest_sha256=s['manifest_sha256'])
    return f,r


def run(output):
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability.pusht_composition_cuda_replay_v2 import CudaReplay
    from experiments.spectral_reachability.d2p_model_call_instrumentation import PredictInstrumentation
    from experiments.spectral_reachability.p3_matched_physical_20261027_v1 import write
    output=Path(output).resolve(); f,prepared=verify_prepared(output)
    write(output/'launch_receipt.json',dict(protocol_sha256=prepared['protocol_sha256']))
    plans=[]; counter=None; failure=None; before=after=None
    jacobian_attempts=jacobian_completed=svd_completed=0
    print('Fixed six contexts: recompile ten full10 spectral suffixes using original Q; choose by model only. No physics or outcome selection.',flush=True)
    try:
        import torch
        base.require_cuda_allocation('cuda'); device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False
        model=base.load_compatible_pretrained(Path(f['policy']),device=device).eval().requires_grad_(False)
        before=checked.tensor_state_hash(model)
        transition=packet.make_transition(model,device); replay=CudaReplay(model,device)
        def values(t): return t.detach().cpu().double().numpy().tolist()
        with PredictInstrumentation(model) as counter:
            for case in f['cases']:
                c=case['context']; method=case['role']+'_'+case['arm']
                emb=torch.tensor(c['latent_history'],dtype=torch.float32,device=device)[None]
                history=torch.tensor(c['action_history'],dtype=torch.float32,device=device)[None]
                def terminal(u): return transition(emb,history,torch.tanh(u).reshape(5,2))[0][0,-1]
                with counter.phase('planning',method=method+'_compile',horizon_tokens=1):
                    with torch.no_grad(): nominal=terminal(emb.new_zeros(10))
                    jacobian_attempts+=1
                    jac=torch.autograd.functional.jacobian(terminal,emb.new_zeros(10).requires_grad_(True),vectorize=True)
                    jacobian_completed+=1
                q,goal=np.asarray(case['q']),np.asarray(case['goal'])
                bank=packet.spectral_bank(values(jac),values(nominal),goal,q)
                svd_completed+=2
                candidates=[]
                with counter.phase('planning',method=method+'_nonlinear_bank',horizon_tokens=1),torch.no_grad():
                    for i,params in enumerate(bank['params']):
                        u=torch.tensor(params,dtype=emb.dtype,device=device)
                        pred=terminal(u)
                        commands=case['prefix']+values(torch.tanh(u).reshape(5,2))
                        candidates.append(dict(candidate_id=f'suffix_{i}',kind=bank['kinds'][i],
                            suffix_params=values(u),commands=commands,predicted_terminal=values(pred),
                            nonlinear_terminal_cost=float(np.mean((q@(goal-np.asarray(values(pred))))**2))))
                winner=min(candidates,key=lambda x:x['nonlinear_terminal_cost'])
                with counter.phase('evaluation',method=method+'_independent_replay',horizon_tokens=1):
                    verified=replay(deepcopy(c),winner['commands'][5:])
                exact=verified['latent_history'][-1]==winner['predicted_terminal']
                plan=dict(**case,nominal_terminal=values(nominal),jacobian=values(jac),bank=bank,
                    candidates=candidates,selected_candidate_id=winner['candidate_id'],independent_replay_exact=exact)
                verify_plan(plan)
                write(output/f'{method}.plan.json',plan); plans.append(plan)
                print(f'{method}: {winner["candidate_id"]}, model Q MSE {winner["nonlinear_terminal_cost"]:.8f}',flush=True)
        torch.cuda.synchronize(device); after=checked.tensor_state_hash(model)
        verify_prepared(output)
        if before!=after or len(plans)!=6 or sum(r['predict_samples'] for r in counter.records)!=78:
            raise ValueError('model/census/work mismatch')
        if any(m.training for m in model.modules()) or any(p.requires_grad for p in model.parameters()):
            raise ValueError('frozen eval contract changed')
    except BaseException as exc:
        failure=dict(type=type(exc).__name__,message=str(exc)); raise
    finally:
        write(output/'report.json',dict(schema=TAG,status='integrity_fail' if failure else 'plans_sealed',
            plans=plans,failure=failure,source_seals=prepared['source_seals'],
            protocol_sha256=prepared['protocol_sha256'],predict=[] if counter is None else counter.records,
            actual_predict_samples=0 if counter is None else sum(r['predict_samples'] for r in counter.records),
            model_unchanged=before is not None and before==after,
            jacobian_attempts=jacobian_attempts,jacobian_materializations=jacobian_completed,
            scalar_vjp_directions_completed=192*jacobian_completed,
            svd_calls_completed=svd_completed,partial_internal_work_unknown=failure is not None,
            new_encode_calls=0,new_physics_steps=0,
            P1_admission=False,efficacy_verified=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','run')); p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    if args.command=='prepare': print(json.dumps(prepare(args.output)))
    else: run(args.output)
