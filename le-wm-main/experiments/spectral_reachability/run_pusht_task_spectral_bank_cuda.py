"""Known-root spectral-prefix generation and full-bank physical calibration.

All model plans sealed before candidate physical execution. Not P1 admission,
independent outcome, equal-budget comparison or a full multi-round controller.
"""
import argparse
import inspect
import json
from pathlib import Path
import numpy as np


def run(compilation, policy, output):
    import torch
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability import pusht_d2p_metric_program_compiler as compiler
    from experiments.spectral_reachability import preflight_pusht_d2p_p1_reconfiguration as p1
    from experiments.spectral_reachability import pusht_task_spectral_bank as spectral
    from experiments.spectral_reachability.audit_pusht_d2p_metric_program_arrays import audit
    from experiments.spectral_reachability.pusht_visible_command_executor import execute
    from experiments.spectral_reachability.pusht_rgb_observability_guard import RgbObservabilityGuard
    from experiments.spectral_reachability.audit_pusht_sealed_task_contract import labels
    from experiments.spectral_reachability.d2p_model_call_instrumentation import PredictInstrumentation
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal
    source, bindings, config = checked.validate_inputs(compilation, policy)
    output=Path(output); output.mkdir(exist_ok=False)
    candidates=[json.loads(Path(p).read_text()) for p in bindings if Path(p).name=='report.json']
    intervention=next(r for r in candidates if 'bank_path' in r.get('protocol',{}))
    oldbank=json.loads(Path(intervention['protocol']['bank_path']).read_text())
    upstream=base.physical._load_preflight(Path(oldbank['upstream_path']))
    protocol=dict(bindings=bindings, roots=[r['root_sha256'] for r in source['rows']],
        prefix_horizon=1,suffix_horizon=1,generated_prefixes=10,damping=.01,beta=1.,
        spectral_radii_atanh=[.25,.75],progress_norm_cap=.75,
        selector='stage + beta own reach; no non-greedy eligibility filtering',
        all_plans_saved_before_physical_execution=True,feedback_every_primitive=True,
        physical_outcomes_used_for_generation=False,known_calibration_roots=True,P1_admission=False)
    files=checked.sources_before_allocate(source,config)
    files += [Path(__file__),Path(spectral.__file__),Path(inspect.getfile(execute)),
              Path(inspect.getfile(RgbObservabilityGuard)),Path(inspect.getfile(labels)),
              Path(inspect.getfile(base.collect_nonterminal_boundary_contact_roots))]
    files=sorted(set(p.resolve() for p in files))
    seals=[create_seal(files=files[i:i+50],args_json=json.dumps(protocol),output_dir=output/f'source_seal_{i//50}') for i in range(0,len(files),50)]
    base.write_new(output/'protocol.json',protocol)
    host=base.require_cuda_allocation('cuda'); device=torch.device('cuda')
    model=base.load_compatible_pretrained(Path(policy),device=device).eval().requires_grad_(False)
    before=base.p0._parameter_hash(model); buffers={k:v.clone() for k,v in model.named_buffers()}
    plans=[]
    # No candidate physical outcomes are read until all roots are compiled.
    with PredictInstrumentation(model) as counter:
        for row in source['rows']:
            c=row['compiled']; initial=c['input_context']
            emb=torch.tensor(initial['emb'],dtype=torch.float32,device=device)
            actions=torch.tensor(initial['actions'],dtype=torch.float32,device=device)
            goal=np.asarray(c['goal_vectors'][0]); q=np.asarray(c['scoring']['metric']['projection'])
            with counter.phase('planning',method='task_spectral_prefix_and_own_suffix',horizon_tokens=2):
                def terminal(flat):
                    return p1._suffix_terminal(model,emb,actions,flat.reshape(1,1,2))[0]
                nominal=emb.new_zeros(2)
                with torch.no_grad(): origin=terminal(nominal)
                jac=torch.autograd.functional.jacobian(terminal,nominal.requires_grad_(True),vectorize=True)
                bank=spectral.synthesize(jacobian=jac.detach().cpu().numpy(),nominal_terminal=origin.cpu().numpy(),goal=goal,projection=q)
                result=compiler.compile_programs(model,emb,actions,c['goal_vectors'],bank['params'],1)
            array_audit=audit(result)
            plan=dict(root_index=row['root_index'],root_sha256=row['root_sha256'],bank=bank,
                root_nominal_terminal=origin.cpu().tolist(),root_jacobian=jac.cpu().tolist(),
                compiled=result,array_audit=array_audit)
            base.write_new(output/f'plan_{row["root_index"]}.json',plan); plans.append(plan)
            print(f'compiled root {row["root_index"]}: selected {result["selected_candidate_id"]}, greedy {result["scoring"]["greedy_candidate_id"]}',flush=True)
        prediction_records=counter.records
    plan_hashes={str((output/f'plan_{p["root_index"]}.json').resolve()):base.digest(output/f'plan_{p["root_index"]}.json') for p in plans}
    base.write_new(output/'all_plans_manifest.json',plan_hashes)
    env=base.gym.make('swm/PushT-v1',render_mode='rgb_array'); transform=base.p0.image_transform()
    rows=[]
    try:
        roots=base.collect_nonterminal_boundary_contact_roots(env,count=4,seed=int(upstream['protocol']['root_seed']),oversample_factor=4)
        if [base.root_fingerprint(r) for r in roots] != protocol['roots']:
            raise ValueError('root identity changed')
        if tuple(env.unwrapped.variation_space['agent']['color'].value.tolist()) != (65,105,225):
            raise ValueError('unsupported renderer')
        refs={r['root_sha256']:r for r in upstream['rows']}
        for root,plan in zip(roots,plans,strict=True):
            key=plan['root_sha256']; branches=[]
            for receipt in plan['compiled']['receipts']:
                frames,_,goalrgb=base.physical._reset_and_warmup(env,root,history_len=3)
                if (base.physical._sha256(goalrgb)!=refs[key]['goal_rgb_sha256'] or
                    base.physical._sha256(np.stack(frames))!=refs[key]['history_rgb_sha256']):
                    raise ValueError('root or goal RGB changed')
                commands=np.repeat(np.asarray(receipt['program_live_commands']),5,axis=0)
                trace=[]
                def record(step,command,transition,rgb):
                    trace.append(dict(step=step,command=command.tolist(),rgb_sha256=base.physical._sha256(rgb),
                        state=base.p0.packets.state_of(env).tolist()))
                execution=execute(env,commands,root_id=key,agent_rgb=(65,105,225),on_observation=record)
                rgb=execution.pop('terminal_rgb'); state=base.p0.packets.state_of(env)
                with torch.no_grad():
                    vector=model.encode({'pixels':base.p0._image_tensor(rgb,transform=transform,device=device)})['emb'][0,0]
                if vector.device.type!='cuda' or not torch.isfinite(vector).all():
                    raise ValueError('finite CUDA final encoding required')
                branch=dict(candidate_id=receipt['candidate_id'],**execution,trace=trace,final_vector=vector.cpu().tolist(),
                    final_state=state.tolist(),official=labels(state,root['goal']),pose=base.physical._pose_label(state,root['goal']))
                branches.append(branch)
                base.write_new(output/f'root_{plan["root_index"]}.{receipt["candidate_id"]}.json',branch)
            row=dict(root_index=plan['root_index'],root_sha256=key,selected_candidate_id=plan['compiled']['selected_candidate_id'],
                greedy_candidate_id=plan['compiled']['scoring']['greedy_candidate_id'],branches=branches)
            rows.append(row);base.write_new(output/f'root_{plan["root_index"]}.json',row)
            print(f'executed root {row["root_index"]}: {len(branches)} branches retained',flush=True)
    finally: env.close()
    branches=[b for r in rows for b in r['branches']]
    gates=dict(all40_retained=len(branches)==40,all_four_plans=len(plans)==4,
        planning_records_complete=len(prediction_records)==4 and all(p['completed'] for p in prediction_records),
        no_execution_errors=all(b['status'] in ('completed','observation_unavailable','environment_terminated') for b in branches),
        parameters_unchanged=before==base.p0._parameter_hash(model),
        buffers_unchanged=all(torch.equal(buffers[k],v) for k,v in model.named_buffers()),
        frozen_eval=all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters()),
        plans_unchanged=all(base.digest(Path(p))==d for p,d in plan_hashes.items()),
        inputs_unchanged=all(base.digest(Path(p))==d for p,d in bindings.items()))
    verified=[verify_seal(output_dir=s['output_dir'],expected_manifest_sha256=s['manifest_sha256']) for s in seals]
    base.write_new(output/'report.json',dict(schema='pusht_task_spectral_bank_cuda.v1',
        status='calibration_completed' if all(gates.values()) else 'integrity_fail',protocol=protocol,host=host,
        gates=gates,rows=rows,predict=prediction_records,plan_hashes=plan_hashes,source_seals=seals,
        source_verification=verified,P1_admission=False))
    if not all(gates.values()):raise RuntimeError('spectral calibration integrity failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('compilation','policy','output'):parser.add_argument('--'+name,type=Path,required=True)
    run(**vars(parser.parse_args()))
