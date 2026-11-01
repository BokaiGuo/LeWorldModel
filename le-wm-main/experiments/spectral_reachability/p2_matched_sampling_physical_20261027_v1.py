"""Execute eight sealed CEM/random H2 plans; physics persisted before encoding.

Default is model-free physical execution, NOT CPU neural fallback. Optional
terminal encoding is host CUDA only and never changes selections or physics.
"""
from copy import deepcopy
import inspect
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p2_matched_sampling_20261027_v1 as sampling
from experiments.spectral_reachability import p3_matched_physical_20261027_v1 as durable
from experiments.spectral_reachability.d2p_composition_witness_v2 import digest as content_digest
from experiments.spectral_reachability.run_pusht_sampling_comparison_20261027_v2 import verify_rgb_context, official_checked

TAG='p2_matched_sampling_physical_20261027_v1'
write=durable.write
sha=durable.digest


def plans_from_saved(report,protocol,compilation):
    """Consumer validates full contexts, all40 candidates and 80+4 receipts."""
    actions=sampling.selected_actions(report,protocol,compilation)
    index={r['root_sha256']:r for r in protocol['prepared']}
    plans=[]
    for source in compilation['rows']:
        key=source['root_sha256']; ready=index[key]; i=source['root_index']
        peers=next(r for r in report['rows'] if r['root_sha256']==key)
        order=sampling.ARMS if i%2==0 else sampling.ARMS[::-1]
        for arm in order:
            saved=next(a for a in peers['rows'] if a['arm']==arm)
            raw=np.asarray(actions[key,arm])
            if raw.shape!=(10,2) or not np.isfinite(raw).all() or np.any(np.abs(raw)>1):
                raise ValueError('sealed bounded H2 commands required')
            plan=dict(root_index=i,root_sha256=key,method=arm,commands=raw.tolist(),
                planned_horizon_tokens=2,planned_commands=10,corrected_search_horizon=2,
                root=deepcopy(ready['root']),action_domain=deepcopy(ready['action_domain']),
                selected_candidate_id=saved['selected_candidate_id'],
                receipt_sha256=saved['witness']['receipt']['receipt_sha256'])
            plan['selection_sha256']=content_digest(plan)
            plans.append(plan)
    if len(plans)!=8 or len({(p['root_sha256'],p['method']) for p in plans})!=8:
        raise ValueError('all eight unique selections required')
    return plans


def prepare(planning):
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    path,protocol,prepared,source=sampling.load_prepared(planning)
    report_path=path/'report.json'
    report=json.loads(report_path.read_text())
    if report.get('source_seals')!=prepared['source_seals']:
        raise ValueError('planning report source seals differ from prepared sources')
    plans=plans_from_saved(report,protocol,source)
    bindings=deepcopy(protocol['bindings'])
    for p in (report_path,path/'protocol.json',path/'prepared.json'):
        bindings[str(p.resolve())]=sha(p)
    candidates=[json.loads(Path(p).read_text()) for p in protocol['bindings'] if Path(p).name=='report.json']
    interventions=[r for r in candidates if 'bank_path' in r.get('protocol',{})]
    if len(interventions)!=1:
        raise ValueError('unique official physical root provenance required')
    intervention=interventions[0]
    bank_path=Path(intervention['protocol']['bank_path']).resolve()
    if sha(bank_path)!=intervention['protocol']['bank_sha256']:
        raise ValueError('source bank changed')
    bank=json.loads(bank_path.read_text())
    upstream_path=Path(bank['upstream_path']).resolve()
    if sha(upstream_path)!=intervention['protocol']['upstream_sha256']:
        raise ValueError('root RGB source changed')
    upstream=base.physical._load_preflight(upstream_path)
    refs={r['root_sha256']:r for r in upstream['rows']}
    if len(refs)!=4 or set(refs)!={p['root_sha256'] for p in plans}:
        raise ValueError('official reference cohort differs')
    for plan in plans:
        reference=refs[plan['root_sha256']]; root=plan['root']
        if (root['root_rgb_sha256']!=reference['history_rgb_sha256'] or
                root['goal_rgb_sha256']!=reference['goal_rgb_sha256']):
            raise ValueError('selection and official root/goal RGB differ')
    for p in (bank_path,upstream_path):bindings[str(p)]=sha(p)
    _,_,config=checked.validate_inputs(protocol['compilation'],protocol['policy'])
    return base,checked,source,config,bindings,plans,upstream,refs,protocol


def physical_branch(*,plan,output,reset,env,state_of,evaluate):
    raw=np.asarray(plan['commands'])
    if (plan['planned_horizon_tokens']!=2 or plan['planned_commands']!=10 or
            raw.shape!=(10,2) or raw.dtype.kind not in 'fi' or not np.isfinite(raw).all() or np.any(np.abs(raw)>1)):
        raise ValueError('exact H2 plan required before environment access')
    return durable.physical_branch(plan=plan,repetition=0,output=output,reset=reset,
        env=env,state_of=state_of,evaluate=evaluate)


def materialize_missing(plans,rows,output):
    """Never silently drop a root/arm after setup failure."""
    existing={(r['root_sha256'],r['method']) for r in rows}
    for plan in plans:
        if (plan['root_sha256'],plan['method']) in existing:continue
        row={k:deepcopy(plan[k]) for k in ('root_index','root_sha256','method','selection_sha256',
                                        'planned_horizon_tokens','planned_commands')}
        row.update(status='blocked',failure='physical setup unavailable',repetition=0,
            attempted_commands=0,completed_commands=0,trace=[],terminal_rgb_path=None,
            full_horizon_completed=False,physical_env_step_attempts=0,physical_env_step_returns=0)
        write(output/f'root_{plan["root_index"]}_{plan["method"]}_replay0.physical.json',row)
        rows.append(row)


def optional_encodings(rows,output,setup):
    """Durable physical report MUST exist before any setup/CUDA/model call."""
    if not (output/'physical_report.json').is_file():
        raise ValueError('persist all physical branches before encoding')
    try:
        encode=setup()
    except (Exception,KeyboardInterrupt) as exc:
        receipts=[]
        for row in rows:
            result=dict(root_sha256=row['root_sha256'],method=row['method'],repetition=0,
                status='encoding_setup_failed',attempted_encode_samples=0,
                failure=durable.failure(exc,'cuda_encoding_setup'))
            write(output/f'root_{row["root_index"]}_{row["method"]}_replay0.encoding.json',result)
            receipts.append(result)
        return receipts
    return durable.encode_saved(rows,output,encode)


def run(*,planning,output,encode_terminals=False,preflight_only=False):
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal,verify_seal
    base,checked,source,config,bindings,plans,upstream,refs,planning_protocol=prepare(planning)
    if preflight_only:
        print(json.dumps(dict(status='preflight_only',branches=len(plans),planned_steps=80,
            encode_terminals=encode_terminals,new_predict_calls=0,P1_admission=False,P2_admission=False)))
        return
    output=Path(output).resolve()
    if TAG not in output.name:raise ValueError('dedicated new output directory required')
    output.mkdir(exist_ok=False)
    protocol=dict(schema=TAG+'.protocol',bindings=bindings,plans=plans,
        planning=str(Path(planning).resolve()),physical_repetitions=1,planned_commands_per_branch=10,
        fixed_alternating_arm_order=True,physical_before_optional_encoding=True,
        encode_terminals=encode_terminals,root_selection_changed=False,
        extra_search_or_training=False,P1_admission=False,P2_admission=False,
        scope='known-root selected CUDA plans; physical calibration, not task efficacy')
    write(output/'protocol.json',protocol)
    rows,failures,seals,encodings=[],[],[],[]
    model=None;before=None;host=None;env=None
    try:
        env=base.gym.make('swm/PushT-v1',render_mode='rgb_array')
        files=checked.sources_before_allocate(source,config)
        files += [Path(__file__).resolve(),Path(durable.__file__),Path(sampling.__file__),
            Path(verify_rgb_context.__code__.co_filename),Path(official_checked.__code__.co_filename),
            Path(env.unwrapped.eval_state.__code__.co_filename),
            Path(durable.RgbObservabilityGuard.__module__.replace('.','/')).with_suffix('.py').resolve(),
            Path(__file__).resolve().parents[2]/f'tests/test_{TAG}.py',
            Path(__file__).resolve().parents[3]/'docs/P2_MATCHED_SAMPLING_PHYSICAL_20261027.md']
        files=sorted(set(p.resolve() for p in files))
        seals=[create_seal(files=files[i:i+40],args_json=json.dumps(dict(protocol_sha256=sha(output/'protocol.json'))),
            output_dir=output/f'source_seal_{i//40}') for i in range(0,len(files),40)]
        write(output/'source_seals.json',seals)
        if tuple(env.unwrapped.variation_space['agent']['color'].value.tolist())!=(65,105,225):
            raise ValueError('unsupported renderer color')
        roots=base.collect_nonterminal_boundary_contact_roots(env,count=4,
            seed=int(upstream['protocol']['root_seed']),oversample_factor=4)
        roots={base.root_fingerprint(r):r for r in roots}
        if len(roots)!=4 or set(roots)!=set(refs):raise ValueError('official roots differ; no replacement')
        for plan in plans:
            root=roots[plan['root_sha256']]
            def reset():
                if base.root_fingerprint(root)!=plan['root_sha256']:
                    raise ValueError('physical root fingerprint changed')
                frames,history,goal=base.physical._reset_and_warmup(env,root,history_len=3)
                identity=verify_rgb_context(frames,goal,refs[plan['root_sha256']])
                identity.update(history=np.asarray(history).tolist(),
                    state=base.p0.packets.state_of(env).tolist(),goal=np.asarray(root['goal']).tolist())
                return identity
            def evaluate(state):
                return dict(official=official_checked(env,state,root['goal']),
                            pose=base.physical._pose_label(state,root['goal']))
            row=physical_branch(plan=plan,output=output,reset=reset,env=env,
                                state_of=base.p0.packets.state_of,evaluate=evaluate)
            rows.append(row)
            print(f'P2 sampling root {plan["root_index"]} {plan["method"]}: '
                  f'{row["status"]} {row["completed_commands"]}/10',flush=True)
    except (Exception,KeyboardInterrupt) as exc:
        failures.append(durable.failure(exc,'physical_setup'))
    finally:
        if env is not None:
            try:env.close()
            except Exception as exc:failures.append(durable.failure(exc,'env_close'))
    materialize_missing(plans,rows,output)
    write(output/'physical_report.json',dict(schema=TAG+'.physical_before_encoding',rows=rows,
        failures=deepcopy(failures),new_predict_calls=0,P1_admission=False,P2_admission=False))
    if encode_terminals:
        def setup():
            nonlocal model,before,host
            import torch
            from PIL import Image
            host=base.require_cuda_allocation('cuda')
            device=torch.device('cuda:0')
            model=base.load_compatible_pretrained(Path(planning_protocol['policy']),device=device).eval().requires_grad_(False)
            before=checked.tensor_state_hash(model)
            def forbid(*a,**k):raise RuntimeError('terminal encoding permits no new predictions')
            model.predict=forbid
            transform=base.p0.image_transform()
            def encode(path):
                with Image.open(path) as im:rgb=np.asarray(im.convert('RGB')).copy()
                with torch.no_grad():
                    vector=model.encode({'pixels':base.p0._image_tensor(rgb,transform=transform,device=device)})['emb'][0,0]
                if vector.device.type!='cuda' or not torch.isfinite(vector).all():
                    raise ValueError('finite HOST CUDA terminal vector required')
                torch.cuda.synchronize(device)
                return vector.cpu().tolist()
            return encode
        encodings=optional_encodings(rows,output,setup)
    checks=[]
    for p,h in bindings.items():
        try:checks.append(dict(path=p,matched=sha(p)==h))
        except Exception as exc:checks.append(dict(path=p,matched=False,error=str(exc)))
    verified=[]
    try:
        prepare(planning)  # recheck planning receipts and all bound sources
        if not seals:raise ValueError('physical source seal missing')
        verified=[verify_seal(output_dir=s['output_dir'],expected_manifest_sha256=s['manifest_sha256']) for s in seals]
    except Exception as exc:failures.append(durable.failure(exc,'final_source_check'))
    actionmap={(p['root_sha256'],p['method']):np.asarray(p['commands']) for p in plans}
    trace_ok=all(len(r['trace'])==r['completed_commands'] and all(
        e['primitive_step']==i+1 and np.array_equal(e['command'],actionmap[r['root_sha256'],r['method']][i])
        for i,e in enumerate(r['trace'])) for r in rows)
    gates=dict(all_eight_retained=len(rows)==8 and len({(r['root_sha256'],r['method']) for r in rows})==8,
        no_execution_errors=all(r['status'] in ('completed','observation_unavailable','environment_terminated') for r in rows),
        trace_matches_selected_prefix=trace_ok,inputs_unchanged=all(c['matched'] for c in checks),
        sources_unchanged=bool(verified),physical_artifacts_persisted=(output/'physical_report.json').is_file(),
        encodings_complete=not encode_terminals or len(encodings)==8 and all(e['status']=='completed' for e in encodings))
    if model is not None:
        gates.update(model_unchanged=before==checked.tensor_state_hash(model),
            frozen_eval=all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters()))
    passed=all(gates.values()) and not failures
    write(output/'report.json',dict(schema=TAG+'.report',status='calibration_recorded' if passed else 'integrity_fail',
        rows=rows,encodings=encodings,failures=failures,gates=gates,source_seals=seals,
        source_verification=verified,binding_checks=checks,host=host,encode_terminals=encode_terminals,
        selected_path_env_step_attempts=sum(r['physical_env_step_attempts'] for r in rows),
        selected_path_env_step_returns=sum(r['physical_env_step_returns'] for r in rows),
        attempted_terminal_encode_samples=sum(e['attempted_encode_samples'] for e in encodings),
        new_predict_calls=0,P1_admission=False,P2_admission=False,efficacy_verified=False,
        planned_horizon_fixed_H2=True,actual_guard_endpoints_may_differ=True,fair_total_budget_comparison=False))
    if not passed:raise RuntimeError('P2 sampling physical integrity failed; all artifacts retained')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--planning',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--encode-terminals',action='store_true')
    p.add_argument('--preflight-only',action='store_true')
    run(**vars(p.parse_args()))
