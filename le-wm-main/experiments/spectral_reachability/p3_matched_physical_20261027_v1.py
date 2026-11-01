"""Saved P3 variable-horizon physical calibration; never scientific admission.

No optimization/model prediction. Physical branch receipts precede optional
HOST CUDA terminal encoding. Existing sealed sources are read-only dependencies.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from experiments.spectral_reachability.pusht_rgb_observability_guard import (
    RgbObservabilityGuard, ObservationUnavailable,
)

TAG = 'p3_matched_physical_20261027_v1'
METHODS = ('p3_adaptive_runtime_fixed', 'p3_adaptive_runtime_adaptive')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def failure(exc, phase):
    return dict(phase=phase, type=type(exc).__name__, message=str(exc))


def selected_plans(report, protocol, correction, compilation):
    """Validate original and corrected census, bank actions and saved witnesses.

    No evaluation labels are used to choose/order plans. Executed H is the best
    path length, NOT the corrected maximum SEARCH horizon.
    """
    from experiments.spectral_reachability.p3_adaptive_runtime_accounting_audit_20261027 import audit
    from experiments.spectral_reachability.run_pusht_program_search_cuda import metric_for
    identity = hashlib.sha256(json.dumps(protocol['protocol'], sort_keys=True).encode()).hexdigest()
    if identity != protocol['protocol_id'] or identity != report['protocol_id']:
        raise ValueError('original protocol hash mismatch')
    expected_correction = audit(report, protocol)
    if any(correction.get(k) != v for k, v in expected_correction.items()):
        raise ValueError('corrective audit does not reproduce from saved receipts')
    roots = protocol['protocol']['roots']
    if (len(roots) != 4 or len(set(roots)) != 4 or
            compilation['protocol']['roots'] != roots):
        raise ValueError('exact frozen four-root cohort required')
    source = {r['root_sha256']: r for r in compilation['rows']}
    if len(compilation['rows']) != 4 or set(source) != set(roots):
        raise ValueError('compilation root census differs')
    index = {(r['root_sha256'], r['method']): r for r in report['rows']}
    if len(report['rows']) != 8 or set(index) != {(r,m) for r in roots for m in METHODS}:
        raise ValueError('all eight distinct saved paths required')
    corrected = {(r['root_sha256'],r['method']): r for r in correction['corrections']}
    plans = []
    for i, root in enumerate(roots):
        compiled = source[root]['compiled']
        initial = compiled['input_context']
        context = dict(layout='states_with_between_state_actions_v1', context_length=3,
            latent_dim=192, action_dim=10, latent_history=initial['emb'][0],
            action_history=initial['actions'][0][1:])
        bank = [np.repeat(np.asarray(r['program_live_commands'][0])[None],5,axis=0)
                for r in compiled['receipts']]
        if len(bank) != 16:
            raise ValueError('complete sixteen-command bank required')
        for method in (METHODS if i % 2 == 0 else METHODS[::-1]):
            row = index[root,method]
            best = row['search']['best_plan']
            path = best['primitive_indices']
            if (len(path) not in (2,3) or best['depth'] != len(path)
                    or any(type(k) is not int or not 0 <= k < 16 for k in path)):
                raise ValueError('sealed H2/H3 path required')
            actions = np.concatenate([bank[k] for k in path])
            if (actions.shape != (5*len(path),2) or not np.isfinite(actions).all()
                    or np.any(np.abs(actions)>1) or not np.array_equal(best['actions'], actions)):
                raise ValueError('exact bounded bank action concatenation required')
            if row['full_replay_exact'] is not True or row['failure'] is not None:
                raise ValueError('saved full model replay required')
            if abs(metric_for(compiled)(best['context']) - best['cost']) > 1e-12:
                raise ValueError('saved goal/projection/terminal cost differs')
            plan = dict(root_index=i, root_sha256=root, method=method,
                primitive_indices=path, planned_horizon_tokens=len(path),
                planned_commands=len(actions), commands=actions.tolist(),
                corrected_search_horizon=corrected[root,method]['actual_horizon'],
                predictor_context=context,
                goal_vectors_sha256=hashlib.sha256(json.dumps(compiled['goal_vectors'],sort_keys=True).encode()).hexdigest(),
                projection_sha256=hashlib.sha256(json.dumps(compiled['scoring']['metric']['projection'],sort_keys=True).encode()).hexdigest(),
                bank=[a.tolist() for a in bank],
                original_prediction_receipt=row['prediction_receipt'])
            plan['selection_sha256'] = hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
            plans.append(plan)
    return plans


def guarded_execute(env, commands, *, root_id, state_of, event):
    """Guard every raw step, including H3. Event sink durably journals attempts.

    Evaluator-only state never enters guard/dispatch. Failed env.step attempts
    remain charged even if no successor render/label exists.
    """
    raw = np.asarray(commands)
    if (raw.dtype.kind not in 'fi' or raw.shape not in ((10,2),(15,2))
            or not np.isfinite(raw).all() or np.any(np.abs(raw)>1)):
        raise ValueError('exact bounded ten/fifteen-step path required')
    guard = RgbObservabilityGuard(root_id=root_id, agent_rgb=(65,105,225))
    attempted = completed = 0
    trace, rgb, last_state, err = [], None, None, None
    status = 'completed'
    try:
        rgb = np.asarray(env.render()).copy()
        last_state = np.asarray(state_of(env)).tolist()
        event(dict(kind='initial',state=last_state),rgb)
        guard.observe(rgb, primitive_step=0)
        for index, command in enumerate(raw):
            authorized = guard.dispatch(command)
            attempted += 1
            event(dict(kind='attempt',primitive_step=index+1,command=command.tolist()),None)
            transition = env.step(authorized)
            completed += 1  # env.step returned: charge before render/label errors.
            event(dict(kind='step_returned',primitive_step=index+1),None)
            if len(transition) != 5:
                raise ValueError('Gymnasium five-field transition required')
            rgb = np.asarray(env.render()).copy()
            last_state = np.asarray(state_of(env)).tolist()
            observation = dict(kind='observation',primitive_step=index+1,
                command=command.tolist(),state=last_state,reward=float(transition[1]),
                terminated=bool(transition[2]),truncated=bool(transition[3]))
            event(observation,rgb)
            trace.append(observation)
            guard.observe(rgb, primitive_step=index+1)
            if transition[2] or transition[3]:
                status = 'environment_terminated'
                break
    except ObservationUnavailable as exc:
        status, err = 'observation_unavailable', failure(exc,'guard')
    except (Exception, KeyboardInterrupt) as exc:
        status, err = 'execution_failed', failure(exc,'execution')
    return dict(status=status,failure=err,attempted_commands=attempted,
        completed_commands=completed,planned_commands=len(raw),
        remaining_commands_not_executed=len(raw)-completed,trace=trace,
        final_state=last_state,guard=guard.receipt(),
        full_horizon_completed=status=='completed' and completed==len(raw),
        actual_endpoint_primitive_steps=completed,
        physical_env_step_attempts=attempted,physical_env_step_returns=completed), rgb


def physical_branch(*, plan, repetition, output, reset, env, state_of, evaluate):
    """Always persist a branch after caught reset/execution/evaluator errors."""
    stem = f'root_{plan["root_index"]}_{plan["method"]}_replay{repetition}'
    row = {k:deepcopy(plan[k]) for k in ('root_index','root_sha256','method',
        'selection_sha256','planned_horizon_tokens','planned_commands','corrected_search_horizon')}
    row.update(repetition=repetition,status='branch_failed',attempted_commands=0,
        completed_commands=0,trace=[],failures=[],full_horizon_completed=False,
        physical_env_step_attempts=0,physical_env_step_returns=0,
        warmup_cost_scope='separate reset/warmup; not included in selected-path counters',
        terminal_rgb_path=None)
    rgb = None
    journal = output / (stem+'.events.jsonl')
    try:
        with journal.open('x') as stream:
            def event(value, image):
                value = deepcopy(value)
                if image is not None:
                    from PIL import Image
                    image_path = output / f'{stem}_rgb_{value.get("primitive_step",0)}.png'
                    # Exclusive open prevents any accidental replay overwrite.
                    with image_path.open('xb') as image_stream:
                        Image.fromarray(image).save(image_stream,format='PNG')
                        image_stream.flush(); os.fsync(image_stream.fileno())
                    value.update(rgb_path=str(image_path),rgb_file_sha256=digest(image_path))
                    row['terminal_rgb_path'] = str(image_path)
                    row['terminal_rgb_sha256'] = digest(image_path)
                stream.write(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')
                stream.flush(); os.fsync(stream.fileno())
            row['context_identity'] = reset()
            result,rgb = guarded_execute(env,plan['commands'],root_id=plan['root_sha256'],
                state_of=state_of,event=event)
            row.update(result)
            if result['failure']:
                row['failures'].append(result['failure'])
            if result['final_state'] is not None:
                row.update(evaluate(result['final_state']))
    except (Exception, KeyboardInterrupt) as exc:
        row['failures'].append(failure(exc,'physical_branch'))
        row['status'] = 'branch_failed'
    row['journal_path'] = str(journal)
    if journal.exists():
        row['journal_sha256'] = digest(journal)
    write(output/(stem+'.physical.json'),row)
    return row


def replay_equal(left,right):
    """Compare physical state/guard/commands, not per-repetition file paths."""
    keys=('status','attempted_commands','completed_commands','planned_commands','trace',
          'final_state','guard','context_identity','official','pose')
    return (all(k in left and k in right and left[k]==right[k] for k in keys)
            and len(left['trace'])==left['completed_commands']==right['completed_commands'])


def encode_saved(rows, output, encode):
    """Read only already-persisted terminal RGB; save failures per branch."""
    receipts=[]
    for row in rows:
        result=dict(root_sha256=row['root_sha256'],method=row['method'],repetition=row['repetition'],
            attempted_encode_samples=0,status='blocked',failure=None)
        try:
            path=row['terminal_rgb_path']
            if not path or digest(path)!=row['terminal_rgb_sha256']:
                raise ValueError('no hash-valid persisted terminal RGB')
            result['attempted_encode_samples']=1
            result['terminal_vector']=encode(Path(path))
            vector=np.asarray(result['terminal_vector'])
            if vector.shape!=(192,) or not np.isfinite(vector).all():
                raise ValueError('finite 192-vector required')
            result['status']='completed'
        except (Exception, KeyboardInterrupt) as exc:
            result.update(status='encoding_failed',failure=failure(exc,'terminal_encoding'))
        receipts.append(result)
        write(output/f'root_{row["root_index"]}_{row["method"]}_replay{row["repetition"]}.encoding.json',result)
    return receipts


def prepare(p3_report, correction, compilation, policy):
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability.d2p_prerun_source_seal import verify_seal
    source,bindings,config=checked.validate_inputs(compilation,policy)
    p3_report,correction=Path(p3_report).resolve(),Path(correction).resolve()
    protocol_path=p3_report.parent/'P3_ADAPTIVE_RUNTIME_protocol_20261027.json'
    report=json.loads(p3_report.read_text()); protocol=json.loads(protocol_path.read_text())
    corrected=json.loads(correction.read_text())
    for p in (p3_report,protocol_path):
        if corrected['inputs'].get(str(p))!=digest(p):
            raise ValueError('corrective audit input binding differs')
    seal=corrected['source_seal']
    verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
    if any(digest(p)!=h for p,h in protocol['protocol']['sources'].items()):
        raise ValueError('original P3 source freeze changed')
    if protocol['protocol']['bindings']!=bindings:
        raise ValueError('P3 compilation/checkpoint bindings differ')
    plans=selected_plans(report,protocol,corrected,source)
    candidates=[json.loads(Path(p).read_text()) for p in bindings if Path(p).name=='report.json']
    interventions=[r for r in candidates if 'bank_path' in r.get('protocol',{})]
    if len(interventions)!=1:
        raise ValueError('unique physical root provenance required')
    bank_path=Path(interventions[0]['protocol']['bank_path']).resolve()
    if digest(bank_path)!=interventions[0]['protocol']['bank_sha256']:
        raise ValueError('physical source bank hash differs')
    bank=json.loads(bank_path.read_text()); upstream_path=Path(bank['upstream_path']).resolve()
    if digest(upstream_path)!=interventions[0]['protocol']['upstream_sha256']:
        raise ValueError('physical root source hash differs')
    upstream=base.physical._load_preflight(upstream_path)
    refs={r['root_sha256']:r for r in upstream['rows']}
    if any(p['root_sha256'] not in refs for p in plans):
        raise ValueError('physical cohort reference missing')
    for p in (p3_report,protocol_path,correction,bank_path,upstream_path):
        bindings[str(p)]=digest(p)
    return base,checked,source,config,bindings,plans,upstream,refs


def run(*,p3_report,correction,compilation,policy,output,encode_terminals=False,preflight_only=False):
    base,checked,source,config,bindings,plans,upstream,refs=prepare(p3_report,correction,compilation,policy)
    from experiments.spectral_reachability.run_pusht_sampling_comparison_20261027_v2 import verify_rgb_context,official_checked
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal,verify_seal
    if preflight_only:
        print(json.dumps(dict(status='preflight_only',branches=len(plans),
            selected_horizons=[p['planned_horizon_tokens'] for p in plans],
            physical_repetitions=2,encode_terminals=encode_terminals,P3_admission=False)))
        return
    output=Path(output).resolve(); output.mkdir(exist_ok=False)
    protocol=dict(schema=TAG+'.protocol',bindings=bindings,plans=plans,physical_repetitions=2,
        endpoints='actual variable H2/H3, no padding or fixed-H comparison',
        encoding_optional=encode_terminals,physical_before_encoding=True,
        root_selection_changed=False,P1_admission=False,P3_admission=False,
        scope='known calibration cohort; no efficacy or total-budget matching claim')
    write(output/'protocol.json',protocol)
    rows,failures,seals,encodings=[],[],[],[]
    env=None; model=None; before=None; host=None
    try:
        env=base.gym.make('swm/PushT-v1',render_mode='rgb_array')
        files=checked.sources_before_allocate(source,config)
        files += [Path(__file__).resolve(),Path(__file__).resolve().parents[2]/f'tests/test_{TAG}.py',
            Path(__file__).resolve().parents[3]/'docs/P3_MATCHED_PHYSICAL_20261027.md',
            Path(RgbObservabilityGuard.__module__.replace('.','/')).with_suffix('.py').resolve(),
            Path(official_checked.__code__.co_filename),Path(env.unwrapped.eval_state.__code__.co_filename)]
        files=sorted(set(files))
        for i in range(0,len(files),50):
            seals.append(create_seal(files=files[i:i+50],args_json=json.dumps(dict(
                protocol_path=str(output/'protocol.json'),protocol_sha256=digest(output/'protocol.json'))),
                output_dir=output/f'source_seal_{i//50}'))
        if tuple(env.unwrapped.variation_space['agent']['color'].value.tolist())!=(65,105,225):
            raise ValueError('unsupported renderer color')
        roots=base.collect_nonterminal_boundary_contact_roots(env,count=4,
            seed=int(upstream['protocol']['root_seed']),oversample_factor=4)
        roots={base.root_fingerprint(r):r for r in roots}
        if len(roots)!=4 or set(roots)!={p['root_sha256'] for p in plans}:
            raise ValueError('physical cohort differs; no replacement roots')
        for plan in plans:
            root=roots[plan['root_sha256']]
            def reset():
                frames,history,goal=base.physical._reset_and_warmup(env,root,history_len=3)
                identity=verify_rgb_context(frames,goal,refs[plan['root_sha256']])
                identity.update(history=np.asarray(history).tolist(),
                    state=base.p0.packets.state_of(env).tolist(),goal=np.asarray(root['goal']).tolist())
                return identity
            def evaluate(state):
                return dict(official=official_checked(env,state,root['goal']),
                    pose=base.physical._pose_label(state,root['goal']))
            for repetition in range(2):
                rows.append(physical_branch(plan=plan,repetition=repetition,output=output,
                    reset=reset,env=env,state_of=base.p0.packets.state_of,evaluate=evaluate))
    except (Exception, KeyboardInterrupt) as exc:
        failures.append(failure(exc,'physical_setup'))
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                failures.append(failure(exc,'env_close'))
    # Materialize every unexecuted census slot before any CUDA allocation.
    existing={(r['root_sha256'],r['method'],r['repetition']) for r in rows}
    for plan in plans:
        for repetition in range(2):
            if (plan['root_sha256'],plan['method'],repetition) not in existing:
                row={k:plan[k] for k in ('root_index','root_sha256','method','planned_horizon_tokens','planned_commands','selection_sha256')}
                row.update(repetition=repetition,status='blocked',failure='physical setup unavailable',
                    attempted_commands=0,completed_commands=0,trace=[],terminal_rgb_path=None,
                    full_horizon_completed=False,physical_env_step_attempts=0,physical_env_step_returns=0)
                write(output/f'root_{plan["root_index"]}_{plan["method"]}_replay{repetition}.physical.json',row)
                rows.append(row)
    write(output/'physical_report.json',dict(rows=rows,failures=failures,P3_admission=False))
    if encode_terminals:
        try:
            import torch
            from PIL import Image
            host=base.require_cuda_allocation('cuda')
            device=torch.device('cuda:0')
            model=base.load_compatible_pretrained(Path(policy),device=device).eval().requires_grad_(False)
            before=checked.tensor_state_hash(model)
            transform=base.p0.image_transform()
            def encode(path):
                with Image.open(path) as image:
                    rgb=np.asarray(image.convert('RGB')).copy()
                with torch.no_grad():
                    vector=model.encode({'pixels':base.p0._image_tensor(rgb,transform=transform,device=device)})['emb'][0,0]
                if vector.device.type!='cuda' or not torch.isfinite(vector).all():
                    raise ValueError('finite HOST CUDA terminal encoding required')
                torch.cuda.synchronize()
                return vector.cpu().tolist()
            encodings=encode_saved(rows,output,encode)
        except (Exception, KeyboardInterrupt) as exc:
            failures.append(failure(exc,'cuda_encoding_setup'))
    comparisons=[]
    for plan in plans:
        peers=[r for r in rows if (r['root_sha256'],r['method'])==(plan['root_sha256'],plan['method'])]
        comparisons.append(dict(root_sha256=plan['root_sha256'],method=plan['method'],
            independent_physical_replay_exact=len(peers)==2 and replay_equal(*peers)))
    checks=[]
    for path,h in bindings.items():
        try:
            checks.append(dict(path=path,matched=digest(path)==h))
        except Exception as exc:
            checks.append(dict(path=path,matched=False,error=str(exc)))
    verified=[]
    try:
        prepare(p3_report,correction,compilation,policy)
        if not seals:
            raise ValueError('execution source seal missing')
        verified=[verify_seal(output_dir=s['output_dir'],expected_manifest_sha256=s['manifest_sha256']) for s in seals]
    except Exception as exc:
        failures.append(failure(exc,'final_source_check'))
    gates=dict(all_sixteen_retained=len(rows)==16,
        no_execution_errors=all(r['status'] in ('completed','observation_unavailable','environment_terminated') for r in rows),
        independent_replays_exact=all(c['independent_physical_replay_exact'] for c in comparisons),
        inputs_unchanged=all(c['matched'] for c in checks),sources_unchanged=bool(verified),
        encodings_complete=not encode_terminals or len(encodings)==16 and all(e['status']=='completed' for e in encodings))
    if model is not None:
        try:
            gates['model_unchanged']=before==checked.tensor_state_hash(model)
            gates['frozen_eval']=all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters())
        except Exception as exc:
            failures.append(failure(exc,'model_check'))
    passed=all(gates.values()) and not failures
    write(output/'report.json',dict(schema=TAG+'.physical',status='calibration_recorded' if passed else 'integrity_fail',
        rows=rows,encodings=encodings,comparisons=comparisons,gates=gates,failures=failures,
        source_seals=seals,source_verification=verified,binding_checks=checks,host=host,
        selected_path_env_step_attempts=sum(r['physical_env_step_attempts'] for r in rows),
        selected_path_env_step_returns=sum(r['physical_env_step_returns'] for r in rows),
        attempted_terminal_encode_samples=sum(e['attempted_encode_samples'] for e in encodings),
        new_predict_calls=0,P1_admission=False,P3_admission=False,efficacy_verified=False,
        fixed_horizon_comparison=False,fair_total_budget_comparison=False))
    if not passed:
        raise RuntimeError('P3 physical integrity failed; all branch artifacts retained')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('p3_report','correction','compilation','policy','output'):
        parser.add_argument('--'+name.replace('_','-'),dest=name,type=Path,required=True)
    parser.add_argument('--encode-terminals',action='store_true')
    parser.add_argument('--preflight-only',action='store_true')
    run(**vars(parser.parse_args()))
