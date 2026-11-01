"""Execute all 80 sealed constant/full candidates; no selection or prediction.

Physical RGB/attempt journals precede optional HOST CUDA prefix/terminal
encoding. Existing sources remain immutable. P1 admission is always false.
"""
from copy import deepcopy
import inspect
import json
from pathlib import Path
import re

import numpy as np

from experiments.spectral_reachability import p3_matched_physical_20261027_v1 as durable
from experiments.spectral_reachability.d2p_composition_witness_v2 import digest as content_digest
from experiments.spectral_reachability.run_pusht_sampling_comparison_20261027_v2 import (
    verify_rgb_context, official_checked,
)

TAG='pusht_full_packet_physical_20261027'
ARMS=('constant','full')
write=durable.write
sha=durable.digest


def array(value,shape):
    a=np.asarray(value)
    if a.shape!=shape or a.dtype.kind not in 'fiu' or not np.isfinite(a).all():
        raise ValueError(f'finite numeric {shape} required')
    return a


def candidate_id(value):
    if type(value) is int and value>=0:
        return str(value)
    if isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9_-]{1,64}',value):
        return value
    raise ValueError('safe integer/string candidate ID required')


def validate_arm(saved,root,compiled,protocol):
    """Recompute original MODEL-only coordinates, solves, costs and ID choices.

    Fixed 1e-6 command tolerance covers torch float32 tanh vs NumPy, not action
    clipping or parameter fitting. Exact saved raw concatenation is still used.
    """
    from experiments.spectral_reachability.pusht_full_packet_chart_20261027 import chart_matrix
    from experiments.spectral_reachability.pusht_full_packet_compiler_20261027 import (
        bounded_ridge,spectral_bank,CAP,DAMPING,RADII,
    )
    if (protocol['damping']!=DAMPING or protocol['norm_cap_atanh_per_packet']!=CAP
            or list(protocol['spectral_radii'])!=list(RADII)):
        raise ValueError('compiler damping/norm protocol differs')
    arm=saved['arm'];basis=chart_matrix(arm);d=basis.shape[1]
    if not np.array_equal(array(saved['chart'],basis.shape),basis):
        raise ValueError('saved chart differs from constant/full basis')
    q=array(compiled['scoring']['metric']['projection'],(192,192))
    goal=array(compiled['goal_vectors'][0],(192,))
    jac=array(root['root_jacobian_full'],(192,10))
    origin=array(root['root_nominal_terminal'],(192,))
    bank=spectral_bank(jac@basis,origin,goal,q)
    if not np.allclose(array(saved['bank']['params'],(10,d)),bank['params'],rtol=0,atol=1e-12):
        raise ValueError('saved spectral bank differs from root response')
    for index,c in enumerate(saved['candidates']):
        if c['kind']!=bank['kinds'][index]:
            raise ValueError('candidate kind/bank ordering differs')
        prefix=array(c['prefix_params'],(d,));suffix=array(c['suffix_params'],(d,))
        if (np.linalg.norm(prefix)>CAP+1e-6 or np.linalg.norm(suffix)>CAP+1e-6
                or not np.allclose(prefix,np.asarray(bank['params'][index],dtype=np.float32),atol=1e-6,rtol=0)):
            raise ValueError('candidate chart params/norm/bank differs')
        for params,field in ((prefix,'prefix_commands'),(suffix,'suffix_commands')):
            expected=np.tanh(basis.astype(np.float32)@params.astype(np.float32)).reshape(5,2)
            if not np.allclose(array(c[field],(5,2)),expected,rtol=0,atol=1e-6):
                raise ValueError('raw commands differ from chart params; no 2D full reconstruction')
        predicted_prefix=array(c['predicted_prefix'],(192,)); predicted_terminal=array(c['predicted_terminal'],(192,))
        nominal=array(c['nominal_terminal'],(192,)); response=array(c['jacobian'],(192,d))
        own=bounded_ridge(q@response,q@(goal-nominal))
        if (not np.allclose(suffix,np.asarray(own['params'],dtype=np.float32),rtol=0,atol=1e-6)
                or c.get('independent_model_replay_exact') is not True):
            raise ValueError('saved suffix solve/replay differs')
        stage=float(np.mean((q@(goal-predicted_prefix))**2))
        terminal=float(np.mean((q@(goal-predicted_terminal))**2))
        values=dict(stage_cost=stage,own_reach_value=own['own_reach_value'],
            objective=stage+own['own_reach_value'],nonlinear_terminal_cost=terminal)
        if any(not np.isclose(c[k],v,rtol=1e-10,atol=1e-12) for k,v in values.items()):
            raise ValueError('original stage/reach/objective/terminal metric differs')
    if (saved['selected_candidate_id']!=min(saved['candidates'],key=lambda c:c['objective'])['candidate_id']
            or saved['greedy_candidate_id']!=min(saved['candidates'],key=lambda c:c['stage_cost'])['candidate_id']):
        raise ValueError('model-only stable selected/greedy IDs differ')


def plans_from_saved(report,compilation):
    if (report.get('status')!='plans_sealed' or not report.get('gates')
            or not all(value is True for value in report['gates'].values())
            or report.get('all_plans_before_physical_execution') is not True):
        raise ValueError('sealed planning report required')
    roots=report['protocol']['roots']
    if len(roots)!=4 or len(set(roots))!=4 or roots!=compilation['protocol']['roots']:
        raise ValueError('exact four official root identities/order required')
    source={r['root_sha256']:r for r in compilation['rows']}
    rows=report['plans']
    if len(rows)!=4 or len(source)!=4 or {r['root_sha256'] for r in rows}!=set(roots):
        raise ValueError('all four unique roots required, no filtering')
    index={r['root_sha256']:r for r in rows}
    plans=[]
    for i,key in enumerate(roots):
        row=index[key]
        if type(row['root_index']) is not int or row['root_index']!=i:
            raise ValueError('root identity/index differs')
        arms=row['arms']
        if len(arms)!=2 or {a['arm'] for a in arms}!=set(ARMS):
            raise ValueError('constant/full arms required')
        armmap={a['arm']:a for a in arms}
        for arm in (ARMS if i%2==0 else ARMS[::-1]):
            saved=armmap[arm]; candidates=saved['candidates']
            ids=[candidate_id(c['candidate_id']) for c in candidates]
            if len(ids)!=10 or len(set(ids))!=10:
                raise ValueError('all ten unique candidates required')
            selected=candidate_id(saved['selected_candidate_id'])
            greedy=candidate_id(saved['greedy_candidate_id'])
            if selected not in ids or greedy not in ids:
                raise ValueError('presealed selected/greedy ID absent')
            validate_arm(saved,row,source[key]['compiled'],report['protocol'])
            zero_ids=[]
            for c in candidates:
                cid=candidate_id(c['candidate_id'])
                raw=array(c['commands'],(10,2))
                prefix=array(c['prefix_commands'],(5,2)); suffix=array(c['suffix_commands'],(5,2))
                if np.any(np.abs(raw)>1) or not np.array_equal(raw,np.concatenate((prefix,suffix))):
                    raise ValueError('exact bounded prefix/suffix concatenation required')
                if arm=='constant' and not np.array_equal(suffix,np.repeat(suffix[:1],5,axis=0)):
                    raise ValueError('constant arm suffix is not constant')
                # Full is consumed verbatim; no tanh/repetition/2D reconstruction.
                for field in ('predicted_prefix','predicted_terminal'):
                    array(c[field],(192,))
                for field in ('stage_cost','own_reach_value','objective','nonlinear_terminal_cost'):
                    array(c[field],())
                if c['kind']=='zero':
                    if np.any(prefix):raise ValueError('declared zero prefix is nonzero')
                    zero_ids.append(cid)
                plan=dict(root_index=i,root_sha256=key,arm=arm,candidate_id=cid,
                    method=arm+'_'+cid,commands=raw.tolist(),planned_horizon_tokens=2,
                    planned_commands=10,corrected_search_horizon=2,
                    selected_candidate_id=selected,greedy_candidate_id=greedy,
                    saved_candidate=deepcopy(c))
                plan['selection_sha256']=content_digest(plan)
                plans.append(plan)
            # Zero means the frozen zero-PREFIX baseline, not a zero suffix or
            # outcome-selected lowest observed task cost.
            if len(zero_ids)!=1:
                raise ValueError('unique saved zero-prefix baseline required')
            declared=saved.get('zero_candidate_id',zero_ids[0])
            if candidate_id(declared)!=zero_ids[0]:
                raise ValueError('declared zero candidate differs from raw zero prefix')
            for plan in plans[-10:]:
                plan['zero_candidate_id']=zero_ids[0]
                plan['selection_sha256']=content_digest({k:v for k,v in plan.items() if k!='selection_sha256'})
    if len(plans)!=80 or len({(p['root_sha256'],p['method']) for p in plans})!=80:
        raise ValueError('complete 80-branch census required')
    return plans


def prepare(planning_report):
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability.d2p_prerun_source_seal import verify_seal
    path=Path(planning_report).resolve(); report=json.loads(path.read_text()); protocol=report['protocol']
    source,bindings,config=checked.validate_inputs(protocol['compilation'],protocol['policy'])
    if protocol['bindings']!=bindings:
        raise ValueError('planning inputs differ from official compilation/policy')
    seals=report.get('source_seals') or ([report['source_seal']] if report.get('source_seal') else [])
    if not seals:
        raise ValueError('compiler must provide prerun source_seals/source_seal')
    for seal in seals:
        verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
    plans=plans_from_saved(report,source)
    bindings=deepcopy(bindings); bindings[str(path)]=sha(path)
    for seal in seals:
        for name in ('manifest.json','args.json'):
            p=Path(seal['output_dir'])/name
            bindings[str(p.resolve())]=sha(p)
    candidates=[json.loads(Path(p).read_text()) for p in protocol['bindings'] if Path(p).name=='report.json']
    interventions=[r for r in candidates if 'bank_path' in r.get('protocol',{})]
    if len(interventions)!=1:
        raise ValueError('unique physical root provenance required')
    intervention=interventions[0]['protocol']; bank_path=Path(intervention['bank_path']).resolve()
    if sha(bank_path)!=intervention['bank_sha256']:
        raise ValueError('official bank changed')
    bank=json.loads(bank_path.read_text()); upstream_path=Path(bank['upstream_path']).resolve()
    if sha(upstream_path)!=intervention['upstream_sha256']:
        raise ValueError('official root source changed')
    upstream=base.physical._load_preflight(upstream_path)
    refs={r['root_sha256']:r for r in upstream['rows']}
    if set(refs)!=set(protocol['roots']) or len(upstream['rows'])!=4:
        raise ValueError('physical reference cohort differs')
    for p in (bank_path,upstream_path): bindings[str(p)]=sha(p)
    return base,checked,source,config,bindings,plans,upstream,refs,protocol


def materialize_missing(plans,rows,output):
    existing={(r['root_sha256'],r['method']) for r in rows}
    for p in plans:
        if (p['root_sha256'],p['method']) in existing: continue
        row={k:p[k] for k in ('root_index','root_sha256','method','selection_sha256','planned_horizon_tokens','planned_commands')}
        row.update(repetition=0,status='blocked',failure='physical setup unavailable',trace=[],
            attempted_commands=0,completed_commands=0,terminal_rgb_path=None,full_horizon_completed=False,
            physical_env_step_attempts=0,physical_env_step_returns=0)
        write(output/f'root_{p["root_index"]}_{p["method"]}_replay0.physical.json',row)
        rows.append(row)


def image_receipt(row,kind):
    """Only observed step5 prefix / step10 full terminal; no padding/fallback.

    Aborted terminal RGB remains in physical artifacts but is NOT a full-H2
    terminal prediction comparison. Prefix may remain comparable if step5 exists.
    """
    step=5 if kind=='prefix' else 10
    if row['completed_commands']<step:
        raise ValueError(f'{kind} blocked: branch ended before step {step}')
    path=Path(row['journal_path'])
    if sha(path)!=row['journal_sha256']:
        raise ValueError('physical journal changed')
    events=[json.loads(line) for line in path.read_text().splitlines()]
    matches=[e for e in events if e['kind']=='observation' and e['primitive_step']==step]
    if len(matches)!=1:
        raise ValueError('unique persisted endpoint observation required')
    event=matches[0]; image=Path(event['rgb_path'])
    if sha(image)!=event['rgb_file_sha256']:
        raise ValueError('persisted endpoint PNG changed')
    return image


def encode_observations(rows,plans,output,setup,metrics=None):
    if not (output/'physical_report.json').is_file():
        raise ValueError('persist all physical branches before CUDA setup')
    encode=None; setup_error=None
    try: encode=setup()
    except (Exception,KeyboardInterrupt) as exc: setup_error=durable.failure(exc,'encoding_setup')
    planmap={(p['root_sha256'],p['method']):p for p in plans}
    receipts=[]
    for row in rows:
        for kind in ('prefix','terminal'):
            receipt=dict(root_sha256=row['root_sha256'],method=row['method'],kind=kind,
                status='blocked',attempted_encode_samples=0,comparable=False,failure=None)
            try:
                path=image_receipt(row,kind)
                receipt.update(rgb_path=str(path),rgb_sha256=sha(path))
                if setup_error is not None:
                    receipt.update(status='encoding_setup_failed',failure=setup_error)
                else:
                    receipt['attempted_encode_samples']=1
                    vector=array(encode(path),(192,))
                    prediction=np.asarray(planmap[row['root_sha256'],row['method']]['saved_candidate'][
                        'predicted_prefix' if kind=='prefix' else 'predicted_terminal'])
                    receipt.update(status='completed',comparable=True,vector=vector.tolist(),
                        prediction_mse=float(np.mean((vector-prediction)**2)))
                    if metrics is not None:
                        goal,q=metrics[row['root_sha256']]
                        receipt.update(observed_Q_goal_cost=float(np.mean((q@(goal-vector))**2)),
                            projected_prediction_mse=float(np.mean((q@(vector-prediction))**2)))
            except (Exception,KeyboardInterrupt) as exc:
                receipt.update(status='encoding_failed' if receipt['attempted_encode_samples'] else 'blocked',
                    failure=durable.failure(exc,'saved_endpoint_encoding'))
            write(output/f'root_{row["root_index"]}_{row["method"]}_{kind}.encoding.json',receipt)
            receipts.append(receipt)
    return receipts


def comparisons(plans,rows):
    index={(r['root_sha256'],r['method']):r for r in rows}
    result=[]
    for root in dict.fromkeys(p['root_sha256'] for p in plans):
        for arm in ARMS:
            p=next(p for p in plans if p['root_sha256']==root and p['arm']==arm)
            roles={}
            for role,field in (('selected','selected_candidate_id'),('greedy','greedy_candidate_id'),('zero','zero_candidate_id')):
                row=index[root,arm+'_'+p[field]]
                roles[role]=dict(candidate_id=p[field],status=row['status'],
                    completed_commands=row['completed_commands'],full_horizon_comparable=row['full_horizon_completed'],
                    official=row.get('official'),pose=row.get('pose'),final_state=row.get('final_state'),
                    physical_file=f'root_{p["root_index"]}_{row["method"]}_replay0.physical.json')
            result.append(dict(root_sha256=root,arm=arm,roles=roles,
                selection_policy='saved model-only IDs and unique zero prefix; no physical reselection'))
    return result


def run(*,planning_report,output,encode_terminals=False,preflight_only=False):
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal,verify_seal
    base,checked,source,config,bindings,plans,upstream,refs,planning_protocol=prepare(planning_report)
    if preflight_only:
        print(json.dumps(dict(status='preflight_only',branches=80,planned_env_steps=800,
            max_prefix_images=80,max_terminal_images=80,new_predict_calls=0,P1_admission=False)))
        return
    output=Path(output).resolve(); output.mkdir(exist_ok=False)
    protocol=dict(schema=TAG+'.protocol',bindings=bindings,planning_report=str(Path(planning_report).resolve()),
        plans=plans,physical_repetitions=1,planned_commands=10,encode_terminals=encode_terminals,
        all_candidates_executed=True,physical_before_encoding=True,P1_admission=False,
        scope='four known roots, 80 model-only sealed candidates, engineering calibration')
    write(output/'protocol.json',protocol); protocol_hash=sha(output/'protocol.json')
    rows,failures,seals,encodings=[],[],[],[]
    env=None; model=None; before=None; host=None
    try:
        env=base.gym.make('swm/PushT-v1',render_mode='rgb_array')
        files=checked.sources_before_allocate(source,config)
        files += [Path(__file__).resolve(),Path(durable.__file__),Path(inspect.getfile(durable.RgbObservabilityGuard)),
            Path(verify_rgb_context.__code__.co_filename),Path(env.unwrapped.eval_state.__code__.co_filename),
            Path(__file__).resolve().parents[2]/f'tests/test_{TAG}.py',
            Path(__file__).resolve().parents[3]/'docs/PUSHT_FULL_PACKET_PHYSICAL_20261027.md']
        files=sorted(set(p.resolve() for p in files))
        seals=[create_seal(files=files[i:i+40],args_json=json.dumps(dict(protocol_sha256=protocol_hash,
            planning_report_sha256=bindings[str(Path(planning_report).resolve())])),
            output_dir=output/f'source_seal_{i//40}') for i in range(0,len(files),40)]
        write(output/'source_seals.json',seals)
        if tuple(env.unwrapped.variation_space['agent']['color'].value.tolist())!=(65,105,225):
            raise ValueError('unsupported renderer')
        roots=base.collect_nonterminal_boundary_contact_roots(env,count=4,
            seed=int(upstream['protocol']['root_seed']),oversample_factor=4)
        roots={base.root_fingerprint(r):r for r in roots}
        if len(roots)!=4 or set(roots)!=set(refs): raise ValueError('physical roots differ')
        for plan in plans:
            root=roots[plan['root_sha256']]
            def reset():
                if base.root_fingerprint(root)!=plan['root_sha256']:raise ValueError('root changed')
                frames,history,goal=base.physical._reset_and_warmup(env,root,history_len=3)
                identity=verify_rgb_context(frames,goal,refs[plan['root_sha256']])
                identity.update(history=np.asarray(history).tolist(),state=base.p0.packets.state_of(env).tolist(),
                    goal=np.asarray(root['goal']).tolist())
                return identity
            def evaluate(state):
                return dict(official=official_checked(env,state,root['goal']),pose=base.physical._pose_label(state,root['goal']))
            row=durable.physical_branch(plan=plan,repetition=0,output=output,reset=reset,env=env,
                state_of=base.p0.packets.state_of,evaluate=evaluate)
            rows.append(row)
            print(f'full packet {plan["root_index"]} {plan["method"]}: {row["status"]} {row["completed_commands"]}/10',flush=True)
    except (Exception,KeyboardInterrupt) as exc: failures.append(durable.failure(exc,'physical_setup'))
    finally:
        if env is not None:
            try:env.close()
            except Exception as exc:failures.append(durable.failure(exc,'env_close'))
    materialize_missing(plans,rows,output)
    write(output/'physical_report.json',dict(rows=rows,failures=deepcopy(failures),P1_admission=False,new_predict_calls=0))
    if encode_terminals:
        def setup():
            nonlocal model,before,host
            import torch
            from PIL import Image
            host=base.require_cuda_allocation('cuda'); device=torch.device('cuda:0')
            model=base.load_compatible_pretrained(Path(planning_protocol['policy']),device=device).eval().requires_grad_(False)
            before=checked.tensor_state_hash(model)
            def forbid(*args,**kwargs):raise RuntimeError('physical encoding forbids model.predict')
            model.predict=forbid
            transform=base.p0.image_transform()
            def encode(path):
                with Image.open(path) as im:rgb=np.asarray(im.convert('RGB')).copy()
                with torch.no_grad():
                    vector=model.encode({'pixels':base.p0._image_tensor(rgb,transform=transform,device=device)})['emb'][0,0]
                if vector.device.type!='cuda' or not torch.isfinite(vector).all():raise ValueError('finite HOST CUDA vector required')
                torch.cuda.synchronize(device)
                return vector.cpu().tolist()
            return encode
        metrics={r['root_sha256']:(np.asarray(r['compiled']['goal_vectors'][0]),
            np.asarray(r['compiled']['scoring']['metric']['projection'])) for r in source['rows']}
        encodings=encode_observations(rows,plans,output,setup,metrics)
    verified=[]; checks=[]
    for path,h in bindings.items():
        try:checks.append(dict(path=path,matched=sha(path)==h))
        except Exception as exc:checks.append(dict(path=path,matched=False,error=str(exc)))
    try:
        prepare(planning_report)
        if sha(output/'protocol.json')!=protocol_hash or not seals:raise ValueError('physical protocol/seal changed')
        verified=[verify_seal(output_dir=s['output_dir'],expected_manifest_sha256=s['manifest_sha256']) for s in seals]
    except Exception as exc:failures.append(durable.failure(exc,'source_check'))
    actionmap={(p['root_sha256'],p['method']):p['commands'] for p in plans}
    gates=dict(all_eighty_retained=len(rows)==80 and len({(r['root_sha256'],r['method']) for r in rows})==80,
        no_execution_errors=all(r['status'] in ('completed','observation_unavailable','environment_terminated') for r in rows),
        trace_matches_saved_commands=all(len(r['trace'])==r['completed_commands'] and all(
            e['primitive_step']==i+1 and e['command']==actionmap[r['root_sha256'],r['method']][i]
            for i,e in enumerate(r['trace'])) for r in rows),
        inputs_unchanged=all(c['matched'] for c in checks),sources_unchanged=bool(verified),
        encodings_complete=not encode_terminals or len(encodings)==160 and all(e['status']=='completed' for e in encodings))
    if model is not None:
        try:
            gates.update(model_unchanged=before==checked.tensor_state_hash(model),
                frozen_eval=all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters()))
        except Exception as exc:failures.append(durable.failure(exc,'model_check'))
    passed=all(gates.values()) and not failures
    write(output/'report.json',dict(schema=TAG+'.report',status='calibration_recorded' if passed else 'integrity_fail',
        rows=rows,encodings=encodings,model_only_selection_comparisons=comparisons(plans,rows),
        failures=failures,gates=gates,source_seals=seals,source_verification=verified,binding_checks=checks,host=host,
        selected_path_env_step_attempts=sum(r['physical_env_step_attempts'] for r in rows),
        selected_path_env_step_returns=sum(r['physical_env_step_returns'] for r in rows),
        attempted_encode_samples=sum(e['attempted_encode_samples'] for e in encodings),
        prefix_comparable=sum(e['comparable'] for e in encodings if e['kind']=='prefix'),
        terminal_comparable=sum(e['comparable'] for e in encodings if e['kind']=='terminal'),
        new_predict_calls=0,P1_admission=False,efficacy_verified=False,significance_claim=False))
    if not passed:raise RuntimeError('full packet physical integrity failed; all 80 branches retained')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--planning-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--encode-terminals',action='store_true')
    p.add_argument('--preflight-only',action='store_true')
    run(**vars(p.parse_args()))
