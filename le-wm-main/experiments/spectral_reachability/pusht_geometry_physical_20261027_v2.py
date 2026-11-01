"""Known-root geometry/stopK physical calibration; no generation or admission.

All 96 saved branches are retained before optional host-CUDA encoding of only
persisted step-5 and step-10 observations. No predictor or physical reselection.
"""
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import pusht_full_packet_physical_20261027 as packet_physical
from experiments.spectral_reachability import pusht_geometry_prefix_20261027_v3 as planning
from experiments.spectral_reachability.pusht_geometry_prefix_20261027_v3 import verify_plans
from experiments.spectral_reachability.d2p_composition_witness_v2 import digest as content_digest
from experiments.spectral_reachability.run_pusht_sampling_comparison_20261027_v2 import (
    verify_rgb_context, official_checked,
)

TAG = 'pusht_geometry_physical_20261027_v2'
ARMS = ('geometry', 'stopK')
durable = packet_physical.durable
write = durable.write
sha = durable.digest
array = packet_physical.array
image_receipt = packet_physical.image_receipt
materialize_missing = packet_physical.materialize_missing


def plans_from_saved(report, source):
    """Delegate mathematical verification, then bind immutable execution receipts."""
    verified = verify_plans(report, source)
    roots = source['protocol']['roots']
    if (len(roots) != 4 or len(set(roots)) != 4
            or report['protocol']['roots'] != roots
            or len(source['rows']) != 4 or not report.get('gates')):
        raise ValueError('exact four-root protocol and nonempty passing gates required')
    rootmap = {r['root_sha256']: r for r in report['plans']}
    candidates = {}
    for i, root in enumerate(roots):
        row = rootmap[root]
        if type(row['root_index']) is not int or row['root_index'] != i:
            raise ValueError('root identity/index differs')
        for arm in row['arms']:
            ids = [c['candidate_id'] for c in arm['candidates']]
            if ids != [f'prefix_{j}' for j in range(12)]:
                raise ValueError('ordered twelve-prefix bank required')
            for candidate in arm['candidates']:
                candidates[root, arm['arm']+'_'+candidate['candidate_id']] = (arm, candidate)
    plans = []
    for saved in verified:
        plan = deepcopy(saved)
        arm, candidate = candidates[plan['root_sha256'], plan['method']]
        raw = array(candidate['commands'], (10, 2))
        if np.any(np.abs(raw) > 1) or not np.array_equal(plan['commands'], raw):
            raise ValueError('consume exact bounded saved commands without reconstruction')
        for field in ('predicted_prefix', 'predicted_terminal', 'nominal_terminal'):
            array(candidate[field], (192,))
        array(candidate['jacobian'], (192, 10))
        array(candidate['prefix_params'], (10,))
        array(candidate['suffix_params'], (10,))
        for field in ('stage_cost', 'own_reach_value', 'objective'):
            array(candidate[field], ())
        plan.update(arm=arm['arm'], saved_candidate=deepcopy(candidate),
            corrected_search_horizon=2, selected_candidate_id=arm['selected_candidate_id'],
            greedy_candidate_id=arm['greedy_candidate_id'], zero_candidate_id='prefix_0')
        plan['selection_sha256'] = content_digest(plan)
        plans.append(plan)
    branch_census(plans, roots=roots)
    return plans


def branch_census(plans, rows=None, *, roots=None):
    """Explicit 4 x 2 x 12 identity census, never only a row-count check."""
    if roots is None:
        roots = list(dict.fromkeys(p['root_sha256'] for p in plans))
    expected = {(r, a+f'_prefix_{i}') for r in roots for a in ARMS for i in range(12)}
    key = lambda p: (p['root_sha256'], p['method'])
    if (len(roots) != 4 or len(set(roots)) != 4 or len(plans) != 96
            or len(expected) != 96 or {key(p) for p in plans} != expected):
        raise ValueError('complete unique all96branchcensus required')
    for p in plans:
        if (p['selection_sha256'] != content_digest({k:v for k,v in p.items() if k != 'selection_sha256'})
                or p['planned_commands'] != 10 or p['planned_horizon_tokens'] != 2):
            raise ValueError('plan selection receipt/horizon changed')
    index = {key(p): p for p in plans}
    if rows is not None:
        if len(rows) != 96 or {key(r) for r in rows} != expected:
            raise ValueError('all96 physical branches must precede encoding')
        for row in rows:
            p = index[key(row)]
            if (row['root_index'] != p['root_index'] or row.get('repetition') != 0
                    or row['selection_sha256'] != p['selection_sha256']
                    or row['planned_commands'] != 10 or row['planned_horizon_tokens'] != 2):
                raise ValueError('physical branch selection/root/horizon binding differs')
    return dict(all96branchcensus=True, planned_branches=96,
        retained_branches=None if rows is None else len(rows),
        roots=[dict(root_sha256=r, arms={a:12 for a in ARMS}) for r in roots],
        branch_ids=[dict(root_sha256=p['root_sha256'], method=p['method'],
            selection_sha256=p['selection_sha256']) for p in plans])


def encode_observations(rows, plans, output, setup, metrics=None):
    """Check the full durable census before invoking the shared endpoint encoder."""
    branch_census(plans, rows)
    physical = json.loads((output/'physical_report.json').read_text())
    if physical['rows'] != rows:
        raise ValueError('persisted physical census differs')
    for row in rows:
        path = output/f'root_{row["root_index"]}_{row["method"]}_replay0.physical.json'
        if json.loads(path.read_text()) != row:
            raise ValueError('persisted physical branch differs')
    return packet_physical.encode_observations(rows, plans, output, setup, metrics)


def comparisons(plans, rows):
    # Reuse the saved-role display logic, with our arms passed locally, never
    # monkeypatch the constant/full helper globals.
    index = {(r['root_sha256'], r['method']): r for r in rows}
    result = []
    for root in dict.fromkeys(p['root_sha256'] for p in plans):
        for arm in ARMS:
            p = next(p for p in plans if p['root_sha256'] == root and p['arm'] == arm)
            roles = {}
            for role, field in (('selected','selected_candidate_id'),
                                ('greedy','greedy_candidate_id'), ('zero','zero_candidate_id')):
                row = index[root, arm+'_'+p[field]]
                roles[role] = dict(candidate_id=p[field], status=row['status'],
                    completed_commands=row['completed_commands'],
                    full_horizon_comparable=row['full_horizon_completed'],
                    official=row.get('official'), pose=row.get('pose'), final_state=row.get('final_state'))
            result.append(dict(root_sha256=root, arm=arm, roles=roles,
                selection_policy='saved model-only IDs; no physical reselection'))
    return result


def sync_directory(output):
    fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


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


def run(*,planning_report,output,encode_terminals=False,preflight_only=False):
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal,verify_seal
    base,checked,source,config,bindings,plans,upstream,refs,planning_protocol=prepare(planning_report)
    if preflight_only:
        print(json.dumps(dict(status='preflight_only',branches=96,planned_env_steps=960,
            max_prefix_images=96,max_terminal_images=96,new_predict_calls=0,P1_admission=False,
            census=branch_census(plans))))
        return
    output=Path(output).resolve()
    if not output.name.startswith(TAG):
        raise ValueError('exclusive output directory basename must start with '+TAG)
    output.mkdir(exist_ok=False)
    protocol=dict(schema=TAG+'.protocol',bindings=bindings,planning_report=str(Path(planning_report).resolve()),
        plans=plans,physical_repetitions=1,planned_commands=10,encode_terminals=encode_terminals,
        all_candidates_executed=True,physical_before_encoding=True,P1_admission=False,
        scope='four known roots, 96 model-only sealed candidates, engineering calibration')
    write(output/'protocol.json',protocol); protocol_hash=sha(output/'protocol.json')
    write(output/'all96branchcensus.planned.json',branch_census(plans))
    sync_directory(output)
    rows,failures,seals,encodings=[],[],[],[]
    env=None; model=None; before=None; host=None
    try:
        env=base.gym.make('swm/PushT-v1',render_mode='rgb_array')
        files=checked.sources_before_allocate(source,config)
        files += [Path(__file__).resolve(),Path(durable.__file__),Path(inspect.getfile(durable.RgbObservabilityGuard)),
            Path(verify_rgb_context.__code__.co_filename),Path(env.unwrapped.eval_state.__code__.co_filename),
            Path(__file__).resolve().parents[2]/f'tests/test_{TAG}.py',
            Path(__file__).resolve().parents[2]/f'docs/{TAG}.md',
            Path(packet_physical.__file__),Path(planning.__file__)]
        planning_snapshot=json.loads(Path(planning_report).read_text())
        planning_seals=planning_snapshot.get('source_seals') or [planning_snapshot['source_seal']]
        for seal in planning_seals:
            manifest=json.loads((Path(seal['output_dir'])/'manifest.json').read_text())
            files += [Path(entry['source']) for entry in manifest['files']]
        files += [Path(verify_seal.__code__.co_filename)]
        files=sorted(set(p.resolve() for p in files))
        seals=[create_seal(files=files[i:i+40],args_json=json.dumps(dict(protocol_sha256=protocol_hash,
            planning_report_sha256=bindings[str(Path(planning_report).resolve())])),
            output_dir=output/f'source_seal_{i//40}') for i in range(0,len(files),40)]
        write(output/'source_seals.json',seals)
        verified_before=[verify_seal(output_dir=s['output_dir'],expected_manifest_sha256=s['manifest_sha256']) for s in seals]
        current=prepare(planning_report)
        if current[4] != bindings or current[5] != plans:
            raise ValueError('planning report/input snapshot changed before physics')
        if sha(output/'protocol.json') != protocol_hash or not verified_before:
            raise ValueError('pre-physics protocol/source integrity failed')
        sync_directory(output)
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
            sync_directory(output)
            print(f'geometry physical {plan["root_index"]} {plan["method"]}: {row["status"]} {row["completed_commands"]}/10',flush=True)
    except (Exception,KeyboardInterrupt) as exc: failures.append(durable.failure(exc,'physical_setup'))
    finally:
        if env is not None:
            try:env.close()
            except Exception as exc:failures.append(durable.failure(exc,'env_close'))
    materialize_missing(plans,rows,output)
    census=branch_census(plans,rows)
    write(output/'all96branchcensus.physical.json',census)
    write(output/'physical_report.json',dict(rows=rows,census=census,failures=deepcopy(failures),P1_admission=False,new_predict_calls=0))
    sync_directory(output)
    if encode_terminals:
        def setup():
            nonlocal model,before,host
            if not seals: raise ValueError('physical source seal missing')
            for seal in seals:
                verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
            current=prepare(planning_report)
            if current[4] != bindings or current[5] != plans:
                raise ValueError('planning report/input snapshot changed before encoding')
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
    gates=dict(all96branchcensus=census['all96branchcensus'],
        all_ninety_six_retained=len(rows)==96 and len({(r['root_sha256'],r['method']) for r in rows})==96,
        no_execution_errors=all(r['status'] in ('completed','observation_unavailable','environment_terminated') for r in rows),
        trace_matches_saved_commands=all(len(r['trace'])==r['completed_commands'] and all(
            e['primitive_step']==i+1 and e['command']==actionmap[r['root_sha256'],r['method']][i]
            for i,e in enumerate(r['trace'])) for r in rows),
        inputs_unchanged=all(c['matched'] for c in checks),sources_unchanged=bool(verified),
        encodings_complete=not encode_terminals or len(encodings)==192 and all(e['status']=='completed' for e in encodings))
    if model is not None:
        try:
            gates.update(model_unchanged=before==checked.tensor_state_hash(model),
                frozen_eval=all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters()))
        except Exception as exc:failures.append(durable.failure(exc,'model_check'))
    passed=all(gates.values()) and not failures
    write(output/'report.json',dict(schema=TAG+'.report',status='calibration_recorded' if passed else 'integrity_fail',
        rows=rows,all96branchcensus=census,encodings=encodings,model_only_selection_comparisons=comparisons(plans,rows),
        failures=failures,gates=gates,source_seals=seals,source_verification=verified,binding_checks=checks,host=host,
        selected_path_env_step_attempts=sum(r['physical_env_step_attempts'] for r in rows),
        selected_path_env_step_returns=sum(r['physical_env_step_returns'] for r in rows),
        attempted_encode_samples=sum(e['attempted_encode_samples'] for e in encodings),
        prefix_comparable=sum(e['comparable'] for e in encodings if e['kind']=='prefix'),
        terminal_comparable=sum(e['comparable'] for e in encodings if e['kind']=='terminal'),
        new_predict_calls=0,P1_admission=False,efficacy_verified=False,significance_claim=False))
    if not passed:raise RuntimeError('geometry physical integrity failed; all 96 branches retained')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--planning-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--encode-terminals',action='store_true')
    p.add_argument('--preflight-only',action='store_true')
    run(**vars(p.parse_args()))

