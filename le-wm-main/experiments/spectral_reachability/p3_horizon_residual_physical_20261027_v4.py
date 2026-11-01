"""Independent mixed-H2/H3 physical consumer; prepare/test are model-free.

Source preparation precedes report binding. Only main explicitly executes
physics; optional HOST CUDA endpoint encoding follows durable physical report.
No old physical consumer runner, planner modification, reselection or admission.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import os
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import p3_horizon_residual_budget_20261027_v4 as planning
from experiments.spectral_reachability import p3_matched_physical_20261027_v1 as durable
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal,verify_seal

NAME='p3_horizon_residual_physical_20261027_v4'
PROJECT=Path(__file__).resolve().parents[2]
PLANNING_PREPARED='results/p3_horizon_residual_budget_20261027_v4_prepared01'
PLANNING_SHA='1374e31190e18fabadc90f8fe8668e3800c863be5083ced8a4fa9bf023a85b98'
PLANNER_SOURCE_SHA='27f8b7f1cabed21d664ded40be0e0640aed9b0367bdfe2c39d377f4c482463c0'
METHODS=planning.METHODS
NORMALIZATION=dict(mean=np.asarray([-.0087,.0068],dtype=np.float32).tolist(),
    std=np.asarray([.2019,.2002],dtype=np.float32).tolist())
io,require,write=planning.io,planning.require,durable.write
same=lambda a,b:io.canonical(a)==io.canonical(b)


def contract():
    return dict(roots=4,rootarms=8,physical_repetitions=1,allowed_horizons=[2,3],
        commands='exact sealed exporter actions, no clipping/casting/reselection',
        guard='RoyalBlue RGB every raw command; retain attempts, returns, render failures and terminations',
        encoding='optional HOST CUDA only, after all durable physical branches and physical_report.json',
        encoding_cap=8,new_predict_calls=0,
        strata=['full_H2_pair','full_H3_pair','full_mixed_H2_H3','partial_same_steps','partial_or_unmatched'],
        no_cross_horizon_efficacy_aggregation=True,total_flops_equivalent=False,
        P1_admission=False,P3_admission=False,efficacy_claim=False)


def archived_exporter(receipt):
    source=Path(planning.__file__).resolve();found=[]
    for seal in receipt['source_seals']:
        verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
        manifest=io.decode((Path(seal['output_dir'])/'manifest.json').read_bytes())
        for entry in manifest['files']:
            if entry['source']==str(source):
                archive=Path(seal['output_dir'])/entry['archive']
                require(entry['sha256']==durable.digest(source)==durable.digest(archive)==PLANNER_SOURCE_SHA,'original sealed planner source required')
                require('\ndef selected_actions(output):' in archive.read_text(),'preexisting sealed exporter required')
                found.append(dict(source=str(source),archive=str(archive),sha256=entry['sha256']))
    require(len(found)==1,'unique archived exporter');return found[0]


def prepare(root,output):
    root=Path(root).resolve();output=Path(output).absolute();require(not output.exists(),'exclusive NEW prepare output')
    body,identity=planning.validate_prepared(root/PLANNING_PREPARED,PLANNING_SHA)
    receipt=io.decode((root/PLANNING_PREPARED/'prepare.json').read_bytes());proof=archived_exporter(receipt)
    previous,_=planning.prior.validate_prepared(body['prior_prepared'],body['prior_prepare_sha256'])
    evidence=io.Evidence(root)
    for p,h in previous['inputs_sha256'].items():evidence.raw(Path(p),h)
    for name in ('prepare.json','protocol.json'):evidence.raw(root/PLANNING_PREPARED/name)
    compilation=evidence.json(root/planning.prior.old.COMPILATION,planning.prior.old.COMPILATION_SHA)
    compiled=io.index(compilation['rows'],('root_sha256',))
    intervention_path=root/'results/pusht_d2p_goal_agent_intervention4_v2_20261026_cuda/report.json'
    intervention=evidence.json(intervention_path,previous['inputs_sha256'][str(intervention_path)])
    bank=evidence.json(Path(intervention['protocol']['bank_path']),intervention['protocol']['bank_sha256'])
    upstream=evidence.json(Path(bank['upstream_path']),intervention['protocol']['upstream_sha256'])
    require(durable.digest(bank['upstream_path'])==bank['upstream_sha256'],'upstream bank binding')
    refs=io.index(upstream['rows'],('root_sha256',));require(set(refs)==set(compiled)=={(r,) for r in body['roots']},'source physical root census')
    for p in body['packets']:
        c=compiled[p['root_sha256'],]['compiled'];original=planning.prior.old.packet(compiled[p['root_sha256'],])
        require(all(same(p[k],original[k]) for k in ('root_sha256','root_context','goal','projection')),'original root/goal0/Q alignment')
        require(same(p['goal'],c['goal_vectors'][0]),'goal0 exact source')
    seeds=[Path(__file__).resolve(),root/'tests'/('test_'+NAME+'.py'),root/'innerdocs'/(NAME+'.md'),
        root/'experiments/spectral_reachability/audit_pusht_d2p_goal_agent_intervention.py',
        root/'experiments/spectral_reachability/run_pusht_sampling_comparison_20261027_v2.py',
        root/'.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py',
        root/'.venv/lib/python3.12/site-packages/stable_worldmodel/envs/__init__.py']
    sources=planning.prior.physical.source_closure(root,seeds)
    for path in sources:evidence.raw(path)
    body_out=dict(schema=NAME+'.protocol',stage='source_prepared_awaiting_report',contract=contract(),
        planner_prepared=str(root/PLANNING_PREPARED),planner_prepare_sha256=PLANNING_SHA,planner_protocol_id=identity,
        exporter_proof=proof,roots=body['roots'],packets=body['packets'],model_bindings=body['model_bindings'],arm_order=body['arm_order'],
        root_seed=upstream['protocol']['root_seed'],root_oversample_factor=4,
        references={r:{k:refs[r,][k] for k in ('root_sha256','history_rgb_sha256','goal_rgb_sha256')} for r in body['roots']},
        inputs_sha256=dict(sorted(evidence.hashes.items())),sources_sha256={str(p):evidence.hashes[str(p)] for p in sources})
    evidence.recheck();output.mkdir(exist_ok=False);write(output/'protocol.json',body_out)
    seals=[create_seal(files=sources[i:i+40],args_json=io.canonical({'consumer':NAME}),output_dir=output/f'source_seal_{i//40}') for i in range(0,len(sources),40)]
    evidence.recheck()
    result=dict(status=body_out['stage'],protocol_sha256=durable.digest(output/'protocol.json'),source_seals=seals,
        model_loaded=False,new_physical_steps=0,new_predict_calls=0,P1_admission=False,P3_admission=False)
    write(output/'prepare.json',result);return result


def validate_prepared(prepared,expected_sha):
    prepared=Path(prepared).absolute();require(durable.digest(prepared/'prepare.json')==expected_sha,'prepare receipt SHA')
    receipt=io.decode((prepared/'prepare.json').read_bytes())
    require(receipt['model_loaded'] is False and receipt['new_physical_steps']==0 and receipt['new_predict_calls']==0,'prepare never executes neural/physics')
    require(durable.digest(prepared/'protocol.json')==receipt['protocol_sha256'],'protocol byte SHA')
    protocol=io.decode((prepared/'protocol.json').read_bytes())
    require(protocol['schema']==NAME+'.protocol' and protocol['contract']==contract(),'consumer contract')
    require(receipt['status']==protocol['stage'],'prepare stage identity')
    for p,h in protocol['inputs_sha256'].items():require(durable.digest(p)==h,'bound input changed: '+p)
    for p,h in protocol['sources_sha256'].items():require(durable.digest(p)==h,'sealed source changed: '+p)
    for seal in receipt['source_seals']:verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
    body,identity=planning.validate_prepared(protocol['planner_prepared'],protocol['planner_prepare_sha256'])
    require(protocol['planner_prepare_sha256']==PLANNING_SHA and identity==protocol['planner_protocol_id'] and
        protocol['roots']==body['roots'] and protocol['packets']==body['packets'] and
        protocol['model_bindings']==body['model_bindings'] and protocol['arm_order']==body['arm_order'],'original planner inputs')
    if protocol['stage']=='report_bound_ready':
        validate_prepared(protocol['source_prepared'],protocol['source_prepare_sha256'])
        require(durable.digest(Path(protocol['planning_output'])/'report.json')==protocol['planning_report_sha256'],'bound planner report bytes')
        plans=plans_from_exports(protocol,planning.selected_actions(protocol['planning_output']))
        require(same(plans,protocol['plans']),'exported selected actions/source witnesses changed')
    else:require(protocol['stage']=='source_prepared_awaiting_report','unknown preparation stage')
    return protocol


def plans_from_exports(protocol,exports):
    expected={(r,m) for r in protocol['roots'] for m in METHODS}
    index=io.index(exports,('root_sha256','mode'));require(set(index)==expected,'eight exact exported rootarms')
    packets=io.index(protocol['packets'],('root_sha256',));plans=[]
    for i,r in enumerate(protocol['roots']):
        for mode in protocol['arm_order'][i]:
            e=index[r,mode];p=packets[r,];h=e['horizon'];raw=np.asarray(e['actions'])
            require(type(h) is int and h in (2,3) and e['root_index']==i and e['protocol_id']==protocol['planner_protocol_id'],'exported root index/horizon/protocol')
            require(raw.shape==(5*h,2) and raw.dtype.kind in 'fi' and np.isfinite(raw).all() and np.all(np.abs(raw)<=1),'all selected raw commands')
            require(e['input_sha256']==p['input_sha256'] and all(same(e[k],p[k]) for k in ('root_context','goal','projection')),'exported root/context/goal/Q')
            require(e['report_sha256']==protocol['planning_report_sha256'],'exported actual report SHA')
            require(planning.digest({k:v for k,v in e.items() if k!='selection_sha256'})==e['selection_sha256'],'exporter selection SHA')
            plan=dict(schema=NAME+'.plan',root_index=i,root_sha256=r,method=mode,
                planned_horizon_tokens=h,planned_commands=5*h,corrected_search_horizon=3,commands=raw.tolist(),
                input_sha256=p['input_sha256'],predictor_context=p['root_context'],goal=p['goal'],projection=p['projection'],
                selected_predicted_context=e['selected_predicted_context'],predicted_Q_cost=e['predicted_Q_cost'],
                planning_report_sha256=e['report_sha256'],planner_protocol_id=e['protocol_id'],export_selection_sha256=e['selection_sha256'],
                goal_vector_sha256=planning.digest(p['goal']),projection_sha256=planning.digest(p['projection']))
            plan['selection_sha256']=planning.digest(plan);plans.append(plan)
    return plans


def bind_report(prepared,expected_sha,planning_output,report_sha,output):
    protocol=validate_prepared(prepared,expected_sha);require(protocol['stage']=='source_prepared_awaiting_report','bind only original source preparation')
    output=Path(output).absolute();planning_output=Path(planning_output).absolute();require(not output.exists(),'exclusive NEW report binding output')
    require(durable.digest(planning_output/'report.json')==report_sha,'explicit actual planner report SHA')
    bound=deepcopy(protocol);bound.update(stage='report_bound_ready',source_prepared=str(Path(prepared).absolute()),source_prepare_sha256=expected_sha,
        planning_output=str(planning_output),planning_report_sha256=report_sha)
    bound['plans']=plans_from_exports(bound,planning.selected_actions(planning_output))
    for record in io.decode((planning_output/'report.json').read_bytes())['roots']:
        bound['inputs_sha256'][record['journal_path']]=record['journal_sha256']
    for name in ('report.json','execution_protocol.json'):bound['inputs_sha256'][str(planning_output/name)]=durable.digest(planning_output/name)
    bound['planned_selected_path_steps']=sum(p['planned_commands'] for p in bound['plans'])
    output.mkdir(exist_ok=False);write(output/'protocol.json',bound)
    receipt=io.decode((Path(prepared)/'prepare.json').read_bytes())
    receipt.update(status='report_bound_ready',protocol_sha256=durable.digest(output/'protocol.json'),plans=8,
        planned_selected_path_steps=bound['planned_selected_path_steps'],export_numpy_svd_calls=8,planning_report_sha256=report_sha)
    write(output/'prepare.json',receipt)
    validate_prepared(output,durable.digest(output/'prepare.json'));return receipt


def validate_plan(plan):
    h=plan['planned_horizon_tokens'];raw=np.asarray(plan['commands'])
    require(plan['schema']==NAME+'.plan' and type(h) is int and h in (2,3) and plan['planned_commands']==5*h and
        raw.shape==(5*h,2) and raw.dtype.kind in 'fi' and np.isfinite(raw).all() and np.all(np.abs(raw)<=1),'exact mixed-H2/H3 plan before environment access')
    require(planning.digest({k:v for k,v in plan.items() if k!='selection_sha256'})==plan['selection_sha256'],'physical selection content SHA')


def validate_reset_history(history,expected):
    """Match original float32 constant-five-step warmup normalization exactly."""
    raw=np.asarray(history)
    require(raw.shape==(3,2) and raw.dtype.kind in 'fi' and np.isfinite(raw).all() and np.all(np.abs(raw)<=1),'full bounded raw warmup history')
    mean,std=(np.asarray(NORMALIZATION[k],dtype=np.float32) for k in ('mean','std'))
    packet=np.repeat(raw.astype(np.float32)[:,None,:],5,axis=1).reshape(3,10)
    tokens=(packet-np.tile(mean,5))/np.tile(std,5)
    require(np.array_equal(tokens[1:],np.asarray(expected)),'normalized warmup between-state tokens differ from predictor context')
    return dict(raw_history=raw.tolist(),normalized_between_history=tokens[1:].tolist())


def validate_roots(roots,refs,fingerprint):
    require(isinstance(roots,list) and len(roots)==4,'all four actual physical roots upfront')
    result={}
    for root in roots:
        require(set(root)=={'state','goal'},'complete physical root state/goal')
        for field in ('state','goal'):
            a=np.asarray(root[field]);require(a.shape==(7,) and a.dtype.kind in 'fi' and np.isfinite(a).all(),'finite original seven-state/goal')
        key=fingerprint(root);require(key in refs and key not in result,'upfront physical root/goal identity')
        result[key]=deepcopy(root)
    require(set(result)==set(refs),'same four root/goal census');return result


def physical_branch(*,plan,output,reset,env,state_of,evaluate):
    validate_plan(plan)
    row=durable.physical_branch(plan=plan,repetition=0,output=Path(output),reset=reset,env=env,state_of=state_of,evaluate=evaluate)
    journal=Path(row['journal_path'])
    events=[io.decode(line) for line in journal.read_bytes().splitlines()] if journal.is_file() else []
    images=[e for e in events if 'rgb_path' in e]
    row['endpoint_fresh']=bool(images and row.get('terminal_rgb_path') and
        images[-1].get('primitive_step',0)==row['physical_env_step_returns'] and
        row['physical_env_step_attempts']==row['physical_env_step_returns'] and
        row['status'] in ('completed','observation_unavailable','environment_terminated'))
    row['endpoint_scope']=f'full_H{plan["planned_horizon_tokens"]}' if row['full_horizon_completed'] else (
        f'partial_H{plan["planned_horizon_tokens"]}' if row['endpoint_fresh'] else 'unknown_or_stale_endpoint')
    write(Path(output)/f'root_{plan["root_index"]}_{plan["method"]}_replay0.endpoint.json',dict(
        root_sha256=plan['root_sha256'],method=plan['method'],selection_sha256=plan['selection_sha256'],
        endpoint_fresh=row['endpoint_fresh'],endpoint_scope=row['endpoint_scope']))
    return row


def verify_branch(plan,row):
    validate_plan(plan)
    require(all(row[k]==plan[k] for k in ('root_sha256','method','selection_sha256','planned_horizon_tokens','planned_commands')),'physical root/selection/horizon identity')
    attempted,returned=(io.count(row[k]) for k in ('physical_env_step_attempts','physical_env_step_returns'))
    require(returned<=attempted<=plan['planned_commands'] and returned==row['completed_commands'] and attempted==row['attempted_commands'],'attempt/return counters')
    require(durable.digest(row['journal_path'])==row['journal_sha256'],'physical journal SHA')
    events=[io.decode(line) for line in Path(row['journal_path']).read_bytes().splitlines()]
    attempts=[e for e in events if e['kind']=='attempt'];returns=[e for e in events if e['kind']=='step_returned'];obs=[e for e in events if e['kind']=='observation']
    require(len(attempts)==attempted and len(returns)==returned and [e['primitive_step'] for e in attempts]==list(range(1,attempted+1)) and
        [e['primitive_step'] for e in returns]==list(range(1,returned+1)),'durable step census/order')
    require(all(same(e['command'],plan['commands'][i]) for i,e in enumerate(attempts)),'all attempted commands match selected path')
    require(len(obs)==len(row['trace']) and all(all(same(e.get(k),v) for k,v in t.items()) for e,t in zip(obs,row['trace'])),'durable observation trace')
    for e in events:
        if 'rgb_path' in e:require(durable.digest(e['rgb_path'])==e['rgb_file_sha256'],'physical RGB SHA')
    if row['full_horizon_completed']:require(row['status']=='completed' and returned==plan['planned_commands'] and row['endpoint_fresh'],'complete actual selected horizon')
    return True


def materialize_missing(plans,rows,output):
    seen={(r['root_sha256'],r['method']) for r in rows}
    for p in plans:
        if (p['root_sha256'],p['method']) in seen:continue
        row={k:deepcopy(p[k]) for k in ('root_index','root_sha256','method','selection_sha256','planned_commands','planned_horizon_tokens')}
        row.update(status='blocked',failure='setup/earlier integrity failure; no commands dispatched',failures=[],
            repetition=0,trace=[],attempted_commands=0,completed_commands=0,physical_env_step_attempts=0,physical_env_step_returns=0,
            full_horizon_completed=False,endpoint_fresh=False,endpoint_scope='unknown_or_stale_endpoint',terminal_rgb_path=None)
        write(Path(output)/f'root_{p["root_index"]}_{p["method"]}_replay0.physical.json',row);rows.append(row)


def comparisons(plans,rows):
    index=io.index(rows,('root_sha256','method'));expected={(p['root_sha256'],p['method']) for p in plans}
    require(set(index)==expected and len(expected)==8,'eight physical rootarms required')
    for p in plans:
        r=index[p['root_sha256'],p['method']]
        require(all(r[k]==p[k] for k in ('selection_sha256','planned_horizon_tokens','planned_commands')),'paired selection/horizon binding')
    pairs=[]
    for root in dict.fromkeys(p['root_sha256'] for p in plans):
        u,d=(index[root,m] for m in METHODS)
        identical='context_identity' in u and 'context_identity' in d and same(u['context_identity'],d['context_identity'])
        full=identical and all(r['full_horizon_completed'] and r['endpoint_fresh'] for r in (u,d))
        horizons=[u['planned_horizon_tokens'],d['planned_horizon_tokens']];same_h=horizons[0]==horizons[1]
        partial=identical and all(r['endpoint_fresh'] for r in (u,d)) and u['physical_env_step_returns']==d['physical_env_step_returns']
        classification=(f'full_H{horizons[0]}_pair' if same_h else 'full_mixed_H2_H3') if full else ('partial_same_steps' if partial else 'partial_or_unmatched')
        pairs.append(dict(root_sha256=root,classification=classification,identical_initial_context=identical,
            selected_horizons=horizons,same_selected_horizon=same_h,actual_steps=[u['physical_env_step_returns'],d['physical_env_step_returns']],
            dynamic_minus_uniform_pose_cost=(d['pose']['pose_cost']-u['pose']['pose_cost']) if full and same_h and all('pose' in r for r in (u,d)) else None,
            scope='descriptive known-root endpoint only; mixed/partial not pooled with same-horizon full paths',P1_admission=False,P3_admission=False))
    return pairs


def optional_encodings(plans,rows,output,setup):
    output=Path(output);require((output/'physical_report.json').is_file(),'durable physical report before CUDA setup')
    plans=io.index(plans,('root_sha256','method'));receipts=[];encode=None;setup_error=None
    with (output/'encoding.events.jsonl').open('x') as stream:
        def event(e):stream.write(io.canonical(e)+'\n');stream.flush();os.fsync(stream.fileno())
        if any(r['endpoint_fresh'] for r in rows):
            event(dict(kind='encoding_setup',completed=False))
            try:encode=setup();event(dict(kind='encoding_setup',completed=True))
            except (Exception,KeyboardInterrupt) as exc:setup_error=durable.failure(exc,'encoding_setup');event(setup_error)
        for r in rows:
            p=plans[r['root_sha256'],r['method']]
            receipt=dict(root_sha256=r['root_sha256'],method=r['method'],selection_sha256=p['selection_sha256'],
                planned_horizon_tokens=p['planned_horizon_tokens'],endpoint_scope=r['endpoint_scope'],
                attempted_encode_samples=0,status='blocked_stale_endpoint',failure=None)
            try:
                if r['endpoint_fresh']:
                    if setup_error is not None:receipt.update(status='encoding_setup_failed',failure=setup_error)
                    else:
                        path=Path(r['terminal_rgb_path']);require(durable.digest(path)==r['terminal_rgb_sha256'],'terminal image SHA')
                        receipt.update(attempted_encode_samples=1,terminal_rgb_sha256=r['terminal_rgb_sha256'])
                        event(dict(kind='endpoint_encode',completed=False,**receipt))
                        vector=planning.prior.array(encode(path),(192,))
                        cost=float(np.mean((np.asarray(p['projection'])@(vector-np.asarray(p['goal'])))**2))
                        receipt.update(status='completed',terminal_vector=vector.tolist(),original_Q_cost=cost,
                            goal_vector_sha256=p['goal_vector_sha256'],projection_sha256=p['projection_sha256'])
            except (Exception,KeyboardInterrupt) as exc:receipt.update(status='encoding_failed',failure=durable.failure(exc,'endpoint_encode'))
            event(dict(kind='endpoint_result',**receipt))
            write(output/f'root_{r["root_index"]}_{r["method"]}_replay0.encoding.json',receipt);receipts.append(receipt)
    return receipts


def execute_physics(prepared,expected_sha,output,encode_terminals=False):
    protocol=validate_prepared(prepared,expected_sha);require(protocol['stage']=='report_bound_ready','bind real completed planner report first')
    plans=protocol['plans'];output=Path(output).absolute();require(not output.exists(),'exclusive NEW physical execution output')
    output.mkdir(exist_ok=False);write(output/'execution_protocol.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected_sha,
        planning_report_sha256=protocol['planning_report_sha256'],encode_terminals=encode_terminals,new_predict_calls=0))
    rows=[];failures=[];encodings=[];env=None;model=None;before=None;host=None;checked=None
    stats=dict(encode_calls=0,encode_samples=0,failed_encode_calls=0,forbidden_predict_attempts=0)
    try:
        from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
        from experiments.spectral_reachability.run_pusht_sampling_comparison_20261027_v2 import verify_rgb_context,official_checked
        env=base.gym.make('swm/PushT-v1',render_mode='rgb_array')
        evaluator=Path(env.unwrapped.eval_state.__code__.co_filename).resolve()
        require(str(evaluator) in protocol['sources_sha256'] and durable.digest(evaluator)==protocol['sources_sha256'][str(evaluator)],'sealed installed evaluator')
        require(tuple(env.unwrapped.variation_space['agent']['color'].value.tolist())==(65,105,225),'RoyalBlue guard contract')
        roots=base.collect_nonterminal_boundary_contact_roots(env,count=4,seed=int(protocol['root_seed']),oversample_factor=protocol['root_oversample_factor'])
        rootmap=validate_roots(roots,protocol['references'],base.root_fingerprint)
        require(all(np.array_equal(np.asarray(getattr(base.p0,'PUBLIC_ACTION_'+k.upper())),NORMALIZATION[k]) for k in ('mean','std')),'original public normalization constants')
        identities={}
        for plan in plans:
            root=rootmap[plan['root_sha256']]
            def reset():
                require(base.root_fingerprint(root)==plan['root_sha256'],'root fingerprint changed')
                frames,history,goal=base.physical._reset_and_warmup(env,root,history_len=3)
                identity=verify_rgb_context(frames,goal,protocol['references'][plan['root_sha256']])
                identity.update(validate_reset_history(history,plan['predictor_context']['action_history']),
                    state=base.p0.packets.state_of(env).tolist(),goal=np.asarray(root['goal']).tolist())
                if plan['root_sha256'] in identities:require(same(identity,identities[plan['root_sha256']]),'paired physical reset/context mismatch')
                identities[plan['root_sha256']]=deepcopy(identity);return identity
            def evaluate(state):return dict(official=official_checked(env,state,root['goal']),pose=base.physical._pose_label(state,root['goal']))
            row=physical_branch(plan=plan,output=output,reset=reset,env=env,state_of=base.p0.packets.state_of,evaluate=evaluate)
            rows.append(row);verify_branch(plan,row)
    except (Exception,KeyboardInterrupt) as exc:failures.append(durable.failure(exc,'physical_setup_or_branch'))
    finally:
        if env is not None:
            try:env.close()
            except Exception as exc:failures.append(durable.failure(exc,'env_close'))
    materialize_missing(plans,rows,output)
    physical=dict(schema=NAME+'.physical_report',rows=rows,failures=deepcopy(failures),
        planning_report_sha256=protocol['planning_report_sha256'],planner_protocol_id=protocol['planner_protocol_id'],
        consumer_prepare_sha256=expected_sha,contract=contract(),new_predict_calls=0,P1_admission=False,P3_admission=False)
    write(output/'physical_report.json',physical)
    physical_sha=durable.digest(output/'physical_report.json')
    if encode_terminals:
        def setup():
            nonlocal model,before,host,checked
            import torch
            from PIL import Image
            from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
            from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
            from experiments.checkpoint_compat import load_compatible_pretrained
            from experiments.gpu_preflight import require_cuda_allocation
            require(durable.digest(output/'physical_report.json')==physical_sha,'durable physical report changed before neural setup')
            for p,h in protocol['model_bindings'].items():require(checked.digest(p)==h,'endpoint model bytes changed')
            policy=Path(next(p for p in protocol['model_bindings'] if Path(p).name=='weights.pt')).parent
            host=require_cuda_allocation('cuda');device=torch.device('cuda:0')
            model=load_compatible_pretrained(str(policy),device=device).eval().requires_grad_(False);before=checked.tensor_state_hash(model)
            def forbidden(*a,**kw):
                stats['forbidden_predict_attempts']+=1;raise RuntimeError('no prediction in physical encoding consumer')
            model.predict=forbidden;transform=p0.image_transform()
            def encode(path):
                require(stats['encode_calls']<8,'eight endpoint cap before neural dispatch')
                with Image.open(path) as image:rgb=np.asarray(image.convert('RGB')).copy()
                tensor=p0._image_tensor(rgb,transform=transform,device=device)
                require(tensor.device.type=='cuda' and tensor.shape[0]==1,'single sample HOST CUDA endpoint only')
                stats['encode_calls']+=1;stats['encode_samples']+=1
                try:
                    with torch.no_grad():vector=model.encode({'pixels':tensor})['emb'][0,0]
                    require(vector.device.type=='cuda' and bool(torch.isfinite(vector).all()),'finite CUDA endpoint vector')
                    torch.cuda.synchronize(device);return vector.cpu().tolist()
                except BaseException:stats['failed_encode_calls']+=1;raise
            return encode
        encodings=optional_encodings(plans,rows,output,setup)
    source_ok=True;model_ok=True
    try:validate_prepared(prepared,expected_sha)
    except Exception as exc:source_ok=False;failures.append(durable.failure(exc,'final_source_verification'))
    if model is not None:
        try:model_ok=before==checked.tensor_state_hash(model)
        except Exception as exc:model_ok=False;failures.append(durable.failure(exc,'final_model_check'))
    pairs=comparisons(plans,rows)
    integrity=not failures and source_ok and model_ok and stats['forbidden_predict_attempts']==0 and all(
        r['status'] in ('completed','observation_unavailable','environment_terminated') and 'pose' in r and 'official' in r for r in rows)
    encoding_ok=not encode_terminals or len(encodings)==8 and all(e['status']=='completed' for e in encodings)
    report=dict(schema=NAME+'.report',status='guarded_calibration_recorded' if integrity and encoding_ok else 'incomplete',
        rows=rows,pairs=pairs,encodings=encodings,failures=failures,source_verified=source_ok,model_unchanged=model_ok,
        optional_encodings_complete=encoding_ok,encoding_stats=stats,host=host,physical_report_sha256=physical_sha,
        planning_report_sha256=protocol['planning_report_sha256'],planner_protocol_id=protocol['planner_protocol_id'],consumer_prepare_sha256=expected_sha,
        physical_env_step_attempts=sum(r['physical_env_step_attempts'] for r in rows),
        physical_env_step_returns=sum(r['physical_env_step_returns'] for r in rows),
        planned_selected_path_steps=protocol['planned_selected_path_steps'],
        attempted_terminal_encode_samples=sum(e['attempted_encode_samples'] for e in encodings),
        planner_cost_scope='upstream actual report, shared charged once; no prediction or AD repeated by consumer',
        new_predict_calls=0,new_jacobian_materializations=0,new_independent_roots=0,
        P1_admission=False,P3_admission=False,efficacy_claim=False,total_flops_equivalent=False)
    write(output/'report.json',report)
    if not integrity or not encoding_ok:raise RuntimeError('incomplete physical/encoding; retain all partials, no automatic retry')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);action=parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare','bind','verify','execute-physics'):action.add_argument('--'+name,action='store_true')
    parser.add_argument('--project-root',type=Path,default=PROJECT);parser.add_argument('--output',type=Path)
    parser.add_argument('--prepared',type=Path);parser.add_argument('--prepare-sha256')
    parser.add_argument('--planning-output',type=Path);parser.add_argument('--report-sha256');parser.add_argument('--encode-terminals',action='store_true')
    args=parser.parse_args();require(args.execute_physics or not args.encode_terminals,'encoding only explicit physics execution')
    if args.prepare:
        require(args.output is not None,'new output');result=prepare(args.project_root,args.output)
    else:
        require(args.prepared is not None and args.prepare_sha256 is not None,'prepared path and receipt SHA')
        if args.verify:
            p=validate_prepared(args.prepared,args.prepare_sha256);result=dict(status='verified',stage=p['stage'])
        elif args.bind:
            require(args.output is not None and args.planning_output is not None and args.report_sha256 is not None,'new binding output and explicit actual report SHA')
            result=bind_report(args.prepared,args.prepare_sha256,args.planning_output,args.report_sha256,args.output)
        else:
            require(args.output is not None,'new physical output');result=execute_physics(args.prepared,args.prepare_sha256,args.output,args.encode_terminals)
    print(io.canonical(result))


if __name__=='__main__':main()
