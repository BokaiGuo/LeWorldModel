"""Independent v3 physical consumer of actual action-Jacobian H3 CUDA plans.

Never consumes v2 reports/endpoints. Shares only immutable low-level durable
execution utilities. Inherited Jacobian acquisition cost remains unknown.

Physics execution and optional CUDA-only endpoint encoding require explicit
--execute-physics and a reviewed prepare SHA. No search or formula changes.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p3_action_jacobian_budget_schedule_20261027_v3 as planning
from experiments.spectral_reachability import p3_matched_physical_20261027_v1 as durable
from experiments.spectral_reachability import p3_quality_cost_audit_20261027_v2 as io
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal,verify_seal

NAME='p3_action_jacobian_physical_consumer_20261027_v3'
REPORT='results/p3_action_jacobian_budget_schedule_20261027_v3_cuda_main_v1/report.json'
REPORT_SHA='c13dab20228fd590c3f9375e29b6be905a8db083942a547c862d97c90cf53921'
PREPARED='results/p3_action_jacobian_budget_schedule_20261027_v3_prepared'
PLANNING_PREPARE_SHA='27988bdb563563bd2cb0ea5ef3f69851627a2cae4d648e1e545f2e68c7d5c134'
METHODS=planning.METHODS
write=planning.write
require=planning.require


def same(a,b):
    return io.canonical(a)==io.canonical(b)


def plans_from_saved(body,report,journals=None):
    """Rebuild scheduler choices from SAVED contexts, not neural prediction.

    Callback returns only a bound recorded child; scheduler independently
    rechecks parent order, raw actions, metric, goal-related SVD, allocation and
    H3 winner. This does NOT validate learned transitions against real physics.
    """
    require(report.get('schema')==planning.NAME+'.report','actual v3 planning report schema required')
    require(same(report['inherited_cost'],planning.inherited_cost()) and
            report['inherited_arrays_charged_once']==4 and report['new_jacobian_materializations']==0 and
            report['total_pipeline_cost_equivalent'] is False,'v3 inherited unknown cost/scope')
    require(same(body['inherited_cost'],planning.inherited_cost()) and body['total_pipeline_cost_equivalent'] is False,
            'v3 protocol inherited cost/scope')
    require(report['status']=='calibration_recorded' and report['failure'] is None and
            report['source_failure'] is None and report['model_unchanged'] is True and
            report['unknown_attempted_work'] is False and report['known_predict_samples']==1176 and
            report['new_independent_roots']==report['new_physical_steps']==0 and
            report['P1_admission'] is False and report['P3_admission'] is False,'complete fixed-H3 CUDA receipt required')
    identity=io.sha(io.canonical(body).encode())
    require(report['protocol_id']==identity,'planning protocol identity')
    require(same(body['contract'],planning.contract()),'frozen scheduling formula/contract changed')
    roots=body['roots'];expected={(r,m) for r in roots for m in METHODS}
    rows=io.index(report['rows'],('root_sha256','mode'))
    require(len(roots)==len(set(roots))==4 and set(rows)==expected,'exact eight root-arm census')
    packets=io.index(body['packets'],('root_sha256',))
    require(set(packets)=={(r,) for r in roots},'packet root census')
    plans=[]
    model=next(h for p,h in body['model_bindings'].items() if Path(p).name=='weights.pt')+':'+next(
        h for p,h in body['model_bindings'].items() if Path(p).name=='config.json')
    for index,root in enumerate(roots):
        packet=packets[root,]
        require(io.sha(io.canonical({k:v for k,v in packet.items() if k!='input_sha256'}).encode())==
                packet['input_sha256'],'packet content SHA')
        for mode in body['arm_order'][index]:
            row=rows[root,mode];receipt=row['prediction_receipt'];method=planning.NAME+'_'+mode
            require(same(row['inherited_cost'],planning.inherited_cost()) and
                    row['total_pipeline_cost_equivalent'] is False and io.count(row['new_jacobian_materializations'])==0,
                    'v3 arm inherited unknown cost and no new derivatives')
            binding=dict(protocol_id=identity,root_id=root,frozen_model_id=model,
                         task_id='pusht_fixed_h3_cached_latent_calibration')
            require(row['status']=='completed' and row['failure'] is None and row['whole_h3_replay_exact'] is True and
                    row['input_sha256']==packet['input_sha256'] and same(row['binding'],binding) and
                    same(receipt['binding'],binding) and receipt['method']==method,'arm/context/task/model binding')
            for k in ('attempted_predict_samples','predict_sample_limit'):
                require(io.count(receipt[k])==147,'147 actual predict samples required')
            require(io.count(receipt['blocked_calls'])==0,'no blocked predictions in successful plan')
            phases=io.index(receipt['phases'],('phase',))
            require(set(phases)=={('planning',),('evaluation',)},'planning/verification census')
            for name,n in (('planning',144),('evaluation',3)):
                phase=phases[name,]
                require(phase['completed'] is True and io.count(phase['failed_predict_calls'])==0 and
                    io.count(phase['predict_samples'])==io.count(phase['predict_calls'])==n and
                    phase['method']==method and type(phase['horizon_tokens']) is int and phase['horizon_tokens']==3 and
                    same(phase['context_lengths'],{'3':n}),'144+3 B1/T3 predict phase receipt')
            for k,n in (('action_encoder_calls',147),('action_encoder_samples',147),
                        ('processed_action_context_tokens',441),('visual_encoder_samples',0),
                        ('runtime_action_backward_calls',0),('runtime_action_vjp_rows',0),
                        ('runtime_input_backward_calls',0),('runtime_input_vjp_rows',0),('new_physical_steps',0)):
                require(io.count(row[k])==n,'runtime resource coordinate: '+k)
            search=row['search'];best=search['best_plan'];path=best['primitive_indices']
            require(type(best['depth']) is int and best['depth']==3 and len(path)==3 and
                    all(type(i) is int and 0<=i<16 for i in path),'three exact integer bank indices required')
            raw=sum([packet['primitive_bank'][i] for i in path],[])
            require(same(raw,best['actions']) and np.asarray(raw).shape==(15,2),'exact raw-bank H3 path reconstruction')
            events=search['events'];cursor=0;contexts={():packet['root_context']};journal=[]
            def recorded_child(context,action):
                nonlocal cursor
                require(cursor<len(events),'missing saved transition receipt')
                e=events[cursor];cursor+=1
                parent=tuple(e['parent_path']);i=e['primitive_index']
                require(parent in contexts and type(i) is int and 0<=i<16 and
                    same(context,contexts[parent]) and np.array_equal(action,np.asarray(packet['primitive_bank'][i])) and
                    e['succeeded'] is True and e['replay_completed'] is True,'saved parent/bank/transition binding')
                child_path=parent+(i,)
                require(child_path not in contexts,'duplicate saved path')
                contexts[child_path]=e['context']
                return deepcopy(e['context'])
            rebuilt=planning.run_search(p=packet,replay=recorded_child,mode=mode,event_sink=journal.append)
            require(cursor==144==len(events) and same(rebuilt,search),'saved-event geometry/schedule/winner recomputation differs')
            if journals is not None:
                require(same(journal,journals[root,mode]),'durable planning journal differs from full saved event order')
            plan=dict(schema=NAME+'.plan',planning_report_sha256=REPORT_SHA,planning_protocol_id=identity,
                inherited_cost=planning.inherited_cost(),inherited_source=deepcopy(packet['inherited_source']),
                root_index=index,root_sha256=root,method=mode,primitive_indices=path,
                planned_horizon_tokens=3,planned_commands=15,corrected_search_horizon=3,commands=raw,
                input_sha256=packet['input_sha256'],prediction_receipt=deepcopy(receipt),
                planning_predict_samples=144,verification_predict_samples=3,upstream_predict_samples=147,
                selected_predicted_context=best['context'],selected_model_cost=best['cost'],
                allocation=search['allocation'],planning_journal_sha256=row['journal_sha256'])
            plan['selection_sha256']=io.sha(io.canonical(plan).encode())
            plans.append(plan)
    # Do not trust the old paired classifier without reconstructing it.
    require(same(planning.compare(report['rows'],roots),report['pairs']) and
            all(p['classification']=='matched_full_H3' for p in report['pairs']),'same-input/action-Jacobian/feedback paired receipt')
    return plans


def source_closure(root,seeds):
    """Read Python ASTs only, never import neural/simulator modules to seal them."""
    packages={'experiments':root/'experiments',
              'stable_worldmodel':root/'.venv/lib/python3.12/site-packages/stable_worldmodel',
              'stable_pretraining':root/'.venv/lib/python3.12/site-packages/stable_pretraining'}
    def locate(name):
        parts=name.split('.')
        if parts[0] not in packages:return None
        p=packages[parts[0]].joinpath(*parts[1:])
        for candidate in (p.with_suffix('.py'),p/'__init__.py'):
            if candidate.is_file():return candidate
        return None
    pending=list(seeds);seen=set()
    while pending:
        path=Path(pending.pop()).absolute()
        if path in seen:continue
        seen.add(path)
        if path.suffix!='.py':continue
        package=None
        for name,directory in packages.items():
            if path.is_relative_to(directory):
                tail=path.relative_to(directory).with_suffix('').parts
                package=[name,*tail[:-1]]
                break
        for node in ast.walk(ast.parse(path.read_text())):
            names=[]
            if isinstance(node,ast.Import):names=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                base=node.module or ''
                if node.level:
                    if package is None:continue
                    base='.'.join(package[:len(package)-node.level+1]+([base] if base else []))
                names=[base,*[base+'.'+a.name for a in node.names if a.name!='*']]
            for name in names:
                candidate=locate(name)
                if candidate is not None and candidate not in seen:pending.append(candidate)
    return sorted(seen)


def prepare(root,output):
    root=Path(root).resolve();output=Path(output).absolute()
    require(not output.exists(),'exclusive NEW prepare output required')
    body,identity=planning.validate_prepared(root/PREPARED,PLANNING_PREPARE_SHA)
    evidence=io.Evidence(root)
    for path,h in body['inputs_sha256'].items():evidence.raw(Path(path),h)
    for name in ('prepare.json','protocol.json'):
        evidence.raw(root/PREPARED/name,PLANNING_PREPARE_SHA if name=='prepare.json' else None)
    original_receipt=evidence.json(root/PREPARED/'prepare.json')
    for seal in original_receipt['source_seals']:evidence.seal(seal)
    report=evidence.json(root/REPORT,REPORT_SHA)
    run_dir=(root/REPORT).parent
    execution=evidence.json(run_dir/'execution_protocol.json')
    require(execution==dict(protocol_id=identity,prepare_sha256=PLANNING_PREPARE_SHA),'CUDA execution prepare binding')
    journals={}
    for row in report['rows']:
        path=run_dir/(row['root_sha256']+'_'+row['mode']+'.json')
        require(same(evidence.json(path),row),'per-arm JSON differs from final report')
        journal=run_dir/(row['root_sha256']+'_'+row['mode']+'.events.jsonl')
        require(str(journal)==row['journal_path'],'planning journal path identity')
        journals[row['root_sha256'],row['mode']]=[io.decode(line) for line in evidence.raw(journal,row['journal_sha256']).splitlines()]
    plans=plans_from_saved(body,report,journals)
    # Reconstruct only physical source references, never use OLD endpoints.
    intervention_path=root/'results/pusht_d2p_goal_agent_intervention4_v2_20261026_cuda/report.json'
    intervention=evidence.json(intervention_path,body['inputs_sha256'][str(intervention_path)])
    bank_path=Path(intervention['protocol']['bank_path'])
    bank=evidence.json(bank_path,intervention['protocol']['bank_sha256'])
    upstream_path=Path(bank['upstream_path'])
    upstream=evidence.json(upstream_path,bank['upstream_sha256'])
    require(io.sha(upstream_path.read_bytes())==intervention['protocol']['upstream_sha256'],'physical root provenance')
    references=io.index(upstream['rows'],('root_sha256',))
    require(set(references)=={(r,) for r in body['roots']},'physical source reference root census')
    module_root=root/'experiments/spectral_reachability'
    seeds=[*map(Path,body['sources_sha256']),Path(__file__).resolve(),Path(durable.__file__).resolve(),
        root/'tests'/('test_'+NAME+'.py'),root/'docs'/(NAME+'.md'),
        module_root/'audit_pusht_d2p_goal_agent_intervention.py',
        module_root/'run_pusht_sampling_comparison_20261027_v2.py',
        root/'.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py',
        root/'.venv/lib/python3.12/site-packages/stable_worldmodel/envs/__init__.py']
    sources=source_closure(root,seeds)
    for path in sources:evidence.raw(path)
    protocol=dict(schema=NAME+'.protocol',planning_report=str(root/REPORT),planning_report_sha256=REPORT_SHA,
        planning_prepare_sha256=PLANNING_PREPARE_SHA,planning_protocol_id=identity,roots=body['roots'],plans=plans,
        root_seed=upstream['protocol']['root_seed'],root_oversample_factor=4,
        references={r:{k:references[r,][k] for k in ('root_sha256','history_rgb_sha256','goal_rgb_sha256')}
                    for r in body['roots']},model_bindings=body['model_bindings'],
        inherited_cost=planning.inherited_cost(),inherited_arrays_charged_once=4,total_pipeline_cost_equivalent=False,
        upstream_predict_samples_charged_once=1176,new_jacobian_materializations=0,
        prepare_numpy_svd_calls=8,
        physical_repetitions=1,planned_selected_path_steps=120,planned_commands_per_branch=15,
        feedback='same frozen open-loop H3 commands; fresh RoyalBlue RGB presence guard every raw step',
        optional_encoding='at most one actual fresh endpoint per arm, 8 total; CUDA only, after durable physical report',
        inputs_sha256=dict(sorted(evidence.hashes.items())),sources_sha256={str(p):evidence.hashes[str(p)] for p in sources},
        new_predict_samples=0,new_independent_roots=0,P1_admission=False,P3_admission=False)
    evidence.recheck()
    output.mkdir(exist_ok=False)
    write(output/'protocol.json',protocol)
    protocol_sha=durable.digest(output/'protocol.json')
    seals=[create_seal(files=sources[i:i+40],args_json=io.canonical(dict(
        protocol_path=str(output/'protocol.json'),protocol_sha256=protocol_sha)),output_dir=output/f'source_seal_{i//40}')
        for i in range(0,len(sources),40)]
    evidence.recheck()
    receipt=dict(status='prepared_not_executed',protocol_path=str(output/'protocol.json'),protocol_sha256=protocol_sha,
        source_seals=seals,source_count=len(sources),plans=8,planned_selected_path_steps=120,
        new_predict_samples=0,new_physical_steps=0,model_loaded=False,calibration_roots=4,new_independent_roots=0,
        P3_admission=False,planning_report_sha256=REPORT_SHA,
        inherited_cost=planning.inherited_cost(),inherited_arrays_charged_once=4,total_pipeline_cost_equivalent=False,
        prepare_numpy_svd_calls=8,new_jacobian_materializations=0,
        external_model_bindings_not_rehashed=body['model_bindings'])
    write(output/'prepare.json',receipt)
    print(io.canonical(receipt))
    return receipt


def validate_prepared(prepared,expected_sha):
    prepared=Path(prepared).absolute();raw=(prepared/'prepare.json').read_bytes()
    require(io.sha(raw)==expected_sha,'prepare receipt SHA256 mismatch')
    receipt=io.decode(raw);path=prepared/'protocol.json'
    require(receipt['status']=='prepared_not_executed' and receipt['model_loaded'] is False and
            receipt['protocol_path']==str(path) and durable.digest(path)==receipt['protocol_sha256'],'protocol hash/path binding')
    protocol=io.decode(path.read_bytes())
    require(protocol['schema']==NAME+'.protocol' and protocol['planning_report_sha256']==REPORT_SHA and
            protocol['planned_commands_per_branch']==15 and protocol['physical_repetitions']==1,'fixed H3 physical contract')
    for p,h in protocol['inputs_sha256'].items():
        require(durable.digest(p)==h,'input/source hash changed: '+p)
    for seal in receipt['source_seals']:
        verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
    require(protocol['planning_prepare_sha256']==PLANNING_PREPARE_SHA and
            same(protocol['inherited_cost'],planning.inherited_cost()) and
            protocol['total_pipeline_cost_equivalent'] is False and protocol['inherited_arrays_charged_once']==4,
            'v3 inherited source/cost binding')
    expected={(r,m) for r in protocol['roots'] for m in METHODS}
    index=io.index(protocol['plans'],('root_sha256','method'))
    require(len(protocol['roots'])==4 and set(index)==expected,'all eight plans required')
    for plan in protocol['plans']:
        require(plan['schema']==NAME+'.plan' and plan['planning_report_sha256']==REPORT_SHA and
                plan['planning_protocol_id']==protocol['planning_protocol_id'] and
                same(plan['inherited_cost'],planning.inherited_cost()),'v3 plan source/cost identity')
        require(io.sha(io.canonical({k:v for k,v in plan.items() if k!='selection_sha256'}).encode())==
                plan['selection_sha256'],'selection hash changed')
    return protocol


def physical_branch(*,plan,output,reset,env,state_of,evaluate):
    require(plan.get('schema')==NAME+'.plan' and plan.get('planning_report_sha256')==REPORT_SHA,
            'v3 plan/report identity required before environment access')
    raw=np.asarray(plan['commands'])
    require(plan['planned_commands']==15 and plan['planned_horizon_tokens']==3 and raw.shape==(15,2) and
            raw.dtype.kind in 'fi' and np.isfinite(raw).all() and np.all(np.abs(raw)<=1),'exact fifteen-step H3 plan before environment access')
    row=durable.physical_branch(plan=plan,repetition=0,output=output,reset=reset,env=env,state_of=state_of,evaluate=evaluate)
    # Returned env.step count does not prove a fresh final render or known state
    # after an exception. Never encode a stale previous RGB as the actual endpoint.
    observations=[]
    if row.get('journal_path') and Path(row['journal_path']).is_file():
        observations=[io.decode(line) for line in Path(row['journal_path']).read_bytes().splitlines()]
    images=[e for e in observations if 'rgb_path' in e]
    row['endpoint_fresh']=bool(images and row.get('terminal_rgb_path') and
        images[-1].get('primitive_step',0)==row['physical_env_step_returns'] and
        row['physical_env_step_attempts']==row['physical_env_step_returns'] and
        row['status'] in ('completed','observation_unavailable','environment_terminated'))
    row['endpoint_scope']='full_H3' if row['full_horizon_completed'] else (
        'guard_partial_or_terminated' if row['endpoint_fresh'] else 'unknown_or_stale_endpoint')
    # durable's immutable branch JSON stays intact; supplemental annotation is new.
    write(Path(output)/f'root_{plan["root_index"]}_{plan["method"]}_replay0.endpoint.json',dict(
        schema=NAME+'.endpoint',planning_report_sha256=REPORT_SHA,
        root_sha256=plan['root_sha256'],method=plan['method'],endpoint_fresh=row['endpoint_fresh'],endpoint_scope=row['endpoint_scope']))
    return row


def verify_branch(plan,row):
    require(row['root_sha256']==plan['root_sha256'] and row['method']==plan['method'] and
            row['selection_sha256']==plan['selection_sha256'],'physical selection binding')
    attempted,returned=io.count(row['physical_env_step_attempts']),io.count(row['physical_env_step_returns'])
    require(returned<=attempted<=15 and returned==row['completed_commands'] and attempted==row['attempted_commands'],
            'physical attempt/return counters')
    journal_path=Path(row['journal_path'])
    require(durable.digest(journal_path)==row['journal_sha256'],'physical journal hash changed')
    events=[io.decode(line) for line in journal_path.read_bytes().splitlines()]
    attempts=[e for e in events if e['kind']=='attempt'];returns=[e for e in events if e['kind']=='step_returned']
    obs=[e for e in events if e['kind']=='observation']
    require(len(attempts)==attempted and len(returns)==returned,'durable attempt/return census')
    require([e['primitive_step'] for e in attempts]==list(range(1,attempted+1)) and
            [e['primitive_step'] for e in returns]==list(range(1,returned+1)),'durable command order')
    require(all(same(e['command'],plan['commands'][i]) for i,e in enumerate(attempts)),'attempted commands differ from selected raw path')
    require(len(obs)==len(row['trace']) and all(all(same(e.get(k),v) for k,v in t.items()) for e,t in zip(obs,row['trace'])),
            'observed physical trace differs from journal')
    for e in events:
        if 'rgb_path' in e:require(durable.digest(e['rgb_path'])==e['rgb_file_sha256'],'physical RGB hash changed')
    if row['full_horizon_completed']:
        require(row['status']=='completed' and returned==15 and row['endpoint_fresh'],'full H3 endpoint unavailable')
    return True


def materialize_missing(plans,rows,output):
    existing={(r['root_sha256'],r['method']) for r in rows}
    for plan in plans:
        if (plan['root_sha256'],plan['method']) in existing:continue
        row={k:deepcopy(plan[k]) for k in ('root_index','root_sha256','method','selection_sha256','planned_commands','planned_horizon_tokens')}
        row.update(status='blocked',failure='physical setup unavailable; no selected-path steps dispatched',failures=[],
            repetition=0,attempted_commands=0,completed_commands=0,trace=[],terminal_rgb_path=None,
            physical_env_step_attempts=0,physical_env_step_returns=0,full_horizon_completed=False,
            endpoint_fresh=False,endpoint_scope='unknown_or_stale_endpoint')
        write(Path(output)/f'root_{plan["root_index"]}_{plan["method"]}_replay0.physical.json',row)
        rows.append(row)


def optional_encodings(rows,output,setup):
    output=Path(output)
    require((output/'physical_report.json').is_file(),'durable physical report required before CUDA encoding')
    setup_error=None;encode=None
    if any(r.get('endpoint_fresh') for r in rows):
        try:encode=setup()
        except (Exception,KeyboardInterrupt) as exc:setup_error=durable.failure(exc,'encoding_setup')
    receipts=[]
    for row in rows:
        receipt=dict(root_sha256=row['root_sha256'],method=row['method'],repetition=0,
            endpoint_scope=row['endpoint_scope'],attempted_encode_samples=0,status='blocked_stale_endpoint',failure=None)
        try:
            if row.get('endpoint_fresh'):
                if setup_error is not None:
                    receipt.update(status='encoding_setup_failed',failure=setup_error)
                else:
                    path=Path(row['terminal_rgb_path'])
                    require(durable.digest(path)==row['terminal_rgb_sha256'],'terminal image hash changed')
                    receipt['attempted_encode_samples']=1
                    vector=encode(path)
                    array=np.asarray(vector)
                    require(array.shape==(192,) and array.dtype.kind in 'fi' and np.isfinite(array).all(),'finite 192D terminal encoding')
                    receipt.update(status='completed',terminal_vector=vector)
        except (Exception,KeyboardInterrupt) as exc:
            receipt.update(status='encoding_failed',failure=durable.failure(exc,'terminal_encoding'))
        write(output/f'root_{row["root_index"]}_{row["method"]}_replay0.encoding.json',receipt)
        receipts.append(receipt)
    return receipts


def comparisons(plans,rows):
    index=io.index(rows,('root_sha256','method'));pairs=[]
    for root in dict.fromkeys(p['root_sha256'] for p in plans):
        u,d=(index[root,m] for m in METHODS)
        same_context=('context_identity' in u and 'context_identity' in d and same(u['context_identity'],d['context_identity']))
        full=same_context and all(r['full_horizon_completed'] for r in (u,d))
        partial=(same_context and all(r.get('endpoint_fresh') for r in (u,d)) and
            u['physical_env_step_returns']==d['physical_env_step_returns'] and not full)
        actual_endpoints=all(r.get('endpoint_fresh') for r in (u,d))
        pairs.append(dict(root_sha256=root,classification='full-path' if full else 'guard-partial' if partial else 'unmatched',
            identical_initial_context=same_context,actual_steps=[u['physical_env_step_returns'],d['physical_env_step_returns']],
            dynamic_minus_uniform_pose_cost=(d['pose']['pose_cost']-u['pose']['pose_cost']) if actual_endpoints and 'pose' in u and 'pose' in d else None,
            delta_scope='descriptive actual endpoints; partial/unmatched are not full-H3 efficacy',P3_admission=False))
    return pairs


def execute_physics(*,prepared,prepare_sha256,output,encode_terminals=False):
    protocol=validate_prepared(prepared,prepare_sha256);plans=protocol['plans']
    output=Path(output).absolute();require(not output.exists(),'exclusive NEW physical output required')
    output.mkdir(exist_ok=False)
    write(output/'execution_protocol.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=prepare_sha256,
        encode_terminals=encode_terminals,physical_repetitions=1,new_predict_calls=0))
    rows=[];failures=[];encodings=[];env=None;model=None;model_before=None;host=None
    stats=dict(encode_calls=0,encode_samples=0,failed_encode_calls=0,forbidden_predict_attempts=0)
    base=None;checked=None
    try:
        # Imports and environment creation occur ONLY after explicit execution.
        from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
        from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
        from experiments.spectral_reachability.run_pusht_sampling_comparison_20261027_v2 import verify_rgb_context,official_checked
        env=base.gym.make('swm/PushT-v1',render_mode='rgb_array')
        evaluator=Path(env.unwrapped.eval_state.__code__.co_filename).resolve()
        require(str(evaluator) in protocol['sources_sha256'] and durable.digest(evaluator)==protocol['sources_sha256'][str(evaluator)],
                'installed evaluator not in prepared source seal')
        require(tuple(env.unwrapped.variation_space['agent']['color'].value.tolist())==(65,105,225),'canonical RoyalBlue guard contract')
        roots=base.collect_nonterminal_boundary_contact_roots(env,count=4,seed=int(protocol['root_seed']),
            oversample_factor=protocol['root_oversample_factor'])
        rootmap={base.root_fingerprint(r):r for r in roots}
        require(len(roots)==len(rootmap)==4 and set(rootmap)==set(protocol['roots']),'same four physical roots; no replacement')
        identities={}
        for plan in plans:
            root=rootmap[plan['root_sha256']]
            def reset():
                require(base.root_fingerprint(root)==plan['root_sha256'],'physical root fingerprint changed')
                frames,history,goal=base.physical._reset_and_warmup(env,root,history_len=3)
                identity=verify_rgb_context(frames,goal,protocol['references'][plan['root_sha256']])
                identity.update(history=np.asarray(history).tolist(),state=base.p0.packets.state_of(env).tolist(),goal=np.asarray(root['goal']).tolist())
                if plan['root_sha256'] in identities:
                    require(same(identity,identities[plan['root_sha256']]),'paired root reset/feedback context changed')
                identities[plan['root_sha256']]=deepcopy(identity)
                return identity
            def evaluate(state):
                return dict(official=official_checked(env,state,root['goal']),pose=base.physical._pose_label(state,root['goal']))
            row=physical_branch(plan=plan,output=output,reset=reset,env=env,state_of=base.p0.packets.state_of,evaluate=evaluate)
            rows.append(row)
            verify_branch(plan,row)
    except (Exception,KeyboardInterrupt) as exc:failures.append(durable.failure(exc,'physical_setup_or_branch'))
    finally:
        if env is not None:
            try:env.close()
            except Exception as exc:failures.append(durable.failure(exc,'env_close'))
    materialize_missing(plans,rows,output)
    write(output/'physical_report.json',dict(schema=NAME+'.physical_report',planning_report_sha256=REPORT_SHA,
        planning_protocol_id=protocol['planning_protocol_id'],inherited_cost=planning.inherited_cost(),
        total_pipeline_cost_equivalent=False,rows=rows,failures=deepcopy(failures),new_predict_calls=0,P3_admission=False))
    if encode_terminals:
        def setup():
            nonlocal model,model_before,host,checked
            import torch
            from PIL import Image
            from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
            from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
            from experiments.checkpoint_compat import load_compatible_pretrained
            from experiments.gpu_preflight import require_cuda_allocation
            for p,h in protocol['model_bindings'].items():require(checked.digest(p)==h,'terminal model binding changed')
            policy=Path(next(p for p in protocol['model_bindings'] if Path(p).name=='weights.pt')).parent
            host=require_cuda_allocation('cuda');device=torch.device('cuda:0')
            model=load_compatible_pretrained(str(policy),device=device).eval().requires_grad_(False)
            model_before=checked.tensor_state_hash(model)
            def forbidden(*a,**kw):
                stats['forbidden_predict_attempts']+=1
                raise RuntimeError('endpoint encoding forbids new predictions')
            model.predict=forbidden
            transform=p0.image_transform()
            def encode(path):
                with Image.open(path) as im:rgb=np.asarray(im.convert('RGB')).copy()
                tensor=p0._image_tensor(rgb,transform=transform,device=device)
                require(tensor.device.type=='cuda','no CPU neural fallback')
                stats['encode_calls']+=1;stats['encode_samples']+=tensor.shape[0]
                try:
                    with torch.no_grad():vector=model.encode({'pixels':tensor})['emb'][0,0]
                    require(vector.device.type=='cuda' and bool(torch.isfinite(vector).all()),'finite CUDA endpoint')
                    torch.cuda.synchronize(device)
                    return vector.cpu().tolist()
                except BaseException:
                    stats['failed_encode_calls']+=1
                    raise
            return encode
        encodings=optional_encodings(rows,output,setup)
    source_ok=True
    try:validate_prepared(prepared,prepare_sha256)
    except Exception as exc:
        source_ok=False;failures.append(durable.failure(exc,'final_source_verification'))
    model_ok=True
    if model is not None:
        try:model_ok=model_before==checked.tensor_state_hash(model)
        except Exception as exc:model_ok=False;failures.append(durable.failure(exc,'final_model_check'))
    pairs=comparisons(plans,rows)
    integrity=(not failures and source_ok and model_ok and stats['forbidden_predict_attempts']==0 and
        all(r['status'] in ('completed','observation_unavailable','environment_terminated') and 'pose' in r and 'official' in r for r in rows))
    encoding_ok=(not encode_terminals or len(encodings)==8 and all(e['status']=='completed' for e in encodings))
    report=dict(schema=NAME+'.report',status='guarded_calibration_recorded' if integrity else 'integrity_fail',
        rows=rows,encodings=encodings,pairs=pairs,failures=failures,source_verified=source_ok,model_unchanged=model_ok,
        optional_encodings_complete=encoding_ok,encoding_stats=stats,host=host,
        planning_report_sha256=REPORT_SHA,planning_protocol_id=protocol['planning_protocol_id'],
        consumer_prepare_sha256=prepare_sha256,upstream_predict_samples_charged_once=1176,new_predict_calls=0,
        inherited_cost=planning.inherited_cost(),inherited_arrays_charged_once=4,new_jacobian_materializations=0,
        total_pipeline_cost_equivalent=False,
        physical_env_step_attempts=sum(r['physical_env_step_attempts'] for r in rows),
        physical_env_step_returns=sum(r['physical_env_step_returns'] for r in rows),
        attempted_terminal_encode_samples=sum(e['attempted_encode_samples'] for e in encodings),
        planned_selected_path_steps=120,physical_repetitions=1,calibration_roots=4,new_independent_roots=0,
        P1_admission=False,P3_admission=False,total_flops_equality=False,efficacy_claim=False)
    write(output/'report.json',report)
    if not integrity or not encoding_ok:raise RuntimeError('physical/encoding incomplete; all branches retained, no retry')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare',action='store_true')
    action.add_argument('--execute-physics',action='store_true')
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--prepared',type=Path)
    parser.add_argument('--prepare-sha256')
    parser.add_argument('--encode-terminals',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.prepare:
        require(not args.encode_terminals,'prepare never enables CUDA encoding')
        prepare(args.project_root,args.output)
    else:
        require(args.prepared is not None and args.prepare_sha256 is not None,'reviewed prepare path/SHA required')
        execute_physics(prepared=args.prepared,prepare_sha256=args.prepare_sha256,output=args.output,
                        encode_terminals=args.encode_terminals)
