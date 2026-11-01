"""Independent fresh eight-arm goal-patch physical consumer, mixed H2/H3.

Source prepare is model-free. Bind only the NEW complete double-replay exporter.
MAIN explicitly runs physics, then optional CUDA encoding of durable endpoints.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import os
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import p3p4_goal_patch_replay_20261027_v1 as replay
from experiments.spectral_reachability import p3_horizon_residual_physical_20261027_v4 as helpers
from experiments.spectral_reachability import p3_matched_physical_20261027_v1 as durable
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal,verify_seal

NAME='p3p4_goal_patch_physical_20261027_v1'
PROJECT=Path(__file__).resolve().parents[2]
REPLAY_PREPARED='results/p3p4_goal_patch_replay_20261027_v1_prepared01'
REPLAY_SHA='a01d6c90e2293eaa5b330d6297139a2c59a68580e1bdc8accb2d520bc8efa7a6'
REFERENCES='results/p3_horizon_residual_physical_20261027_v4_source_prepared01'
REFERENCES_SHA='c5eee78ded753d1c2b955335b97e218d420671e1d2d35636aae8303a8811622b'
io,require,write,digest=replay.io,replay.require,replay.write,replay.digest
read_json=replay.read_json
same=lambda a,b:io.canonical(a)==io.canonical(b)


def contract():
    return dict(roots=4,arms=list(replay.ARMS),branches=8,repetitions=1,horizons=[2,3],planned_commands=90,
        actions='all 5H raw commands from new source-bound double replay; no clipping/reselection',
        controls='fresh old_Q AND fresh goal_patch_Q, never reuse old physical endpoint',
        guard='RoyalBlue RGB upfront and every raw command; durable attempts/returns/observations and partials',
        reset='four roots+goals upfront; exact RGB/full state and normalized incoming warmup tokens',
        encoding='optional HOST CUDA after durable entire physical_report; endpoint encoder+existing projector each cap8',
        strata=['full_H2_pair','full_H3_pair','full_mixed_H2_H3','partial_same_steps','partial_or_unmatched'],
        metrics='original official evaluator and pose; both original Q and goal-only patch Q on identical fresh endpoint vector',
        new_predict_calls=0,new_jacobians=0,new_vjp_directions=0,new_svds=0,
        total_resource_matched=False,P1_admission=False,P3_admission=False,P4_admission=False,efficacy_claim=False)


def prepare(root,output):
    root=Path(root).resolve();output=Path(output).absolute();require(not output.exists(),'exclusive NEW source preparation')
    p=replay.validate_prepared(root/REPLAY_PREPARED,REPLAY_SHA)
    refs=helpers.validate_prepared(root/REFERENCES,REFERENCES_SHA)
    require([v['root_sha256'] for v in p['plans'][::2]]==refs['roots'],'same four source roots')
    original=io.index(refs['packets'],('root_sha256',))
    for plan in p['plans']:
        packet=original[plan['root_sha256'],]
        require(all(same(plan[k],packet[k]) for k in ('root_context','goal','input_sha256')) and same(plan['old_projection'],packet['projection']),'root/goal/context/original Q source')
    seeds=[Path(__file__).resolve(),root/'tests'/('test_'+NAME+'.py'),root/'innerdocs'/(NAME+'.md'),
        root/'experiments/spectral_reachability/audit_pusht_d2p_goal_agent_intervention.py',
        root/'experiments/spectral_reachability/run_pusht_sampling_comparison_20261027_v2.py',
        root/'.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py',
        root/'.venv/lib/python3.12/site-packages/stable_worldmodel/envs/__init__.py']
    sources=replay.metric.old.prior.physical.source_closure(root,seeds)
    protocol=dict(schema=NAME+'.protocol',stage='source_prepared_awaiting_replay',contract=contract(),
        replay_prepared=str(root/REPLAY_PREPARED),replay_prepare_sha256=REPLAY_SHA,
        reference_prepared=str(root/REFERENCES),reference_prepare_sha256=REFERENCES_SHA,
        expected_exports=p['plans'],references=refs['references'],root_seed=refs['root_seed'],root_oversample_factor=4,
        model_bindings=p['model_bindings'],inherited_resources=p['inherited_resources'],
        sources_sha256={str(path):io.sha(path.read_bytes()) for path in sources})
    output.mkdir(exist_ok=False);write(output/'protocol.json',protocol)
    seals=[create_seal(files=sources[i:i+40],args_json=io.canonical(dict(consumer=NAME)),output_dir=output/f'source_seal_{i//40}') for i in range(0,len(sources),40)]
    receipt=dict(status=protocol['stage'],protocol_sha256=io.sha((output/'protocol.json').read_bytes()),source_seals=seals,new_neural_calls=0,new_physical_steps=0)
    write(output/'prepare.json',receipt);return receipt


def validate_prepared(prepared,sha):
    prepared=Path(prepared).absolute();require(io.sha((prepared/'prepare.json').read_bytes())==sha,'prepare SHA')
    receipt=read_json(prepared/'prepare.json');p=read_json(prepared/'protocol.json')
    require(io.sha((prepared/'protocol.json').read_bytes())==receipt['protocol_sha256'] and p['schema']==NAME+'.protocol' and p['contract']==contract() and p['stage']==receipt['status'],'protocol contract/bytes/stage')
    for path,h in p['sources_sha256'].items():require(io.sha(Path(path).read_bytes())==h,'sealed source drift: '+path)
    for s in receipt['source_seals']:verify_seal(output_dir=s['output_dir'],expected_manifest_sha256=s['manifest_sha256'])
    base=replay.validate_prepared(p['replay_prepared'],p['replay_prepare_sha256']);refs=helpers.validate_prepared(p['reference_prepared'],p['reference_prepare_sha256'])
    require(p['replay_prepare_sha256']==REPLAY_SHA and p['reference_prepare_sha256']==REFERENCES_SHA and
        p['expected_exports']==base['plans'] and p['model_bindings']==base['model_bindings'] and p['inherited_resources']==base['inherited_resources'] and
        p['references']==refs['references'] and p['root_seed']==refs['root_seed'],'upstream replay and independent roots/goals')
    if p['stage']=='report_bound_ready':
        validate_prepared(p['source_prepared'],p['source_prepare_sha256'])
        require(io.sha((Path(p['replay_output'])/'report.json').read_bytes())==p['replay_report_sha256'],'actual replay report bytes')
        require(p['plans']==plans_from_exports(p,replay.selected_actions(p['replay_output'])),'source-bound fresh double witnesses')
    else:require(p['stage']=='source_prepared_awaiting_replay','known prepare stage')
    return p


def plans_from_exports(protocol,exports):
    index=io.index(exports,('root_sha256','arm'));expected=protocol['expected_exports']
    require(set(index)=={(p['root_sha256'],p['arm']) for p in expected} and len(index)==8,'strict eight root-arm census')
    plans=[]
    for source in expected:
        e=index[source['root_sha256'],source['arm']]
        require(all(same(e[k],v) for k,v in source.items()) and e['replay_report_sha256']==protocol['replay_report_sha256'],'complete bank/horizon/root/goal/bothQ binding')
        require(digest({k:v for k,v in e.items() if k!='export_sha256'})==e['export_sha256'],'new exporter content digest')
        p=dict(schema=NAME+'.plan',root_index=e['root_index'],root_sha256=e['root_sha256'],method=e['arm'],
            planned_horizon_tokens=e['horizon'],planned_commands=5*e['horizon'],corrected_search_horizon=3,
            commands=e['actions'],predictor_context=e['root_context'],goal=e['goal'],projection=e['projection'],old_projection=e['old_projection'],
            selected_predicted_context=e['context'],predicted_old_Q_cost=e['old_Q_cost'],predicted_goal_patch_Q_cost=e['goal_patch_Q_cost'],
            whole_bank_sha256=e['bank_sha256'],metric_sha256=e['metric_sha256'],input_sha256=e['input_sha256'],
            export_sha256=e['export_sha256'],replay_report_sha256=e['replay_report_sha256'],replay_witness=e['replay_witness'],
            replay_shared_resources=e['replay_shared_resources'],inherited_resources=e['inherited_resources'])
        p['selection_sha256']=digest(p);validate_plan(p);plans.append(p)
    require(sum(p['planned_commands'] for p in plans)==90,'exact mixed horizon physical step upper bound')
    return plans


def bind_report(prepared,sha,replay_output,report_sha,output):
    p=validate_prepared(prepared,sha);require(p['stage']=='source_prepared_awaiting_replay','bind only original source preparation')
    output=Path(output).absolute();replay_output=Path(replay_output).absolute();require(not output.exists(),'exclusive new report-binding path')
    require(io.sha((replay_output/'report.json').read_bytes())==report_sha,'explicit actual new replay report SHA')
    bound=dict(deepcopy(p),stage='report_bound_ready',source_prepared=str(Path(prepared).absolute()),source_prepare_sha256=sha,
        replay_output=str(replay_output),replay_report_sha256=report_sha)
    bound['plans']=plans_from_exports(bound,replay.selected_actions(replay_output))
    output.mkdir(exist_ok=False);write(output/'protocol.json',bound)
    receipt=dict(read_json(Path(prepared)/'prepare.json'),status=bound['stage'],protocol_sha256=io.sha((output/'protocol.json').read_bytes()))
    write(output/'prepare.json',receipt);validate_prepared(output,io.sha((output/'prepare.json').read_bytes()));return receipt


def validate_plan(p):
    h=p['planned_horizon_tokens'];raw=np.asarray(p['commands'])
    require(p['schema']==NAME+'.plan' and type(h) is int and h in (2,3) and p['planned_commands']==5*h and p['method'] in replay.ARMS and
        raw.shape==(5*h,2) and raw.dtype.kind in 'fi' and np.isfinite(raw).all() and np.all(np.abs(raw)<=1),'valid exact mixed-H physical actions before env access')
    require(digest({k:v for k,v in p.items() if k!='selection_sha256'})==p['selection_sha256'],'physical selection content SHA')


def physical_branch(*,plan,output,reset,env,state_of,evaluate):
    validate_plan(plan);output=Path(output)
    r=durable.physical_branch(plan=plan,repetition=0,output=output,reset=reset,env=env,state_of=state_of,evaluate=evaluate)
    events=[io.decode(s) for s in Path(r['journal_path']).read_bytes().splitlines()]
    images=[e for e in events if 'rgb_path' in e]
    r['endpoint_fresh']=bool(images and r.get('terminal_rgb_path') and images[-1].get('primitive_step',0)==r['physical_env_step_returns'] and
        r['physical_env_step_attempts']==r['physical_env_step_returns'] and r['status'] in ('completed','observation_unavailable','environment_terminated'))
    r['endpoint_scope']=('full_H' if r['full_horizon_completed'] else 'partial_H')+str(plan['planned_horizon_tokens']) if r['endpoint_fresh'] else 'unknown_or_stale_endpoint'
    write(output/f'root_{plan["root_index"]}_{plan["method"]}_replay0.endpoint.json',r)
    return r


def verify_branch(p,r):
    validate_plan(p)
    require(all(r[k]==p[k] for k in ('root_index','root_sha256','method','selection_sha256','planned_horizon_tokens','planned_commands')),'physical exact root-arm identity')
    a,b=(io.count(r[k]) for k in ('physical_env_step_attempts','physical_env_step_returns'))
    require(b<=a<=p['planned_commands'] and a==r['attempted_commands'] and b==r['completed_commands'],'step counters')
    path=Path(r['journal_path']);require(io.sha(path.read_bytes())==r['journal_sha256'],'journal bytes')
    events=[io.decode(s) for s in path.read_bytes().splitlines()]
    attempts=[e for e in events if e['kind']=='attempt'];returns=[e for e in events if e['kind']=='step_returned'];obs=[e for e in events if e['kind']=='observation']
    require(len(attempts)==a and len(returns)==b and [e['primitive_step'] for e in attempts]==list(range(1,a+1)) and [e['primitive_step'] for e in returns]==list(range(1,b+1)),'durable step census/order')
    require(all(same(e['command'],p['commands'][i]) for i,e in enumerate(attempts)),'all raw command witnesses')
    require(len(obs)==len(r['trace']) and all(all(same(e.get(k),v) for k,v in t.items()) for e,t in zip(obs,r['trace'])),'observation traces')
    for e in events:
        if 'rgb_path' in e:require(io.sha(Path(e['rgb_path']).read_bytes())==e['rgb_file_sha256'],'physical RGB bytes')
    if r['full_horizon_completed']:require(r['status']=='completed' and b==p['planned_commands'] and r['endpoint_fresh'],'full horizon guard')


def materialize_missing(plans,rows,output):
    seen={(r['root_sha256'],r['method']) for r in rows}
    for p in plans:
        if (p['root_sha256'],p['method']) in seen:continue
        r={k:deepcopy(p[k]) for k in ('root_index','root_sha256','method','selection_sha256','planned_commands','planned_horizon_tokens')}
        r.update(status='blocked',failures=['setup/earlier integrity failure; no dispatch'],repetition=0,trace=[],attempted_commands=0,completed_commands=0,
            physical_env_step_attempts=0,physical_env_step_returns=0,full_horizon_completed=False,endpoint_fresh=False,endpoint_scope='unknown_or_stale_endpoint',terminal_rgb_path=None)
        write(Path(output)/f'root_{p["root_index"]}_{p["method"]}_replay0.physical.json',r);rows.append(r)


def comparisons(plans,rows):
    index=io.index(rows,('root_sha256','method'));require(set(index)=={(p['root_sha256'],p['method']) for p in plans} and len(index)==8,'eight physical rootarms')
    for p in plans:require(all(index[p['root_sha256'],p['method']][k]==p[k] for k in ('selection_sha256','planned_horizon_tokens','planned_commands')),'paired plan identity')
    pairs=[]
    for root in dict.fromkeys(p['root_sha256'] for p in plans):
        a,b=(index[root,arm] for arm in replay.ARMS);hs=[a['planned_horizon_tokens'],b['planned_horizon_tokens']]
        identical='context_identity' in a and 'context_identity' in b and same(a['context_identity'],b['context_identity'])
        full=identical and all(r['full_horizon_completed'] and r['endpoint_fresh'] for r in (a,b))
        partial=identical and all(r['endpoint_fresh'] for r in (a,b)) and a['physical_env_step_returns']==b['physical_env_step_returns']
        classification=(f'full_H{hs[0]}_pair' if hs[0]==hs[1] else 'full_mixed_H2_H3') if full else ('partial_same_steps' if partial else 'partial_or_unmatched')
        pairs.append(dict(root_sha256=root,classification=classification,selected_horizons=hs,identical_initial_context=identical,
            actual_steps=[a['physical_env_step_returns'],b['physical_env_step_returns']],
            new_minus_old_pose_cost=b['pose']['pose_cost']-a['pose']['pose_cost'] if full and hs[0]==hs[1] and all('pose' in r for r in (a,b)) else None,
            scope='known-root descriptive only; mixed horizons and partials never pooled as full same-H'))
    return pairs


def optional_encodings(plans,rows,output,setup):
    output=Path(output);require((output/'physical_report.json').is_file(),'all physical receipts durable before any CUDA setup')
    index=io.index(plans,('root_sha256','method'));receipts=[];encode=None;setup_error=None
    with (output/'encoding.events.jsonl').open('x') as stream:
        def event(e):stream.write(io.canonical(e)+'\n');stream.flush();os.fsync(stream.fileno())
        if any(r['endpoint_fresh'] for r in rows):
            event(dict(kind='encoding_setup',completed=False))
            try:encode=setup(event);event(dict(kind='encoding_setup',completed=True))
            except (Exception,KeyboardInterrupt) as exc:setup_error=durable.failure(exc,'encoding_setup');event(setup_error)
        for r in rows:
            p=index[r['root_sha256'],r['method']]
            receipt=dict(root_sha256=r['root_sha256'],method=r['method'],selection_sha256=p['selection_sha256'],endpoint_scope=r['endpoint_scope'],
                planned_horizon_tokens=p['planned_horizon_tokens'],attempted_encode_samples=0,status='blocked_stale_endpoint',failure=None)
            try:
                if r['endpoint_fresh']:
                    if setup_error is not None:receipt.update(status='encoding_setup_failed',failure=setup_error)
                    else:
                        path=Path(r['terminal_rgb_path']);require(io.sha(path.read_bytes())==r['terminal_rgb_sha256'],'fresh terminal bytes')
                        receipt.update(attempted_encode_samples=1,terminal_rgb_sha256=r['terminal_rgb_sha256']);event(dict(kind='endpoint_encode',completed=False,**receipt))
                        z=replay.metric.old.prior.array(encode(path),(192,))
                        receipt.update(status='completed',terminal_vector=z.tolist(),
                            actual_old_Q_cost=replay.metric.native_cost(z,p['goal'],p['old_projection']),
                            actual_goal_patch_Q_cost=replay.metric.native_cost(z,p['goal'],p['projection']),
                            predicted_old_Q_cost=p['predicted_old_Q_cost'],predicted_goal_patch_Q_cost=p['predicted_goal_patch_Q_cost'],
                            whole_bank_sha256=p['whole_bank_sha256'],metric_sha256=p['metric_sha256'])
            except (Exception,KeyboardInterrupt) as exc:receipt.update(status='encoding_failed',failure=durable.failure(exc,'endpoint_encode'))
            event(dict(kind='endpoint_result',**receipt));write(output/f'root_{r["root_index"]}_{r["method"]}_replay0.encoding.json',receipt);receipts.append(receipt)
    return receipts


def execute_physics(prepared,sha,output,encode_terminals=False):
    protocol=validate_prepared(prepared,sha);require(protocol['stage']=='report_bound_ready','bind NEW actual replay first')
    plans=protocol['plans'];output=Path(output).absolute();require(not output.exists(),'exclusive NEW physics output')
    output.mkdir(exist_ok=False);write(output/'execution_protocol.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=sha,replay_report_sha256=protocol['replay_report_sha256'],encode_terminals=encode_terminals))
    rows=[];failures=[];encodings=[];env=None;model=None;before=None;checked=None;host=None
    stats=dict(encoder_attempts=0,encoder_completed=0,projector_attempts=0,projector_completed=0,forbidden_predict_attempts=0)
    try:
        from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
        from experiments.spectral_reachability.run_pusht_sampling_comparison_20261027_v2 import verify_rgb_context,official_checked
        env=base.gym.make('swm/PushT-v1',render_mode='rgb_array');evaluator=Path(env.unwrapped.eval_state.__code__.co_filename).resolve()
        require(str(evaluator) in protocol['sources_sha256'] and io.sha(evaluator.read_bytes())==protocol['sources_sha256'][str(evaluator)],'sealed installed official evaluator')
        require(tuple(env.unwrapped.variation_space['agent']['color'].value.tolist())==(65,105,225),'RoyalBlue guard contract')
        roots=base.collect_nonterminal_boundary_contact_roots(env,count=4,seed=int(protocol['root_seed']),oversample_factor=protocol['root_oversample_factor'])
        rootmap=helpers.validate_roots(roots,protocol['references'],base.root_fingerprint)
        identities={}
        for p in plans:
            root=rootmap[p['root_sha256']]
            def reset():
                require(base.root_fingerprint(root)==p['root_sha256'],'root fingerprint')
                frames,history,goal=base.physical._reset_and_warmup(env,root,history_len=3)
                identity=verify_rgb_context(frames,goal,protocol['references'][p['root_sha256']])
                identity.update(helpers.validate_reset_history(history,p['predictor_context']['action_history']),state=base.p0.packets.state_of(env).tolist(),goal=np.asarray(root['goal']).tolist())
                if p['root_sha256'] in identities:require(same(identity,identities[p['root_sha256']]),'paired full reset/context mismatch')
                identities[p['root_sha256']]=deepcopy(identity);return identity
            def evaluate(state):return dict(official=official_checked(env,state,root['goal']),pose=base.physical._pose_label(state,root['goal']))
            r=physical_branch(plan=p,output=output,reset=reset,env=env,state_of=base.p0.packets.state_of,evaluate=evaluate);rows.append(r);verify_branch(p,r)
    except (Exception,KeyboardInterrupt) as exc:failures.append(durable.failure(exc,'physical_setup_or_branch'))
    finally:
        if env is not None:
            try:env.close()
            except Exception as exc:failures.append(durable.failure(exc,'env_close'))
    materialize_missing(plans,rows,output)
    physical=dict(schema=NAME+'.physical_report',rows=rows,failures=deepcopy(failures),consumer_prepare_sha256=sha,replay_report_sha256=protocol['replay_report_sha256'],contract=contract())
    write(output/'physical_report.json',physical);physical_sha=io.sha((output/'physical_report.json').read_bytes())
    if encode_terminals:
        def setup(event):
            nonlocal model,before,checked,host
            import torch
            from PIL import Image
            from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
            from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
            from experiments.checkpoint_compat import load_compatible_pretrained
            from experiments.gpu_preflight import require_cuda_allocation
            require(io.sha((output/'physical_report.json').read_bytes())==physical_sha,'physical report durable before CUDA')
            for path,h in protocol['model_bindings'].items():require(checked.digest(path)==h,'unchanged endpoint model')
            policy=Path(next(path for path in protocol['model_bindings'] if Path(path).name=='weights.pt')).parent
            host=require_cuda_allocation('cuda');device=torch.device('cuda:0');model=load_compatible_pretrained(str(policy),device=device).eval().requires_grad_(False)
            before=checked.tensor_state_hash(model)
            def forbid(*a,**kw):stats['forbidden_predict_attempts']+=1;event(dict(kind='forbidden_predict'));raise ValueError('no predict in endpoint encoding')
            model.predict=forbid
            for name,shape in (('encoder',(1,3,224,224)),('projector',(1,192))):
                original=getattr(model,name).forward
                def counted(*a,_name=name,_shape=shape,_original=original,**kw):
                    require(a[0].device.type=='cuda' and tuple(a[0].shape)==_shape and stats[_name+'_attempts']<8,'single CUDA endpoint encoder/projector cap')
                    index=stats[_name+'_attempts'];stats[_name+'_attempts']+=1;event(dict(kind=_name,index=index,completed=False))
                    try:r=_original(*a,**kw);stats[_name+'_completed']+=1;event(dict(kind=_name,index=index,completed=True));return r
                    except BaseException as exc:event(dict(kind=_name,index=index,completed=False,error=str(exc)));raise
                getattr(model,name).forward=counted
            transform=p0.image_transform()
            def encode(path):
                with Image.open(path) as image:rgb=np.asarray(image.convert('RGB')).copy()
                with torch.no_grad():z=model.encode({'pixels':p0._image_tensor(rgb,transform=transform,device=device)})['emb'][0,0]
                require(z.device.type=='cuda' and bool(torch.isfinite(z).all()),'finite native endpoint192');torch.cuda.synchronize(device);return z.cpu().tolist()
            return encode
        encodings=optional_encodings(plans,rows,output,setup)
    source_ok=model_ok=True
    try:validate_prepared(prepared,sha)
    except Exception as exc:source_ok=False;failures.append(durable.failure(exc,'final_source_check'))
    if model is not None:
        try:model_ok=before==checked.tensor_state_hash(model)
        except Exception as exc:model_ok=False;failures.append(durable.failure(exc,'final_model_check'))
    integrity=not failures and source_ok and model_ok and stats['forbidden_predict_attempts']==0 and all(r['status'] in ('completed','observation_unavailable','environment_terminated') and 'pose' in r and 'official' in r for r in rows)
    encoded=not encode_terminals or len(encodings)==8 and all(e['status']=='completed' for e in encodings)
    result=dict(schema=NAME+'.report',status='guarded_calibration_recorded' if integrity and encoded else 'incomplete',rows=rows,pairs=comparisons(plans,rows),encodings=encodings,failures=failures,
        physical_report_sha256=physical_sha,replay_report_sha256=protocol['replay_report_sha256'],consumer_prepare_sha256=sha,
        model_unchanged=model_ok,source_verified=source_ok,encoding_stats=stats,host=host,optional_encodings_complete=encoded,
        physical_env_step_attempts=sum(r['physical_env_step_attempts'] for r in rows),physical_env_step_returns=sum(r['physical_env_step_returns'] for r in rows),
        planned_selected_path_steps=90,inherited_resources=protocol['inherited_resources'],replay_shared_resources=plans[0]['replay_shared_resources'],
        root_search_warmup_cost='separate, not included in selected-path90 bound; no total resource equivalence',contract=contract(),
        new_predict_calls=0,P1_admission=False,P3_admission=False,P4_admission=False,efficacy_claim=False)
    write(output/'report.json',result)
    if not integrity or not encoded:raise RuntimeError('incomplete physical/encoding; preserve partials, no retry')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare','bind','verify','execute-physics'):group.add_argument('--'+name,action='store_true')
    parser.add_argument('--project-root',type=Path,default=PROJECT);parser.add_argument('--output',type=Path);parser.add_argument('--prepared',type=Path);parser.add_argument('--prepare-sha256')
    parser.add_argument('--replay-output',type=Path);parser.add_argument('--report-sha256');parser.add_argument('--encode-terminals',action='store_true');a=parser.parse_args()
    require(a.execute_physics or not a.encode_terminals,'encoding only explicit physical execution')
    if a.prepare:require(a.output is not None,'new output');r=prepare(a.project_root,a.output)
    else:
        require(a.prepared is not None and a.prepare_sha256 is not None,'reviewed receipt SHA')
        if a.verify:p=validate_prepared(a.prepared,a.prepare_sha256);r=dict(status='verified',stage=p['stage'])
        elif a.bind:
            require(a.output is not None and a.replay_output is not None and a.report_sha256 is not None,'new output and actual report SHA');r=bind_report(a.prepared,a.prepare_sha256,a.replay_output,a.report_sha256,a.output)
        else:require(a.output is not None,'new output');r=execute_physics(a.prepared,a.prepare_sha256,a.output,a.encode_terminals)
    print(io.canonical({k:v for k,v in r.items() if k not in ('source_seals','rows','encodings','inherited_resources')}))


if __name__=='__main__':main()
