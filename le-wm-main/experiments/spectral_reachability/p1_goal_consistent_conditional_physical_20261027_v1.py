"""Fresh fixed-H2 conditional-compiler physical consumer, MAIN execution only.

Source prepare and report bind are model/physics-free. No inherited endpoint.
Eight exact programs; durable RoyalBlue guard; optional eight CUDA encodings.
"""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
import os
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import p3_goal_patch_consistent_compiler_20261027_v1 as planning
from experiments.spectral_reachability import p3_horizon_residual_physical_20261027_v4 as helpers
from experiments.spectral_reachability import p3_matched_physical_20261027_v1 as durable
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal,verify_seal

NAME='p1_goal_consistent_conditional_physical_20261027_v1'
PROJECT=Path(__file__).resolve().parents[2]
PLANNING_PREPARED='results/p3_goal_patch_consistent_compiler_20261027_v1_prepared'
PLANNING_SHA='2b32137d1e7facf79335b25c6e6acf976ed10162073b938bff1262b385ba1425'
PLANNING_SOURCE_SHA='006398e5aaf0f6e901a01c2d6779a977c31875413b967fec552245332002396d'
REFERENCES='results/p3_horizon_residual_physical_20261027_v4_source_prepared01'
REFERENCES_SHA='c5eee78ded753d1c2b955335b97e218d420671e1d2d35636aae8303a8811622b'
ARMS=planning.ARMS
io,require,write,digest=planning.io,planning.require,planning.write,planning.digest
same=lambda a,b:io.canonical(a)==io.canonical(b)


def contract():
    return dict(roots=4,branches=8,arms=list(ARMS),repetitions=1,horizon=2,commands_per_branch=10,
        selected_path_command_cap=80,warmup_step_cap=120,warmup_steps_per_branch=15,
        root_collection_steps=0,root_collection_resets='separately metered; original count4 oversample4',
        guard='unchanged durable RoyalBlue upfront and every raw command; no blind continuation',
        reset='original independent four state/goal pairs; original RGB stack/goal and normalized history exact',
        actions='new conditional compiler selected_actions only; all eight execute even ties; no clipping/reselection',
        feedback='RGB guard only; no replanning or physical-state feedback to action selection',
        metrics='same installed official evaluator/pose for both arms; each endpoint vector scored with BOTH sealed Qs',
        encoding='optional CUDA only, after ALL physical receipts; each fresh full/partial endpoint once',
        encoder_attempt_cap=8,existing_projector_attempt_cap=8,encoder_shape=[1,3,224,224],projector_shape=[1,192],
        stale_endpoints='never encode; failed env.step/render cannot reuse previous RGB',
        strata=['full_H2_pair','partial_same_steps','partial_or_unmatched'],
        inherited_planning_predicts=528,inherited_planning_action_encoders=528,inherited_jacobians=88,inherited_svds=96,
        bind_export_saved_array_svds=96,endpoint_new_predict_calls=0,endpoint_new_jacobians=0,
        independent_history_audit='all80 candidates, all8 winners/exports and104 saved replay contexts; exact float32 tokens and full layout',
        inherited_metric_cost=planning.contract()['inherited_metric_cost'],total_resource_matched=False,
        new_independent_roots=0,P1_admission=False,P3_admission=False,P4_admission=False,efficacy_claim=False)


def original_inputs(root):
    root=Path(root).resolve()
    require(io.sha(Path(planning.__file__).read_bytes())==PLANNING_SOURCE_SHA,'unchanged sealed conditional compiler source')
    body,identity=planning.validate_prepared(root/PLANNING_PREPARED,PLANNING_SHA)
    refs=helpers.validate_prepared(root/REFERENCES,REFERENCES_SHA)
    require(body['roots']==refs['roots'] and body['model_bindings']==refs['model_bindings'],'original roots/model bindings')
    old=io.index(refs['packets'],('root_sha256',))
    for p in body['packets']:
        source=old[p['root_sha256'],]
        require(p['source_input_sha256']==source['input_sha256'] and
            all(same(p[k],source[k]) for k in ('root_context','goal')) and
            same(p['projections']['old_Q'],source['projection']),'original context/goal0/old Q input')
    return dict(compiler_protocol_id=identity,roots=body['roots'],packets=body['packets'],arm_order=body['arm_order'],
        model_bindings=body['model_bindings'],references=refs['references'],root_seed=refs['root_seed'],root_oversample_factor=4)


def public_stats():
    """Read source literals without importing torch or the neural module."""
    values={}
    for node in ast.parse((PROJECT/'experiments/run_pusht_counterfactual.py').read_text()).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name in ('PUBLIC_ACTION_MEAN','PUBLIC_ACTION_STD'):
                require(isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='np.asarray','literal public normalizer')
                values[name]=np.asarray(ast.literal_eval(node.value.args[0]),dtype=np.float32)
    require(set(values)=={'PUBLIC_ACTION_MEAN','PUBLIC_ACTION_STD'} and
        all(v.shape==(2,) and np.isfinite(v).all() for v in values.values()) and
        np.all(values['PUBLIC_ACTION_STD']>0),'source public mean/std')
    require(np.array_equal(values['PUBLIC_ACTION_MEAN'],helpers.NORMALIZATION['mean']) and
        np.array_equal(values['PUBLIC_ACTION_STD'],helpers.NORMALIZATION['std']),'same frozen warmup/transition normalizer')
    return np.tile(values['PUBLIC_ACTION_MEAN'],5),np.tile(values['PUBLIC_ACTION_STD'],5)


def tokens(commands):
    raw=np.asarray(commands)
    require(raw.shape in ((5,2),(10,2)) and raw.dtype.kind in 'fi' and np.isfinite(raw).all() and
        np.all(np.abs(raw)<=1),'one/two exact bounded time-major packets')
    mean,std=public_stats()
    return (raw.astype(np.float32).reshape(-1,10)-mean)/std


def context_witness(initial,commands,after):
    """Independent normalized content and rolling chronology, not shape-only.

    Does not infer new latent values: checks retained old states and the exact
    between-state action tokens under the sealed production float32 formula.
    """
    from experiments.spectral_reachability.d2p_composition_witness_v2 import _context
    for c in (initial,after):
        _context(c)
        require(c['context_length']==3 and c['latent_dim']==192 and c['action_dim']==10,'official complete context dimensions')
    require(set(initial)==set(after) and all(same(initial[k],after[k]) for k in initial
        if k not in ('latent_history','action_history')),'full context layout/static metadata')
    t=tokens(commands);h=len(t)
    expected=np.concatenate((np.asarray(initial['action_history'],dtype=np.float32),t))[-2:]
    require(np.array_equal(after['action_history'],expected),'exact normalized action history CONTENT/order')
    require(same(after['latent_history'][:3-h],initial['latent_history'][h:]),'retained latent history rolling chronology')


def verify_compiler_histories(protocol,compiler_output,exports):
    """Read-only independent safety gate over ALL successful v1 saved contexts.

    Supplements v1's shape-only history check; no changed v1 or nonexistent v2.
    Zero SVD/model calls, no outcome-based candidate selection.
    """
    compiler_output=Path(compiler_output)
    raw=(compiler_output/'report.json').read_bytes()
    require(io.sha(raw)==protocol['compiler_report_sha256'],'history audit actual report binding')
    report=io.decode(raw)
    require(report['schema']==planning.NAME+'.report' and report['status']=='cuda_completed_h2_prerequisite','actual v1 complete source')
    rows=io.index(report['rows'],('root_sha256','arm'));roots=io.index(report['roots'],('root_sha256',))
    packets=io.index(protocol['packets'],('root_sha256',));exported=io.index(exports,('root_sha256','arm'))
    expected={(r,a) for r in protocol['roots'] for a in ARMS}
    require(set(rows)==set(exported)==expected and set(roots)=={(r,) for r in protocol['roots']},'full history-audit census')
    counts=dict(candidates=0,winner_contexts=0,replay_contexts=0,selected_exports=0)
    for root in protocol['roots']:
        initial=packets[root,]['root_context'];record=roots[root,]
        path=Path(record['journal_path']);require(io.sha(path.read_bytes())==record['journal_sha256'],'history audit journal SHA')
        journal=[io.decode(line) for line in path.read_bytes().splitlines()]
        for arm in ARMS:
            c=rows[root,arm]['compilation'];require(len(c['candidates'])==10,'ALL ten conditional candidates')
            for index,candidate in enumerate(c['candidates']):
                require(candidate['index']==index and candidate['completed'] is True,'ordered complete candidate')
                context_witness(initial,candidate['commands'],candidate['context'])
                require(same(candidate['context']['latent_history'][1:],
                    [candidate['predicted_prefix'],candidate['predicted_terminal']]),'stage/terminal history content')
                counts['candidates']+=1
            winner=c['winner']
            for context in (winner['context'],c['whole_replay_context'],c['segmented_replay_context']):
                context_witness(initial,winner['commands'],context);counts['winner_contexts']+=1
            e=exported[root,arm]
            context_witness(initial,e['actions'],e['selected_predicted_context']);counts['selected_exports']+=1
            require(same(e['actions'],winner['commands']) and same(e['selected_predicted_context'],winner['context']),'selected fresh program/context')
            replays=[e for e in journal if e.get('arm')==arm and e['kind']=='replay' and e['completed']]
            require([e['phase'] for e in replays]==['candidate_'+str(k) for k in range(10)]+
                ['winner_whole','winner_segment_0','winner_segment_1'],'all candidate and winner replay phases')
            for k,e in enumerate(replays):
                origin=replays[11]['context'] if k==12 else initial
                context_witness(origin,e['commands'],e['context']);counts['replay_contexts']+=1
    require(counts==dict(candidates=80,winner_contexts=24,replay_contexts=104,selected_exports=8),'complete independent content census')
    return dict(schema=NAME+'.compiler_history_audit',**counts,normalizer_mean_std=[v.tolist() for v in public_stats()],
        compiler_report_sha256=protocol['compiler_report_sha256'],new_neural_calls=0,new_svds=0,
        checks='exact float32 tokens; complete layout/static metadata; retained latent chronology; all candidates and selected',all_checks=True)


def prepare(root,output):
    root=Path(root).resolve();output=Path(output).absolute();require(not output.exists(),'exclusive NEW source prepare')
    body=original_inputs(root)
    seeds=[Path(__file__).resolve(),root/'tests'/('test_'+NAME+'.py'),root/'innerdocs'/(NAME+'.md'),
        root/'experiments/spectral_reachability/audit_pusht_d2p_goal_agent_intervention.py',
        root/'experiments/spectral_reachability/run_pusht_sampling_comparison_20261027_v2.py',
        root/'.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py',
        root/'.venv/lib/python3.12/site-packages/stable_worldmodel/envs/__init__.py']
    sources=planning.old.prior.physical.source_closure(root,seeds)
    protocol=dict(schema=NAME+'.protocol',stage='source_prepared_awaiting_compiler_report',project_root=str(root),
        contract=contract(),compiler_prepared=str(root/PLANNING_PREPARED),compiler_prepare_sha256=PLANNING_SHA,
        reference_prepared=str(root/REFERENCES),reference_prepare_sha256=REFERENCES_SHA,**body,
        sources_sha256={str(p):io.sha(p.read_bytes()) for p in sources})
    output.mkdir(exist_ok=False);write(output/'protocol.json',protocol)
    seals=[create_seal(files=sources[i:i+40],args_json=io.canonical(dict(consumer=NAME)),output_dir=output/f'source_seal_{i//40}')
        for i in range(0,len(sources),40)]
    receipt=dict(status=protocol['stage'],protocol_sha256=io.sha((output/'protocol.json').read_bytes()),source_seals=seals,
        model_loaded=False,new_neural_calls=0,new_physical_steps=0,P1_admission=False,P3_admission=False)
    write(output/'prepare.json',receipt);validate_prepared(output,io.sha((output/'prepare.json').read_bytes()));return receipt


def validate_prepared(prepared,expected_sha):
    prepared=Path(prepared).absolute();raw=(prepared/'prepare.json').read_bytes()
    require(io.sha(raw)==expected_sha,'prepare SHA');receipt=io.decode(raw)
    raw=(prepared/'protocol.json').read_bytes();require(io.sha(raw)==receipt['protocol_sha256'],'protocol bytes')
    p=io.decode(raw)
    require(p['schema']==NAME+'.protocol' and p['contract']==contract() and p['stage']==receipt['status'],'own schema/contract/stage')
    require(p['compiler_prepare_sha256']==PLANNING_SHA and p['reference_prepare_sha256']==REFERENCES_SHA and
        p['compiler_prepared']==str(Path(p['project_root'])/PLANNING_PREPARED) and
        p['reference_prepared']==str(Path(p['project_root'])/REFERENCES),'frozen source preparation paths')
    for key,value in original_inputs(p['project_root']).items():require(same(p[key],value),'original physical input binding: '+key)
    for path,h in p['sources_sha256'].items():require(io.sha(Path(path).read_bytes())==h,'consumer source drift: '+path)
    for seal in receipt['source_seals']:verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
    if p['stage']=='report_bound_ready':
        original=validate_prepared(p['source_prepared'],p['source_prepare_sha256'])
        require(original['stage']=='source_prepared_awaiting_compiler_report','nonrecursive source stage')
        for path,h in p['compiler_outputs_sha256'].items():require(io.sha(Path(path).read_bytes())==h,'bound compiler output drift')
        require(io.sha((Path(p['compiler_output'])/'report.json').read_bytes())==p['compiler_report_sha256'],'explicit actual report SHA')
        exports=planning.selected_actions(p['compiler_output'])
        require(same(p['plans'],plans_from_exports(p,exports)),'fresh compiler exports/actions changed')
        require(p['history_content_audit']==verify_compiler_histories(p,p['compiler_output'],exports),'independent ALL-candidate history content changed')
    else:require(p['stage']=='source_prepared_awaiting_compiler_report','known stage')
    return p


def plans_from_exports(protocol,exports):
    index=io.index(exports,('root_sha256','arm'));packets=io.index(protocol['packets'],('root_sha256',))
    require(set(index)=={(r,a) for r in protocol['roots'] for a in ARMS} and len(index)==8,'exact eight new compiler exports')
    plans=[]
    for i,root in enumerate(protocol['roots']):
        packet=packets[root,]
        for arm in protocol['arm_order'][i]:
            e=index[root,arm]
            require(e['source_schema']==planning.NAME+'.report' and type(e['horizon']) is int and e['horizon']==2 and
                e['root_index']==i and e['protocol_id']==protocol['compiler_protocol_id'],'new fixed-H2 source identity')
            require(e['report_sha256']==protocol['compiler_report_sha256'] and e['model_bindings']==protocol['model_bindings'],'model and real compiler report')
            require(all(same(e[k],packet[k]) for k in ('root_context','goal','input_sha256','source_input_sha256')) and
                same(e['projection'],packet['projections'][arm]) and e['q_sha256']==digest(packet['projections'][arm]) and
                e['active_metric_id']==packet['active_metric_ids'][arm],'source/active Q goal/context identity')
            require(digest({k:v for k,v in e.items() if k!='selection_sha256'})==e['selection_sha256'],'compiler export content hash')
            context_witness(packet['root_context'],e['actions'],e['selected_predicted_context'])
            plan=dict(schema=NAME+'.plan',root_index=i,root_sha256=root,method=arm,planned_horizon_tokens=2,
                planned_commands=10,corrected_search_horizon=2,commands=deepcopy(e['actions']),predictor_context=packet['root_context'],
                goal=packet['goal'],projections=packet['projections'],active_metric_id=e['active_metric_id'],
                input_sha256=e['input_sha256'],source_input_sha256=e['source_input_sha256'],
                selected_predicted_context=e['selected_predicted_context'],predicted_active_Q_cost=e['predicted_Q_cost'],
                compiler_report_sha256=e['report_sha256'],compiler_protocol_id=e['protocol_id'],
                export_selection_sha256=e['selection_sha256'],model_bindings=e['model_bindings'])
            plan['selection_sha256']=digest(plan);validate_plan(plan);plans.append(plan)
    require(sum(p['planned_commands'] for p in plans)==80,'eight full H2 paths, no duplicate removal')
    return plans


def bind_report(prepared,expected_sha,compiler_output,report_sha,output):
    p=validate_prepared(prepared,expected_sha);require(p['stage']=='source_prepared_awaiting_compiler_report','bind original source prepare only')
    output=Path(output).absolute();compiler_output=Path(compiler_output).absolute();require(not output.exists(),'exclusive NEW report bind')
    raw=(compiler_output/'report.json').read_bytes();require(io.sha(raw)==report_sha,'explicit real compiler report SHA')
    report=io.decode(raw)
    bound=dict(deepcopy(p),stage='report_bound_ready',source_prepared=str(Path(prepared).absolute()),source_prepare_sha256=expected_sha,
        compiler_output=str(compiler_output),compiler_report_sha256=report_sha)
    exports=planning.selected_actions(compiler_output)
    bound['plans']=plans_from_exports(bound,exports)
    bound['history_content_audit']=verify_compiler_histories(bound,compiler_output,exports)
    paths=[compiler_output/'report.json',compiler_output/'execution_protocol.json']
    paths.extend(Path(r['journal_path']) for r in report['roots'])
    paths.extend(compiler_output/f'root_{i}.json' for i in range(4))
    paths.extend(compiler_output/f'root_{i}_{a}.json' for i in range(4) for a in ARMS)
    bound['compiler_outputs_sha256']={str(path):io.sha(path.read_bytes()) for path in paths}
    output.mkdir(exist_ok=False);write(output/'protocol.json',bound)
    receipt=io.decode((Path(prepared)/'prepare.json').read_bytes())
    receipt.update(status=bound['stage'],protocol_sha256=io.sha((output/'protocol.json').read_bytes()),
        compiler_report_sha256=report_sha,plans=8,selected_path_command_cap=80,
        array_export_verification_svds_per_call=96)
    write(output/'prepare.json',receipt);validate_prepared(output,io.sha((output/'prepare.json').read_bytes()));return receipt


def validate_plan(p):
    raw=np.asarray(p['commands'])
    require(p['schema']==NAME+'.plan' and p['method'] in ARMS and type(p['planned_horizon_tokens']) is int and
        p['planned_horizon_tokens']==2 and p['planned_commands']==10 and p['corrected_search_horizon']==2 and
        raw.shape==(10,2) and raw.dtype.kind in 'fi' and np.isfinite(raw).all() and np.all(np.abs(raw)<=1),'exact bounded H2 plan before environment access')
    require(set(p['projections'])==set(ARMS),'both Qs on each fresh endpoint')
    for q in p['projections'].values():planning.validate_q(q)
    require(digest({k:v for k,v in p.items() if k!='selection_sha256'})==p['selection_sha256'],'physical selection SHA')


class StepMeter:
    """Pre-dispatch caps; selected-path, original warmup and collection separate."""
    def __init__(self,env,sink):
        self.env=env;self.sink=sink;self.phase='root_collection';self.binding={}
        self.counts={};self.reset_counts={};self.branch_counts={}
    def __getattr__(self,name):return getattr(self.env,name)
    def bind(self,plan):
        self.binding={k:plan[k] for k in ('root_sha256','method','selection_sha256')}
    def call(self,name,table,args,kwargs):
        require(self.phase in ('root_collection','warmup','selected'),'known physical phase')
        c=table.setdefault(self.phase,dict(attempts=0,returns=0))
        key=(self.binding.get('root_sha256'),self.binding.get('method'),self.phase,name)
        b=self.branch_counts.setdefault(key,dict(attempts=0,returns=0))
        limit={'root_collection':0,'warmup':120,'selected':80}[self.phase]
        local={'root_collection':0,'warmup':15,'selected':10}[self.phase]
        if name=='step' and (c['attempts']>=limit or b['attempts']>=local):
            self.sink(dict(kind='env_step_blocked',phase=self.phase,**self.binding))
            raise ValueError('physical step budget exceeded before dispatch')
        c['attempts']+=1;b['attempts']+=1
        event=dict(kind='env_'+name,phase=self.phase,index=c['attempts']-1,completed=False,**self.binding)
        if name=='step':event['command']=np.asarray(args[0]).tolist()
        self.sink(deepcopy(event))
        try:
            result=getattr(self.env,name)(*args,**kwargs);c['returns']+=1;b['returns']+=1;event['completed']=True
            return result
        except BaseException as exc:event['error']=type(exc).__name__+': '+str(exc);raise
        finally:self.sink(deepcopy(event))
    def step(self,*args,**kwargs):return self.call('step',self.counts,args,kwargs)
    def reset(self,*args,**kwargs):return self.call('reset',self.reset_counts,args,kwargs)
    def receipt(self):return dict(steps=deepcopy(self.counts),resets=deepcopy(self.reset_counts))


def endpoint_state(plan,row,events):
    images=[e for e in events if 'rgb_path' in e]
    fresh=bool(images and row.get('terminal_rgb_path')==images[-1]['rgb_path'] and
        images[-1].get('primitive_step',0)==row['physical_env_step_returns'] and
        row['physical_env_step_attempts']==row['physical_env_step_returns'] and
        row['status'] in ('completed','observation_unavailable','environment_terminated'))
    return fresh,('full_H2' if row['full_horizon_completed'] else 'partial_H2') if fresh else 'unknown_or_stale_endpoint'


def physical_branch(*,plan,output,reset,env,state_of,evaluate):
    validate_plan(plan);output=Path(output)
    row=durable.physical_branch(plan=plan,repetition=0,output=output,reset=reset,env=env,state_of=state_of,evaluate=evaluate)
    events=[io.decode(line) for line in Path(row['journal_path']).read_bytes().splitlines()]
    row['endpoint_fresh'],row['endpoint_scope']=endpoint_state(plan,row,events)
    row.update(schema=NAME+'.branch',commands=deepcopy(plan['commands']),active_metric_id=plan['active_metric_id'],
        export_selection_sha256=plan['export_selection_sha256'],compiler_report_sha256=plan['compiler_report_sha256'])
    write(output/f'root_{plan["root_index"]}_{plan["method"]}.endpoint.json',row);return row


def verify_branch(plan,row):
    validate_plan(plan)
    require(row['schema']==NAME+'.branch' and all(row[k]==plan[k] for k in
        ('root_index','root_sha256','method','selection_sha256','planned_horizon_tokens','planned_commands',
         'commands','active_metric_id','export_selection_sha256','compiler_report_sha256')),'own physical plan/branch identity')
    a,b=(io.count(row[k]) for k in ('physical_env_step_attempts','physical_env_step_returns'))
    require(b<=a<=10 and a==row['attempted_commands'] and b==row['completed_commands'],'nonrefundable selected-step counters')
    path=Path(row['journal_path']);require(io.sha(path.read_bytes())==row['journal_sha256'],'physical journal SHA')
    events=[io.decode(line) for line in path.read_bytes().splitlines()]
    attempts=[e for e in events if e['kind']=='attempt'];returns=[e for e in events if e['kind']=='step_returned'];obs=[e for e in events if e['kind']=='observation']
    require(len(attempts)==a and len(returns)==b and [e['primitive_step'] for e in attempts]==list(range(1,a+1)) and
        [e['primitive_step'] for e in returns]==list(range(1,b+1)),'durable attempt/return census/order')
    require(all(same(e['command'],plan['commands'][i]) for i,e in enumerate(attempts)) and
        len(obs)==len(row['trace']) and all(all(same(e.get(k),v) for k,v in t.items()) for e,t in zip(obs,row['trace'],strict=True)),'raw command and observation trace identity')
    from experiments.spectral_reachability.pusht_rgb_observability_guard import RgbObservabilityGuard,ObservationUnavailable
    from PIL import Image
    guard=RgbObservabilityGuard(root_id=plan['root_sha256'],agent_rgb=(65,105,225));abstained=False
    for e in events:
        if 'rgb_path' in e:
            require(io.sha(Path(e['rgb_path']).read_bytes())==e['rgb_file_sha256'],'saved physical RGB bytes')
            with Image.open(e['rgb_path']) as image:rgb=np.asarray(image).copy()
            try:guard.observe(rgb,primitive_step=e.get('primitive_step',0))
            except ObservationUnavailable:abstained=True
        if e['kind']=='attempt':
            require(not abstained,'no dispatch after guard abstention');guard.dispatch(e['command'])
    if 'guard' in row:require(row['guard']==guard.receipt(),'independent saved-RGB guard replay')
    require((row['endpoint_fresh'],row['endpoint_scope'])==endpoint_state(plan,row,events),'actual endpoint freshness, no stale fallback')
    if row['endpoint_fresh']:require(row['terminal_rgb_sha256']==events[-1]['rgb_file_sha256'],'terminal corresponds to latest physical RGB')
    if row['full_horizon_completed']:require(row['status']=='completed' and b==10 and row['endpoint_fresh'],'full H2 only')
    return True


def materialize_missing(plans,rows,output):
    seen={(r['root_sha256'],r['method']) for r in rows}
    for p in plans:
        if (p['root_sha256'],p['method']) in seen:continue
        row={k:deepcopy(p[k]) for k in ('root_index','root_sha256','method','selection_sha256','planned_commands','planned_horizon_tokens')}
        row.update(schema=NAME+'.branch',status='blocked',failures=['setup/earlier integrity failure; no dispatch'],
            repetition=0,trace=[],physical_env_step_attempts=0,physical_env_step_returns=0,attempted_commands=0,completed_commands=0,
            full_horizon_completed=False,endpoint_fresh=False,endpoint_scope='unknown_or_stale_endpoint',terminal_rgb_path=None)
        write(Path(output)/f'root_{p["root_index"]}_{p["method"]}.blocked.json',row);rows.append(row)


def comparisons(plans,rows):
    index=io.index(rows,('root_sha256','method'))
    require(set(index)=={(p['root_sha256'],p['method']) for p in plans} and len(index)==8,'all eight physical root/arms retained')
    for p in plans:
        require(all(index[p['root_sha256'],p['method']][k]==p[k] for k in
            ('selection_sha256','planned_horizon_tokens','planned_commands')),'paired physical plan identity')
    pairs=[]
    for root in dict.fromkeys(p['root_sha256'] for p in plans):
        old,new=(index[root,a] for a in ARMS)
        identical='context_identity' in old and 'context_identity' in new and same(old['context_identity'],new['context_identity'])
        full=identical and all(r['full_horizon_completed'] and r['endpoint_fresh'] and r['physical_env_step_returns']==10 for r in (old,new))
        partial=identical and all(r['endpoint_fresh'] for r in (old,new)) and old['physical_env_step_returns']==new['physical_env_step_returns']
        pairs.append(dict(root_sha256=root,classification='full_H2_pair' if full else 'partial_same_steps' if partial else 'partial_or_unmatched',
            identical_initial_context=identical,actual_steps=[old['physical_env_step_returns'],new['physical_env_step_returns']],
            new_minus_old_pose_cost=new['pose']['pose_cost']-old['pose']['pose_cost'] if full and all('pose' in r for r in (old,new)) else None,
            new_minus_old_official_distance=new['official']['official_state_distance']-old['official']['official_state_distance']
                if full and all('official' in r for r in (old,new)) else None,
            scope='known-root descriptive complete H2 only; all partials retained; no admission'))
    return pairs


def empty_encoding_stats():
    return dict(encoder_attempts=0,encoder_completed=0,projector_attempts=0,projector_completed=0,
        forbidden_predict_attempts=0,forbidden_action_encoder_attempts=0,blocked_module_entries=0)


def restore_hooks(restore,failures):
    """Cleanup cannot erase an earlier durable physical/encoding failure."""
    for owner,name,original in reversed(restore):
        try:setattr(owner,name,original)
        except BaseException as exc:failures.append(dict(durable.failure(exc,'hook_restore'),attribute=name))


def frozen_model(model,device):
    require(getattr(device,'type',None)=='cuda' and all(not m.training for m in model.modules()) and
        all(not p.requires_grad and p.device==device for p in model.parameters()) and
        all(b.device==device for b in model.buffers()),'entire endpoint model frozen eval on CUDA')


class EncodingCap:
    def __init__(self,stats,event):self.stats=stats;self.event=event
    def dispatch(self,name,callback,args,kwargs):
        require(name in ('encoder','projector'),'known endpoint module')
        shape=(1,3,224,224) if name=='encoder' else (1,192);x=args[0]
        if getattr(getattr(x,'device',None),'type',None)!='cuda' or tuple(x.shape)!=shape or self.stats[name+'_attempts']>=8:
            self.stats['blocked_module_entries']+=1;self.event(dict(kind='blocked_module',module=name))
            raise ValueError('single-sample CUDA endpoint module cap/shape required')
        index=self.stats[name+'_attempts'];self.stats[name+'_attempts']+=1
        e=dict(kind=name,index=index,completed=False);self.event(deepcopy(e))
        try:
            result=callback(*args,**kwargs);self.stats[name+'_completed']+=1;e['completed']=True
            return result
        except BaseException as exc:e['error']=type(exc).__name__+': '+str(exc);raise
        finally:self.event(deepcopy(e))


def optional_encodings(plans,rows,output,setup):
    output=Path(output);require((output/'physical_report.json').is_file(),'all physical reports durable before CUDA setup')
    index=io.index(plans,('root_sha256','method'));receipts=[];encode=None;setup_error=None
    with (output/'encoding.events.jsonl').open('x') as stream:
        binding={}
        def event(e):stream.write(io.canonical(dict(e,**binding))+'\n');stream.flush();os.fsync(stream.fileno())
        if any(r['endpoint_fresh'] for r in rows):
            event(dict(kind='encoding_setup',completed=False))
            try:encode=setup(event);event(dict(kind='encoding_setup',completed=True))
            except (Exception,KeyboardInterrupt) as exc:setup_error=durable.failure(exc,'encoding_setup');event(setup_error)
        for row in rows:
            p=index[row['root_sha256'],row['method']]
            binding.update(root_sha256=row['root_sha256'],method=row['method'],selection_sha256=p['selection_sha256'])
            r=dict(**binding,schema=NAME+'.encoding',endpoint_scope=row['endpoint_scope'],attempted_encode_samples=0,
                status='blocked_stale_endpoint',failure=None,active_metric_id=p['active_metric_id'])
            try:
                if row['endpoint_fresh']:
                    if setup_error is not None:r.update(status='encoding_setup_failed',failure=setup_error)
                    else:
                        path=Path(row['terminal_rgb_path']);require(io.sha(path.read_bytes())==row['terminal_rgb_sha256'],'fresh terminal RGB bytes')
                        r.update(attempted_encode_samples=1,terminal_rgb_sha256=row['terminal_rgb_sha256']);event(dict(kind='endpoint_encode',**r))
                        z=planning.compiler.finite(encode(path),(192,))
                        r.update(status='completed',terminal_vector=z.tolist(),
                            actual_Q_costs={a:planning.cost(z,p['goal'],p['projections'][a]) for a in ARMS},
                            q_sha256={a:digest(p['projections'][a]) for a in ARMS})
            except (Exception,KeyboardInterrupt) as exc:r.update(status='encoding_failed',failure=durable.failure(exc,'endpoint_encode'))
            event(dict(kind='endpoint_result',**r));write(output/f'root_{row["root_index"]}_{row["method"]}.encoding.json',r);receipts.append(r)
    return receipts


def execute_physics(prepared,expected_sha,output,encode_terminals=False):
    protocol=validate_prepared(prepared,expected_sha);require(protocol['stage']=='report_bound_ready','bind complete fresh compiler report first')
    plans=protocol['plans']
    for p in plans:validate_plan(p)
    require(len(plans)==8 and sum(p['planned_commands'] for p in plans)==80,'exact fixed-H2 eight-arm budget')
    output=Path(output).absolute();require(not output.exists(),'exclusive NEW physical output');output.mkdir(exist_ok=False)
    write(output/'execution_protocol.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected_sha,
        compiler_report_sha256=protocol['compiler_report_sha256'],encode_terminals=encode_terminals))
    rows=[];failures=[];encodings=[];env=None;meter=None;model=None;before=None;checked=None;host=None;restore=[]
    stats=empty_encoding_stats()
    with (output/'physical_budget.events.jsonl').open('x') as stream:
        def journal(e):stream.write(io.canonical(e)+'\n');stream.flush();os.fsync(stream.fileno())
        try:
            from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
            from experiments.spectral_reachability.run_pusht_sampling_comparison_20261027_v2 import verify_rgb_context,official_checked
            env=base.gym.make('swm/PushT-v1',render_mode='rgb_array')
            evaluator=Path(env.unwrapped.eval_state.__code__.co_filename).resolve()
            require(str(evaluator) in protocol['sources_sha256'] and io.sha(evaluator.read_bytes())==protocol['sources_sha256'][str(evaluator)],'source-sealed installed official evaluator')
            require(tuple(env.unwrapped.variation_space['agent']['color'].value.tolist())==(65,105,225),'unchanged RoyalBlue contract')
            meter=StepMeter(env,journal)
            roots=base.collect_nonterminal_boundary_contact_roots(meter,count=4,seed=int(protocol['root_seed']),oversample_factor=4)
            rootmap=helpers.validate_roots(roots,protocol['references'],base.root_fingerprint);identities={}
            for p in plans:
                root=rootmap[p['root_sha256']];meter.bind(p)
                def reset():
                    meter.phase='warmup'
                    require(base.root_fingerprint(root)==p['root_sha256'],'original independent physical root/goal')
                    prior=meter.counts.get('warmup',{}).get('attempts',0)
                    frames,history,goal=base.physical._reset_and_warmup(meter,root,history_len=3)
                    require(meter.counts['warmup']['attempts']-prior==15,'exact original warmup15')
                    identity=verify_rgb_context(frames,goal,protocol['references'][p['root_sha256']])
                    identity.update(helpers.validate_reset_history(history,p['predictor_context']['action_history']),
                        state=base.p0.packets.state_of(meter).tolist(),goal=np.asarray(root['goal']).tolist(),
                        source_input_sha256=p['source_input_sha256'])
                    if p['root_sha256'] in identities:require(same(identity,identities[p['root_sha256']]),'paired exact initial root/frame/context')
                    identities[p['root_sha256']]=deepcopy(identity);meter.phase='selected';return identity
                def evaluate(state):return dict(official=official_checked(meter,state,root['goal']),pose=base.physical._pose_label(state,root['goal']))
                row=physical_branch(plan=p,output=output,reset=reset,env=meter,state_of=base.p0.packets.state_of,evaluate=evaluate)
                rows.append(row);verify_branch(p,row)
        except (Exception,KeyboardInterrupt) as exc:failures.append(durable.failure(exc,'physical_setup_or_integrity'))
        finally:
            if env is not None:
                try:env.close()
                except BaseException as exc:failures.append(durable.failure(exc,'env_close'))
    materialize_missing(plans,rows,output)
    resources=meter.receipt() if meter is not None else dict(steps={},resets={})
    physical=dict(schema=NAME+'.physical_report',rows=rows,failures=deepcopy(failures),physical_resources=resources,
        physical_budget_journal_sha256=io.sha((output/'physical_budget.events.jsonl').read_bytes()),
        consumer_prepare_sha256=expected_sha,compiler_report_sha256=protocol['compiler_report_sha256'],contract=contract())
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
            require(io.sha((output/'physical_report.json').read_bytes())==physical_sha,'durable entire physical report before model load')
            for path,h in protocol['model_bindings'].items():require(checked.digest(path)==h,'original frozen endpoint model bytes')
            policy=Path(next(path for path in protocol['model_bindings'] if Path(path).name=='weights.pt')).parent
            host=require_cuda_allocation('cuda');device=torch.device('cuda:0')
            model=load_compatible_pretrained(str(policy),device=device).eval().requires_grad_(False)
            frozen_model(model,device);before=checked.tensor_state_hash(model)
            counter=EncodingCap(stats,event)
            def patch(owner,name,value):restore.append((owner,name,getattr(owner,name)));setattr(owner,name,value)
            def forbidden(name):
                def call(*a,**k):
                    stats['forbidden_'+name+'_attempts']+=1;event(dict(kind='forbidden_'+name));raise ValueError('endpoint encoding forbids '+name)
                return call
            patch(model,'predict',forbidden('predict'));patch(model.action_encoder,'forward',forbidden('action_encoder'))
            for name in ('encoder','projector'):
                original=getattr(model,name).forward
                patch(getattr(model,name),'forward',lambda *a,_name=name,_original=original,**k:counter.dispatch(_name,_original,a,k))
            transform=p0.image_transform()
            def encode(path):
                with Image.open(path) as image:rgb=np.asarray(image.convert('RGB')).copy()
                with torch.no_grad():z=model.encode({'pixels':p0._image_tensor(rgb,transform=transform,device=device)})['emb'][0,0]
                require(z.shape==(192,) and z.device.type=='cuda' and bool(torch.isfinite(z).all()),'finite frozen native endpoint192')
                torch.cuda.synchronize(device);return z.cpu().tolist()
            return encode
        try:encodings=optional_encodings(plans,rows,output,setup)
        except (Exception,KeyboardInterrupt) as exc:failures.append(durable.failure(exc,'encoding_driver'))
        finally:
            restore_hooks(restore,failures)
    source_ok=True;model_ok=None
    try:validate_prepared(prepared,expected_sha)
    except Exception as exc:source_ok=False;failures.append(durable.failure(exc,'final_source_check'))
    if model is not None:
        try:model_ok=before is not None and before==checked.tensor_state_hash(model)
        except Exception as exc:model_ok=False;failures.append(durable.failure(exc,'final_model_check'))
    selected=resources['steps'].get('selected',dict(attempts=0,returns=0))
    steps_ok=selected['attempts']==sum(r['physical_env_step_attempts'] for r in rows)<=80 and selected['returns']==sum(r['physical_env_step_returns'] for r in rows)
    if not steps_ok:failures.append(dict(phase='physical_budget',type='CounterMismatch',message='durable selected branch/meter counts disagree'))
    integrity=not failures and source_ok and model_ok is not False and all(r['status'] in ('completed','observation_unavailable','environment_terminated') and
        'pose' in r and 'official' in r for r in rows)
    fresh=sum(r['endpoint_fresh'] for r in rows)
    encoded=not encode_terminals or (len(encodings)==8 and all(e['status']==('completed' if r['endpoint_fresh'] else 'blocked_stale_endpoint')
        for e,r in zip(encodings,rows,strict=True)) and stats['encoder_attempts']==stats['encoder_completed']==
        stats['projector_attempts']==stats['projector_completed']==fresh and
        stats['forbidden_predict_attempts']==stats['forbidden_action_encoder_attempts']==stats['blocked_module_entries']==0)
    result=dict(schema=NAME+'.report',status='guarded_calibration_recorded' if integrity and encoded else 'incomplete',
        rows=rows,pairs=comparisons(plans,rows),encodings=encodings,failures=failures,physical_resources=resources,
        physical_report_sha256=physical_sha,compiler_report_sha256=protocol['compiler_report_sha256'],consumer_prepare_sha256=expected_sha,
        encoding_stats=stats,optional_encodings_complete=encoded,model_unchanged=model_ok,
        model_loaded=model is not None,model_hash_verified=model_ok is True,
        source_verified=source_ok,host=host,physical_env_step_attempts=selected['attempts'],physical_env_step_returns=selected['returns'],
        planned_selected_path_steps=80,contract=contract(),new_predict_calls=0,P1_admission=False,P3_admission=False,P4_admission=False,efficacy_claim=False)
    write(output/'report.json',result)
    if not integrity or not encoded:raise RuntimeError('incomplete: preserve all physical/encoding partials; no automatic retry')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare','bind','verify','execute-physics'):group.add_argument('--'+name,action='store_true')
    parser.add_argument('--project-root',type=Path,default=PROJECT);parser.add_argument('--output',type=Path)
    parser.add_argument('--prepared',type=Path);parser.add_argument('--prepare-sha256');parser.add_argument('--compiler-output',type=Path)
    parser.add_argument('--report-sha256');parser.add_argument('--encode-terminals',action='store_true');args=parser.parse_args()
    require(args.execute_physics or not args.encode_terminals,'encoding only explicit main physical execution')
    if args.prepare:
        require(args.output is not None,'new source output');result=prepare(args.project_root,args.output)
    else:
        require(args.prepared is not None and args.prepare_sha256 is not None,'reviewed prepare path/SHA')
        if args.verify:result=dict(status='verified',stage=validate_prepared(args.prepared,args.prepare_sha256)['stage'])
        elif args.bind:
            require(args.output is not None and args.compiler_output is not None and args.report_sha256 is not None,'actual compiler output/hash and exclusive bind output')
            result=bind_report(args.prepared,args.prepare_sha256,args.compiler_output,args.report_sha256,args.output)
        else:
            require(args.output is not None,'exclusive physical output');result=execute_physics(args.prepared,args.prepare_sha256,args.output,args.encode_terminals)
    print(io.canonical({k:v for k,v in result.items() if k not in ('source_seals','rows','pairs','encodings')}))


if __name__=='__main__':main()
