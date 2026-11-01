"""New GRASP-H1 raw-plan physical consumer. Prepare never imports physics/torch.

Reuse only low-level durable guard/root utilities, never old reports/endpoints.
Out-of-box plans are rejected whole, without reset/dispatch/clipping. No neural
model is loaded, no fresh predictions/encodings, and no old endpoint is reused.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p2_grasp_h1_cuda_runner_20261027_v1 as planning
from experiments.spectral_reachability import p3_matched_physical_20261027_v1 as durable
from experiments.spectral_reachability import p3_quality_cost_audit_20261027_v2 as io
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal,verify_seal

NAME='p2_grasp_h1_guarded_physical_20261027_v1'
PREPARED='results/p2_grasp_h1_cuda_runner_20261027_v1_prepared'
PLANNING_PREPARE_SHA='70f452342f574f94f3171372958d4542d4b5f4a3ce95095382ff8bf1b816d8a2'
DEFAULT_REPORT='results/p2_grasp_h1_cuda_runner_20261027_v1_cuda_main_v1/report.json'
require=io.require
write=planning.write


def same(a,b): return io.canonical(a)==io.canonical(b)


def validate_plan(p):
    require(p['schema']==NAME+'.plan' and p['method'] in planning.METHODS,'new consumer schema/method required')
    require(p['planned_commands']==15 and p['planned_horizon_tokens']==3 and
        p['corrected_search_horizon']==3,'exact fifteen-step plan required')
    require(same(planning.export_actions(p['normalized_actions'])['commands'],p['commands']),'raw normalization/action path differs')
    supported=all(abs(v)<=1 for cmd in p['commands'] for v in cmd)
    require(p['physical_box_supported'] is supported and p['clipped'] is False and
        p['dispatch_status']==('eligible' if supported else 'rejected_raw_out_of_box'),'whole-plan raw-box classification differs')
    require(io.sha(io.canonical(p['commands']).encode())==p['command_sha256'],'raw command hash changed')
    require(io.sha(io.canonical({k:v for k,v in p.items() if k!='selection_sha256'}).encode())==p['selection_sha256'],'selection hash changed')


def plans_from_saved(body,report,report_sha):
    require(report['schema']==planning.NAME+'.report' and report['status']=='h1_calibration_recorded' and
        report['failure'] is None and report['model_unchanged'] is True,'complete actual H1 CUDA report required')
    require(report['prepare_sha256']==PLANNING_PREPARE_SHA and report['protocol_id']==io.sha(io.canonical(body).encode()) and
        same(report['contract'],planning.contract()) and same(body['contract'],planning.contract()),'frozen H1 source/config/goal contract')
    require(report['new_physical_steps']==report['new_independent_roots']==0 and
        report['P0_admission'] is report['P2_admission'] is False,'calibration scope')
    rows=io.index(report['rows'],('root_sha256','method'));roots=body['roots']
    require(len(roots)==len(set(roots))==4 and set(rows)=={(r,m) for r in roots for m in planning.METHODS},'all eight root-arm slots required')
    require(same(report['paired_h1'],planning.paired_h1(report['rows'],roots)),'H1 pair accounting differs')
    packet_index=io.index(body['packets'],('root_sha256',));plans=[]
    for i,root in enumerate(roots):
        packet=packet_index[root,];planning.validate_packet(packet)
        for method in (planning.METHODS if i%2==0 else planning.METHODS[::-1]):
            row=rows[root,method];receipt=row['receipt']
            require(row['status']=='encoded_h1_completed_not_physical' and row['failure'] is None,'all completed CUDA arm receipts required')
            require(same(row['plan'],planning.export_actions(row['plan']['normalized_actions'])),'exact exported normalized/raw plan required')
            expected={}
            for phase in planning.phases(method)+['real_h3_diagnostic']*3:
                expected[phase]=expected.get(phase,0)+1
            require(same(row['transition_counts'],{phase:dict(requested=n,completed=n,failed=0) for phase,n in expected.items()}),'transition phase accounting')
            require(receipt['backward_requested']==receipt['backward_completed']==900,'900 backward requests/completions')
            require(len(receipt['iterations'])==(750 if method=='grasp_h1' else 900),'optimizer iteration census')
            if method=='grasp_h1':
                require(same(receipt['config'],planning.contract()['config']) and
                    receipt['sync_after_iterations_1based']==planning.core.budget(planning.core.Config())['sync_after_iterations_1based'], 'GRASP fixed configuration/sync schedule')
                require(same(receipt['transitions_requested'],dict(soft=2250,sync=450,verification=3)) and
                    same(receipt['transitions_completed'],receipt['transitions_requested']),'GRASP core phase ledger')
                events=receipt['events']
                require(len(events)==2703 and [e['phase'] for e in events]==planning.phases(method) and
                    all(e['status']=='completed' and e['batch_samples']==1 for e in events),'GRASP core transition event census')
            tokens=planning.contract()['processed_context_tokens_per_arm'][method]+9
            for name in ('predictor','action_encoder'):
                require(same(row['module_work'][name],dict(requested_calls=2706,completed_calls=2706,
                    requested_samples=2706,completed_samples=2706,requested_context_tokens=tokens,completed_context_tokens=tokens)), 'actual model hook accounting')
            require(all(v==0 for v in row['module_work']['encoder'].values()),'no upstream fresh encoder calls')
            for key in ('verification',):
                require(planning.finite_array(row[key],(4,192)) and row[key][0]==packet['context'][-1],'H1 verification start/full latent identity')
            cost=float(np.mean((np.asarray(row['verification'][-1])-packet['goal'])**2))
            require(np.isclose(cost,row['h1_terminal_mse'],atol=1e-6,rtol=1e-5),'H1 full-latent MSE recomputation')
            diag=row['real_h3_diagnostic']
            require(diag['included_in_h1_comparison'] is False and planning.finite_array(diag['states'],(4,192)) and
                diag['states'][0]==packet['context'][-1],'separate real-H3 diagnostic identity')
            p=dict(schema=NAME+'.plan',planning_report_sha256=report_sha,planning_protocol_id=report['protocol_id'],
                root_index=i,root_sha256=root,method=method,input_sha256=packet['input_sha256'],
                goal_rgb_sha256=packet['goal_rgb_sha256'],planned_horizon_tokens=3,corrected_search_horizon=3,
                upstream_h1_transitions=2703,upstream_real_h3_diagnostic_transitions=3,upstream_backward_calls=900,
                dispatch_status='eligible' if row['plan']['physical_box_supported'] else 'rejected_raw_out_of_box',**deepcopy(row['plan']))
            p['selection_sha256']=io.sha(io.canonical(p).encode());validate_plan(p);plans.append(p)
    return plans


def verify_planning_journal(path,report):
    """Bounded streaming: actual journal slightly exceeds Evidence's 32MB cap.

    Exact event order, all actions, per-transition phases, module context tokens
    and per-arm final exports are checked, not only a claimed journal digest.
    """
    require(Path(path).is_file() and not Path(path).is_symlink(),'regular planning journal required')
    require(Path(path).stat().st_size<=128*1024*1024,'bounded planning journal')
    expected_order=[(r['root_sha256'],r['method']) for r in report['rows']]
    schedules={m:planning.phases(m)+['real_h3_diagnostic']*3 for m in planning.METHODS}
    rows=io.index(report['rows'],('root_sha256','method'));started=[];active=None;state=None
    h=hashlib.sha256();size=0
    def finish():
        require(state is not None and state['n']==2706 and state['pending'] is None and state['final'],'incomplete planning journal arm')
    with Path(path).open('rb') as stream:
        for sequence,line in enumerate(stream):
            size+=len(line)
            require(sequence<200000 and size<=128*1024*1024 and len(line)<=65536,'bounded journal events')
            h.update(line);e=io.decode(line)
            require(e['sequence']==sequence,'planning journal sequence')
            key=(e['root_sha256'],e['method']);kind=e['kind']
            if kind=='arm_started':
                if active is not None:finish()
                require(len(started)<8 and key==expected_order[len(started)],'planning arm order/census')
                active=key;started.append(key);state=dict(n=0,pending=None,final=False,phase_counts={})
                continue
            require(key==active,'planning event root/arm differs')
            row=rows[key];schedule=schedules[key[1]]
            if kind=='transition_requested':
                n=state['n'];phase=schedule[n] if n<len(schedule) else None
                require(state['pending'] is None and e['index']==n and e['phase']==phase,'planning transition phase/index')
                require(planning.finite_array(e['normalized_action'],(1,10)),'finite submitted normalized action')
                k=state['phase_counts'].get(phase,0);state['phase_counts'][phase]=k+1
                length=1 if phase=='soft' else 3 if phase=='real_h3_diagnostic' else k%3+1
                if phase in ('verification','real_h3_diagnostic'):
                    require(same(e['normalized_action'][0],row['plan']['normalized_actions'][k%3]),'final plan differs from actually verified/diagnosed actions')
                state['pending']=dict(phase=phase,index=n,stage=0,context_tokens=length)
            elif kind in ('module_requested','module_completed'):
                pending=state['pending'];require(pending is not None,'module event outside transition')
                stages=[('module_requested','action_encoder'),('module_completed','action_encoder'),('module_requested','predictor'),('module_completed','predictor')]
                require(pending['stage']<4 and (kind,e['module'])==stages[pending['stage']] and
                    e['batch']==1 and e['context_tokens']==pending['context_tokens'],'actual module order/batch/context tokens')
                pending['stage']+=1
            elif kind=='transition_completed':
                pending=state['pending']
                require(pending is not None and pending['stage']==4 and e['phase']==pending['phase'] and e['index']==pending['index'],'planning transition completion')
                state['pending']=None;state['n']+=1
            elif kind=='final_plan':
                require(state['n']==2703 and state['pending'] is None and not state['final'] and same(e['plan'],row['plan']),'durable final plan')
                state['final']=True
            else:raise ValueError('unexpected/failure planning journal event: '+kind)
    finish();require(started==expected_order,'complete planning journal census')
    require(h.hexdigest()==report['journal_sha256'],'planning journal SHA mismatch')
    return dict(sha256=h.hexdigest(),bytes=size,arms=len(started),transitions=21648)


def source_closure(root,seeds):
    packages={'experiments':root/'experiments','stable_worldmodel':root/'.venv/lib/python3.12/site-packages/stable_worldmodel',
        'stable_pretraining':root/'.venv/lib/python3.12/site-packages/stable_pretraining'}
    pending=list(seeds);seen=set()
    while pending:
        p=Path(pending.pop()).absolute()
        if p in seen:continue
        seen.add(p)
        if p.suffix!='.py':continue
        package=None
        for name,directory in packages.items():
            if p.is_relative_to(directory):package=[name,*p.relative_to(directory).with_suffix('').parts[:-1]];break
        for node in ast.walk(ast.parse(p.read_text())):
            names=[]
            if isinstance(node,ast.Import):names=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                base=node.module or ''
                if node.level:
                    if package is None:continue
                    base='.'.join(package[:len(package)-node.level+1]+([base] if base else []))
                names=[base,*[base+'.'+a.name for a in node.names if a.name!='*']]
            for name in names:
                parts=name.split('.')
                if parts[0] not in packages:continue
                candidate=packages[parts[0]].joinpath(*parts[1:])
                for q in (candidate.with_suffix('.py'),candidate/'__init__.py'):
                    if q.is_file() and q not in seen:pending.append(q);break
    return sorted(seen)


def prepare(root,output,report_path,report_sha):
    root=Path(root).resolve();output=Path(output).absolute();report_path=Path(report_path).absolute()
    require(not output.exists(),'exclusive NEW prepare output required')
    e=io.Evidence(root);report=e.json(report_path,report_sha)
    body,identity=planning.validate_prepared(root/PREPARED,PLANNING_PREPARE_SHA)
    for p,h in body['inputs_sha256'].items():e.raw(Path(p),h)
    for name in ('prepare.json','protocol.json'):e.raw(root/PREPARED/name)
    original=e.json(root/PREPARED/'prepare.json');e.seal(original['source_seal'])
    plans=plans_from_saved(body,report,report_sha)
    started=e.json(report_path.parent/'started.json')
    require(started['protocol_id']==identity and started['prepare_sha256']==PLANNING_PREPARE_SHA and
        same(started['rows'],planning.slots(body['roots'])),'actual started eight-slot census')
    for i,row in enumerate(report['rows']):
        require(same(e.json(report_path.parent/f'arm_{i:02d}.json'),row),'per-arm actual report differs')
    journal_path=report_path.parent/'events.jsonl'
    journal=verify_planning_journal(journal_path,report)
    goals=e.json(root/planning.GOALS,body['inputs_sha256'][str(root/planning.GOALS)])
    bank_path=Path(goals['protocol']['bank_path']);bank=e.json(bank_path,goals['protocol']['bank_sha256'])
    upstream_path=Path(bank['upstream_path']);upstream=e.json(upstream_path,bank['upstream_sha256'])
    require(io.sha(upstream_path.read_bytes())==goals['protocol']['upstream_sha256'],'root reconstruction source')
    refs=io.index(upstream['rows'],('root_sha256',))
    require(set(refs)=={(r,) for r in body['roots']},'physical source roots')
    for packet in body['packets']:
        ref=refs[packet['root_sha256'],]
        require(ref['history_rgb_sha256']==packet['history_rgb_sha256'] and ref['goal_rgb_sha256']==packet['goal_rgb_sha256'],'physical helper original goal must equal independent goal0, never substitute')
    module=root/'experiments/spectral_reachability'
    seeds=[*map(Path,body['sources_sha256']),Path(__file__).resolve(),Path(durable.__file__).resolve(),
        root/'tests'/('test_'+NAME+'.py'),root/'docs'/(NAME+'.md'),
        module/'run_pusht_d2p_p1_physical_calibration.py',root/'experiments/audit_pusht_reachability_horizon.py',
        root/'.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py',
        root/'.venv/lib/python3.12/site-packages/stable_worldmodel/envs/__init__.py']
    sources=source_closure(root,seeds)
    for p in sources:e.raw(p)
    eligible=sum(p['dispatch_status']=='eligible' for p in plans)
    protocol=dict(schema=NAME+'.protocol',planning_report=str(report_path),planning_report_sha256=report_sha,
        planning_protocol_id=identity,planning_prepare_sha256=PLANNING_PREPARE_SHA,
        planning_journal_path=str(journal_path),planning_journal=journal,
        roots=body['roots'],packets=body['packets'],plans=plans,
        root_seed=upstream['protocol']['root_seed'],root_oversample_factor=4,
        max_selected_path_steps=15*eligible,max_warmup_steps=15*eligible,scheduled_arms=8,eligible_arms=eligible,
        selected_path_step_cap_per_arm=15,physical_repetitions=1,
        upstream_h1_transition_samples_charged_once=21624,upstream_diagnostic_samples_charged_once=24,
        upstream_backward_calls_charged_once=7200,inherited_encoding_cost='unknown',
        new_model_loads=0,new_neural_encodes=0,new_predict_samples=0,
        model_bindings=body['model_bindings'],
        feedback='canonical RoyalBlue presence guard before every raw selected-path command; stop on absence/termination/error; no replanning',
        metric_scope='physical real-15-step path of H1-planned commands; not equal-budget H3 optimization',
        inputs_sha256=dict(sorted(e.hashes.items())),sources_sha256={str(p):e.hashes[str(p)] for p in sources},
        P0_admission=False,P2_admission=False,new_independent_roots=0)
    e.recheck();require(planning.digest(journal_path)==journal['sha256'],'planning journal changed during prepare')
    output.mkdir(exist_ok=False);write(output/'protocol.json',protocol)
    protocol_sha=planning.digest(output/'protocol.json')
    seals=[create_seal(files=sources[i:i+40],args_json=io.canonical(dict(protocol_path=str(output/'protocol.json'),protocol_sha256=protocol_sha)),
        output_dir=output/f'source_seal_{i//40}') for i in range(0,len(sources),40)]
    e.recheck()
    receipt=dict(schema=NAME+'.prepare',status='prepared_not_executed',protocol_sha256=protocol_sha,source_seals=seals,
        source_count=len(sources),scheduled_arms=8,eligible_arms=eligible,rejected_arms=8-eligible,
        max_selected_path_steps=15*eligible,max_warmup_steps=15*eligible,new_physical_steps=0,new_predictions=0,
        model_loaded=False,planning_report_sha256=report_sha,P0_admission=False,P2_admission=False)
    write(output/'prepare.json',receipt)
    return receipt


def validate_prepared(prepared,expected_sha):
    prepared=Path(prepared).absolute();raw=(prepared/'prepare.json').read_bytes()
    require(io.sha(raw)==expected_sha,'prepare receipt SHA mismatch');receipt=io.decode(raw)
    require(receipt['status']=='prepared_not_executed' and receipt['model_loaded'] is False,'prepared only receipt')
    require(planning.digest(prepared/'protocol.json')==receipt['protocol_sha256'],'protocol SHA mismatch')
    p=io.decode((prepared/'protocol.json').read_bytes())
    require(p['schema']==NAME+'.protocol' and p['scheduled_arms']==8 and p['physical_repetitions']==1 and
        p['selected_path_step_cap_per_arm']==15 and p['new_predict_samples']==p['new_neural_encodes']==p['new_model_loads']==0,'fixed guarded consumer contract')
    for path,h in p['inputs_sha256'].items():require(planning.digest(path)==h,'input/source changed: '+path)
    require(planning.digest(p['planning_journal_path'])==p['planning_journal']['sha256'],'planning journal changed')
    for seal in receipt['source_seals']:verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
    rows=io.index(p['plans'],('root_sha256','method'))
    require(len(p['roots'])==len(set(p['roots']))==4 and set(rows)=={(r,m) for r in p['roots'] for m in planning.METHODS},'all eight frozen selections')
    for plan in p['plans']:validate_plan(plan)
    return p


def rgb_sha(rgb): return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def check_context(packet,frames,history,goal):
    stack=np.stack(frames);goal=np.asarray(goal);history=np.asarray(history)
    require(stack.shape==(3,224,224,3) and stack.dtype==np.uint8 and goal.shape==(224,224,3) and goal.dtype==np.uint8,'actual RGB history/goal shape')
    require(rgb_sha(stack)==packet['history_rgb_sha256'] and rgb_sha(goal)==packet['goal_rgb_sha256'],'source RGB/independent goal0 identity differs')
    require(history.shape==(3,2) and np.isfinite(history).all(),'actual raw warmup action history')
    raw=np.repeat(history[1:,None,:],5,axis=1)
    norm=((raw.astype(np.float32)-np.array(planning.MEAN,np.float32))/np.array(planning.STD,np.float32)).reshape(2,10)
    require(np.allclose(norm,np.asarray(packet['history_actions']),rtol=0,atol=2e-6),'actual between-frame normalized history differs')
    return dict(root_sha256=packet['root_sha256'],history_rgb_sha256=rgb_sha(stack),goal_rgb_sha256=rgb_sha(goal),
        input_sha256=packet['input_sha256'],history=history.tolist(),normalized_between_history=norm.tolist(),
        latent_scope='inherited sealed full192; no neural re-encoding in physical consumer')


class StepMeter:
    """Counts reset/warmup/setup separately from selected-path physical calls."""
    def __init__(self,env,sink,limits=None):
        self.env=env;self.sink=sink;self.phase='root_collection';self.counts={};self.reset_counts={};self.limits=limits or {}
        self.binding={}
    def __getattr__(self,name): return getattr(self.env,name)
    def call(self,method,table,*args,**kwargs):
        c=table.setdefault(self.phase,dict(attempts=0,returns=0))
        if method=='step' and self.phase in self.limits and c['attempts']>=self.limits[self.phase]:
            self.sink(dict(kind='env_step_blocked',phase=self.phase,reason='budget',**self.binding))
            raise ValueError('physical step budget exhausted before dispatch')
        c['attempts']+=1
        self.sink(dict(kind='env_'+method+'_attempt',phase=self.phase,command=np.asarray(args[0]).tolist() if method=='step' else None,**self.binding))
        out=getattr(self.env,method)(*args,**kwargs);c['returns']+=1
        self.sink(dict(kind='env_'+method+'_returned',phase=self.phase,**self.binding))
        return out
    def step(self,*a,**kw):return self.call('step',self.counts,*a,**kw)
    def reset(self,*a,**kw):return self.call('reset',self.reset_counts,*a,**kw)


def empty_row(plan,status):
    return dict(root_index=plan['root_index'],root_sha256=plan['root_sha256'],method=plan['method'],
        selection_sha256=plan['selection_sha256'],commands=deepcopy(plan['commands']),
        status=status,failures=[],repetition=0,planned_commands=15,planned_horizon_tokens=3,
        physical_env_step_attempts=0,physical_env_step_returns=0,attempted_commands=0,completed_commands=0,
        remaining_commands_not_executed=15,full_horizon_completed=False,endpoint_fresh=False,
        endpoint_scope='no_endpoint',terminal_rgb_path=None,trace=[])


def physical_branch(*,plan,output,reset,env,state_of,evaluate):
    validate_plan(plan);output=Path(output)
    if plan['dispatch_status']!='eligible':
        row=empty_row(plan,'rejected_raw_out_of_box')
        write(output/f'root_{plan["root_index"]}_{plan["method"]}.rejected.json',row)
        return row
    row=durable.physical_branch(plan=plan,repetition=0,output=output,reset=reset,env=env,state_of=state_of,evaluate=evaluate)
    events=[]
    if row.get('journal_path') and Path(row['journal_path']).exists():
        events=[io.decode(line) for line in Path(row['journal_path']).read_bytes().splitlines()]
    images=[e for e in events if 'rgb_path' in e]
    row['commands']=deepcopy(plan['commands'])
    row['endpoint_fresh']=bool(images and row.get('terminal_rgb_path') and images[-1].get('primitive_step',0)==row['physical_env_step_returns'] and
        row['physical_env_step_attempts']==row['physical_env_step_returns'] and row['status'] in ('completed','observation_unavailable','environment_terminated'))
    row['endpoint_scope']='full_15' if row['full_horizon_completed'] else 'guard_partial_or_terminated' if row['endpoint_fresh'] else 'unknown_or_stale_endpoint'
    write(output/f'root_{plan["root_index"]}_{plan["method"]}.endpoint.json',row)
    return row


def verify_branch(plan,row):
    require(row['root_sha256']==plan['root_sha256'] and row['method']==plan['method'] and row['selection_sha256']==plan['selection_sha256'],'physical plan identity')
    a,b=io.count(row['physical_env_step_attempts']),io.count(row['physical_env_step_returns'])
    require(0<=b<=a<=15 and a==row['attempted_commands'] and b==row['completed_commands'],'attempt/return accounting')
    require(same(row['commands'],plan['commands']),'original raw commands preserved')
    if row['status'] in ('rejected_raw_out_of_box','blocked'):
        require(a==b==0 and not row['endpoint_fresh'] and not row['full_horizon_completed'],'non-dispatched branch accounting')
        return True
    path=Path(row['journal_path']);require(planning.digest(path)==row['journal_sha256'],'physical journal hash')
    events=[io.decode(line) for line in path.read_bytes().splitlines()]
    attempts=[e for e in events if e['kind']=='attempt'];returns=[e for e in events if e['kind']=='step_returned']
    require(len(attempts)==a and len(returns)==b and [e['primitive_step'] for e in attempts]==list(range(1,a+1)) and
        [e['primitive_step'] for e in returns]==list(range(1,b+1)),'durable command attempt/return order')
    require(all(same(e['command'],plan['commands'][i]) for i,e in enumerate(attempts)),'exact selected raw path')
    obs=[e for e in events if e['kind']=='observation']
    require(same([{k:v for k,v in e.items() if k not in ('rgb_path','rgb_file_sha256')} for e in obs],row['trace']),'observation trace journal differs')
    for e in events:
        if 'rgb_path' in e:require(planning.digest(e['rgb_path'])==e['rgb_file_sha256'],'saved RGB file hash')
    if row['full_horizon_completed']:require(b==15 and row['status']=='completed' and row['endpoint_fresh'],'full fifteen-step completion only')
    return True


def materialize_missing(plans,rows,output):
    existing={(r['root_sha256'],r['method']) for r in rows}
    for plan in plans:
        if (plan['root_sha256'],plan['method']) in existing:continue
        row=empty_row(plan,'rejected_raw_out_of_box' if plan['dispatch_status']!='eligible' else 'blocked')
        write(Path(output)/f'root_{plan["root_index"]}_{plan["method"]}.undispatched.json',row);rows.append(row)


def comparisons(roots,rows):
    index=io.index(rows,('root_sha256','method'));out=[]
    for root in roots:
        a,b=(index[root,m] for m in planning.METHODS)
        context='context_identity' in a and 'context_identity' in b and same(a['context_identity'],b['context_identity'])
        full=context and all(r['full_horizon_completed'] and r['endpoint_fresh'] for r in (a,b))
        partial=context and all(r['endpoint_fresh'] for r in (a,b)) and a['physical_env_step_returns']==b['physical_env_step_returns'] and not full
        out.append(dict(root_sha256=root,classification='full-path' if full else 'guard-partial' if partial else 'unmatched',
            steps=[a['physical_env_step_returns'],b['physical_env_step_returns']],arm_status={r['method']:r['status'] for r in (a,b)},
            grasp_minus_action_adam_pose_cost=a['pose']['pose_cost']-b['pose']['pose_cost'] if full and all('pose' in r for r in (a,b)) else None,
            scope='complete matched actual15 only; partials/rejected not pooled; H1-planned paths, not equal-budget H3',P2_admission=False))
    return out


def execute_physics(prepared,prepare_sha,output):
    p=validate_prepared(prepared,prepare_sha);output=Path(output).absolute();output.mkdir(exist_ok=False)
    rows=[];failures=[];env=None;meter=None
    write(output/'started.json',dict(prepare_sha256=prepare_sha,plans=p['plans'],new_predict_samples=0))
    with planning.Journal(output/'setup.events.jsonl') as sink:
        try:
            # Persist all rejections BEFORE constructing an environment.
            for plan in p['plans']:
                if plan['dispatch_status']!='eligible':
                    rows.append(physical_branch(plan=plan,output=output,reset=None,env=None,state_of=None,evaluate=None))
            if any(plan['dispatch_status']=='eligible' for plan in p['plans']):
                import gymnasium as gym
                import stable_worldmodel  # noqa: F401, environment registration only
                from experiments.audit_pusht_reachability_horizon import collect_nonterminal_boundary_contact_roots
                from experiments.pusht_protocol_contracts import root_fingerprint
                from experiments.spectral_reachability.run_pusht_d2p_p1_physical_calibration import _reset_and_warmup,_pose_label
                from experiments.spectral_reachability.pusht_d2p_packet_contract import state_of
                from experiments.spectral_reachability.audit_pusht_sealed_task_contract import labels
                env=gym.make('swm/PushT-v1',render_mode='rgb_array');meter=StepMeter(env,sink,
                    limits=dict(selected=p['max_selected_path_steps'],warmup=p['max_warmup_steps']))
                evaluator=Path(env.unwrapped.eval_state.__code__.co_filename).resolve()
                require(str(evaluator) in p['sources_sha256'] and planning.digest(evaluator)==p['sources_sha256'][str(evaluator)],'evaluator source seal')
                require(tuple(env.unwrapped.variation_space['agent']['color'].value.tolist())==(65,105,225),'canonical RoyalBlue guard contract')
                roots=collect_nonterminal_boundary_contact_roots(meter,count=4,seed=int(p['root_seed']),oversample_factor=4)
                rootmap={root_fingerprint(r):r for r in roots}
                require(len(roots)==len(rootmap)==4 and set(rootmap)==set(p['roots']),'same four source roots, no replacement')
                identities={}
                for plan in p['plans']:
                    if plan['dispatch_status']!='eligible':continue
                    root=rootmap[plan['root_sha256']];packet=next(x for x in p['packets'] if x['root_sha256']==plan['root_sha256'])
                    meter.binding=dict(root_sha256=plan['root_sha256'],method=plan['method'])
                    counts_before=deepcopy(meter.counts)
                    def reset():
                        require(root_fingerprint(root)==plan['root_sha256'],'root state/goal fingerprint changed')
                        meter.phase='warmup';initial=meter.counts.get('warmup',{}).get('attempts',0)
                        frames,history,goal=_reset_and_warmup(meter,root,history_len=3)
                        require(meter.counts.get('warmup',{}).get('attempts',0)-initial==15,'exact actual fifteen warmup steps')
                        identity=check_context(packet,frames,history,goal)
                        identity.update(state=state_of(meter).tolist(),goal=np.asarray(root['goal']).tolist())
                        if plan['root_sha256'] in identities:require(same(identity,identities[plan['root_sha256']]),'paired actual source context differs')
                        identities[plan['root_sha256']]=deepcopy(identity);meter.phase='selected'
                        return identity
                    def evaluate(state):
                        official=labels(state,root['goal']);success,distance=env.unwrapped.eval_state(np.asarray(root['goal']),np.asarray(state))
                        require(bool(success)==official['official_success'] and np.isclose(distance,official['official_state_distance'],rtol=0,atol=1e-10),'installed official evaluator agreement')
                        return dict(official=official,pose=_pose_label(state,root['goal']))
                    row=physical_branch(plan=plan,output=output,reset=reset,env=meter,state_of=state_of,evaluate=evaluate)
                    row['environment_step_counts']={phase:{k:meter.counts.get(phase,{}).get(k,0)-counts_before.get(phase,{}).get(k,0)
                        for k in ('attempts','returns')} for phase in ('warmup','selected')}
                    rows.append(row)
                    write(output/f'root_{plan["root_index"]}_{plan["method"]}.budget.json',dict(root_sha256=plan['root_sha256'],
                        method=plan['method'],environment_step_counts=row['environment_step_counts']))
                    verify_branch(plan,row)
        except (Exception,KeyboardInterrupt) as exc:failures.append(durable.failure(exc,'physical_setup_or_branch'))
        finally:
            if env is not None:
                try:env.close()
                except Exception as exc:failures.append(durable.failure(exc,'close'))
    materialize_missing(p['plans'],rows,output)
    index=io.index(rows,('root_sha256','method'));rows=[index[x['root_sha256'],x['method']] for x in p['plans']]
    try:validate_prepared(prepared,prepare_sha);source_ok=True
    except Exception as exc:source_ok=False;failures.append(durable.failure(exc,'final_source_check'))
    selected_attempts=sum(r['physical_env_step_attempts'] for r in rows);selected_returns=sum(r['physical_env_step_returns'] for r in rows)
    step_counts=meter.counts if meter else {}
    try:
        require(step_counts.get('selected',dict(attempts=0,returns=0))==dict(attempts=selected_attempts,returns=selected_returns),'independent selected step meter differs')
        require(selected_attempts<=p['max_selected_path_steps'] and step_counts.get('warmup',{}).get('attempts',0)<=p['max_warmup_steps'],'physical path/warmup budget')
    except Exception as exc:failures.append(durable.failure(exc,'physical_meter_check'))
    allowed=('completed','observation_unavailable','environment_terminated','rejected_raw_out_of_box')
    integrity=source_ok and not failures and all(r['status'] in allowed for r in rows)
    report=dict(schema=NAME+'.report',status='guarded_calibration_recorded' if integrity else 'integrity_fail',rows=rows,
        pairs=comparisons(p['roots'],rows),failures=failures,source_verified=source_ok,
        consumer_prepare_sha256=prepare_sha,planning_report_sha256=p['planning_report_sha256'],
        physical_env_step_attempts=selected_attempts,physical_env_step_returns=selected_returns,
        all_environment_step_counts=step_counts,all_environment_reset_counts=meter.reset_counts if meter else {},
        total_environment_step_attempts=sum(c['attempts'] for c in step_counts.values()),
        total_environment_step_returns=sum(c['returns'] for c in step_counts.values()),
        scheduled_arms=8,eligible_arms=p['eligible_arms'],rejected_arms=8-p['eligible_arms'],
        upstream_h1_transition_samples_charged_once=21624,upstream_diagnostic_samples_charged_once=24,
        upstream_backward_calls_charged_once=7200,inherited_encoding_cost='unknown',
        new_predict_samples=0,new_neural_encodes=0,new_model_loads=0,new_independent_roots=0,
        full_pipeline_equal_compute=False,P0_admission=False,P2_admission=False,efficacy_claim=False)
    write(output/'report.json',report)
    if not integrity:raise RuntimeError('physical integrity failed; all slots/partials retained, no retry')
    return report


def main():
    parser=argparse.ArgumentParser(__doc__);action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare',action='store_true');action.add_argument('--execute-physics',action='store_true')
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--planning-report',type=Path,default=Path(DEFAULT_REPORT))
    parser.add_argument('--planning-report-sha256');parser.add_argument('--prepared',type=Path)
    parser.add_argument('--prepare-sha256');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.prepare:
        require(planning.hexsha(args.planning_report_sha256),'explicit actual planning report SHA required')
        prepare(args.root,args.output,args.planning_report,args.planning_report_sha256)
        print(io.canonical(dict(prepare=str(args.output.absolute()/'prepare.json'),sha256=planning.digest(args.output/'prepare.json'))))
    else:
        require(args.prepared and args.prepare_sha256,'reviewed physical prepare path/SHA required')
        execute_physics(args.prepared,args.prepare_sha256,args.output)


if __name__=='__main__':main()
