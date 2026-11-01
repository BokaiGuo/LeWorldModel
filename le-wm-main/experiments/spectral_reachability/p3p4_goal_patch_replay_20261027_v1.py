"""Fresh double CUDA replay of goal-metric common-bank selected winners.

Main-thread predeclared under authorized implementation. No model/physics in
prepare, verify or selected_actions. No old physical endpoint is consumed.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import p3p4_goal_patch_metric_20261027_v1 as metric
from experiments.spectral_reachability import pusht_composition_cuda_replay_v2 as adapter
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal

NAME='p3p4_goal_patch_replay_20261027_v1'
PROJECT=Path(__file__).resolve().parents[2]
METRIC_PREPARED='results/p3p4_goal_patch_metric_20261027_v1_prepared01'
METRIC_SHA='c1447c9b0170a38af180498cfea2d0ad31cabcf720d9da23dc1dcc1068aa6de8'
METRIC_RUN='results/p3p4_goal_patch_metric_20261027_v1_cuda_main01'
ARMS=('old_Q','goal_patch_Q')
io,require,write=metric.io,metric.require,metric.write
digest=metric.old.digest


def contract():
    return dict(arms=list(ARMS),roots=4,branches=8,replays='fresh whole then independently segmented from original context',
        exact='entire three latent boundaries and two normalized action packets; also equal saved candidate',
        selection='unchanged complete common bank, cost then horizon then index; no reselection',
        caps='per path H; per arm 2H; per root sum(2H); global36 for each neural entry type',
        predict_samples=36,action_encoder_samples=36,context_tokens_each=108,visual_encoder_samples=0,
        jacobians=0,vjp_directions=0,execution_svds=0,validation_saved_array_svds_per_reconstruction=16,
        inherited='planner actual resources and metric80 encoder+80 projector+8 SVD remain separate shared costs; cached context cost unknown, not zero',
        predict_includes='native predictor and pred_proj; not separately hooked or FLOP measured',
        no_refunds=True,no_padding=True,total_resource_matched=False,new_physical_steps=0,
        P1_admission=False,P3_admission=False,P4_admission=False,efficacy_claim=False)


def read_json(path):return io.decode(Path(path).read_bytes())


def validate_metric(root):
    """Rebuild all four metrics and the WHOLE union, never just winners (16 SVD)."""
    root=Path(root).resolve();prepared=root/METRIC_PREPARED;run=root/METRIC_RUN
    p=metric.validate_prepared(prepared,METRIC_SHA);binding=read_json(run/'execution_protocol.json')
    require(binding==dict(prepared=str(prepared),prepare_sha256=METRIC_SHA),'metric execution binding')
    r=read_json(run/'report.json')
    require(r['schema']==metric.NAME+'.report' and r['status']=='model_only_metric_compiled' and r['failure'] is None and r['model_unchanged'] is True,'complete unchanged metric compilation')
    require(r['preparation_sha256']==METRIC_SHA and all(r[k] is False for k in ('P1_admission','P3_admission','P4_admission','physical_ready','root_cause_established','efficacy_claim')),'metric scope')
    counts=dict(actual_encoder_calls=80,actual_encoder_samples=80,completed_encoder_calls=80,
        actual_existing_projector_calls=80,completed_existing_projector_calls=80,actual_neural_module_entries=160,
        attempted_metric_svds=8,completed_metric_svds=8,forbidden_predict_attempts=0,new_predict_calls=0,new_jacobians=0,new_vjp_directions=0)
    require(all(r[k]==v for k,v in counts.items()),'metric resource counts')
    events=[io.decode(line.encode()) for line in (run/'compile.events.jsonl').read_text().splitlines()]
    require(io.sha((run/'compile.events.jsonl').read_bytes())==r['compile_journal_sha256'],'compile journal bytes')
    for kind,n in (('encoder',80),('existing_projector',80)):
        actual=[e for e in events if e['kind']==kind]
        expected=[dict(kind=kind,index=i,completed=c) for i in range(n) for c in (False,True)]
        require(actual==expected,'complete ordered compilation entry journal')
    require(len([e for e in events if e['kind']=='svd' and not e['completed']])==8 and
        len([e for e in events if e['kind']=='svd' and e['completed']])==8,'metric SVD journal')
    metric.old.selected_actions(p['planner_output'])
    prior=read_json(Path(p['planner_output'])/'report.json');rows=io.index(r['rows'],('root_index','root_sha256'))
    require(set(rows)=={(i,v['root_sha256']) for i,v in enumerate(p['packets'])},'four metric roots exactly')
    plans=[]
    for i,packet in enumerate(p['packets']):
        row=rows[i,packet['root_sha256']];goal=p['goals'][i]
        require(row==read_json(run/f'root_{i}.json') and row['catalog']==goal['catalog'] and row['actual_encoder_samples']==20 and row['physical_ready'] is False,'durable root metric witness')
        rebuilt=metric.compile_metric(row['goal_vectors'],goal['legacy_goal_vectors'])
        require(rebuilt==row['metric'],'reconstructed metric')
        bank={}
        for source in prior['rows']:
            if source['root_sha256']!=packet['root_sha256']:continue
            for c in source['search']['candidates']:
                key=(c['horizon'],c['index'])
                require(key not in bank or bank[key]==c,'shared complete candidate identity')
                bank[key]=c
        selection=metric.select_common_bank(packet,rebuilt,[bank[k] for k in sorted(bank)])
        require(selection==row['selection'],'entire common bank and both Q costs/winners')
        for arm in (ARMS if i%2==0 else ARMS[::-1]):
            winner=selection['winners'][arm]
            plan=dict(root_index=i,root_sha256=packet['root_sha256'],arm=arm,**deepcopy(winner),
                root_context=packet['root_context'],goal=packet['goal'],old_projection=packet['projection'],
                projection=rebuilt['projection'],input_sha256=packet['input_sha256'],
                common_bank_candidates=selection['common_candidates'],bank_cost_evaluations=selection['cost_evaluations'],bank_key_comparisons=selection['key_comparisons'],
                bank_sha256=digest(selection),metric_sha256=digest(rebuilt),metric_report_sha256=io.sha((run/'report.json').read_bytes()))
            plan['selection_sha256']=digest(plan);plans.append(plan)
    require(sum(2*x['horizon'] for x in plans)==36,'fixed selected replay budget')
    bindings={str(run/name):io.sha((run/name).read_bytes()) for name in
        ['report.json','execution_protocol.json','compile.events.jsonl']+[f'root_{i}.json' for i in range(4)]}
    return plans,p,bindings,dict(planner=prior['resources'],metric=counts,cached_input_history_cost='unknown; excluded not zero')


def prepare(root,output):
    root=Path(root).resolve();output=Path(output).absolute();require(not output.exists(),'exclusive new preparation')
    plans,p,bindings,inherited=validate_metric(root)
    sources=metric.old.prior.physical.source_closure(root,[Path(__file__).resolve(),root/'tests'/('test_'+NAME+'.py'),root/'innerdocs'/(NAME+'.md')])
    protocol=dict(schema=NAME+'.protocol',contract=contract(),project_root=str(root),plans=plans,bindings=bindings,
        model_bindings=p['model_bindings'],inherited_resources=inherited,sources_sha256={str(s):io.sha(s.read_bytes()) for s in sources})
    output.mkdir(exist_ok=False);write(output/'protocol.json',protocol)
    seals=[create_seal(files=sources[i:i+40],args_json=io.canonical(dict(protocol_sha256=digest(protocol))),output_dir=output/f'source_seal_{i//40}') for i in range(0,len(sources),40)]
    receipt=dict(status='prepared_not_executed',protocol_sha256=io.sha((output/'protocol.json').read_bytes()),source_seals=seals,contract=contract())
    write(output/'prepare.json',receipt);return receipt


def validate_prepared(prepared,sha):
    prepared=Path(prepared).absolute();require(io.sha((prepared/'prepare.json').read_bytes())==sha,'prepare SHA')
    receipt=read_json(prepared/'prepare.json');require(io.sha((prepared/'protocol.json').read_bytes())==receipt['protocol_sha256'],'protocol bytes')
    p=read_json(prepared/'protocol.json');require(p['schema']==NAME+'.protocol' and p['contract']==contract(),'frozen replay contract')
    for path,h in {**p['bindings'],**p['sources_sha256']}.items():require(io.sha(Path(path).read_bytes())==h,'bound source/input drift: '+path)
    for s in receipt['source_seals']:verify_seal(output_dir=s['output_dir'],expected_manifest_sha256=s['manifest_sha256'])
    plans,source,bindings,inherited=validate_metric(p['project_root'])
    require(plans==p['plans'] and bindings==p['bindings'] and source['model_bindings']==p['model_bindings'] and inherited==p['inherited_resources'],'complete source reconstruction')
    return p


class ReplayCap:
    """Nonrefundable, pred-dispatch path/arm/root/global cap; durable failures."""
    def __init__(self,model,plans,journal):
        self.model,self.plans,self.journal=model,plans,journal;self.role=None;self.events=[];self.blocked=0;self.forbidden=0

    @contextmanager
    def scope(self,plan,path):
        require(self.role is None and path in ('whole','segmented'),'explicit independent replay scope')
        self.role=(plan['root_index'],plan['arm'],path,plan['horizon'])
        try:yield
        finally:self.role=None

    def dispatch(self,kind,callback,args,kwargs):
        require(self.role is not None,'unscoped neural call');i,arm,path,h=self.role;x=args[0]
        require(getattr(getattr(x,'device',None),'type',None)=='cuda' and tuple(x.shape)==((1,3,192) if kind=='predict' else (1,3,10)),'single CUDA full-context input')
        entries=[e for e in self.events if e['kind']==kind]
        rootcap=sum(2*p['horizon'] for p in self.plans if p['root_index']==i)
        if len(entries)>=36 or sum(e['root_index']==i for e in entries)>=rootcap or sum(e['root_index']==i and e['arm']==arm for e in entries)>=2*h or sum(e['root_index']==i and e['arm']==arm and e['path']==path for e in entries)>=h:
            self.blocked+=1;self.journal(dict(kind='blocked',role=list(self.role),entry=kind));raise ValueError('replay cap before dispatch')
        e=dict(kind=kind,index=len(self.events),root_index=i,arm=arm,path=path,context_tokens=3,completed=False)
        self.events.append(e);self.journal(deepcopy(e))
        try:result=callback(*args,**kwargs);e['completed']=True;return result
        except BaseException as exc:e['error']=type(exc).__name__+': '+str(exc);raise
        finally:self.journal(deepcopy(e))

    def __enter__(self):
        self.original=(self.model.predict,self.model.action_encoder.forward,self.model.encoder.forward)
        self.model.predict=lambda *a,**kw:self.dispatch('predict',self.original[0],a,kw)
        self.model.action_encoder.forward=lambda *a,**kw:self.dispatch('action_encoder',self.original[1],a,kw)
        def forbid(*a,**kw):
            self.forbidden+=1;self.journal(dict(kind='forbidden_visual_encoder'));raise ValueError('no visual encoder in cached replay')
        self.model.encoder.forward=forbid;return self

    def __exit__(self,*exc):self.model.predict,self.model.action_encoder.forward,self.model.encoder.forward=self.original


def replay_plan(plan,replay,guard,journal):
    row=dict(root_index=plan['root_index'],root_sha256=plan['root_sha256'],arm=plan['arm'],selection_sha256=plan['selection_sha256'],status='attempted')
    journal(deepcopy(row))
    try:
        with guard.scope(plan,'whole'):whole=replay(deepcopy(plan['root_context']),plan['actions'])
        row['whole_context']=whole;journal(deepcopy(row))
        segmented=deepcopy(plan['root_context']);segments=[]
        with guard.scope(plan,'segmented'):
            for packet in np.asarray(plan['actions']).reshape(-1,5,2):
                segmented=replay(segmented,packet);segments.append(deepcopy(segmented));row['segments']=deepcopy(segments);journal(deepcopy(row))
        require(whole==segmented==plan['context'],'exact saved/whole/independently segmented full context')
        for key,q in (('old_Q_cost',plan['old_projection']),('goal_patch_Q_cost',plan['projection'])):
            require(metric.native_cost(whole['latent_history'][-1],plan['goal'],q)==plan[key],'both Q replay witnesses')
        row.update(status='completed',segmented_context=segmented,horizon=plan['horizon'],actions=plan['actions'],old_Q_cost=plan['old_Q_cost'],goal_patch_Q_cost=plan['goal_patch_Q_cost'])
    except (Exception,KeyboardInterrupt) as exc:row.update(status='failed',failure=dict(type=type(exc).__name__,message=str(exc)))
    journal(deepcopy(row));return row


def validate_ledger(events,plans,rows,journal):
    expected=[]
    for plan in plans:
        for path in ('whole','segmented'):
            for _ in range(plan['horizon']):
                for kind in ('action_encoder','predict'):
                    expected.append(dict(kind=kind,index=len(expected),root_index=plan['root_index'],arm=plan['arm'],path=path,context_tokens=3,completed=True))
    require(events==expected,'exact complete neural census/order')
    require([v for v in journal if v.get('kind') in ('action_encoder','predict')]==[v for e in events for v in (dict(e,completed=False),e)],'durable exact neural attempt/completion census')
    require(not any(v.get('kind') in ('blocked','forbidden_visual_encoder') for v in journal),'no blocked/forbidden calls')
    require(len(rows)==8,'all eight fresh replay branches')
    for p,r in zip(plans,rows,strict=True):
        require(r['status']=='completed' and r['selection_sha256']==p['selection_sha256'] and (r['root_index'],r['root_sha256'],r['arm'])==(p['root_index'],p['root_sha256'],p['arm']),'complete exact root-arm pairing')
        require(r['horizon']==p['horizon'] and r['actions']==p['actions'] and r['whole_context']==r['segmented_context']==r['segments'][-1]==p['context'] and len(r['segments'])==p['horizon'],'entire context/action/horizon witness')
        require(all(r[k]==p[k] for k in ('old_Q_cost','goal_patch_Q_cost')),'old/new Q witnesses')


def execute_cuda(prepared,sha,output):
    p=validate_prepared(prepared,sha);output=Path(output).absolute();require(not output.exists(),'exclusive NEW replay execution')
    output.mkdir(exist_ok=False);write(output/'execution_protocol.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=sha))
    rows=[];events=[];failure=None;before=after=None;host=None
    with (output/'replay.events.jsonl').open('x') as stream:
        def journal(e):stream.write(io.canonical(e)+'\n');stream.flush();os.fsync(stream.fileno())
        try:
            from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
            from experiments.checkpoint_compat import load_compatible_pretrained
            from experiments.gpu_preflight import require_cuda_allocation
            import torch
            for path,h in p['model_bindings'].items():require(checked.digest(path)==h,'original model bytes')
            host=require_cuda_allocation('cuda');device=torch.device('cuda:0')
            policy=Path(next(path for path in p['model_bindings'] if Path(path).name=='weights.pt')).parent
            model=load_compatible_pretrained(str(policy),device=device).eval().requires_grad_(False)
            before=checked.tensor_state_hash(model);replay=adapter.CudaReplay(model,device)
            with ReplayCap(model,p['plans'],journal) as guard:
                for plan in p['plans']:
                    row=replay_plan(plan,replay,guard,journal);rows.append(row);write(output/f'root_{plan["root_index"]}.{plan["arm"]}.json',row)
                    if row['status']!='completed':raise RuntimeError('branch failed; no retry/refund')
            events=guard.events;torch.cuda.synchronize(device);after=checked.tensor_state_hash(model)
        except (Exception,KeyboardInterrupt) as exc:
            failure=dict(type=type(exc).__name__,message=str(exc));journal(dict(kind='execution_failure',**failure))
            if 'guard' in locals():events=guard.events
    for plan in p['plans'][len(rows):]:
        row=dict(root_index=plan['root_index'],root_sha256=plan['root_sha256'],arm=plan['arm'],selection_sha256=plan['selection_sha256'],status='blocked',failure=failure)
        rows.append(row);write(output/f'root_{plan["root_index"]}.{plan["arm"]}.json',row)
    try:
        validate_prepared(prepared,sha)
        validate_ledger(events,p['plans'],rows,[io.decode(s.encode()) for s in (output/'replay.events.jsonl').read_text().splitlines()])
        require(before is not None and before==after,'unchanged frozen model')
    except Exception as exc:failure=failure or dict(type=type(exc).__name__,message=str(exc))
    result=dict(schema=NAME+'.report',status='cuda_completed' if failure is None else 'incomplete',failure=failure,
        rows=rows,events=events,model_unchanged=before is not None and before==after,host=host,contract=contract(),
        journal_sha256=io.sha((output/'replay.events.jsonl').read_bytes()),inherited_resources=p['inherited_resources'],
        resources={k:dict(attempted=sum(e['kind']==k for e in events),completed=sum(e['kind']==k and e['completed'] for e in events),context_tokens=3*sum(e['kind']==k for e in events)) for k in ('predict','action_encoder')},
        P1_admission=False,P3_admission=False,P4_admission=False,physical_ready=failure is None,new_physical_steps=0)
    write(output/'report.json',result)
    if failure is not None:raise RuntimeError('incomplete replay retained; no automatic retry')
    return result


def selected_actions(output):
    """Readonly complete-bank + fresh-witness exporter, present BEFORE sealing."""
    output=Path(output).absolute();b=read_json(output/'execution_protocol.json');p=validate_prepared(b['prepared'],b['prepare_sha256']);r=read_json(output/'report.json')
    require(r['schema']==NAME+'.report' and r['status']=='cuda_completed' and r['failure'] is None and r['model_unchanged'] is True and r['physical_ready'] is True,'fresh complete CUDA replay required')
    require(r['contract']==contract() and all(r[k] is False for k in ('P1_admission','P3_admission','P4_admission')) and r['new_physical_steps']==0 and r['inherited_resources']==p['inherited_resources'],'scope/inherited costs')
    require(io.sha((output/'replay.events.jsonl').read_bytes())==r['journal_sha256'],'replay journal binding')
    journal=[io.decode(s.encode()) for s in (output/'replay.events.jsonl').read_text().splitlines()]
    validate_ledger(r['events'],p['plans'],r['rows'],journal)
    require(r['resources']=={k:dict(attempted=36,completed=36,context_tokens=108) for k in ('predict','action_encoder')},'exact replay costs')
    plans=[]
    for plan,row in zip(p['plans'],r['rows'],strict=True):
        require(row==read_json(output/f'root_{plan["root_index"]}.{plan["arm"]}.json'),'durable branch witness')
        exported=dict(deepcopy(plan),replay_witness=deepcopy(row),replay_report_sha256=io.sha((output/'report.json').read_bytes()),
            replay_journal_sha256=r['journal_sha256'],inherited_resources=p['inherited_resources'],replay_shared_resources=r['resources'],
            export_validation_saved_array_svds=16,P1_admission=False,P3_admission=False,P4_admission=False)
        exported['export_sha256']=digest(exported);plans.append(exported)
    return plans


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare','verify','execute-cuda'):group.add_argument('--'+name,action='store_true')
    parser.add_argument('--project-root',type=Path,default=PROJECT);parser.add_argument('--prepared',type=Path);parser.add_argument('--prepare-sha256');parser.add_argument('--output',type=Path);a=parser.parse_args()
    if a.prepare:require(a.output is not None,'new output');r=prepare(a.project_root,a.output)
    else:
        require(a.prepared is not None and a.prepare_sha256 is not None,'reviewed prepare SHA')
        if a.verify:validate_prepared(a.prepared,a.prepare_sha256);r=dict(status='verified')
        else:require(a.output is not None,'new output');r=execute_cuda(a.prepared,a.prepare_sha256,a.output)
    print(io.canonical({k:v for k,v in r.items() if k not in ('rows','events','source_seals','inherited_resources')}))


if __name__=='__main__':main()
