"""E1 calibration physical adapter. MAIN-only physics; saved prepare/consume.
Four dispatched plans plus two unknown learned slots. No neural operations.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import moving_prediction_matched_sampling_physical_20261027_v1 as primitives
from experiments.spectral_reachability import agent_experiment1_common_controller_20261028_v1 as producer

NAME='agent_experiment1_common_controller_physical_20261028_v1'
AUTH='MAIN_E1_COMMON_CONTROLLER_CALIBRATION_PHYSICS_ONLY'
ROOT,kernel,source=primitives.ROOT,primitives.kernel,primitives.source
require,sha,read,raw=primitives.require,primitives.sha,primitives.read,primitives.raw
fields,absolute,merge=primitives.fields,primitives.absolute,primitives.merge
ARMS,LEARNED=producer.ARMS,producer.LEARNED
CAP,FLOOR,FINAL,PREPARE_CAP=primitives.CAP,primitives.FLOOR,primitives.FINAL,primitives.PREPARE_CAP
SOURCE_SHA=primitives.SOURCE_SHA
REPORT_SHA='bc3cd74c85f878cdfc7b170d0b5b555827e0dc4a59cdaea6fb5781e71f03767f'
CUDA_PREPARE_SHA='8c472485d44c40ce3d95f8043d2c58f9ec2694fef74786a840d1699d5a1a86bd'
STRICT_SHA='b4b72dea9d782c6d2e1b93f50523a1089e7f3d894e2e2c1171fc13f9b8cd425d'
PRIMITIVES_SHA='50552d50ce3d97c9a25f938ef2b47e99bc66c88fa1644659611b5bb32843884a'
PRODUCER_SEALS=('c34a303c5e2df56a2002f195118cc2272a7b4d10afb79f8ed63d730af1a732a9',
    '2a160139a19cc8c802e5eac8aeea48093f16f60d8948ef129ac1edbcb3589544',
    'c58f9a491f1674da5f8cd82e76dc83bf22a2da42a3ed26e1dc61252dbac072bc')
Store,Ops=primitives.Store,primitives.Ops


def ownconsume(entry):
    fields(entry,'authority output report_sha256 prepare_sha256 strict_receipt strict_sha256')
    require(entry['authority']==producer.NAME,'only common-controller authority')
    require(entry['report_sha256']==REPORT_SHA and entry['prepare_sha256']==CUDA_PREPARE_SHA and
        entry['strict_sha256']==STRICT_SHA,'MAIN actual report/prepare/strict seals')
    frozen={str(f.resolve()):h for f,h in zip((Path(producer.__file__),ROOT/'tests'/('test_'+producer.NAME+'.py'),
        ROOT/'innerdocs'/(producer.NAME+'.md')),PRODUCER_SEALS,strict=True)}
    for f,h in frozen.items():require(sha(f)==h,'frozen common-controller source/tests/doc')
    output=Path(absolute(entry['output']));expected=primitives.hash_literal(entry['report_sha256'])
    primitives.hash_literal(entry['prepare_sha256']);require(sha(output/'report.json')==expected,'external actual report SHA')
    receipt=producer.consume(output,expected)
    strict=Path(absolute(entry['strict_receipt']))
    require(sha(strict)==entry['strict_sha256'] and read(strict,CAP)==receipt,'external MAIN strict receipt exact recomputation')
    require(receipt['schema']==producer.NAME+'.consume' and receipt['status']=='common_controller_calibration_verified' and
        receipt['resources']==producer.budget(576) and receipt['predictor_samples']==dict(attempted=576,completed=576) and
        receipt['dispatch_denominator']==4 and receipt['scientific_slots']==6 and receipt['learned_unknown_slots']==2 and
        receipt['calibration_only'] is True and receipt['P1_admission'] is False,'strict four-dispatch/six-slot producer receipt')
    report=read(output/'report.json',CAP);prepared=Path(absolute(report['prepared']))
    require(report['prepare_sha256']==entry['prepare_sha256']==sha(prepared),'external actual compiler prepare SHA')
    p=producer.verify(prepared,entry['prepare_sha256']);plans=deepcopy(receipt['plans'])
    require([(r['source_slot'],r['arm']) for r in plans]==[(s,a) for s in (0,1) for a in ARMS],'all four selected plans')
    bindings={};merge(bindings,p['bindings_sha256']);merge(bindings,frozen)
    merge(bindings,{str(prepared):entry['prepare_sha256'],str(output/'report.json'):expected})
    merge(bindings,{str(strict):entry['strict_sha256']})
    for name,h in report['artifact_sha256'].items():
        require(Path(name).name==name and name not in ('','.','..'),'flat compiler artifact');merge(bindings,{str(output/name):h})
    for row in plans:
        s=row['source_slot'];slot=p['slots'][s]
        require(type(s) is int and all(row[k]==slot[k] for k in ('source_slot','task_id','blind_sha256','task_export','task_sha256','task_contract_sha256')) and
            row['target']==p['targets'][s]['targets'][row['arm']] and row['whole_run_report_sha256']==expected,'original targets/task/history')
        content={k:v for k,v in row.items() if k not in ('selection_sha256','whole_run_report_sha256')}
        require(producer.core.digest(content)==row['selection_sha256'],'original compiler selection digest')
        kernel.commands(row['commands']);producer.vec(row['target'])
    for f,h in bindings.items():require(sha(f)==h,'compiler closure')
    return plans,bindings


def selection(m,data):
    fields(m,'schema sourcepack sourcepack_sha256 source_prepared source_prepare_sha256 roots arms authority selection_fixed_before_physics physical_outcome_selection wall_seconds')
    require(m['schema']==NAME+'.selection' and m['sourcepack_sha256']==SOURCE_SHA and m['arms']==list(ARMS) and
        m['selection_fixed_before_physics'] is True and m['physical_outcome_selection'] is False,'fixed original roots/four engineering plans')
    require(type(m['wall_seconds']) is int and 0<m['wall_seconds']<=600,'wall<=600')
    require(len(data['references'])==2 and m['roots']==[dict(root_slot=i,root_sha256=r['root_sha256']) for i,r in enumerate(data['references'])] and
        all(type(r['root_slot']) is int for r in m['roots']),'all original root identities')
    rows,bindings=ownconsume(m['authority']);plans=[]
    for row in rows:
        s=row['source_slot'];primitives.helper.task_bridge(row,data['references'][s],bindings)
        plan=dict(root_slot=s,root_sha256=data['references'][s]['root_sha256'],arm=row['arm'],
            commands=kernel.commands(row['commands']).tolist(),task_id=row['task_id'],blind_sha256=row['blind_sha256'],
            task_export=row['task_export'],task_sha256=row['task_sha256'],task_contract_sha256=row['task_contract_sha256'],
            target=deepcopy(row['target']),predicted_contexts=deepcopy(row['contexts']),
            model_terminal_cost=row['model_terminal_cost'],prefix_target_cost=row['prefix_target_cost'],tracking_cost=row['tracking_cost'],
            compiler_authority=producer.NAME,compiler_report=str(Path(m['authority']['output'])/'report.json'),
            compiler_report_sha256=m['authority']['report_sha256'],compiler_selection_sha256=row['selection_sha256'],
            own_plan_sha256=source.digest(row))
        plan['selection_digest']=source.digest(plan);plans.append(plan)
    require([(r['root_slot'],r['arm']) for r in plans]==[(s,a) for s in (0,1) for a in ARMS],'no subset/reselection')
    return plans,bindings


def contract(wall=600):
    require(type(wall) is int and 0<wall<=600,'wall<=600')
    c=kernel.low.contract(4,wall);c['caps'].update(environment_construct=4,environment_close=4)
    c.update(arms=list(ARMS),scientific_slots=6,learned_unknown_slots=2,output_cap_bytes=CAP,
        disk_floor_bytes=FLOOR,final_reserve_bytes=FINAL,prepare_cap_bytes=PREPARE_CAP,
        warmup_raw=kernel.moving.warmup_commands().tolist(),sourcepack_sha256=SOURCE_SHA,
        inherited_controller=dict(predict_API=576,predictor_samples=576,action_encoder=576,pred_proj=576),
        inherited_cost='separate compiler closure; proposal/encoding inherited separately, not equal total compute',
        tracking='per-step native7 + saved RGB; latent192 error null until separate CUDA evaluation',
        new_neural_calls=0,training=False,with_target=False,retries=0,padding=0,extra_replay=0,
        calibration_only=True,P1_admission=False,scientific_admission=False,independent_confirmation=False,
        wall_scope='Store construction through dispatch/write checks; no native preemption')
    return c


def material(manifest,expected):
    path=Path(absolute(str(manifest)));primitives.hash_literal(expected)
    require(path.stat().st_size<=PREPARE_CAP and sha(path)==expected,'bounded external manifest SHA')
    require(sha(primitives.__file__)==PRIMITIVES_SHA and sha(primitives.helper.__file__)==primitives.HELPER_SHA,'sealed physical helpers')
    for f,h in primitives.helper.SEALED_HELPERS.items():require(sha(f)==h,'sealed recipe dependencies')
    m=read(path);data=kernel.load_source(absolute(m['sourcepack']),m['sourcepack_sha256'],absolute(m['source_prepared']),m['source_prepare_sha256'])
    plans,bindings=selection(m,data);merge(bindings,data['bindings']);merge(bindings,{str(path):expected})
    native,h=primitives.helper.native_constructor_contract();merge(bindings,{native:h})
    files=primitives.physical_files()+[Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    for f in files:merge(bindings,{str(f.resolve()):sha(f)})
    for f,h in bindings.items():require(sha(f)==h,'final bindings')
    return dict(schema=NAME+'.prepare',manifest=str(path),manifest_sha256=expected,plans=plans,bindings=bindings,
        contract=contract(m['wall_seconds']),sourcepack_sha256=SOURCE_SHA,source_prepare_id=source.digest(data['plan']))


def prepare(manifest,expected,output):
    p=material(manifest,expected);payload=raw(p);path=Path(absolute(str(output)))
    require(len(payload)<=PREPARE_CAP and not path.exists() and not path.is_symlink() and
        shutil.disk_usage(path.parent).free>=FLOOR+len(payload),'NEW <=256KiB prepare / floor')
    with path.open('xb') as f:require(f.write(payload)==len(payload),'short prepare');f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(payload))


def verify(path,expected):
    path=Path(absolute(str(path)));primitives.hash_literal(expected);require(sha(path)==expected,'external physical prepare SHA')
    p=read(path);require(p==material(p['manifest'],p['manifest_sha256']) and sha(path)==expected,'entire physical closure');return p


class BranchStore:
    """Add native tracking receipts without altering renderer/kernel/controller."""
    def __init__(self,store,row,plan):self.store,self.row,self.plan=store,row,plan
    def __getattr__(self,key):return getattr(self.store,key)
    def emit(self,kind,**values):
        self.store.emit(kind,**values)
        if kind=='fresh_observation' and values['receipt']['phase']=='plan':
            cp=values['receipt'];state=source.vector(cp['state']);goal=source.vector(self.row['goal']) if 'goal' in self.row else None
            # plan0 capture precedes row.goal assignment: use the bound evaluator goal supplied by execute.
            if goal is None:goal=source.vector(self.row['tracking_goal'])
            receipt=dict(step=cp['step'],png=cp['png'],pixel_sha256=cp['pixel_sha256'],state=state.tolist(),
                native=kernel.metric(goal,state),target_sha256=producer.core.digest(self.plan['target']),
                model_prefix_target_cost=self.plan['prefix_target_cost'],model_terminal_cost=self.plan['model_terminal_cost'],
                observed_latent192=None,latent_tracking_error=None,
                model_successor=self.plan['predicted_contexts'][cp['step']//5-1] if cp['step'] in (5,10) else None)
            self.row['tracking'].append(receipt)
            self.store.emit('tracking_observation',branch=values['branch'],root_slot=self.plan['root_slot'],arm=self.plan['arm'],
                selection_digest=self.plan['selection_digest'],receipt=receipt)


def rows_for(p):
    return [dict(branch_index=i,root_slot=x['root_slot'],arm=x['arm'],selected_plan=deepcopy(x),selection_digest=x['selection_digest'],
        status='not_started',first_error=None,secondary_errors=[],warmup_returned=0,own_attempted=0,own_returned=0,
        initial_exact=False,full_horizon=False,native_success=None,terminated=None,truncated=None,initial_metrics=None,
        final_metrics=None,final_state=None,endpoint_scope='not_started',PNG=0,checkpoints=[],tracking=[],constructor_internal_work='not_attempted')
        for i,x in enumerate(p['plans'])]


def scientific_rows(rows):
    out=[]
    for s in (0,1):
        for a in producer.ALL_ARMS:
            if a==LEARNED:out.append(dict(root_slot=s,arm=a,status='unknown',commands=None,native_success=None,resources=None,
                error='no certified learned prior; not dispatched, not a failure rate'))
            else:
                r=next(r for r in rows if (r['root_slot'],r['arm'])==(s,a))
                out.append(dict(root_slot=s,arm=a,status=r['status'],branch_index=r['branch_index'],native_success=r['native_success']))
    return out


def execute(p,data,store,factory,*,authorization,execution_binding=None):
    require(authorization==AUTH,'MAIN physical authorization')
    require(p['contract']==contract(p['contract']['wall_seconds']) and len(data['references'])==2 and
        [(x['root_slot'],x['arm']) for x in p['plans']]==[(s,a) for s in (0,1) for a in ARMS],'fixed four plans')
    for plan in p['plans']:
        require(type(plan['root_slot']) is int and plan['root_sha256']==data['references'][plan['root_slot']]['root_sha256'] and
            plan['selection_digest']==source.digest({k:v for k,v in plan.items() if k!='selection_digest'}),'sealed root/commands/target')
        kernel.commands(plan['commands']);producer.vec(plan['target'])
    rows=rows_for(p);ops=Ops(p,store)
    r=dict(schema=NAME+'.report',status='running',rows=rows,errors=[],contract=p['contract'],
        sourcepack_sha256=p['sourcepack_sha256'],source_prepare_id=p['source_prepare_id'],fixed_branch_denominator=4,
        scientific_slot_denominator=6,learned_unknown_slots=2,
        prepared=None if execution_binding is None else execution_binding['prepared'],
        prepare_sha256=None if execution_binding is None else execution_binding['prepare_sha256'],
        new_neural_calls=0,new_independent_samples=0,training=False,calibration_only=True,P1_admission=False,scientific_admission=False)
    def error(stage,exc):r['errors'].append(dict(stage=stage,type=type(exc).__name__,message=str(exc)))
    try:
        store.emit('run_started',prepare=p,execution_binding=execution_binding)
        for i,(plan,row) in enumerate(zip(p['plans'],rows,strict=True)):
            env=None;holder=[];ops.where=dict(branch=i,root_slot=plan['root_slot'],arm=plan['arm'],selection_digest=plan['selection_digest'])
            def construct():
                row['constructor_internal_work']='unknown';value=factory();holder.append(value);return value
            try:
                env=ops.call('environment_construct',construct);kernel.moving.validate_env(env.unwrapped);body=env.unwrapped
                require(all(getattr(body,k,None) is None for k in ('space','render_buffer','latest_action')),'pristine constructor')
                require(body.render_action is False and body.render_size==224 and body.render_mode=='rgb_array','target-free RGB224')
                row.update(status='running',constructor_internal_work='sealed_native_pristine_no_physics',tracking_goal=data['references'][plan['root_slot']]['goal'].tolist())
                with kernel.native_hooks(env,ops,row):
                    kernel.branch(env,data['references'][plan['root_slot']],data['plan'],plan,row,ops,BranchStore(store,row,plan),i)
            except BaseException as exc:kernel.retain(row,exc,'setup_or_branch')
            finally:
                if env is None and holder:env=holder[0]
                if env is not None:primitives.close_env(env,row,ops,store)
                steps=[x for x in ops.receipts if x.get('branch')==i and x['resource']=='public_step']
                row['own_attempted']=sum(x.get('phase')=='plan' for x in steps)
                row['own_returned']=sum(x.get('phase')=='plan' and x['status']=='returned' for x in steps)
                row['warmup_returned']=sum(x.get('phase')=='warmup' and x['status']=='returned' for x in steps)
                row['full_horizon']=row['own_returned']==10
                if row['status']=='failed':row['endpoint_scope']='failed_partial'
                store.emit('branch_closed',receipt=row)
    except BaseException as exc:error('execution',exc)
    try:
        for f,h in p['bindings'].items():require(sha(f)==h,'final source drift')
    except BaseException as exc:error('final_bindings',exc)
    clean=not r['errors'] and all(x['status'] in ('horizon_completed','native_success','native_stopped') for x in rows)
    if clean:
        try:
            c=ops.counts;steps=sum(x['warmup_returned']+x['own_returned'] for x in rows)
            require(all(x['attempted']==x['completed'] for x in c.values()) and c['environment_construct']['completed']==c['environment_close']['completed']==4 and
                c['reset']['completed']==8 and c['set_state']['completed']==16 and c['goal_render']['completed']==8 and c['goal_bind']['completed']==4 and
                c['public_step']['completed']==steps and c['physics_step']['completed']==16+10*steps and
                c['explicit_render']['completed']==sum(x['PNG'] for x in rows),'actual earlystop/full census no padding')
        except BaseException as exc:error('final_census',exc);clean=False
    r.update(status='common_controller_physical_calibration_recorded' if clean else 'partial_or_failed',counts=ops.counts,
        actual_operation_receipts=ops.receipts,scientific_rows=scientific_rows(rows),full_horizons=sum(x['full_horizon'] for x in rows),
        native_successes=sum(x['status']=='native_success' for x in rows),engineering_failures=sum(x['status']=='failed' for x in rows),
        total_internal_physics_scope='unknown_constructor_work' if any(x['constructor_internal_work']=='unknown' for x in rows) else 'hook_observed_native_work')
    try:
        store.emit('run_finished',status=r['status'],counts=ops.counts,errors=r['errors'])
        r.update(journal_sha256=sha(store.path/'journal.jsonl'),journal_chain_tail=store.previous)
    except BaseException as exc:error('final_journal',exc);r['status']='partial_or_failed'
    try:store.put('report.json',r,final=True)
    except BaseException as exc:error('final_report',exc);r['status']='partial_or_failed'
    return r


def run(prepared,expected,output,factory,*,authorization):
    require(authorization==AUTH,'MAIN authorization before I/O')
    p=verify(prepared,expected);m=read(p['manifest'])
    data=kernel.load_source(m['sourcepack'],m['sourcepack_sha256'],m['source_prepared'],m['source_prepare_sha256'])
    store=Store(output,p['contract']['wall_seconds']);binding=dict(prepared=absolute(str(prepared)),prepare_sha256=expected)
    store.put('execution_binding.json',binding)
    return execute(p,data,store,factory,authorization=AUTH,execution_binding=binding)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run'))
    for k in ('manifest','manifest-sha','prepared','prepare-sha','output'):parser.add_argument('--'+k)
    parser.add_argument('--authorize-physics',action='store_true');a=parser.parse_args()
    if a.command=='prepare':r=prepare(a.manifest,a.manifest_sha,a.output)
    elif a.command=='verify':r=dict(schema=verify(a.prepared,a.prepare_sha)['schema'],status='verified_no_execution')
    else:
        require(a.authorize_physics,'MAIN explicit physics authorization')
        def factory():
            import gymnasium as gym
            import stable_worldmodel
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False,render_action=False)
        r=run(a.prepared,a.prepare_sha,a.output,factory,authorization=AUTH)
    print(source.canonical({k:v for k,v in r.items() if k not in ('rows','scientific_rows','actual_operation_receipts')}),flush=True)
    if a.command=='run' and r['status']!='common_controller_physical_calibration_recorded':raise SystemExit(2)


if __name__=='__main__':main()
