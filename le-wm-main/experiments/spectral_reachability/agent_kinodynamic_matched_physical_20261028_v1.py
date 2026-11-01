"""Two fixed matched-domain beam plans, MAIN-only moving-source physics.

Own authority/schema/denominator. No planner/model, outcomes-based selection,
old execute reuse or scientific admission. prepare/verify/consume are saved-only.
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
from experiments.spectral_reachability import agent_kinodynamic_matched_domain_20261028_v1 as producer

NAME='agent_kinodynamic_matched_physical_20261028_v1'
AUTH='MAIN_MATCHED_DOMAIN_BEAM_MOVING_PHYSICS_ONLY'
ARM=producer.ARM
ROOT,kernel,source=primitives.ROOT,primitives.kernel,primitives.source
require,sha,read,raw=primitives.require,primitives.sha,primitives.read,primitives.raw
fields,absolute,merge=primitives.fields,primitives.absolute,primitives.merge
CAP,FLOOR,FINAL,PREPARE_CAP=primitives.CAP,primitives.FLOOR,primitives.FINAL,primitives.PREPARE_CAP
SOURCE_SHA=primitives.SOURCE_SHA
PRIMITIVES_SHA='50552d50ce3d97c9a25f938ef2b47e99bc66c88fa1644659611b5bb32843884a'
PRODUCER_SHA='8e6d03cdbae69382cd87bcb7f443fb882023b7930ef138bd02aa37f0950d46b8'
PRODUCER_TEST_SHA='e724f10a97a66996fa4539c0fe9d86299f67cfccc5705809211854afb886594d'
PRODUCER_DOC_SHA='eca8217eca8c6980f3fa8ce3201f994321eb80a08e250594dd605afd3079987e'
REPORT_SHA='d4976aeca357b5324c74be8b2b7e3bfc254e82b834b3acfe1b0ab90a1ef4614c'
CUDA_PREPARE_SHA='de3db628c5bd15f62f700fa56c917582d5e9fa27cf3cb979abf25a181d5e8dc9'
hash_literal=primitives.hash_literal
Store=primitives.Store
Ops=primitives.Ops


def ownconsume(entry):
    fields(entry,'authority output report_sha256 source_sha256 tests_sha256 innerdoc_sha256')
    require(entry['authority']==producer.NAME and entry['report_sha256']==REPORT_SHA and
        entry['source_sha256']==PRODUCER_SHA and entry['tests_sha256']==PRODUCER_TEST_SHA and
        entry['innerdoc_sha256']==PRODUCER_DOC_SHA,'fixed actual new beam authority, no old rawbank')
    frozen={str(Path(producer.__file__).resolve()):PRODUCER_SHA,
        str(ROOT/'tests'/('test_'+producer.NAME+'.py')):PRODUCER_TEST_SHA,
        str(ROOT/'innerdocs'/(producer.NAME+'.md')):PRODUCER_DOC_SHA}
    for p,h in frozen.items():require(sha(p)==h,'producer final three seals')
    output=Path(absolute(entry['output']));require(sha(output/'report.json')==REPORT_SHA,'actual beam report SHA')
    receipt=producer.consume(output,REPORT_SHA)
    require(receipt['schema']==producer.NAME+'.consume' and receipt['errors']==[] and
        receipt['P1_admission'] is False and receipt['equal_total_compute'] is False and
        receipt['official_reproduction'] is False and receipt['resources']==producer.budget(288) and
        receipt['predictor_samples']==dict(attempted=288,completed=288),'actual288 beam calls/samples, not old sampler')
    rows=deepcopy(receipt['plans']);require([(x['source_slot'],x['arm']) for x in rows]==[(0,ARM),(1,ARM)],'fixed two plans')
    report=read(output/'report.json',CAP);prepared=Path(absolute(report['prepared']))
    require(report['prepare_sha256']==CUDA_PREPARE_SHA and sha(prepared)==CUDA_PREPARE_SHA,'actual beam prepare SHA')
    p=read(prepared);require(p['schema']==producer.NAME+'.prepare','own beam prepare')
    bindings={};merge(bindings,p['bindings_sha256']);merge(bindings,frozen)
    merge(bindings,{str(prepared):CUDA_PREPARE_SHA,str(output/'report.json'):REPORT_SHA})
    for name,h in report['artifact_sha256'].items():
        require(Path(name).name==name and name not in ('','.','..'),'flat beam artifacts')
        merge(bindings,{str(output/name):h})
    for row in rows:
        s=row['source_slot'];slot=p['slots'][s]
        require(type(s) is int and row['schema']==producer.NAME+'.model_plan' and row['whole_run_report_sha256']==REPORT_SHA and
            row['with_target'] is False and row['selection_uses_physics'] is False and row['P1_admission'] is False and
            all(row[k]==slot[k] for k in ('source_slot','task_id','blind_sha256','task_export','task_sha256')),'own task/slot authority')
        content={k:v for k,v in row.items() if k not in ('selection_sha256','whole_run_report_sha256')}
        require(producer.core.digest(content)==row['selection_sha256'],'original beam selected-plan digest')
        b=producer.bank(s);path=row['selected_path']
        require(len(path)==2 and all(type(i) is int and 0<=i<20 for i in path) and row['bank_sha256']==producer.core.digest(b) and
            row['commands']==sum([b['commands'][i] for i in path],[]),'fixed bank full10 action witness')
    for p,h in bindings.items():require(sha(p)==h,'saved beam source/model/task/artifact closure')
    return rows,bindings


def task_bridge(row,ref,bindings):
    # The producer plan intentionally has no generic logits/task_contract fields.
    # Derive these ONLY for validation from its bound bank/path and bound export,
    # never alter the original plan or original selection seal.
    from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as encoder
    value=encoder.load_task(row['task_export'],row['task_sha256'])
    t=value['gn_task'];b=producer.bank(row['source_slot'])
    bridge=dict(row,task_contract_sha256=t['task_contract_sha256'],logits=[b['logits'][i] for i in row['selected_path']])
    primitives.helper.task_bridge(bridge,ref,bindings)
    return t['task_contract_sha256']


def selection(m,data):
    fields(m,'schema sourcepack sourcepack_sha256 source_prepared source_prepare_sha256 roots arms authorities selection_fixed_before_physics physical_outcome_selection wall_seconds')
    require(m['schema']==NAME+'.selection' and m['sourcepack_sha256']==SOURCE_SHA and m['arms']==[ARM] and
        m['selection_fixed_before_physics'] is True and m['physical_outcome_selection'] is False,'own fixed two beam branches')
    require(type(m['wall_seconds']) is int and 0<m['wall_seconds']<=600,'wall<=600')
    require(len(data['references'])==2 and m['roots']==[dict(root_slot=i,root_sha256=r['root_sha256']) for i,r in enumerate(data['references'])]
        and all(type(r['root_slot']) is int for r in m['roots']),'original two moving roots/order')
    require(len(m['authorities'])==1,'one beam authority')
    rows,bindings=ownconsume(m['authorities'][0]);plans=[]
    for row in rows:
        s=row['source_slot'];ref=data['references'][s];tc=task_bridge(row,ref,bindings)
        plan=dict(root_slot=s,root_sha256=ref['root_sha256'],arm=ARM,commands=kernel.commands(row['commands']).tolist(),
            task_id=row['task_id'],blind_sha256=row['blind_sha256'],task_export=row['task_export'],task_sha256=row['task_sha256'],
            task_contract_sha256=tc,compiler_authority=producer.NAME,compiler_report_sha256=REPORT_SHA,
            compiler_report=str(Path(m['authorities'][0]['output'])/'report.json'),
            compiler_selection_sha256=row['selection_sha256'],own_plan_sha256=source.digest(row),
            selected_path=row['selected_path'],bank_sha256=row['bank_sha256'])
        plan['selection_digest']=source.digest(plan);plans.append(plan)
    require([(p['root_slot'],p['arm']) for p in plans]==[(0,ARM),(1,ARM)],'all2 plans no subset')
    return plans,bindings


def contract(wall=600):
    require(type(wall) is int and 0<wall<=600,'wall<=600')
    c=kernel.low.contract(2,wall);c['caps'].update(environment_construct=2,environment_close=2)
    c.update(arms=[ARM],output_cap_bytes=CAP,disk_floor_bytes=FLOOR,final_reserve_bytes=FINAL,prepare_cap_bytes=PREPARE_CAP,
        warmup_raw=kernel.moving.warmup_commands().tolist(),sourcepack_sha256=SOURCE_SHA,
        selection_scope='two preselected matched-domain beam plans; not old rawbank or sampling arms',
        inherited_prediction_budget=dict(predict_API=288,predictor_samples=288,action_encoder=288,pred_proj=288),
        inherited_cost='separate model report; API and modules overlap, never add; not equal FLOPs',
        wall_scope='Store creation then dispatch/write checkpoints; cannot preempt blocked native call',
        new_neural_calls=0,training=False,retries=0,padding=0,extra_replay=0,with_target=False,
        P1_admission=False,scientific_admission=False,independent_confirmation=False,equal_total_compute=False,equal_FLOPs=False)
    return c


def material(manifest,expected):
    path=Path(absolute(str(manifest)));hash_literal(expected)
    require(sha(primitives.__file__)==PRIMITIVES_SHA,'frozen metering/store primitives')
    require(path.stat().st_size<=PREPARE_CAP and sha(path)==expected,'external bounded manifest SHA')
    for f,h in primitives.helper.SEALED_HELPERS.items():require(sha(f)==h,'frozen moving recipe helpers')
    require(sha(primitives.helper.__file__)==primitives.HELPER_SHA,'frozen extended helper')
    m=read(path);data=kernel.load_source(absolute(m['sourcepack']),m['sourcepack_sha256'],
        absolute(m['source_prepared']),m['source_prepare_sha256'])
    plans,bindings=selection(m,data);merge(bindings,data['bindings']);merge(bindings,{str(path):expected})
    native,h=primitives.helper.native_constructor_contract();merge(bindings,{native:h})
    files=primitives.physical_files()+[Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    for f in files:merge(bindings,{str(f.resolve()):sha(f)})
    for f,h in bindings.items():require(sha(f)==h,'final input/source/model seals')
    return dict(schema=NAME+'.prepare',manifest=str(path),manifest_sha256=expected,plans=plans,bindings=bindings,
        contract=contract(m['wall_seconds']),sourcepack_sha256=SOURCE_SHA,source_prepare_id=source.digest(data['plan']))


def prepare(manifest,expected,output):
    p=material(manifest,expected);payload=raw(p);path=Path(absolute(str(output)))
    require(len(payload)<=PREPARE_CAP and not path.exists() and not path.is_symlink() and
        shutil.disk_usage(path.parent).free>=FLOOR+len(payload),'NEW small prepare+200MiB floor')
    with path.open('xb') as f:require(f.write(payload)==len(payload),'short prepare');f.flush();os.fsync(f.fileno())
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(payload))


def verify(path,expected):
    path=Path(absolute(str(path)));hash_literal(expected);require(sha(path)==expected,'external physical prepare SHA')
    p=read(path);require(p==material(p['manifest'],p['manifest_sha256']) and sha(path)==expected,'entire rebuilt closure');return p


def execute(p,data,store,factory,*,authorization,execution_binding=None):
    require(authorization==AUTH,'MAIN authorization')
    require(p['contract']==contract(p['contract']['wall_seconds']) and len(data['references'])==2 and
        [(x['root_slot'],x['arm']) for x in p['plans']]==[(0,ARM),(1,ARM)],'fixed2 denominator')
    for plan in p['plans']:
        require(type(plan['root_slot']) is int and plan['root_sha256']==data['references'][plan['root_slot']]['root_sha256'] and
            plan['selection_digest']==source.digest({k:v for k,v in plan.items() if k!='selection_digest'}),'own plan/root identity')
        kernel.commands(plan['commands'])
    rows=[dict(branch_index=i,root_slot=x['root_slot'],arm=x['arm'],selected_plan=deepcopy(x),selection_digest=x['selection_digest'],
        status='not_started',first_error=None,secondary_errors=[],warmup_returned=0,own_attempted=0,own_returned=0,
        initial_exact=False,full_horizon=False,native_success=None,terminated=None,truncated=None,initial_metrics=None,
        final_metrics=None,final_state=None,endpoint_scope='not_started',PNG=0,checkpoints=[],constructor_internal_work='not_attempted')
        for i,x in enumerate(p['plans'])]
    r=dict(schema=NAME+'.report',status='running',rows=rows,errors=[],contract=p['contract'],
        sourcepack_sha256=p['sourcepack_sha256'],source_prepare_id=p['source_prepare_id'],fixed_branch_denominator=2,
        prepared=None if execution_binding is None else execution_binding['prepared'],
        prepare_sha256=None if execution_binding is None else execution_binding['prepare_sha256'],
        planner_report_sha256=REPORT_SHA,new_neural_calls=0,new_independent_samples=0,training=False,
        scientific_admission=False,P1_admission=False,P2_admission=False,independent_confirmation=False,
        equal_total_compute=False,equal_FLOPs=False)
    ops=Ops(p,store)
    def error(stage,exc):r['errors'].append(dict(stage=stage,type=type(exc).__name__,message=str(exc)))
    try:
        store.emit('run_started',prepare=p,execution_binding=execution_binding)
        for i,(plan,row) in enumerate(zip(p['plans'],rows,strict=True)):
            env=None;holder=[];ops.where=dict(branch=i,root_slot=plan['root_slot'],arm=ARM,selection_digest=plan['selection_digest'])
            def construct():
                row['constructor_internal_work']='unknown';value=factory();holder.append(value);return value
            try:
                env=ops.call('environment_construct',construct);kernel.moving.validate_env(env.unwrapped);body=env.unwrapped
                require(all(getattr(body,k,None) is None for k in ('space','render_buffer','latest_action')),'pristine constructor before hooks')
                require(body.render_action is False and body.render_size==224 and body.render_mode=='rgb_array','RGB224 target-free')
                row.update(status='running',constructor_internal_work='sealed_native_pristine_no_physics')
                with kernel.native_hooks(env,ops,row):
                    kernel.branch(env,data['references'][plan['root_slot']],data['plan'],plan,row,ops,store,i)
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
        for f,h in p['bindings'].items():require(sha(f)==h,'final binding drift: '+f)
    except BaseException as exc:error('final_bindings',exc)
    clean=not r['errors'] and all(x['status'] in ('horizon_completed','native_success','native_stopped') for x in rows)
    if clean:
        try:
            c=ops.counts;steps=sum(x['warmup_returned']+x['own_returned'] for x in rows)
            require(all(x['attempted']==x['completed'] for x in c.values()) and
                c['environment_construct']['completed']==c['environment_close']['completed']==2 and
                c['reset']['completed']==4 and c['set_state']['completed']==8 and c['goal_render']['completed']==4 and
                c['goal_bind']['completed']==2 and c['public_step']['completed']==steps and
                c['physics_step']['completed']==8+10*steps and c['explicit_render']['completed']==sum(x['PNG'] for x in rows),
                'actual full/earlystop census, no padded operations')
        except BaseException as exc:error('final_census',exc);clean=False
    r.update(status='matched_domain_beam_physical_recorded_not_admitted' if clean else 'partial_or_failed',counts=ops.counts,
        actual_operation_receipts=ops.receipts,full_horizons=sum(x['full_horizon'] for x in rows),
        native_successes=sum(x['status']=='native_success' for x in rows),engineering_failures=sum(x['status']=='failed' for x in rows),
        not_started=sum(x['status']=='not_started' for x in rows),early_stops=sum(x['status'] in ('native_success','native_stopped') and not x['full_horizon'] for x in rows),
        total_internal_physics_scope='unknown_constructor_work' if any(x['constructor_internal_work']=='unknown' for x in rows) else 'hook_observed_native_work')
    try:
        store.emit('run_finished',status=r['status'],counts=ops.counts,errors=r['errors'])
        r.update(journal_sha256=sha(store.path/'journal.jsonl'),journal_chain_tail=store.previous)
    except BaseException as exc:error('final_journal',exc);r['status']='partial_or_failed'
    try:store.put('report.json',r,final=True)
    except BaseException as exc:error('final_report',exc);r['status']='partial_or_failed'
    return r


def run(prepared,expected,output,factory,*,authorization):
    require(authorization==AUTH,'MAIN authorization before I/O/factory')
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
    elif a.command=='verify':r=dict(schema=verify(a.prepared,a.prepare_sha)['schema'],status='verified_no_dispatch')
    else:
        require(a.authorize_physics,'MAIN explicit physics authorization')
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # registration only
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False,render_action=False)
        r=run(a.prepared,a.prepare_sha,a.output,factory,authorization=AUTH)
    print(source.canonical({k:v for k,v in r.items() if k not in ('rows','actual_operation_receipts')}),flush=True)
    if a.command=='run' and r['status']!='matched_domain_beam_physical_recorded_not_admitted':raise SystemExit(2)


if __name__=='__main__':main()
