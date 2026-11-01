"""Fixed U/D/S/N x moving2 selected physical replay; MAIN execution only.

Full executable recipe consumer with MAIN-confirmed final scheduler authority.
Never import an arbitrary supplied consumer; no neural/physics during prepare.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import importlib
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import p1_moving_extended_baseline_physical_20261027_v1 as helper

NAME='moving_budget_scheduler_physical_20261027_v1'
AUTH='MAIN_MOVING_BUDGET_SCHEDULER_PHYSICS_ONLY'
AUTHORITY='moving_history_budget_scheduler_cuda_20261027_v1'
# MAIN-confirmed final source/tests/doc; not inferred from an unfinished file.
AUTHORITY_SHA='f4f4d5e013e3fc3aac896da58faaafadcf5e2d4e53f5f2f7b61203707e09ff9d'
AUTHORITY_TEST_SHA='58f2acf38164dff6699948f1c2c6d5f9b4b5391e03984b4b60a6142ac17db0c0'
AUTHORITY_DOC_SHA='303405900b416475956f1ecb74b08e4c8b6e20394df5693807234371c995cac1'
ARMS=['U_uniform','D_geometry','S_swapped','N_no_signal_uniform']
kernel=helper.kernel
ROOT=helper.ROOT
require,sha,read,raw,source=helper.require,helper.sha,helper.read,helper.raw,helper.source
fields,absolute=helper.fields,helper.absolute
CAP,FLOOR,FINAL,PREPARE_CAP=16*1024**2,200*1024**2,512*1024,256*1024
SOURCE_SHA=helper.SOURCE_SHA
PREREG=ROOT.parent/'docs/D2P_MOVING_BUDGET_SCHEDULER_PREREG_20261027.md'
PREREG_SHA='8e8cd4091dbaed5991dc119638ffdc00c8aed0852d0ed4c54cc6b1b3a82009bf'
HELPER_SHA='a0f7348ebe8c6e061b5d63a1ba4a8ecb064a26e7f7a1ef4be57fe8b07270d799'


def fixed_arms(value):
    require(isinstance(value,list) and value==ARMS,'ALL four ordered prereg strategies, no subset/reordering');return list(value)


def authority_files():
    return {str(ROOT/'experiments/spectral_reachability'/(AUTHORITY+'.py')):AUTHORITY_SHA,
        str(ROOT/'tests'/('test_'+AUTHORITY+'.py')):AUTHORITY_TEST_SHA,
        str(ROOT/'innerdocs'/(AUTHORITY+'.md')):AUTHORITY_DOC_SHA}


def ownconsume(entry):
    fields(entry,'authority output report_sha256')
    require(entry['authority']==AUTHORITY,'only fixed scheduler own consumer')
    require(isinstance(AUTHORITY_SHA,str) and len(AUTHORITY_SHA)==64,'pending final scheduler authority freeze; no prepare/release')
    path=ROOT/'experiments/spectral_reachability'/(AUTHORITY+'.py')
    for f,h in authority_files().items():require(sha(f)==h,'final scheduler source/tests/doc SHA before import')
    output=Path(absolute(entry['output']));h=entry['report_sha256']
    require(type(h) is str and len(h)==64 and sha(output/'report.json')==h,'explicit external actual scheduler report SHA')
    module=importlib.import_module('experiments.spectral_reachability.'+AUTHORITY)
    require(Path(module.__file__).resolve()==path.resolve() and sha(path)==AUTHORITY_SHA,'own scheduler module identity')
    receipt=module.consume(output,h)
    require(receipt['schema']==AUTHORITY+'.consume' and receipt['status']=='four_policy_commands_verified_not_admitted' and receipt['errors']==[] and
        receipt['P1_admission'] is False and receipt['P3_admission'] is False and receipt['independent_confirmation'] is False and
        receipt['compute_advantage'] is False and receipt['walltime_advantage'] is False and isinstance(receipt['plans'],list) and len(receipt['plans'])==8,
        'complete eight-plan own receipt, no scientific upgrade')
    rows=deepcopy(receipt['plans']);seen=set()
    for row in rows:
        s,a=row['source_slot'],row['arm']
        require(type(s) is int and s in (0,1) and a in ARMS and (s,a) not in seen,'stable scheduler roots/strategies, no duplicates');seen.add((s,a))
        require(row['schema']==AUTHORITY+'.model_plan' and row['whole_run_report_sha256']==h and row['physical_execution'] is False and
            row['selection_uses_physics'] is False and row['P1_admission'] is False and row['P3_admission'] is False,'blind model_plan provenance')
        content={k:v for k,v in row.items() if k not in ('selection_sha256','whole_run_report_sha256')}
        require(source.digest(content)==row['selection_sha256'],'own original model_plan canonical seal');kernel.commands(row['commands'])
    require(seen=={(s,a) for s in (0,1) for a in ARMS},'ALL eight new selected winners, no dedup')
    report=read(output/'report.json');prepared=Path(absolute(report['prepared']))
    require(sha(prepared)==report['prepare_sha256'],'scheduler own prepare seal')
    p=read(prepared);require(p['schema']==AUTHORITY+'.prepare' and len(p['slots'])==2,'fixed scheduler prepare schema/roots')
    for row in rows:
        slot=p['slots'][row['source_slot']]
        require(slot['source_slot']==row['source_slot'] and all(row[k]==slot[k] for k in (
            'root_sha256','task_id','blind_sha256','task_export','task_sha256','task_contract_sha256','bank_sha256')),'own slot/root/task/bank join')
    bindings=dict(p['bindings_sha256']);additions={**authority_files(),str(prepared):report['prepare_sha256'],str(output/'report.json'):h}
    for name,h in report['artifact_sha256'].items():
        require(Path(name).name==name and name not in ('','.','..'),'flat scheduler artifact')
        additions[str(output/name)]=h
    for f,h in additions.items():
        require(f not in bindings or bindings[f]==h,'nonconflicting scheduler bindings');bindings[f]=h
    for f,h in bindings.items():require(sha(f)==h,'own complete saved scheduler closure')
    return rows,bindings


def task_bridge(row,ref,bindings):
    # This helper is schema-independent and ONLY loads moving history, never zero-only physics.
    helper.previous.task_bridge(row,ref,bindings)
    from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as encoder
    t=encoder.load_task(row['task_export'],row['task_sha256'])['gn_task']
    require(row['root_sha256']==ref['root_sha256'] and row['task_contract_sha256']==t['task_contract_sha256'] and
        np.array_equal(t['projection'],np.eye(192)) and t['packet_logit_cap']==float(.75*np.sqrt(5)), 'original moving root/native192 Q/action support')
    kernel.commands(row['commands'])  # literal raw10; no atanh, repair, secondary optimizer or /100


def selection(m,data):
    fields(m,'schema sourcepack sourcepack_sha256 source_prepared source_prepare_sha256 roots arms authorities selection_fixed_before_physics physical_outcome_selection wall_seconds')
    require(m['schema']==NAME+'.selection' and m['sourcepack_sha256']==SOURCE_SHA and m['selection_fixed_before_physics'] is True and
        m['physical_outcome_selection'] is False,'own prereg no-outcome selection')
    fixed_arms(m['arms']);require(type(m['wall_seconds']) is int and 0<m['wall_seconds']<=600,'wall<=600')
    require(len(data['references'])==2 and m['roots']==[dict(root_slot=i,root_sha256=r['root_sha256']) for i,r in enumerate(data['references'])] and
        all(type(r['root_slot']) is int for r in m['roots']),'original two roots, order and identity')
    require(isinstance(m['authorities'],list) and len(m['authorities'])==1 and m['authorities'][0]['authority']==AUTHORITY,'one scheduler authority only')
    entry=m['authorities'][0];rows,bindings=ownconsume(entry);plans={}
    for row in rows:
        s,a=row['source_slot'],row['arm']
        require(type(s) is int and s in (0,1) and a in ARMS and (s,a) not in plans,'no duplicated or unknown selected slot')
        task_bridge(row,data['references'][s],bindings)
        plan=dict(root_slot=s,root_sha256=row['root_sha256'],arm=a,commands=kernel.commands(row['commands']).tolist(),
            task_id=row['task_id'],blind_sha256=row['blind_sha256'],task_export=row['task_export'],task_sha256=row['task_sha256'],
            task_contract_sha256=row['task_contract_sha256'],compiler_authority=AUTHORITY,compiler_report=str(Path(absolute(entry['output']))/'report.json'),
            compiler_report_sha256=entry['report_sha256'],compiler_selection_sha256=row['selection_sha256'],own_plan_sha256=source.digest(row))
        plan['selection_digest']=source.digest(plan);plans[s,a]=plan
    require(set(plans)=={(s,a) for s in (0,1) for a in ARMS},'fixed eight-branch denominator')
    return [plans[s,a] for s in (0,1) for a in ARMS],bindings


def contract(wall):
    require(type(wall) is int and 0<wall<=600,'wall<=600')
    c=kernel.low.contract(8,wall);c['caps'].update(environment_construct=8,environment_close=8)
    c.update(arms=list(ARMS),output_cap_bytes=CAP,final_reserve_bytes=FINAL,prepare_cap_bytes=PREPARE_CAP,
        warmup_raw=kernel.moving.warmup_commands().tolist(),prereg_sha256=PREREG_SHA,
        scope='known moving2 x prereg four strategies; no feedback, dedup, subset, old endpoint or admission',
        constructor_internal_work='unknown on partial constructor exception; hook counts are observed only',
        new_neural_calls=0,RGB_encoding='future independent producer; not part of this run',
        inherited_planning='own scheduler full report closure; distinct from physical budget, not total-compute matched')
    return c


def material(manifest,expected):
    path=Path(absolute(str(manifest)));require(path.stat().st_size<=PREPARE_CAP and sha(path)==expected,'external bounded manifest SHA')
    require(sha(helper.__file__)==HELPER_SHA and sha(PREREG)==PREREG_SHA,'frozen helper/prereg')
    m=read(path)
    require(m['schema']==NAME+'.selection' and m['sourcepack_sha256']==SOURCE_SHA,'own fixed moving selection')
    for k in ('sourcepack','source_prepared'):absolute(m[k])
    data=kernel.load_source(m['sourcepack'],m['sourcepack_sha256'],m['source_prepared'],m['source_prepare_sha256'])
    plans,bindings=selection(m,data)
    for f,h in data['bindings'].items():
        require(f not in bindings or bindings[f]==h,'nonconflicting original source binding');bindings[f]=h
    for f,h in ((str(path),expected),(str(PREREG),PREREG_SHA),helper.native_constructor_contract()):
        require(f not in bindings or bindings[f]==h,'nonconflicting input/constructor seal');bindings[f]=h
    for f in helper.physical_files()+[Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]:bindings[str(f.resolve())]=sha(f)
    for f,h in bindings.items():require(sha(f)==h,'final material source/model/task/artifact seals')
    return dict(schema=NAME+'.prepare',manifest=str(path),manifest_sha256=expected,plans=plans,contract=contract(m['wall_seconds']),
        bindings=bindings,sourcepack_sha256=SOURCE_SHA,source_prepare_id=source.digest(data['plan']))


def prepare(manifest,expected,output):
    p=material(manifest,expected);payload=raw(p);path=Path(absolute(str(output)))
    require(len(payload)<=PREPARE_CAP and shutil.disk_usage(path.parent).free>=FLOOR+len(payload),'NEW<=256KiB prepare +200MiB')
    with path.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(payload))


def verify(path,expected):
    require(Path(path).stat().st_size<=PREPARE_CAP and sha(path)==expected,'external prepare SHA/size');p=read(path)
    require(p==material(p['manifest'],p['manifest_sha256']) and sha(path)==expected,'own full reconstruction/TOCTOU');return p


class Store(helper.Store):
    """Same16MiB/512KiB final/200MiB floor, durable flat artifacts."""
    def __init__(self,path):super().__init__(path)


def execute(p,data,store,factory,*,authorization,execution_binding=None):
    require(authorization==AUTH,'MAIN explicit physics authorization')
    require(p['contract']==contract(p['contract']['wall_seconds']) and len(data['references'])==2 and
        [(x['root_slot'],x['arm']) for x in p['plans']]==[(s,a) for s in (0,1) for a in ARMS],'ALL eight prereg branches')
    for x in p['plans']:
        require(x['selection_digest']==source.digest({k:v for k,v in x.items() if k!='selection_digest'}),'fixed selected command seal');kernel.commands(x['commands'])
    rows=[dict(branch_index=i,root_slot=x['root_slot'],arm=x['arm'],selection_digest=x['selection_digest'],selected_plan=deepcopy(x),
        status='not_started',first_error=None,secondary_errors=[],warmup_returned=0,own_attempted=0,own_returned=0,initial_exact=False,
        full_horizon=False,native_success=False,terminated=False,truncated=False,endpoint_scope='not_started',PNG=0,checkpoints=[],
        constructor_internal_work='not_attempted') for i,x in enumerate(p['plans'])]
    result=dict(schema=NAME+'.report',status='running',rows=rows,errors=[],contract=p['contract'],sourcepack_sha256=p['sourcepack_sha256'],
        source_prepare_id=p['source_prepare_id'],fixed_branch_denominator=8,P1_admission=False,P2_admission=False,P3_admission=False,
        independent_confirmation=False,compute_advantage=False,training=False,new_neural_calls=0,
        prepared=None if execution_binding is None else execution_binding['prepared'],prepare_sha256=None if execution_binding is None else execution_binding['prepare_sha256'])
    ops=kernel.Ops(p,store)
    def error(stage,e):result['errors'].append(dict(stage=stage,type=type(e).__name__,message=str(e)))
    try:
        store.emit('run_started',prepare=p,execution_binding=execution_binding)
        for i,(plan,row) in enumerate(zip(p['plans'],rows,strict=True)):
            env=None;holder=[];ops.where=dict(branch=i,root_slot=plan['root_slot'],arm=plan['arm'],selection_digest=plan['selection_digest'])
            def construct():
                row['constructor_internal_work']='unknown'
                value=factory();holder.append(value);return value
            try:
                store.check(FINAL);env=ops.call('environment_construct',construct);kernel.moving.validate_env(env.unwrapped)
                require(all(getattr(env.unwrapped,k,None) is None for k in ('space','render_buffer','latest_action')),'pristine native constructor before hooks')
                require(env.unwrapped.render_action is False and env.unwrapped.render_size==224 and env.unwrapped.render_mode=='rgb_array','native target-free RGB224')
                row['constructor_internal_work']='sealed_native_pristine_no_physics';row['status']='running'
                with kernel.native_hooks(env,ops,row):kernel.branch(env,data['references'][plan['root_slot']],data['plan'],plan,row,ops,store,i)
            except BaseException as e:kernel.retain(row,e,'setup_or_branch')
            finally:
                if env is None and holder:env=holder[0]
                if env is not None:helper.close_env(env,row,ops,store)
                if row['status']=='failed':row['endpoint_scope']='failed_partial'
                store.emit('branch_closed',receipt=row)
    except BaseException as e:error('execution',e)
    try:
        for f,h in p['bindings'].items():require(sha(f)==h,'final binding: '+f)
    except BaseException as e:error('final_bindings',e)
    clean=not result['errors'] and all(r['status'] in ('horizon_completed','native_success','native_stopped') for r in rows)
    if clean:
        try:
            c=ops.counts;steps=sum(r['warmup_returned']+r['own_returned'] for r in rows)
            require(all(v['attempted']==v['completed'] for v in c.values()) and c['environment_construct']['completed']==c['environment_close']['completed']==8 and
                c['reset']['completed']==16 and c['set_state']['completed']==32 and c['goal_render']['completed']==16 and c['goal_bind']['completed']==8 and
                c['public_step']['completed']==steps and c['physics_step']['completed']==32+10*steps and c['explicit_render']['completed']==sum(r['PNG'] for r in rows),'actual native census, no fake fullcount on partial')
        except BaseException as e:error('final_census',e);clean=False
    result.update(status='moving_budget_scheduler_physical_recorded' if clean else 'partial_or_failed',counts=ops.counts,
        full_horizons=sum(r['full_horizon'] for r in rows),native_successes=sum(r['status']=='native_success' for r in rows),wall_seconds=time.monotonic()-ops.started,
        total_internal_physics_scope='unknown_constructor_work' if any(r['constructor_internal_work']=='unknown' for r in rows) else 'hook_observed_native_work')
    try:
        store.emit('run_finished',status=result['status'],counts=ops.counts,errors=result['errors'])
        result.update(journal_sha256=sha(store.path/'journal.jsonl'),journal_chain_tail=store.previous)
    except BaseException as e:error('final_journal',e);result['status']='partial_or_failed'
    try:store.put('report.json',result,final=True)
    except BaseException as e:error('final_report',e);result['status']='partial_or_failed'
    return result


def run(prepared,expected,output,factory,*,authorization):
    require(authorization==AUTH,'MAIN authorization before input/output/factory')
    p=verify(prepared,expected);m=read(p['manifest'])
    data=kernel.load_source(m['sourcepack'],m['sourcepack_sha256'],m['source_prepared'],m['source_prepare_sha256'])
    store=Store(output);binding=dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected);store.put('execution_binding.json',binding)
    return execute(p,data,store,factory,authorization=authorization,execution_binding=binding)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','verify','run'))
    for k in ('manifest','manifest-sha','prepared','prepare-sha','output'):p.add_argument('--'+k)
    p.add_argument('--authorize-physics',action='store_true');a=p.parse_args()
    if a.command=='prepare':r=prepare(a.manifest,a.manifest_sha,a.output)
    elif a.command=='verify':r=dict(schema=verify(a.prepared,a.prepare_sha)['schema'])
    else:
        require(a.authorize_physics,'MAIN CLI authorization')
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False,render_action=False)
        r=run(a.prepared,a.prepare_sha,a.output,factory,authorization=AUTH)
    print(source.canonical({k:v for k,v in r.items() if k!='rows'}))
    if a.command=='run' and r['status']!='moving_budget_scheduler_physical_recorded':raise SystemExit(2)


if __name__=='__main__':main()
