"""MAIN-only fixed moving baseline physical consumer, including GRASP/SRA.

Saved-only own-consume at prepare, exact original moving recipe at run.
No model execution, old physical outcome reuse, selection or admission.
"""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
import importlib
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import p1_moving_baseline_physical_20261027_v1 as previous

NAME='p1_moving_extended_baseline_physical_20261027_v1'
AUTH='MAIN_MOVING_EXTENDED_BASELINE_SELECTED_PHYSICS_ONLY'
kernel=previous.kernel
ROOT=kernel.ROOT
require,sha,read,raw,source=kernel.require,kernel.sha,kernel.read,kernel.raw,kernel.source
fields,absolute=previous.fields,previous.absolute
CAP,FLOOR,FINAL,PREPARE_CAP=16*1024**2,200*1024**2,512*1024,256*1024
SOURCE_SHA=previous.SOURCE_SHA
AUTHORITIES={
    'grasp':('p1_moving_grasp_baseline_cuda_20261027_v1',
        '52d3157574e4e2726f6d88b43bd7e5dc804ce79730b65512888e63e0dd7c7106',('GRASP_H3_bounded',),'rows'),
    'no_composition':('p1_moving_no_composition_baseline_cuda_20261027_v1',
        '8c709bed26d88e093b0293482be07cfc34702fb095b2fe23a22edadbbb985963',('joint_H2_spectral_packet_mapped',),'rows')}
ARMS=['GRASP_H3_bounded','joint_H2_spectral_packet_mapped']
APPROVED_REPORTS={
    'grasp':'7593940c1503416f42ce87a5d8135e8512763121e027c0ef98dbe614bfb31f91',
    'no_composition':'ec39956ca0c90cc5dbf955b1a65bf300b68d166627a748a82314a2176a481c34'}
SEALED_HELPERS={previous.__file__:'19b367debdfcc47032de8828dd0ab7124171c553d6c3f7b0a225ca86820cc3e0',
    kernel.__file__:'44d90599eab881c773d5efa09177dfe6b395fa3f4eef41070340681fea5ec96e'}


def checked_arms(arms):
    require(isinstance(arms,list) and arms==ARMS,'fixed BOTH ordered arms, no outcome subset')
    return list(arms)


def ownconsume(entry):
    fields(entry,'authority output report_sha256')
    require(type(entry['authority']) is str and entry['authority'] in AUTHORITIES,'unknown authority')
    name,expected,allowed,key=AUTHORITIES[entry['authority']]
    path=ROOT/'experiments/spectral_reachability'/(name+'.py')
    require(sha(path)==expected,'frozen own authority before import')
    output=Path(absolute(entry['output']));h=entry['report_sha256']
    require(h==APPROVED_REPORTS[entry['authority']],'MAIN approved fixed actual report')
    require(sha(output/'report.json')==h,'external complete actual report SHA')
    module=importlib.import_module('experiments.spectral_reachability.'+name)
    require(Path(module.__file__).resolve()==path.resolve() and sha(path)==expected,'own module identity')
    receipt=module.consume(output,h)
    require(receipt['schema']==name+'.consume' and receipt['P1_admission'] is False and
        isinstance(receipt[key],list) and len(receipt[key])>0,'nonempty saved-only own consume')
    rows=deepcopy(receipt[key]);seen=set()
    for row in rows:
        s,a=row['source_slot'],row['arm']
        require(type(s) is int and s in (0,1) and a in allowed and (s,a) not in seen,'strict own roots/arms, no duplicate')
        seen.add((s,a));kernel.commands(row['commands'])
        if key=='rows':
            require(row['schema']==name+'.model_plan' and row['whole_run_report_sha256']==h and
                row['physical_execution'] is False and row['P1_admission'] is False and row['with_target'] is False,'own model_plan provenance')
            content={k:v for k,v in row.items() if k not in ('selection_sha256','whole_run_report_sha256')}
            require(source.digest(content)==row['selection_sha256'],'original own selected plan digest')
    require(seen=={(s,a) for s in (0,1) for a in {a for _,a in seen}},'complete two-root own receipt')
    report=read(output/'report.json');prepared=Path(absolute(report['prepared']))
    require(sha(prepared)==report['prepare_sha256'],'own compiler prepare SHA')
    p=read(prepared);bindings=dict(p['bindings_sha256'])
    additions={str(prepared):report['prepare_sha256'],str(output/'report.json'):h,str(path.resolve()):expected}
    for n,h in report['artifact_sha256'].items():
        require(Path(n).name==n and n not in ('','.','..'),'flat own compiler artifact')
        additions[str(output/n)]=h
    for f,h in additions.items():
        require(f not in bindings or bindings[f]==h,'nonconflicting compiler seals');bindings[f]=h
    for f,h in bindings.items():require(sha(f)==h,'saved ownconsume closure')
    return rows,bindings


def task_bridge(plan,ref,bindings):
    from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as encoder
    path=absolute(plan['task_export']);value=encoder.load_task(path,plan['task_sha256']);e=encoder.read(path)
    task=value['gn_task'];tc=e['task_contract']
    require(type(plan['source_slot']) is int and e['source_slot']==plan['source_slot'] and
        plan['blind_sha256']==e['blind_sha256']==ref['blind_sha256'] and plan['task_id']==value['task_id'],'exact moving source/task/slot identity')
    require(tc['pixel_sha256']==[kernel.moving.image_sha(a) for a in [*ref['history'],ref['goal_rgb']]] and
        tc['raw_action_sha256']==kernel.moving.image_sha(ref['raw_history']) and
        kernel.moving.exact(np.asarray(task['root_raw_action_history'],np.float32),kernel.raw_history(ref['raw_history'])),
        'exact original moving RGB/goal/raw warmup, no zero substitution')
    require(np.array_equal(task['projection'],np.eye(192)) and task['packet_logit_cap']==float(.75*np.sqrt(5)), 'original native192 Q/action cap')
    if 'schema' in plan:
        require(plan['task_contract_sha256']==task['task_contract_sha256'],'own model_plan task contract')
        x=np.asarray(plan['logits']);commands=kernel.commands(plan['commands'])
        require(x.shape==(2,10) and np.isfinite(x).all() and (np.linalg.norm(x,axis=1)<=task['packet_logit_cap']).all() and
            np.array_equal(commands,np.tanh(x.astype(np.float32)).reshape(10,2)),'original raw10 logits/tanh provenance')
    require(path not in bindings or bindings[path]==plan['task_sha256'],'nonconflicting task export seal');bindings[path]=plan['task_sha256']


def selection(manifest,data):
    fields(manifest,'schema sourcepack sourcepack_sha256 source_prepared source_prepare_sha256 roots arms authorities selection_fixed_before_physics physical_outcome_selection wall_seconds')
    require(manifest['schema']==NAME+'.selection' and manifest['selection_fixed_before_physics'] is True and
        manifest['physical_outcome_selection'] is False and manifest['sourcepack_sha256']==SOURCE_SHA,'own predeclared moving selection')
    require(type(manifest['wall_seconds']) is int and 0<manifest['wall_seconds']<=600,'wall<=600')
    require(len(data['references'])==2 and manifest['roots']==[dict(root_slot=i,root_sha256=r['root_sha256']) for i,r in enumerate(data['references'])] and
        all(type(r['root_slot']) is int for r in manifest['roots']),'both original roots, stable order')
    arms=checked_arms(manifest['arms'])
    required=[k for k,(_,_,names,_) in AUTHORITIES.items() if any(a in arms for a in names)]
    require(isinstance(manifest['authorities'],list) and [e['authority'] for e in manifest['authorities']]==required,'exact used authority order')
    selected={};bindings={}
    for entry in manifest['authorities']:
        rows,bound=ownconsume(entry);allowed=AUTHORITIES[entry['authority']][2];seen=set()
        for row in rows:
            s,a=row['source_slot'],row['arm']
            require(type(s) is int and s in (0,1) and a in allowed and (s,a) not in seen,'strict roots/arms/no duplicates');seen.add((s,a))
            task_bridge(row,data['references'][s],bindings)
            if a not in arms:continue
            plan=dict(root_slot=s,root_sha256=data['references'][s]['root_sha256'],arm=a,
                commands=kernel.commands(row['commands']).tolist(),task_id=row['task_id'],blind_sha256=row['blind_sha256'],
                task_export=row['task_export'],task_sha256=row['task_sha256'],compiler_authority=entry['authority'],
                compiler_report=str(Path(absolute(entry['output']))/'report.json'),compiler_report_sha256=entry['report_sha256'],
                own_plan_sha256=source.digest(row))
            plan['selection_digest']=source.digest(plan);selected[s,a]=plan
        require(seen=={(s,a) for s in (0,1) for a in {a for _,a in seen}},'complete two-root receipt, no dropped arm root')
        for f,h in bound.items():
            require(f not in bindings or bindings[f]==h,'nonconflicting authority/task bindings');bindings[f]=h
    require(set(selected)=={(s,a) for s in (0,1) for a in arms},'fixed ALL roots x ALL selected arms')
    return [selected[s,a] for s in (0,1) for a in arms],bindings


def contract(arms,wall):
    arms=checked_arms(arms);n=2*len(arms);c=kernel.low.contract(n,wall)
    c['caps'].update(environment_construct=n,environment_close=n)
    c.update(arms=arms,output_cap_bytes=CAP,final_reserve_bytes=FINAL,prepare_cap_bytes=PREPARE_CAP,
        warmup_raw=kernel.moving.warmup_commands().tolist(),
        lifecycle='construct entry/return; cleanup close attempted even after wall/journal failure; no physics in sealed native constructor',
        scope='fixed old moving2 calibration tasks, not independent confirmation; no outcome selection/new neural/P1 admission',
        partial_denominator='ALL2xselected arms including not_started, failed, native_stopped; never padded',
        inherited_model_cost='compiler own reports/prepare/artifact closures sealed separately, no equal-total-compute claim')
    return c


def physical_files():
    # previous scans the real static kernel closure; own source adds no new runtime primitives.
    return sorted(set(previous.physical_files()+[Path(__file__).resolve()]))


def native_constructor_contract():
    """No simulator import: the exact local native constructor must be inert."""
    path=kernel.low.builder.ENV;expected=kernel.low.builder.ENV_SHA
    require(sha(path)==expected,'original native constructor source SHA')
    tree=ast.parse(Path(path).read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PushT')
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    forbidden={'reset','step','render','_set_state','_set_goal_state','_setup'}
    require(not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in forbidden
        for n in ast.walk(init)),'native constructor has no reset/physics/render dispatch')
    return str(path.resolve()),expected


def material(manifest,expected):
    path=Path(absolute(str(manifest)));require(path.stat().st_size<PREPARE_CAP and sha(path)==expected,'bounded external manifest SHA')
    m=read(path)
    for f,h in SEALED_HELPERS.items():require(sha(f)==h,'frozen physical recipe/helper')
    fields(m,'schema sourcepack sourcepack_sha256 source_prepared source_prepare_sha256 roots arms authorities selection_fixed_before_physics physical_outcome_selection wall_seconds')
    require(m['schema']==NAME+'.selection' and m['sourcepack_sha256']==SOURCE_SHA,'only own manifest/approved moving source')
    for key in ('sourcepack','source_prepared'):absolute(m[key])
    data=kernel.load_source(m['sourcepack'],m['sourcepack_sha256'],m['source_prepared'],m['source_prepare_sha256'])
    plans,bindings=selection(m,data)
    for f,h in data['bindings'].items():
        require(f not in bindings or bindings[f]==h,'nonconflicting source bindings');bindings[f]=h
    bindings[str(path)]=expected
    native,h=native_constructor_contract()
    require(native not in bindings or bindings[native]==h,'nonconflicting native constructor seal');bindings[native]=h
    for f in physical_files()+[ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]:bindings[str(f.resolve())]=sha(f)
    for f,h in bindings.items():require(sha(f)==h,'final prepare source/input seals')
    return dict(schema=NAME+'.prepare',manifest=str(path),manifest_sha256=expected,plans=plans,
        contract=contract(m['arms'],m['wall_seconds']),bindings=bindings,sourcepack_sha256=SOURCE_SHA,source_prepare_id=source.digest(data['plan']))


def prepare(manifest,expected,output):
    p=material(manifest,expected);payload=raw(p);path=Path(absolute(str(output)))
    require(len(payload)<=PREPARE_CAP and shutil.disk_usage(path.parent).free>=FLOOR+len(payload),'prepare<=256KiB +200MiB floor')
    with path.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(payload))


def verify(path,expected):
    require(Path(path).stat().st_size<=PREPARE_CAP and sha(path)==expected,'external bounded prepare SHA');p=read(path)
    require(p==material(p['manifest'],p['manifest_sha256']) and sha(path)==expected,'full own reconstruction/TOCTOU');return p


class Store(previous.Store):
    def __init__(self,output):
        self.path=Path(output).absolute()
        require(not self.path.exists() and shutil.disk_usage(self.path.parent).free>=FLOOR+CAP,'NEW16MiB +200MiB floor')
        self.path.mkdir();self.previous=None;self.index=0
    def check(self,size=4096):
        used=sum(f.stat().st_size for f in self.path.iterdir() if f.is_file())
        require(used+size<=CAP-FINAL and shutil.disk_usage(self.path).free>=FLOOR+size,'16MiB/200MiB gate')
    def write(self,name,value,append=False,final=False):
        require(Path(name).name==name and name not in ('','.','..') and not (self.path/name).is_symlink(),'flat output')
        used=sum(f.stat().st_size for f in self.path.iterdir() if f.is_file())
        require(used+len(value)<=CAP-(0 if final else FINAL) and shutil.disk_usage(self.path).free>=FLOOR+len(value),'16MiB incl partial/final +200MiB')
        with (self.path/name).open('ab' if append else 'xb') as f:f.write(value);f.flush();os.fsync(f.fileno())
        fd=os.open(self.path,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        return sha(self.path/name)


def close_env(env,row,ops,store):
    """Cleanup must run even when the journal or wall cap is already exhausted."""
    count=ops.counts['environment_close'];count['attempted']+=1
    try:store.emit('operation_attempted',resource='environment_close',**ops.where)
    except BaseException as e:kernel.retain(row,e,'close_attempt_journal')
    try:env.close();count['completed']+=1
    except BaseException as e:kernel.retain(row,e,'close')
    else:
        try:store.emit('operation_completed',resource='environment_close',**ops.where)
        except BaseException as e:kernel.retain(row,e,'close_return_journal')


def execute(p,data,store,factory,*,authorization,execution_binding=None):
    require(authorization==AUTH,'MAIN explicit physics authorization')
    arms=checked_arms(p['contract']['arms'])
    require(p['contract']==contract(arms,p['contract']['wall_seconds']) and len(data['references'])==2 and
        [(x['root_slot'],x['arm']) for x in p['plans']]==[(s,a) for s in (0,1) for a in arms],'full fixed denominator/contract')
    for plan in p['plans']:
        require(plan['selection_digest']==source.digest({k:v for k,v in plan.items() if k!='selection_digest'}),'sealed selected plan')
        kernel.commands(plan['commands'])
    rows=[dict(branch_index=i,root_slot=x['root_slot'],arm=x['arm'],selection_digest=x['selection_digest'],selected_plan=deepcopy(x),
        status='not_started',first_error=None,secondary_errors=[],warmup_returned=0,own_attempted=0,own_returned=0,initial_exact=False,
        full_horizon=False,native_success=False,terminated=False,truncated=False,endpoint_scope='not_started',PNG=0,checkpoints=[]) for i,x in enumerate(p['plans'])]
    result=dict(schema=NAME+'.report',status='running',rows=rows,errors=[],contract=p['contract'],
        sourcepack_sha256=p['sourcepack_sha256'],source_prepare_id=p['source_prepare_id'],fixed_branch_denominator=len(rows),
        prepared=None if execution_binding is None else execution_binding['prepared'],
        prepare_sha256=None if execution_binding is None else execution_binding['prepare_sha256'],
        P1_admission=False,P2_admission=False,training=False,new_neural_calls=0,new_independent_samples=0)
    ops=kernel.Ops(p,store)
    def error(stage,e):result['errors'].append(dict(stage=stage,type=type(e).__name__,message=str(e)))
    try:
        store.emit('run_started',prepare=p,execution_binding=execution_binding)
        for i,(plan,row) in enumerate(zip(p['plans'],rows,strict=True)):
            env=None;holder=[];ops.where=dict(branch=i,root_slot=plan['root_slot'],arm=plan['arm'],selection_digest=plan['selection_digest'])
            def construct():
                value=factory();holder.append(value);return value
            try:
                store.check(FINAL);env=ops.call('environment_construct',construct)
                kernel.moving.validate_env(env.unwrapped)
                require(all(getattr(env.unwrapped,key,None) is None for key in ('space','render_buffer','latest_action')),'native constructor has not reset/stepped/rendered before hooks')
                require(env.unwrapped.render_action is False and env.unwrapped.render_size==224 and env.unwrapped.render_mode=='rgb_array','native target-free RGB224')
                row['status']='running'
                with kernel.native_hooks(env,ops,row):kernel.branch(env,data['references'][plan['root_slot']],data['plan'],plan,row,ops,store,i)
            except BaseException as e:kernel.retain(row,e,'setup_or_branch')
            finally:
                if env is None and holder:env=holder[0]  # completion-journal failure must not leak a returned env
                if env is not None:close_env(env,row,ops,store)
                if row['status']=='failed':row['endpoint_scope']='failed_partial'
                store.emit('branch_closed',receipt=row)
    except BaseException as e:error('execution',e)
    try:
        for f,h in p['bindings'].items():require(sha(f)==h,'final binding: '+f)
    except BaseException as e:error('final_bindings',e)
    clean=not result['errors'] and all(r['status'] in ('horizon_completed','native_success','native_stopped') for r in rows)
    if clean:
        try:
            c=ops.counts;n=len(rows);steps=sum(r['warmup_returned']+r['own_returned'] for r in rows)
            require(all(v['attempted']==v['completed'] for v in c.values()) and c['environment_construct']['completed']==c['environment_close']['completed']==n and
                c['reset']['completed']==2*n and c['set_state']['completed']==4*n and c['goal_render']['completed']==2*n and c['goal_bind']['completed']==n and
                c['public_step']['completed']==steps and c['physics_step']['completed']==4*n+10*steps and c['explicit_render']['completed']==sum(r['PNG'] for r in rows),'actual full/partial census')
        except BaseException as e:error('final_census',e);clean=False
    result.update(status='moving_extended_baseline_physical_recorded' if clean else 'partial_or_failed',counts=ops.counts,
        full_horizons=sum(r['full_horizon'] for r in rows),native_successes=sum(r['status']=='native_success' for r in rows),wall_seconds=time.monotonic()-ops.started)
    try:
        store.emit('run_finished',status=result['status'],counts=ops.counts,errors=result['errors'])
        result.update(journal_sha256=sha(store.path/'journal.jsonl'),journal_chain_tail=store.previous)
    except BaseException as e:error('final_journal',e);result['status']='partial_or_failed'
    try:store.put('report.json',result,final=True)
    except BaseException as e:error('final_report',e);result['status']='partial_or_failed'
    return result


def run(prepared,expected,output,factory,*,authorization):
    require(authorization==AUTH,'MAIN authorization before I/O/factory')
    p=verify(prepared,expected);m=read(p['manifest'])
    data=kernel.load_source(m['sourcepack'],m['sourcepack_sha256'],m['source_prepared'],m['source_prepare_sha256'])
    store=Store(output);binding=dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected)
    store.put('execution_binding.json',binding)
    return execute(p,data,store,factory,authorization=authorization,execution_binding=binding)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run'))
    for key in ('manifest','manifest-sha','prepared','prepare-sha','output'):parser.add_argument('--'+key)
    parser.add_argument('--authorize-physics',action='store_true');a=parser.parse_args()
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
    if a.command=='run' and r['status']!='moving_extended_baseline_physical_recorded':raise SystemExit(2)


if __name__=='__main__':main()
