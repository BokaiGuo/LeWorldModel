"""Own fixed eight-branch consumer of the sealed rank scheduler commands.

No model execution or replanning. Source evaluator metadata never enters CUDA
scheduler. Native recipe/hooks are immutable primitives, not old run adapters.
"""
import argparse
from copy import deepcopy
from io import BytesIO
import os
from pathlib import Path
import shutil
import time

import numpy as np

from experiments.spectral_reachability import chat_p3_20261027_cuda as scheduler
from experiments.spectral_reachability import p1_moving_extended_baseline_physical_20261027_v1 as helper

kernel=helper.kernel
ROOT=scheduler.ROOT
NAME='chat_p3_20261027_physical'
ARMS=scheduler.ARMS
AUTH='chat_p3_20261027_fixed_eight_physics'
SCHEDULER_OUTPUT=ROOT/'results/chat_p3_20261027_cuda_run01'
SCHEDULER_SHA='6df856e36b145162e6a8d5ce5f141880a49afe88f43d3cc88f8b98ad790b6d08'
require,read,sha=helper.require,helper.read,helper.sha
FLAGS=scheduler.FLAGS


def contract():
    c=kernel.low.contract(8,600)
    c['caps'].update(environment_construct=8,environment_close=8)
    c.update(arms=list(ARMS),output_cap_bytes=16*1024**2,final_reserve_bytes=512*1024,
             own_schema=NAME,warmup_raw=kernel.moving.warmup_commands().tolist(),
             scope='fixed known moving2, no feedback/dedup/retry/selection; not fresh confirmation')
    return c


def select_plans(receipt,data,bindings):
    require(receipt['schema']==scheduler.NAME+'.consume' and receipt['status']=='actual_rank_queues_verified_not_admitted', 'own verified rank producer')
    require(all(receipt[k] is False for k in FLAGS), 'scientific gate closed')
    rows=receipt['plans'];by_key={}
    for row in rows:
        s,a=row['source_slot'],row['arm']
        require(type(s) is int and s in (0,1) and a in ARMS and (s,a) not in by_key, 'strict unique two roots four arms')
        require(row['schema']==scheduler.NAME+'.model_plan' and row['whole_run_report_sha256']==SCHEDULER_SHA and row['physical_execution'] is False and row['selection_uses_physics'] is False, 'selected blind commands only')
        require(kernel.source.digest({k:v for k,v in row.items() if k not in ('selection_sha256','whole_run_report_sha256')})==row['selection_sha256'], 'producer selection seal')
        helper.previous.task_bridge(row,data['references'][s],bindings)
        require(row['root_sha256']==data['references'][s]['root_sha256'], 'source root identity')
        plan=dict(root_slot=s,arm=a,root_sha256=row['root_sha256'],commands=kernel.commands(row['commands']).tolist(),
                  task_id=row['task_id'],blind_sha256=row['blind_sha256'],task_export=row['task_export'],task_sha256=row['task_sha256'],
                  task_contract_sha256=row['task_contract_sha256'],compiler_report_sha256=SCHEDULER_SHA,
                  compiler_selection_sha256=row['selection_sha256'],own_plan_sha256=kernel.source.digest(row))
        plan['selection_digest']=kernel.source.digest(plan);by_key[s,a]=plan
    require(set(by_key)=={(s,a) for s in (0,1) for a in ARMS}, 'entire eight denominator')
    return [by_key[s,a] for s in (0,1) for a in ARMS]


def material():
    receipt=scheduler.consume(SCHEDULER_OUTPUT,SCHEDULER_SHA)
    sr=read(SCHEDULER_OUTPUT/'report.json'); sp=read(sr['prepared'])
    bindings=dict(sp['bindings_sha256']);bindings[sr['prepared']]=sr['prepare_sha256']
    bindings[str(SCHEDULER_OUTPUT/'report.json')]=SCHEDULER_SHA
    bindings.update({str(SCHEDULER_OUTPUT/n):h for n,h in sr['artifact_sha256'].items()})
    data=kernel.load_source(scheduler.old.SOURCEPACK,scheduler.old.SOURCEPACK_SHA,
                            scheduler.old.SOURCE_PREPARE,scheduler.old.SOURCE_PREPARE_SHA)
    for f,h in data['bindings'].items():
        require(f not in bindings or bindings[f]==h,'nonconflicting original source');bindings[f]=h
    plans=select_plans(receipt,data,bindings)
    files=helper.physical_files()+[Path(__file__),ROOT/'tests/chat_p3_20261027_test_physical.py',
                                   ROOT/'innerdocs/chat_p3_20261027_physical_prereg.md']
    for f in files:bindings[str(f.resolve())]=sha(f)
    native,h=helper.native_constructor_contract(); bindings[native]=h
    for f,h in bindings.items():require(sha(f)==h,'fixed source/input SHA '+f)
    p=dict(schema=NAME+'.prepare',plans=plans,contract=contract(),bindings=bindings,
           sourcepack_sha256=scheduler.old.SOURCEPACK_SHA,source_prepare_id=kernel.source.digest(data['plan']),
           root_goals=[r['goal'].tolist() for r in data['references']],**FLAGS)
    return p,data


def prepare(output):
    p,_=material(); path=scheduler.exclusive(output); raw=scheduler.rule.canonical(p)
    require(len(raw)<=256*1024 and shutil.disk_usage(path.parent).free>=helper.FLOOR+len(raw), 'small prepare/diskfloor')
    with path.open('xb') as f:
        require(f.write(raw)==len(raw),'short prepare');f.flush();os.fsync(f.fileno())
    return dict(prepared=str(path),sha256=sha(path))


def verify(path,expected):
    require(expected is not None and sha(path)==expected,'external physical prepare SHA')
    p,data=material();require(p==read(path),'own physical material equality')
    return p,data


def execute(p,data,store,factory,authorization,binding):
    require(authorization==AUTH and p['contract']==contract(), 'explicit own physical authorization')
    require([(v['root_slot'],v['arm']) for v in p['plans']]==[(s,a) for s in (0,1) for a in ARMS], 'fixed root-major order')
    for plan in p['plans']:
        require(plan['selection_digest']==kernel.source.digest({k:v for k,v in plan.items() if k!='selection_digest'}),'unchanged command seal')
    rows=[dict(branch_index=i,root_slot=x['root_slot'],arm=x['arm'],selection_digest=x['selection_digest'],selected_plan=deepcopy(x),
               status='not_started',first_error=None,secondary_errors=[],warmup_returned=0,own_attempted=0,own_returned=0,
               initial_exact=False,full_horizon=False,native_success=False,terminated=False,truncated=False,
               endpoint_scope='not_started',PNG=0,checkpoints=[],constructor_internal_work='not_attempted')
          for i,x in enumerate(p['plans'])]
    result=dict(schema=NAME+'.report',status='running',rows=rows,errors=[],contract=contract(),
                fixed_branch_denominator=8,prepared=binding['prepared'],prepare_sha256=binding['prepare_sha256'],
                scheduler_report_sha256=SCHEDULER_SHA,new_neural_calls=0,**FLAGS)
    ops=kernel.Ops(p,store)
    def error(stage,exc):result['errors'].append(dict(stage=stage,type=type(exc).__name__,message=str(exc)))
    try:
        store.emit('run_started',prepare=p,execution_binding=binding)
        for i,(plan,row) in enumerate(zip(p['plans'],rows,strict=True)):
            env=None;holder=[]
            ops.where=dict(branch=i,root_slot=plan['root_slot'],arm=plan['arm'],selection_digest=plan['selection_digest'])
            def construct():
                row['constructor_internal_work']='unknown';value=factory();holder.append(value);return value
            try:
                store.check(helper.FINAL);env=ops.call('environment_construct',construct)
                kernel.moving.validate_env(env.unwrapped)
                require(all(getattr(env.unwrapped,k,None) is None for k in ('space','render_buffer','latest_action')), 'pristine native constructor')
                require(env.unwrapped.render_action is False and env.unwrapped.render_size==224 and env.unwrapped.render_mode=='rgb_array', 'original target-free RGB')
                row['constructor_internal_work']='sealed_native_pristine_no_physics';row['status']='running'
                with kernel.native_hooks(env,ops,row):
                    kernel.branch(env,data['references'][plan['root_slot']],data['plan'],plan,row,ops,store,i)
            except BaseException as exc:kernel.retain(row,exc,'setup_or_branch')
            finally:
                if env is None and holder:env=holder[0]
                if env is not None:helper.close_env(env,row,ops,store)
                if row['status']=='failed':row['endpoint_scope']='failed_partial'
                store.emit('branch_closed',receipt=row)
    except BaseException as exc:error('execution',exc)
    try:
        for f,h in p['bindings'].items():require(sha(f)==h,'final binding '+f)
    except BaseException as exc:error('final_bindings',exc)
    clean=not result['errors'] and all(r['status'] in ('horizon_completed','native_success','native_stopped') for r in rows)
    if clean:
        try:
            steps=sum(r['warmup_returned']+r['own_returned'] for r in rows)
            c=ops.counts
            require(all(v['attempted']==v['completed'] for v in c.values()) and c['environment_construct']['completed']==c['environment_close']['completed']==8 and c['reset']['completed']==16 and c['set_state']['completed']==32 and c['goal_render']['completed']==16 and c['goal_bind']['completed']==8 and c['public_step']['completed']==steps and c['physics_step']['completed']==32+10*steps and c['explicit_render']['completed']==sum(r['PNG'] for r in rows),'actual full native operation census')
        except BaseException as exc:error('native_census',exc);clean=False
    result.update(status='rank_physical_recorded_not_admitted' if clean else 'partial_or_failed',counts=ops.counts,
                  full_horizons=sum(r['full_horizon'] for r in rows),native_successes=sum(r['native_success'] for r in rows),
                  wall_seconds=time.monotonic()-ops.started,
                  total_internal_physics_scope='unknown_constructor_work' if any(r['constructor_internal_work']=='unknown' for r in rows) else 'hook_observed_native_work')
    try:
        store.emit('run_finished',status=result['status'],counts=ops.counts,errors=result['errors'])
        result.update(journal_sha256=sha(store.path/'journal.jsonl'),journal_chain_tail=store.previous)
    except BaseException as exc:error('final_journal',exc);result['status']='partial_or_failed'
    try:store.put('report.json',result,final=True)
    except BaseException as exc:error('final_report',exc);result['status']='partial_or_failed'
    return result


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'physics must be explicitly enabled')
    path=scheduler.exclusive(output);p,data=verify(prepared,expected)
    def factory():
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False,render_action=False)
    store=helper.Store(path);binding=dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected)
    store.put('execution_binding.json',binding)
    result=execute(p,data,store,factory,authorization,binding)
    require(result['status']=='rank_physical_recorded_not_admitted','partial preserved; no retry')
    return dict(output=str(path),sha256=sha(path/'report.json'),status=result['status'])


def validate_saved(p,r,events,frame_reader):
    require(r['schema']==NAME+'.report' and r['status']=='rank_physical_recorded_not_admitted' and r['errors']==[] and r['fixed_branch_denominator']==8 and r['contract']==contract(), 'complete own physical report')
    require(all(r[k] is False for k in FLAGS) and r['new_neural_calls']==0 and r['scheduler_report_sha256']==SCHEDULER_SHA,'no scientific upgrades')
    require(events[0]['kind']=='run_started' and events[0]['prepare']==p and events[-1]['kind']=='run_finished' and events[-1]['status']==r['status'] and events[-1]['counts']==r['counts'] and events[-1]['errors']==[], 'journal full boundaries')
    require(len(r['rows'])==8 and [(v['root_slot'],v['arm']) for v in r['rows']]==[(s,a) for s in (0,1) for a in ARMS], 'complete fixed denominator')
    observed={k:dict(attempted=0,completed=0) for k in contract()['caps']}
    for e in events:
        if e['kind'] in ('operation_attempted','operation_completed'):
            require(e['resource'] in observed,'known operation')
            observed[e['resource']]['attempted' if e['kind']=='operation_attempted' else 'completed']+=1
    require(observed==r['counts'] and all(v['attempted']==v['completed'] for v in observed.values()),'actual journal/native operation count')
    closed=[e['receipt'] for e in events if e['kind']=='branch_closed']
    require(closed==r['rows'], 'branch receipts exact')
    filenames=[]
    for i,(row,plan) in enumerate(zip(r['rows'],p['plans'],strict=True)):
        require(row['branch_index']==i and row['selected_plan']==plan and row['selection_digest']==plan['selection_digest'] and row['first_error'] is None and row['secondary_errors']==[], 'selected commands and branch health')
        require(row['initial_exact'] and row['constructor_internal_work']=='sealed_native_pristine_no_physics', 'exact original moving source')
        transitions=[e for e in events if e['kind']=='returned_transition' and e['branch']==i and e['phase']=='plan']
        require(len(transitions)==row['own_returned']==row['own_attempted'] and [e['step'] for e in transitions]==list(range(1,1+len(transitions))), 'unbroken own execution prefix')
        for e in transitions:
            require(e['command']==plan['commands'][e['step']-1] and e['returned_goal_state']==p['root_goals'][plan['root_slot']], 'literal selected command and unchanged goal')
        require(row['goal']==p['root_goals'][plan['root_slot']] and row['initial_metrics']==kernel.metric(row['goal'],row['root_state']), 'original goal/root metrics')
        require(row['final_state']==transitions[-1]['state'] and row['final_metrics']==kernel.metric(row['goal'],row['final_state']), 'actual final metrics recomputed')
        require(row['native_success']==row['final_metrics']['native_success'] and row['terminated']==transitions[-1]['terminated'] and row['truncated']==transitions[-1]['truncated'], 'native endpoint semantics')
        require(row['full_horizon']==(row['own_returned']==10) and row['warmup_returned']==10, 'full10 separate from native stop')
        images=[e['receipt'] for e in events if e['kind']=='fresh_observation' and e['branch']==i]
        guards=[e for e in events if e['kind']=='guard_receipt' and e['branch']==i]
        require(len(images)==len(guards)==row['PNG'] and all(e['accepted'] and e['render_preserved_dynamics'] for e in guards),'all frame guards')
        for im in images:
            frame=frame_reader(im['png'],im['file_sha256']);filenames.append(im['png'])
            require(kernel.moving.image_sha(frame)==im['pixel_sha256'] and frame.dtype==np.uint8 and frame.shape==(224,224,3) and np.any(np.all(frame==[65,105,225],axis=-1)), 'actual target-free PNG/canvas')
            require(im['state']==im['observation_state'] and np.all((np.asarray(im['state'])[:4]>=0)&(np.asarray(im['state'])[:4]<=512)), 'actual state/image guard')
        checkpoints=[im for im in images if im['phase']=='plan' and im['step'] in (0,5,10)]
        require(checkpoints==row['checkpoints'], 'only actual saved checkpoints, no padded missing')
    require(len(filenames)==len(set(filenames)), 'independent branch image files no dedup')
    # No partial or stopped endpoint enters full-H2 paired differences.
    full=all(v['full_horizon'] and not v['truncated'] for v in r['rows'])
    diffs=None
    if full:
        metrics={a:[next(v for v in r['rows'] if v['root_slot']==s and v['arm']==a)['final_metrics'] for s in (0,1)] for a in ARMS}
        fields=('native_distance','native_success','wrapped_allxy_success','object_position_error','wrapped_object_angle_error')
        diffs={a:{k:[float(metrics['D_rank'][s][k])-float(metrics[a][s][k]) for s in (0,1)] for k in fields} for a in ('U','S_rank','N')}
    return dict(status='actual_rank_physical_verified_not_admitted',full_horizons=sum(v['full_horizon'] for v in r['rows']),
                native_successes=sum(v['native_success'] for v in r['rows']),D_minus_control=diffs,
                native_full10_pair_available=full,filenames=filenames,counts=observed,**FLAGS)


def consume(output,expected):
    output=Path(output).absolute();require(expected is not None and sha(output/'report.json')==expected,'external physical report SHA')
    r=read(output/'report.json');p,_=verify(r['prepared'],r['prepare_sha256'])
    require(read(output/'execution_binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'execution binding')
    require(sha(output/'journal.jsonl')==r['journal_sha256'],'journal SHA')
    events=kernel.journal(output/'journal.jsonl',r['journal_chain_tail'])
    def frame_reader(name,expected):
        from PIL import Image
        require(Path(name).name==name and sha(output/name)==expected,'flat actual image SHA')
        with Image.open(BytesIO((output/name).read_bytes())) as im:return np.asarray(im.convert('RGB')).copy()
    value=validate_saved(p,r,events,frame_reader)
    require({f.name for f in output.iterdir()}==set(value['filenames'])|{'report.json','journal.jsonl','execution_binding.json'},'exact saved artifact census')
    require(sha(output/'report.json')==expected,'final report unchanged')
    value.update(schema=NAME+'.consume',report_sha256=expected,wall_seconds=r['wall_seconds'])
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','verify','run','consume'])
    parser.add_argument('--prepared',type=Path);parser.add_argument('--sha256');parser.add_argument('--output',type=Path)
    parser.add_argument('--authorize-physics',action='store_true');parser.add_argument('--receipt',type=Path)
    a=parser.parse_args()
    if a.action=='prepare':value=prepare(a.output)
    elif a.action=='verify':verify(a.prepared,a.sha256);value=dict(status='verified',sha256=a.sha256)
    elif a.action=='run':value=run(a.prepared,a.sha256,a.output,AUTH if a.authorize_physics else None)
    else:
        value=consume(a.output,a.sha256)
        if a.receipt:
            path=scheduler.exclusive(a.receipt)
            with path.open('xb') as f:f.write(scheduler.rule.canonical(value));f.flush();os.fsync(f.fileno())
            value={k:v for k,v in value.items() if k!='filenames'}|dict(receipt=str(path),sha256=sha(path))
    print(scheduler.rule.canonical(value).decode(),flush=True)


if __name__=='__main__':main()
