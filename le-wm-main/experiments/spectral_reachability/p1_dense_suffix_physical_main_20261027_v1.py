"""Prospective persistent suffix interventions, not an outcome-selected controller.

Test whether sparse one-command probes missed contact responses. Both radius and
temporal support differ at the larger radius; do not claim a pure ablation.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import inspect
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p1_same_prefix_physical_main_20261027_v1 as sparse

base, source, core = sparse.base, sparse.source, sparse.core
ROOT, require = sparse.ROOT, sparse.require
NAME = 'p1_dense_suffix_physical_main_20261027_v1'
RADII = (.5, 1.5)
DIRECTIONS = ((1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1))
CAP, FLOOR = sparse.CAP, sparse.FLOOR


def suffixes():
    items = [dict(index=0, radius=0., direction=None, logits=[0.]*10)]
    for radius in RADII:
        for direction in DIRECTIONS:
            d = np.asarray(direction, float); d /= np.linalg.norm(d)
            x = np.tile(d, 5) * radius / np.sqrt(5)
            # Float32 inward rounding, not an unrecorded projection after selection.
            x = x.astype(np.float32)
            if np.linalg.norm(x.astype(float)) > radius:
                x *= np.float32(1-4*np.finfo(np.float32).eps)
            items.append(dict(index=len(items), radius=radius, direction=list(direction), logits=x.astype(float).tolist()))
    return items


def contract():
    return dict(roots=2, roles=list(sparse.ROLES), branches=102, suffixes=17,
        radii=list(RADII), directions=[list(x) for x in DIRECTIONS],
        support='same command direction at all5 suffix times; 10 independent coordinates remain legal',
        reset_cap=204, step_cap=1020, render_cap=1122, png_cap=306,
        order='original source slot,zero/bank_greedy/two_stage,zero then radius/direction',
        output_cap_bytes=CAP, floor_bytes=FLOOR, neural_calls=0, training=False,
        original_Q_unchanged=True, fitted_controller=False, P1_admission=False, independent_samples=0,
        partial_policy='retain all102 planned branches; honor native termination/fresh RGB guard; no retries',
        limitation='new fixed diagnostic after sparse outcomes; not independent confirmation, not pure radius-controlled ablation, not global reachability')


def collect():
    parent = base.verify(sparse.PARENT, sparse.PARENT_SHA)
    witness = base.witness.verify(Path(parent['inputs']['arguments']['witness_prepared']),
                                  parent['inputs']['arguments']['witness_prepare_sha'])
    bindings = {str(sparse.PARENT/'prepare.json'): sparse.PARENT_SHA,
                **parent['sources'], **parent['inputs']['bindings']}
    plans, roles = [], []
    for old in [x for x in parent['inputs']['plans'] if x['arm'] == 'two_stage']:
        task = next(t['task'] for t in witness['inputs']['tasks'] if t['root_sha256'] == old['root_sha256'])
        for role in sparse.choose_prefixes(task, base.read(old['origin'])):
            roles.append(dict(root_sha256=old['root_sha256'], task_id=core.task_id(task), **role))
            for probe in suffixes():
                x = np.asarray(probe['logits'], np.float32)
                require(np.linalg.norm(x.astype(float)) <= task['packet_logit_cap'], 'original legal suffix ball')
                commands = np.concatenate((np.asarray(role['prefix'], np.float32), np.tanh(x).reshape(5,2)))
                base.raw_commands(commands)
                plan = deepcopy(old)
                plan.update(arm=role['role']+'_dense_'+str(probe['index']), role=role['role'],
                    bank_index=role['bank_index'], probe=probe, commands=commands.astype(float).tolist(),
                    prediction_sha256=None, origin='fixed_persistent_suffix_not_model_winner',
                    parent_selection_sha256=old['selection_sha256'],
                    search_cost=dict(new_neural_calls=0, inherited_search_separate=True))
                plan.pop('selection_sha256'); plan['selection_sha256'] = source.digest(plan); plans.append(plan)
    require(len(plans)==102 and len(roles)==6, 'all fixed roots/roles/probes')
    base.bindings_check(bindings)
    return dict(plans=plans, roles=roles, source_plan=parent['inputs']['source_plan'], bindings=bindings)


def material():
    paths = set(sparse.own_sources()) | {str(Path(__file__).resolve()),
        str(ROOT/'tests'/('test_'+NAME+'.py')), str(ROOT/'innerdocs'/(NAME+'.md'))}
    return dict(schema=NAME+'.prepare', contract=contract(), inputs=collect(),
                sources={p:base.sha(p) for p in sorted(paths)})


def prepare(output):
    p = material(); raw = (source.canonical(p)+'\n').encode(); path = Path(output)
    require(len(raw)<1024**2 and shutil.disk_usage(path.parent).free>=FLOOR+len(raw), 'bounded prepare/floor')
    with path.open('xb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
    return dict(prepared=str(path), bytes=len(raw), prepare_sha256=base.sha(path))


def verify(path, expected):
    require(base.sha(path)==expected, 'external dense prepare SHA')
    p = base.read(path); require(p==material(), 'frozen complete dense protocol/source binding'); return p


def analyze(plans, rows, goals):
    require(len(plans)==len(rows)==102, 'all102 planned denominator')
    groups = []
    for start in range(0,102,17):
        ps, rs = plans[start:start+17], rows[start:start+17]; center = rs[0]
        require([p['probe'] for p in ps]==suffixes(), 'exact complete probe bank')
        entries = []
        for plan,row in zip(ps,rs,strict=True):
            require(all(row[k]==plan[k] for k in ('root_sha256','arm','selection_sha256')), 'plan/row identity')
            entry = dict(probe_index=plan['probe']['index'], status=row['status'], response=None)
            if row.get('status')==center.get('status')=='completed' and row.get('full_horizon') and center.get('full_horizon'):
                require(row['checkpoints'][1]['state']==center['checkpoints'][1]['state'] and
                    row['checkpoints'][1]['rgb_sha256']==center['checkpoints'][1]['rgb_sha256'] and
                    row['body_dynamics']['5']==center['body_dynamics']['5'], 'same recorded prefix dynamics/RGB')
                d = sparse.geometric_delta(row['final_state'],center['final_state'])
                goal = source.vector(goals[plan['root_sha256']])
                entry['response'] = dict(endpoint_delta_state7=d.tolist(), object_xy_delta=d[2:4].tolist(),
                    wrapped_object_angle_delta=float(d[4]),
                    object_position_error_change=float(np.linalg.norm(source.vector(row['final_state'])[2:4]-goal[2:4])-
                        np.linalg.norm(source.vector(center['final_state'])[2:4]-goal[2:4])),
                    wrapped_angle_error_change=float(row['pose_diagnostics']['independent']['wrapped_angle_error']-
                        center['pose_diagnostics']['independent']['wrapped_angle_error']),
                    native_success=bool(row['official_evaluation']['success']))
            entries.append(entry)
        groups.append(dict(root_sha256=ps[0]['root_sha256'], role=ps[0]['role'], planned=17,
            completed=sum(r.get('status')=='completed' and r.get('full_horizon',False) for r in rs), responses=entries))
    return dict(groups=groups,P1_admission=False,controller_selected=False,scope=contract()['limitation'])


def run(prepared, expected, output, factory, *, authorize_physics=False):
    require(authorize_physics is True, 'MAIN explicit physics authorization')
    p=verify(prepared,expected); store=sparse.Store(output); plans=p['inputs']['plans']; goals={}; errors=[]
    rows=[dict(root_sha256=x['root_sha256'],arm=x['arm'],selection_sha256=x['selection_sha256'],
               status='not_started',first_error=None,secondary_errors=[]) for x in plans]
    counts={k:dict(attempted=0,completed=0) for k in ('reset','step','render')}
    try:
        store.emit('run_started',prepare_sha256=expected,contract=contract())
        for index,plan in enumerate(plans):
            row=rows[index]
            env=None
            try:
                store.check(262144); env=factory(); body=env.unwrapped
                path=inspect.getsourcefile(type(body)); sp=p['inputs']['source_plan']
                require(path is not None and sp['source_closure'].get(str(Path(path).resolve()))==sp['environment_sha256'], 'native environment source')
                require(body.relative is True and body.action_scale==100 and body.render_size==224 and body.render_mode=='rgb_array', 'native raw mapping')
                goals[plan['root_sha256']]=source.load_raw_source(plan['raw_source_file'],sp)['goal'].tolist()
                store.active_body=body; row=base.branch(plan,sp,env,store,counts)
                row['body_dynamics']={str(k):v for k,v in store.dynamics.get(plan['selection_sha256'],{}).items()}
            except (Exception,KeyboardInterrupt) as exc:
                row['status']='failed'
                error=dict(type=type(exc).__name__,message=str(exc))
                if row.get('first_error') is None: row['first_error']=error
                else: row.setdefault('secondary_errors',[]).append(error)
            finally:
                if env is not None:
                    try: env.close()
                    except (Exception,KeyboardInterrupt) as exc:
                        row['status']='failed'
                        error=dict(stage='close',type=type(exc).__name__,message=str(exc))
                        if row.get('first_error') is None: row['first_error']=error
                        else: row.setdefault('secondary_errors',[]).append(error)
                store.active_body=None; rows[index]=row; store.emit('branch_closed',receipt=row)
        store.json('analysis.json',analyze(plans,rows,goals))
    except (Exception,KeyboardInterrupt) as exc: errors.append(dict(stage='execution',type=type(exc).__name__,message=str(exc)))
    try: verify(prepared,expected)
    except Exception as exc: errors.append(dict(stage='final_binding',type=type(exc).__name__,message=str(exc)))
    complete=len(rows)==102 and not errors and all(r.get('status')=='completed' and r.get('full_horizon') for r in rows)
    if complete:
        try:
            require(counts=={k:dict(attempted=n,completed=n) for k,n in dict(reset=204,step=1020,render=1122).items()}, 'complete operation census')
        except Exception as exc: errors.append(dict(stage='counts',type=type(exc).__name__,message=str(exc))); complete=False
    status='persistent_suffix_interventions_completed' if complete else 'partial_or_failed'
    result=dict(schema=NAME+'.report',status=status,prepare_sha256=expected,contract=contract(),rows=rows,counts=counts,errors=errors,
        journal_sha256=None,new_neural_calls=0,P1_admission=False,new_independent_samples=0)
    try: store.emit('run_finished',status=status,counts=counts,errors=errors)
    except (Exception,KeyboardInterrupt) as exc:
        errors.append(dict(stage='final_journal',type=type(exc).__name__,message=str(exc))); result['status']='partial_or_failed'
    try: result['journal_sha256']=base.sha(store.path/'journal.jsonl')
    except (Exception,KeyboardInterrupt) as exc:
        errors.append(dict(stage='journal_hash',type=type(exc).__name__,message=str(exc))); result['status']='partial_or_failed'
    try: store.json('report.json',result)
    except (Exception,KeyboardInterrupt) as exc:
        errors.append(dict(stage='final_report',type=type(exc).__name__,message=str(exc))); result['status']='partial_or_failed'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('command',choices=('prepare','verify','run'))
    parser.add_argument('--prepared'); parser.add_argument('--prepare-sha'); parser.add_argument('--output'); parser.add_argument('--authorize-physics',action='store_true')
    a=parser.parse_args()
    if a.command=='prepare': result=prepare(a.output)
    elif a.command=='verify': result=dict(schema=verify(a.prepared,a.prepare_sha)['schema'])
    else:
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224)
        result=run(a.prepared,a.prepare_sha,a.output,factory,authorize_physics=a.authorize_physics)
    print(source.canonical({k:v for k,v in result.items() if k!='rows'}))
    if a.command=='run' and result['status']!='persistent_suffix_interventions_completed': raise SystemExit(2)


if __name__=='__main__': main()
