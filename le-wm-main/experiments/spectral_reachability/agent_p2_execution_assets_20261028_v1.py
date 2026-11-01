"""Known-root P2 action-library handoff and MAIN-only calibration physics.

No planner rerun/reselection, no neural calls and no formal scientific release.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import agent_p2_guard_saved_consume_20261028_v1 as saved
from experiments.spectral_reachability import p2_composition_predict_frontier_physical_20261027_v1 as physical

NAME='agent_p2_execution_assets_20261028_v1'
ROOT=saved.ROOT
require,sha,read=saved.require,saved.sha,saved.read
encode=saved.producer.encode
digest=saved.C.digest
ARMS=saved.native.ARMS
LIBRARY=ROOT/'results/agent_p2_guard_saved_consume_20261028_v1_MAIN02.json'
LIBRARY_SHA='c7d183f5b18d1bfacd81f855025efdbbb403f97e5442650970080766db17088b'
REPORT=ROOT/'results/agent_p2_runtime_guard_20261028_v1_known_MAIN01/report.json'
REPORT_SHA='0cfbe37a7783c6df331305e95e3c058a9b006f72bb1ca5b59d24fae047ee23a0'
PREPARED=ROOT/'results/agent_p2_runtime_guard_20261028_v1_prepared01.json'
PREPARE_SHA='02269ae5d08d1f645e9bcf2473a7c7cbb2a16f3125c3fad62c23859380cc6c48'
PHYSICAL_SHA='281f6e8dd422bfa73ddff45ef0ca551ec7f6f1b23c35fca9835fde553f3f8110'
SAVED_SHA='e70886cd383f4f55b3809599ab3c7b33cbbbdc67998bc3df2e39e0ec54372cb2'
FLOOR=200*1024**2
OUTPUT_CAP=100*1024**2
PREPARE_CAP=2*1024**2
WALL=900


def contract():
    return dict(roots=4,arms=list(ARMS),branches=8,raw_steps_per_branch=10,selected_step_cap=80,
        warmup_steps_per_branch=15,warmup_step_cap=120,root_collection_steps=0,
        reset='original sealed deterministic collector seed/oversample, then original reset_and_warmup; exact RGB/context guards',
        root_collection_resets='attempt/return census; original collector has no new finite reset cap here',
        internal_physics_substeps=None,reset_internal_work=None,
        arm_order='root-major; even root original arm order, odd reversed',
        source='original FOUR frontier-v2 known roots; NOT the TWO moving roots',
        goal='original goal0 and original agent+object native success; no object-only replacement',
        metric='original source_Q and original projection; no fitting',
        selection='sealed all80 joint/all40 conditional winner; no physical outcome input or reselection',
        new_predict=0,new_encoder=0,new_projector=0,AD=0,training=False,CPU_neural_fallback=False,
        output_cap_bytes=OUTPUT_CAP,floor_bytes=FLOOR,prepare_cap_bytes=PREPARE_CAP,
        wall_seconds=WALL,wall_scope='between branch and metered public calls; not preemptive watchdog',
        guard='original RoyalBlue fresh RGB guard; failing step may have no post-exception frame',
        known_root_calibration=True,independent_cohort=False,equal_full_resource_cost=False,
        P1_admission=False,P2_admission=False,formal_execution_authorized=False)


def make_plans(library,cases,compilation,rows,normalization):
    """Pure explicit join; old physical plan schema is ONLY a low-level helper ABI."""
    require(len(cases)==len(rows)==4 and len(library['selected_plans'])==8,'fixed4x2 census')
    index={}
    for item in library['selected_plans']:
        key=(item['root_slot'],item['arm'])
        require(type(item['root_slot']) is int and key not in index,'unique literal root/arm')
        index[key]=item
    require(set(index)=={(i,a) for i in range(4) for a in ARMS},'all8 arms, no subset')
    out=[]
    for i,(case,row) in enumerate(zip(cases,rows,strict=True)):
        r=case['root_sha256'];c=compilation[r]
        require(case['root_index']==i and row['root_slot']==i and row['root_sha256']==r,
            'exact original root order')
        require(len(row['results'])==2 and [x['arm'] for x in row['results']]==list(ARMS),'native result arm order')
        initial=case['root']['context']
        expected=dict(physical.planner.META,latent_history=c['input_context']['emb'][0],
            action_history=c['input_context']['actions'][0][1:])
        require(initial==expected and digest(c['goal_vectors'][0])==case['goal_vector_sha256'] and
            digest(c['scoring']['metric']['projection'])==case['projection_sha256'] and
            c['scoring']['metric']['sha256']==case['metric_sha256'],'original goal/Q/full history')
        for arm in (ARMS if i%2==0 else ARMS[::-1]):
            item=index[i,arm];result=row['results'][list(ARMS).index(arm)]
            best=min(result['attempts'],key=lambda x:x['cost'])
            winner=result['winner'];w=best['witness'];receipt=w['receipt']
            require(item['root_sha256']==r and item['report_sha256']==library['report_sha256'] and
                item['physical_execution_authorized'] is False and item['selection_digest']==digest(best) and
                item['candidate_id']==best['candidate_id']==winner['candidate_id'] and
                winner['witness']==w and winner['independent_full_replay_exact'] is True and
                item['native_receipt_sha256']==receipt['receipt_sha256'] and
                item['commands']==best['actions']==receipt['actions'],'authoritative chosen raw10/witness join')
            raw=np.asarray(item['commands']);require(raw.shape==(10,2) and np.isfinite(raw).all() and
                np.array_equal(raw,raw.astype(np.float32)) and np.all(np.abs(raw)<=1),'literal raw10 float32, no cast/clip')
            require([len(e['actions']) for e in w['edges']]==([10] if arm==ARMS[0] else [5,5]) and
                w['edges'][0]['input_context']==initial,'one full10 edge vs two causal5 edges')
            terminal=receipt['terminal_context']
            physical.planner.check_history(initial,terminal,item['commands'],normalization)
            p=dict(schema=physical.NAME+'.plan',root_index=i,root_sha256=r,method=arm,metric_label='source_Q',
                schedule='predict_frontier',active_metric_id=case['metric_sha256'],planned_horizon_tokens=2,
                planned_commands=10,commands=deepcopy(item['commands']),goal=deepcopy(c['goal_vectors'][0]),
                predictor_context=deepcopy(initial),selected_predicted_context=deepcopy(terminal))
            p['selection_sha256']=physical.digest(p);physical.validate_plan(p)
            handoff=dict(schema=NAME+'.plan',root_slot=i,root_sha256=r,arm=arm,source_selection_digest=item['selection_digest'],
                source_report_sha256=library['report_sha256'],candidate_id=item['candidate_id'],
                native_receipt_sha256=item['native_receipt_sha256'],execution_plan=p,
                P1_admission=False,P2_admission=False,known_root_calibration=True)
            handoff['handoff_sha256']=digest(handoff);out.append(handoff)
    return out


def build():
    require(sha(saved.__file__)==SAVED_SHA and sha(physical.__file__)==PHYSICAL_SHA,'frozen helper authority')
    require(sha(LIBRARY)==LIBRARY_SHA and sha(REPORT)==REPORT_SHA and sha(PREPARED)==PREPARE_SHA,'actual fixed library/report/prepare')
    lib=read(LIBRARY);report=read(REPORT);wrapper=read(PREPARED)
    require(lib['schema']==saved.NAME+'.consume' and lib['status']=='terminal_saved_evidence_verified_NOT_physical_release' and
        lib['report_sha256']==REPORT_SHA and lib['prepare_sha256']==PREPARE_SHA and
        lib['root_denominator']==4 and lib['arm_denominator']==8 and lib['candidate_denominator']==480 and
        lib['physical_execution_authorized'] is lib['P1_admission'] is lib['P2_admission'] is False,'accepted saved library, no scientific authority')
    saved.terminal_report(report,PREPARE_SHA,report['main_review_sha256'])
    saved.producer.verify(wrapper,PREPARE_SHA)
    _,body,_=saved.native.load_prepared(wrapper['native_directory'],wrapper['native_prepared_sha256'])
    comp={r['root_sha256']:r['compiled'] for r in read(body['context']['compilation'])['rows']}
    rows=[read(REPORT.parent/f'root_{i}.json') for i in range(4)]
    plans=make_plans(lib,body['context']['cases'],comp,rows,body['context']['normalization'])
    refs=physical.helpers.validate_prepared(ROOT/physical.REFERENCES,physical.REFERENCES_SHA)
    packets={x['root_sha256']:x for x in refs['packets']}
    require(set(packets)=={x['root_sha256'] for x in plans},'same four physical source roots')
    for case in body['context']['cases']:
        r=case['root_sha256'];packet=packets[r];c=comp[r];ref=refs['references'][r]
        require(packet['root_context']==case['root']['context'] and packet['goal']==c['goal_vectors'][0] and
            packet['projection']==c['scoring']['metric']['projection'] and
            case['root']['root_rgb_sha256']==ref['history_rgb_sha256'] and
            case['root']['goal_rgb_sha256']==ref['goal_rgb_sha256'],'source restoration RGB/goal/Q/history')
    require({str(Path(body['context']['policy']).resolve()/n) for n in ('weights.pt','config.json')}==set(refs['model_bindings']),
        'same source model paths')
    bindings={**lib['bindings'],**lib['consumer_sources'],**refs['inputs_sha256'],**refs['sources_sha256'],**refs['model_bindings']}
    # Explicit reference receipt/protocol plus complete existing helper import closure.
    paths=[LIBRARY,REPORT,PREPARED,ROOT/physical.REFERENCES/'prepare.json',ROOT/physical.REFERENCES/'protocol.json',
        Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    paths+=physical.source_files(ROOT)
    for path in paths:
        path=Path(path).resolve();h=sha(path)
        require(str(path) not in bindings or bindings[str(path)]==h,'conflicting source binding')
        bindings[str(path)]=h
    for path,h in bindings.items():require(sha(path)==h,'bound asset/source drift: '+path)
    return dict(schema=NAME+'.prepare',contract=contract(),plans=plans,bindings=bindings,
        library_path=str(LIBRARY),library_sha256=LIBRARY_SHA,planner_report_sha256=REPORT_SHA,
        runtime_prepare_sha256=PREPARE_SHA,root_seed=refs['root_seed'],references=refs['references'],
        inherited_resources=lib['unique_resources'],model_bindings=refs['model_bindings'],
        source_library_is_not_moving_two_roots=True,formal_execution_authorized=False,P1_admission=False,P2_admission=False)


def write(path,value):
    path=Path(path);raw=encode(value)
    require(not path.exists() and not path.is_symlink(),'exclusive NEW artifact')
    with path.open('xb') as f:require(f.write(raw)==len(raw),'short metadata write');f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def prepare(output):
    p=build();raw=encode(p);path=Path(output).absolute()
    require(len(raw)<=PREPARE_CAP and shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'2MiB prepare /200MiB floor')
    write(path,p);return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(raw),new_neural_calls=0,new_physics_steps=0)


def verify(path,expected):
    saved.contract.prior.graph.digest(expected);require(sha(path)==expected,'external prepare SHA')
    p=read(path,PREPARE_CAP);require(p==build(),'entire fresh reconstructed asset/plan closure');return p


def initial_rows(p):
    return [dict(root_slot=x['root_slot'],arm=x['arm'],handoff_sha256=x['handoff_sha256'],
        status='not_started',physical=None) for x in p['plans']]


def execute(prepared,expected,output,review_sha):
    """MAIN-only explicit physics command, never called by prepare/verify/tests."""
    saved.contract.prior.graph.digest(review_sha);p=verify(prepared,expected);output=Path(output).absolute()
    require(not output.exists() and not output.is_symlink() and shutil.disk_usage(output.parent).free>=FLOOR+OUTPUT_CAP,
        'NEW100MiB output +200MiB reserve')
    output.mkdir();started=time.monotonic();rows=initial_rows(p);errors=[];env=meter=stream=None
    def check():
        require(time.monotonic()-started<=WALL,'900s public-call/branch boundary wall')
        used=sum(f.stat().st_size for f in output.iterdir() if f.is_file())
        require(used<OUTPUT_CAP-1024**2 and shutil.disk_usage(output).free>=FLOOR+1024**2,'output/floor boundary gate')
    try:
        write(output/'execution.json',dict(schema=NAME+'.execution',prepared=str(Path(prepared).absolute()),
            prepare_sha256=expected,main_review_sha256=review_sha,planner_report_sha256=REPORT_SHA,
            known_root_calibration=True,P1_admission=False,P2_admission=False))
        from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
        from experiments.spectral_reachability.run_pusht_sampling_comparison_20261027_v2 import verify_rgb_context,official_checked
        env=base.gym.make('swm/PushT-v1',render_mode='rgb_array')
        evaluator=Path(env.unwrapped.eval_state.__code__.co_filename).resolve()
        require(p['bindings'].get(str(evaluator))==sha(evaluator),'bound native evaluator')
        require(tuple(env.unwrapped.variation_space['agent']['color'].value.tolist())==(65,105,225),'RoyalBlue source')
        stream=(output/'physical_budget.jsonl').open('x')
        def event(e):
            check();raw=encode(e)+b'\n';require(stream.write(raw.decode())==len(raw),'short budget journal');stream.flush();os.fsync(stream.fileno())
        meter=physical.PhysicalMeter(env,event)
        roots=base.collect_nonterminal_boundary_contact_roots(meter,count=4,seed=int(p['root_seed']),oversample_factor=4)
        rootmap=physical.helpers.validate_roots(roots,p['references'],base.root_fingerprint);identities={}
        for asset,row in zip(p['plans'],rows,strict=True):
            check();plan=asset['execution_plan'];root=rootmap[plan['root_sha256']];row['status']='attempted';meter.bind(plan)
            def reset():
                meter.phase='warmup';before=deepcopy(meter.steps['warmup'])
                require(base.root_fingerprint(root)==plan['root_sha256'],'exact original root')
                frames,history,goal=base.physical._reset_and_warmup(meter,root,history_len=3)
                require(all(meter.steps['warmup'][k]-before[k]==15 for k in ('attempts','returns')),'exact15 warmup steps')
                identity=verify_rgb_context(frames,goal,p['references'][plan['root_sha256']])
                identity.update(physical.helpers.validate_reset_history(history,plan['predictor_context']['action_history']),
                    state=base.p0.packets.state_of(env).tolist(),goal=np.asarray(root['goal']).tolist())
                r=plan['root_sha256'];require(r not in identities or identities[r]==identity,'paired root reset exact')
                identities[r]=deepcopy(identity);meter.phase='selected';return identity
            def evaluate(state):
                return dict(official=official_checked(env,state,root['goal']),pose=base.physical._pose_label(state,root['goal']))
            result=physical.physical_branch(plan=plan,output=output,reset=reset,env=physical.DiskGuardEnv(meter,output),
                state_of=lambda proxy:base.p0.packets.state_of(proxy.env),evaluate=evaluate)
            row.update(status='returned',physical=result);physical.verify_branch(plan,result)
            require(result['status'] in ('completed','observation_unavailable','environment_terminated'),'engineering fault stops remaining arms')
    except BaseException as exc:errors.append(dict(stage='execution',type=type(exc).__name__,message=str(exc)))
    finally:
        for obj,method,stage in ((env,'close','env_close'),(stream,'close','journal_close')):
            if obj is not None:
                try:getattr(obj,method)()
                except BaseException as exc:errors.append(dict(stage=stage,type=type(exc).__name__,message=str(exc)))
        try:verify(prepared,expected)
        except BaseException as exc:errors.append(dict(stage='final_source',type=type(exc).__name__,message=str(exc)))
        for row in rows:
            if row['status']=='attempted':row['status']='failed_partial'
        hashes={}
        try:
            for f in output.iterdir():hashes[f.name]=sha(f)
        except BaseException as exc:errors.append(dict(stage='hashes',type=type(exc).__name__,message=str(exc)))
        census=meter.receipt() if meter is not None else None
        if meter is not None and meter.journal_errors:errors+=meter.journal_errors
        terminal=not errors and all(x['status']=='returned' for x in rows)
        report=dict(schema=NAME+'.report',status='calibration_recorded' if terminal else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,rows=rows,errors=errors,
            first_error=errors[0] if errors else None,artifact_sha256=hashes,physical_calls=census,
            contract=contract(),root_denominator=4,arm_denominator=8,model_loaded=False,new_neural_calls=0,
            native_success_count=sum(bool(x['physical'].get('official',{}).get('official_success',False)) for x in rows if x['physical']),
            P1_admission=False,P2_admission=False,known_root_calibration=True)
        try:write(output/'report.json',report)
        except BaseException as exc:terminal=False;print('final_report_error',repr(exc),flush=True)
    require(terminal,'partial evidence retained; no automatic retry');return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run-calibration'))
    for k in ('output','prepared','prepare-sha','main-review-sha'):parser.add_argument('--'+k)
    a=parser.parse_args()
    if a.command=='prepare':r=prepare(a.output)
    elif a.command=='verify':r=dict(status='verified_no_dispatch',plans=len(verify(a.prepared,a.prepare_sha)['plans']))
    else:r=execute(a.prepared,a.prepare_sha,a.output,a.main_review_sha)
    print(encode({k:v for k,v in r.items() if k not in ('rows','artifact_sha256')}).decode(),flush=True)


if __name__=='__main__':main()
