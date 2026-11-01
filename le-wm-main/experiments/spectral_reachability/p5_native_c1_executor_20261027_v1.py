"""Main-only native executor for Ohm bridge v1. Default source prepare is non-physical.

Consumes actual frozen_branches receipts, never generates or chooses a packet.
Three independent envs; exact eleven-frame prefix gate before ANY suffix action.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

NAME = 'p5_native_c1_executor_20261027_v1'
BRIDGE = 'p5_prospective_control_bridge_20261027_v1'
BRIDGE_SHA = '80662a845f0416a9024ee0dfe171b44195628061cd429a11a8cfcda286a172c0'
PANEL = 'p5_jepa_factual_panel_20261027_v3'
PANEL_MANIFEST_SHA = '67b719d332a9405b6f4f6dd1adfdf2f5bdbd5c2e8bede1e16d73360a6470d516'
PANEL_SOURCE_SHA = '0d4f1d983ab46f78bc21b79161de1105ed92f9e6ca2c910ab014667bb30019ef'
ROOT = Path(__file__).resolve().parents[2]
ARMS = ('compiled','nominal','factual_reference')
NAMES = ('rgb.npy','state.npy','proprio.npy','full_public_state.json','transition.json')
SEED = 2026102701


class IntegrityFailure(ValueError):
    """Prefix/source identity failure: no further suffix dispatch."""


class EnvironmentTermination(ValueError):
    """Native done: retain endpoint as returned, never pad it."""


def require(ok,message):
    if not ok:
        raise ValueError(message)


def sha(path):
    path=Path(path)
    require(path.is_file() and not any(p.is_symlink() for p in (path,*path.parents)),'regular nonsymlink file required')
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    sha(path)
    require(Path(path).stat().st_size<=4*1024**2,'JSON byte bound')
    def pairs(items):
        out={}
        for k,v in items:
            require(k not in out,'duplicate JSON key')
            out[k]=v
        return out
    value=json.loads(Path(path).read_text(),object_pairs_hook=pairs,
                    parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
    json.dumps(value,allow_nan=False)
    return value


def import_bound(path,expected,tag):
    require(sha(path)==expected,'source SHA rejected: '+str(path))
    spec=importlib.util.spec_from_file_location(NAME+'_'+tag,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dependencies(root):
    panel_manifest=root/f'results/{PANEL}/manifest_frozen.json'
    require(sha(panel_manifest)==PANEL_MANIFEST_SHA,'panel helper manifest rejected')
    pm=read_json(panel_manifest)
    for path,h in pm['seals'].items():
        require(sha(path)==h,'inherited helper/source drift')
    panel=import_bound(root/f'experiments/spectral_reachability/{PANEL}.py',PANEL_SOURCE_SHA,'sealed_budget_helpers')
    v2,base,_=panel.bound(root)
    bridge=import_bound(root/f'experiments/spectral_reachability/{BRIDGE}.py',BRIDGE_SHA,'bridge')
    require(bridge.contract()['seed']==SEED and tuple(bridge.contract()['execution']['arms'])==ARMS,'unsupported bridge contract')
    prior=root/f'results/{BRIDGE}_prepare_v1/protocol.json'
    old=read_json(prior)
    inputs=bridge.saved.Inputs()
    inputs.seal(old['audit_source_seal'])
    inputs.stable()
    seals={**pm['seals'],**inputs.hashes,str(panel_manifest):PANEL_MANIFEST_SHA,str(prior):sha(prior)}
    return panel,v2,base,bridge,seals


def source_prepare(root):
    _,_,_,bridge,seals=dependencies(root)
    for path in (Path(__file__).resolve(),root/f'tests/test_{NAME}.py',root.parent/f'docs/{NAME}.md'):
        seals[str(path)]=sha(path)
    return dict(schema=NAME,status='prepared_executor_source_no_branch_authority',contract=bridge.contract(),seals=seals,
                arms=list(ARMS),seed=SEED,physics_executed=False,neural_executed=False,C2_established=False,P5_admission=False,
                missing_authority='main must supply actual bridge freeze_receipt/all_branches/selected_packet after independent-goal CUDA compile; source prepare grants no dispatch')


def ready_prepare(root,source_manifest_path,freeze_dir):
    source=read_json(source_manifest_path)
    require(source==source_prepare(root),'executor source manifest drift')
    _,_,_,bridge,_=dependencies(root)
    freeze_dir=Path(freeze_dir).resolve()
    inputs=bridge.saved.Inputs()
    paths={name:freeze_dir/(name+'.json') for name in ('freeze_receipt','all_branches','selected_packet')}
    for path in paths.values():
        sha(path)
    freeze,all_branches,selected=(inputs.json(paths[name]) for name in ('freeze_receipt','all_branches','selected_packet'))
    require(freeze['schema']==BRIDGE+'.freeze' and freeze['status']=='all_three_branches_durable_not_executed'
            and freeze['census']=={a:'planned_not_constructed' for a in ARMS} and freeze['new_physics_steps']==freeze['new_neural_calls']==0,'actual three-branch freeze required')
    require(inputs.hash(paths['all_branches'])==freeze['all_branches_sha256'] and inputs.hash(paths['selected_packet'])==freeze['selected_packet_sha256'],'freeze file hashes differ')
    prepared_path=Path(all_branches['prepared_path']);receipt_path=Path(all_branches['compiler_receipt_path'])
    require(sha(prepared_path)==all_branches['prepared_file_sha256'] and sha(receipt_path)==all_branches['compiler_receipt_file_sha256'],'compiler/protocol FILE hashes differ')
    prepared,receipt=inputs.json(prepared_path),inputs.json(receipt_path)
    require(prepared['schema']==BRIDGE and prepared['contract']==bridge.contract() and freeze['source_seal']==prepared['audit_source_seal'],'exact bridge source/contract required')
    inputs.seal(prepared['audit_source_seal'])
    args=inputs.json(Path(prepared['audit_source_seal']['output_dir'])/'args.json')
    values=[None if args[k]=='None' else Path(args[k]) for k in ('goal_spec','compiler_source','executor_source')]
    rebuilt=bridge.prepare(*values);rebuilt['audit_source_seal']=prepared['audit_source_seal']
    require(rebuilt==prepared,'ready bridge does not reproduce')
    require(prepared['runner_sources']['executor']==dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),'bridge must name this exact reviewed executor source')
    branches=bridge.frozen_branches(prepared,receipt)
    require(all_branches['branches']==branches,'three-branch packet census differs from frozen_branches')
    require(selected==dict(packet=receipt['selected_env_packet'],selection_sha256=bridge.digest(receipt)),'selected packet is not the frozen replayed packet')
    seals=dict(source['seals'])
    for bank in (prepared['bindings'],receipt['artifact_bindings'],inputs.hashes):
        for path,h in bank.items():
            require(path not in seals or seals[path]==h,'input/source hash conflict')
            require(sha(path)==h,'ready input/source drift')
            seals[path]=h
    seals[str(source_manifest_path.resolve())]=sha(source_manifest_path)
    inputs.stable()
    return dict(schema=NAME,status='ready_three_branch_native_execution_not_run',source_manifest=str(source_manifest_path.resolve()),freeze_dir=str(freeze_dir),seals=seals,
                contract=bridge.contract(),branches=branches,prefix_identity=prepared['prefix_identity'],native_constructor=prepared['native_constructor'],
                native_environment_source=prepared['native_environment_source'],goal_pose=prepared['goal']['object_goal_pose'],
                selected_packet=selected,physics_executed=False,neural_executed=False,C2_established=False,P5_admission=False)


def verify_ready(root,manifest):
    require(manifest==ready_prepare(root,Path(manifest['source_manifest']),Path(manifest['freeze_dir'])),'ready execution manifest/source drift')


def prefix_gate(path,expected,record):
    if record['frame']!=expected['frame']:
        raise IntegrityFailure('prefix frame order rejected')
    for name in NAMES:
        native=path/name;factual=Path(expected['files'][name]['path']);wanted=expected['files'][name]['sha256']
        if (sha(native)!=wanted or sha(factual)!=wanted or record['files'][name]['sha256']!=wanted
                or native.stat().st_size>200_000 or factual.stat().st_size>200_000
                or native.read_bytes()!=factual.read_bytes()):
            raise IntegrityFailure('prefix byte mismatch frame'+str(expected['frame'])+' '+name)


def object_metrics(public):
    body=public['bodies']['block'];agent=public['bodies']['agent']
    goal=np.asarray(public['env']['goal_pose']['values'],np.float64)
    require(goal.tolist()==[256.,256.,float(np.pi/4)],'unmodified native goal required')
    theta=float(body['angle'])-goal[2]
    coverage=public['env']['coverage_arr']
    return dict(agent_xy=agent['position'],agent_velocity=agent['velocity'],object_xy=body['position'],object_angle_native_rad=body['angle'],goal_pose=goal.tolist(),
                object_xy_goal_error=float(np.linalg.norm(np.array(body['position'])-goal[:2])),object_angle_goal_error=float(abs(np.arctan2(np.sin(theta),np.cos(theta)))),
                native_final_coverage=coverage[-1] if coverage else None,native_max_coverage=max(coverage) if coverage else None,
                C2_established=False,P5_admission=False)


def collect_arm(panel,base,writer,cls,branch,manifest,out,journal,calls,records,gates):
    """No selection, normalization, clipping, snapshot restore or model calls."""
    env=None
    arm=branch['arm'];dest=out/arm
    prefix=np.asarray(branch['prefix_env_commands'],np.float64)
    suffix=np.asarray(branch['suffix_env_packet'],np.float64)
    require(prefix.shape==(10,2) and suffix.shape==(5,2) and branch['seed']==SEED,'exact frozen branch packet shapes/seed')
    require(manifest['native_constructor']==base.CTOR,'constructor must match factual root exactly')
    try:
        env=base.operation(journal,calls,arm,'construct',dict(config=manifest['native_constructor']),lambda:cls(**manifest['native_constructor']))
        writer.save_json(dest/'constructor_rng.json',base.rng_state(env))
        base.operation(journal,calls,arm,'seed',dict(seed=SEED),lambda:env.seed(SEED))
        writer.save_json(dest/'pre_reset_rng.json',base.rng_state(env))
        obs,state=base.operation(journal,calls,arm,'reset',dict(seed=SEED,expected_space_steps=1,dt=.01),env.reset)
        record=panel.capture(base,writer,env,obs,state,dict(kind='reset'),dest/'frame_00')
        records.append(record);journal.append(dict(phase='assets_durable',arm=arm,record=record))
        prefix_gate(dest/'frame_00',manifest['prefix_identity'][0],record)
        gates.append(0);journal.append(dict(phase='prefix_byte_gate_passed',frame=0,arm=arm))
        for frame,u in enumerate(np.concatenate((prefix,suffix)),start=1):
            if frame>10:
                require(gates==list(range(11)),'all eleven prefix gates required BEFORE suffix')
            pos=np.array(env.agent.position,np.float64)
            command=dict(frame=frame,action_index=frame-1,packet_index=(frame-1)//5,step_within_packet=(frame-1)%5,
                         raw_env_argument=u.tolist(),pixel_displacement=(100*u).tolist(),agent_xy_before=pos.tolist(),absolute_target=(pos+100*u).tolist(),
                         expected_substeps=10,dt=.01,pre_public_sha256=record['files']['full_public_state.json']['sha256'])
            writer.save_json(dest/f'command_{frame:02d}.json',command)
            obs,reward,done,info=base.operation(journal,calls,arm,'step',command,lambda:env.step(u.copy()))
            record=panel.capture(base,writer,env,obs,info['state'],dict(kind='step',reward=reward,done=done,info=info),dest/f'frame_{frame:02d}')
            records.append(record);journal.append(dict(phase='assets_durable',arm=arm,record=record))
            if frame<=10:
                prefix_gate(dest/f'frame_{frame:02d}',manifest['prefix_identity'][frame],record)
                gates.append(frame);journal.append(dict(phase='prefix_byte_gate_passed',frame=frame,arm=arm))
            if done:
                raise EnvironmentTermination('native done at frame'+str(frame)+'; no endpoint padding')
    finally:
        if env is not None:
            original=sys.exc_info()[1]
            try:
                base.operation(journal,calls,arm,'close',{},env.close)
            except Exception as exc:
                journal.append(dict(phase='close_failed',arm=arm,error=type(exc).__name__+': '+str(exc),
                                    original_error=None if original is None else type(original).__name__+': '+str(original)),final=True)
                if original is None:
                    raise


def run(root,manifest,manifest_path,expected_sha,out):
    require(sha(manifest_path)==expected_sha,'ready manifest FILE SHA required')
    verify_ready(root,manifest)
    panel,_,base,_,_=dependencies(root)
    require(not out.exists() and out.is_absolute() and out.parent.resolve()==root/f'results/{NAME}','exclusive direct executor result child required')
    require(not any(p.is_symlink() for p in (out.parent,*out.parent.parents)),'output symlink rejected')
    writer=panel.BudgetIO(base);writer.mkdir(out)
    rows=[dict(arm=arm,status='not_run',frames=0,prefix_gate_frames=[]) for arm in ARMS]
    journals={}
    for arm in ARMS:
        writer.mkdir(out/arm);journals[arm]=writer.journal(out/arm/'journal.jsonl')
    # ALL packets/branches and authority durable before ANY native runtime import.
    writer.save_json(out/'all_branches.json',dict(branches=manifest['branches'],ready_manifest_file_sha256=expected_sha))
    writer.save_json(out/'selected_packet.json',manifest['selected_packet'])
    writer.save_json(out/'dispatch_intent.json',dict(rows=rows,contract=manifest['contract'],ready_manifest_file_sha256=expected_sha))
    writer.save_json(out/'progress_00.json',dict(rows=rows))
    for branch in manifest['branches']:
        writer.save_array(out/branch['arm']/'frozen_env_commands.npy',np.concatenate((np.asarray(branch['prefix_env_commands'],np.float64),np.asarray(branch['suffix_env_packet'],np.float64))))
    global_error=None;cls=None;pymunk=None
    try:
        runtime,parent=base.load_runtime(root)
        probe=runtime.probe(root,parent,root/f'results/{base.RUNTIME}/manifest_frozen.json')
        writer.save_json(out/'runtime_probe.json',probe)
        require(probe['status']=='native_import_ready_physics_not_executed','strict isolated runtime probe rejected')
        native=manifest['native_environment_source']
        require(native['path']==str(root/runtime.ENV_REL),'native source path differs')
        module=import_bound(Path(native['path']),native['sha256'],'unmodified_native')
        cls=module.PushTEnv
        require(Path(cls.__init__.__code__.co_filename).resolve()==Path(native['path']),'native constructor provenance')
        import pymunk  # main-only confirmed run path; never source/ready prepare.
    except Exception as exc:
        global_error=type(exc).__name__+': '+str(exc)
    for index,(row,branch) in enumerate(zip(rows,manifest['branches']),start=1):
        arm=row['arm'];journal=journals[arm];records=[];gates=[];calls=None;owned=False
        status,error='not_run',global_error
        if global_error is None:
            try:
                verify_ready(root,manifest)
                journal.append(dict(phase='preflight_intent',branch=branch,ready_manifest_file_sha256=expected_sha))
                calls=base.NativeCalls(cls,pymunk);calls.start();owned=True
                try:
                    collect_arm(panel,base,writer,cls,branch,manifest,out,journal,calls,records,gates)
                finally:
                    calls.stop();owned=False
                expected={k:v//3 for k,v in manifest['contract']['execution']['expected'].items()}
                require(len(records)==16 and gates==list(range(11)),'complete gated trajectory required')
                require(all(calls.counts.get(k,0)==v for k,v in expected.items()) and all(e['dt']==.01 for e in calls.space_steps),'native151step/call census mismatch')
                status='completed'
            except IntegrityFailure as exc:
                status,error='integrity_failure',str(exc)
            except EnvironmentTermination as exc:
                status,error='environment_termination',str(exc)
            except Exception as exc:
                status,error='guard_abort',type(exc).__name__+': '+str(exc)
            finally:
                if owned:
                    calls.stop()
        row.update(status=status,error=error,frames=len(records),prefix_gate_frames=gates,
                   native_call_receipts=calls.receipt() if calls else None,physics_executed=bool(calls and calls.counts.get('space_step',0)))
        if len(records)==16:
            try:
                row['endpoint']=object_metrics(read_json(out/arm/'frame_15/full_public_state.json'))
            except Exception as exc:
                row['endpoint_error']=type(exc).__name__+': '+str(exc)
                row['status']='guard_abort'
        source_after=False
        try:
            verify_ready(root,manifest);source_after=True
        except Exception as exc:
            global_error='source drift: '+str(exc)
            row['source_error']=global_error
            row['status']='integrity_failure'
        row['source_hashes_unchanged']=source_after
        journal.append(dict(phase='terminal',status=row['status'],summary_sha256=hashlib.sha256(base.canonical(row)).hexdigest()),final=True)
        journal.close();row['journal_sha256']=sha(out/arm/'journal.jsonl')
        writer.save_json(out/arm/'report.json',row,final=True)
        row['report_sha256']=sha(out/arm/'report.json')
        writer.save_json(out/f'progress_{index:02d}.json',dict(rows=rows),final=True)
    result=dict(schema=NAME,status='native_c1_three_branch_execution_recorded' if all(r['status']=='completed' for r in rows) else 'native_c1_three_branch_execution_incomplete',
                rows=rows,ready_manifest_file_sha256=expected_sha,global_error=global_error,physics_executed=any(r['physics_executed'] for r in rows),neural_executed=False,
                C1_effectiveness_claim=False,C2_established=False,P5_admission=False,
                scope='fixed independently replayed branches only; no packet reselection, observed-target fitting, root replacement or control-benefit claim')
    writer.save_json(out/'report.json',result,final=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','ready','run'],nargs='?',default='prepare')
    parser.add_argument('--source-manifest',type=Path,default=ROOT/f'results/{NAME}/manifest_source_frozen.json')
    parser.add_argument('--freeze-dir',type=Path)
    parser.add_argument('--manifest',type=Path,default=ROOT/f'results/{NAME}/manifest_ready_frozen.json')
    parser.add_argument('--manifest-sha256')
    parser.add_argument('--confirm-physics',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/f'results/{NAME}/native_main_v1')
    args=parser.parse_args()
    if args.command=='prepare':
        value=source_prepare(ROOT)
    elif args.command=='ready':
        if args.freeze_dir is None:
            parser.error('--freeze-dir required')
        value=ready_prepare(ROOT,args.source_manifest,args.freeze_dir)
    else:
        if not args.confirm_physics or not args.manifest_sha256:
            parser.error('main-only --confirm-physics and --manifest-sha256 required')
        value=run(ROOT,read_json(args.manifest),args.manifest,args.manifest_sha256,args.output)
    print(json.dumps(value,sort_keys=True,indent=2,allow_nan=False))
    if value['status']=='native_c1_three_branch_execution_incomplete':
        raise SystemExit(2)


if __name__=='__main__':
    main()
