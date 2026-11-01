"""Fixed nonzero factual pilot: prepare by default, main-only physics run.

Uses sealed v1 durable capture/profiling helpers, never v1 outcome reports.
No physics/neural import or execution in prepare or module import.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

NAME='p5_jepa_factual_nonzero_20261027_v2'
PARENT='p5_jepa_factual_pilot_20261027_v1'
PARENT_MANIFEST_SHA='00bdcadc6f02e8cb519fe3de84583a6b92df575df6c0a255f68a41d1385f108b'
PARENT_SOURCE_SHA='1f67254e90bd7e231c7433e0b8891e3955f24c3f8864eb69f3b49403b3116632'
ROOT=Path(__file__).resolve().parents[2]
SEED=2026102701
FRAMES=[0,5,10,15]
ARMS=['actual','normalized_zero','negative_normalized','reverse_steps']
SCHEDULE=(
    ((.12,.04),(.08,-.06),(-.05,.09),(.11,.03),(-.04,-.07)),
    ((-.07,.10),(.03,.13),(.09,-.02),(-.11,.05),(.06,-.08)),
    ((.05,-.12),(-.09,.07),(.13,.02),(.02,-.05),(-.06,.11)),
)


def sha(path):
    p=Path(path)
    if p.is_symlink() or not p.is_file():raise ValueError('regular non-symlink file required')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def parent(root):
    path=root/f'results/{PARENT}/manifest_frozen.json'
    if sha(path)!=PARENT_MANIFEST_SHA:raise ValueError('v1 manifest SHA rejected')
    value=json.loads(path.read_text())
    for p,h in value['seals'].items():
        if sha(p)!=h:raise ValueError('v1 bound source drift: '+p)
    source=root/f'experiments/spectral_reachability/{PARENT}.py'
    if sha(source)!=PARENT_SOURCE_SHA:raise ValueError('v1 helper source SHA rejected')
    return path,value,source


def helpers(root):
    _,_,path=parent(root)
    spec=importlib.util.spec_from_file_location(NAME+'_sealed_v1_helpers',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def native_stats(root):
    path=root/'external/jepa-wms/app/plan_common/datasets/pusht_dset.py'
    values={}
    for n in ast.parse(path.read_text()).body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            key=n.targets[0].id
            if key in ('ACTION_MEAN','ACTION_STD','PROPRIO_MEAN','PROPRIO_STD'):
                if not isinstance(n.value,ast.Call) or ast.unparse(n.value.func)!='torch.tensor':raise ValueError('stats not literal native tensor')
                values[key]=np.asarray(ast.literal_eval(n.value.args[0]),np.float32)
    expected=dict(ACTION_MEAN=[-.0087,.0068],ACTION_STD=[.2019,.2002],
        PROPRIO_MEAN=[236.6155,264.5674,-2.93032027,2.54307914],
        PROPRIO_STD=[101.1202,87.0112,74.84556075,74.14009094])
    for key,value in expected.items():
        if key not in values or not np.array_equal(values[key],np.array(value,np.float32)):
            raise ValueError('native normalization drift: '+key)
    return values


def action_inputs(stats):
    u=np.array(SCHEDULE,np.float64).reshape(15,2)
    pixels=100.*u
    dataset_pixels=pixels.astype(np.float32)
    packets=((dataset_pixels/np.float32(100)-stats['ACTION_MEAN'])/stats['ACTION_STD']).reshape(3,10)
    controls=np.repeat(packets[None],4,axis=0)
    controls[1,-1]=0
    controls[2,-1]*=-1
    controls[3,-1]=packets[-1].reshape(5,2)[::-1].reshape(10)
    if any(np.array_equal(controls[i,-1],controls[j,-1]) for i in range(4) for j in range(i)):
        raise ValueError('fixed last-packet controls not distinct')
    return dict(raw_env_commands=u,raw_pixel_displacements=pixels,dataset_pixel_displacements=dataset_pixels,
        normalized_packets=packets,controls_packets=controls)


def protocol():
    return dict(schema=NAME,seed=SEED,roles=['a','b'],controls_per_env=15,
        schedule=np.array(SCHEDULE,np.float64).tolist(),schedule_units='native env.step argument u, float64; internal pixel displacement=100*u; NOT velocity or normalized coordinates',
        schedule_selection='fixed before nonzero execution; no reading zero/nonzero outcomes or action search; no replacement/contact filtering',
        root='same first fixed fresh seed as v1, independently generated and replayed; not an unseen root/cohort',
        execution='two independently constructed source-native envs, seed fixed BEFORE reset; reset .01s then15controls with10substeps each',
        expected=dict(env_construct=2,seed=4,reset=2,control_step=30,space_step=302,native_render=32,native_get_obs=32,close=2),
        frames=FRAMES,context_frames=[0,5,10],target_frame=15,packet_action_indices=[[0,5],[5,10],[10,15]],
        frame_action_alignment='frame00 is post-reset; command01=u[0] advances frame00->01; packet0 advances frame00->05, packet1 05->10, packet2 10->15',
        normalization='float32 native dataset convention: raw_pixel32=(100*u64).astype(float32); z=(raw_pixel32/float32(100)-mean32)/std32; never divide native env u by100 again',
        normalization_precision='actual physics always consumes frozen u64; projection to dataset float32 and inverse error recorded, not exact raw action reconstruction',
        downstream_arms=ARMS,controls_scope='model inputs only, first two packets preserved; no physical counterfactual branches executed',
        native_io='sealed v1 capture/journal/profile helpers; unmodified official source, no monkeypatch; raw RGB/state/proprio/full-public-state every frame0..15 retained',
        identity='exact16paired RGB/state/proprio/full-public-state/transition SHA comparisons, no tolerances; all mismatches retained',
        journal='durable command and hash-chained intent BEFORE construct/seed/reset/step/close; no interrupted-intent resume',
        snapshot_boundary='full DECLARED public state and RNG, NOT private contact-cache serialization or arbitrary snapshot restoration',
        neural_execution=False,action_fidelity_claim=False,control_benefit_claim=False,P5_admission=False)


def prepare(root):
    path,value,_=parent(root)
    stats=native_stats(root);inputs=action_inputs(stats)
    seals=dict(value['seals'])
    for p in [path,Path(__file__).resolve(),root/'tests'/f'test_{NAME}.py',root.parent/'docs'/f'{NAME}.md']:
        seals[str(p)]=sha(p)
    return dict(schema=NAME,status='prepared_no_physics',protocol=protocol(),seals=seals,
        parent_manifest_sha256=PARENT_MANIFEST_SHA,native_stats={k:v.tolist() for k,v in stats.items()},
        action_inputs={k:dict(dtype=v.dtype.str,shape=list(v.shape),values=v.tolist()) for k,v in inputs.items()},
        physics_executed=False,neural_executed=False,P5_admission=False)


def verify(root,manifest):
    if manifest!=prepare(root):raise ValueError('nonzero source/protocol/schedule seal rejected')


def collect_trajectory(base,cls,role,out,journal,calls,inputs):
    env=None;records=[];branch=out/role;base.mkdir(branch)
    try:
        env=base.operation(journal,calls,role,'construct',dict(config=base.CTOR),lambda:cls(**base.CTOR))
        base.save_json(branch/'constructor_rng.json',base.rng_state(env))
        base.operation(journal,calls,role,'seed',dict(seed=SEED),lambda:env.seed(SEED))
        base.save_json(branch/'pre_reset_rng.json',base.rng_state(env))
        obs,state=base.operation(journal,calls,role,'reset',dict(seed=SEED,expected_space_steps=1,dt=.01),env.reset)
        record=base.capture(env,obs,state,dict(kind='reset'),branch/'frame_00');records.append(record)
        journal.append(dict(phase='assets_durable',role=role,record=record))
        for frame,u in enumerate(inputs['raw_env_commands'],start=1):
            agent=np.array(env.agent.position,np.float64)
            command=dict(frame=frame,action_index=frame-1,packet_index=(frame-1)//5,step_within_packet=(frame-1)%5,
                raw_env_argument=u.tolist(),pixel_displacement=(100*u).tolist(),agent_xy_before=agent.tolist(),
                absolute_target=(agent+100*u).tolist(),expected_substeps=10,dt=.01,
                pre_public_sha256=record['files']['full_public_state.json']['sha256'])
            base.save_json(branch/f'command_{frame:02d}.json',command)
            obs,reward,done,info=base.operation(journal,calls,role,'step',command,lambda:env.step(u.copy()))
            record=base.capture(env,obs,info['state'],dict(kind='step',reward=reward,done=done,info=info),branch/f'frame_{frame:02d}')
            records.append(record);journal.append(dict(phase='assets_durable',role=role,record=record))
            if done:raise ValueError('unexpected done, no replacement')
        return records
    finally:
        if env is not None:base.operation(journal,calls,role,'close',{},env.close)


def bounded_array(path,shape,dtype,digest):
    if sha(path)!=digest or Path(path).stat().st_size>200_000:raise ValueError('raw native asset SHA/size rejected')
    a=np.load(path,allow_pickle=False,max_header_size=1024)
    if a.shape!=shape or a.dtype!=np.dtype(dtype) or not np.isfinite(a).all():raise ValueError('native asset dtype/shape/finite rejected')
    return a


def export_inputs(base,branch,records,inputs,stats):
    dest=branch/'consumer';base.mkdir(dest)
    for name,a in inputs.items():base.save_array(dest/(name+'.npy'),a)
    states=[];prop=[];frame_paths=[]
    for frame in FRAMES:
        raw=branch/f'frame_{frame:02d}';record=records[frame]
        if record['frame']!=frame:raise ValueError('consumer frame identity rejected')
        states.append(bounded_array(raw/'state.npy',(7,),np.float32,record['files']['state.npy']['sha256']))
        prop.append(bounded_array(raw/'proprio.npy',(4,),np.float32,record['files']['proprio.npy']['sha256']))
        # RGB stays in raw frame file, no duplicate/resized/transformed copy.
        bounded_array(raw/'rgb.npy',(224,224,3),np.uint8,record['files']['rgb.npy']['sha256'])
        frame_paths.append(dict(frame=frame,rgb=f'{branch.name}/frame_{frame:02d}/rgb.npy',
            state=f'{branch.name}/frame_{frame:02d}/state.npy',proprio=f'{branch.name}/frame_{frame:02d}/proprio.npy',
            full_public_state=f'{branch.name}/frame_{frame:02d}/full_public_state.json',files=record['files']))
    states=np.stack(states);prop=np.stack(prop)
    normalized=(prop-stats['PROPRIO_MEAN'])/stats['PROPRIO_STD']
    for name,a in dict(states_frames=states,raw_proprio_frames=prop,normalized_proprio_frames=normalized,context_proprio=normalized[:3].copy()).items():
        base.save_array(dest/(name+'.npy'),a)
    projected_u=inputs['dataset_pixel_displacements']/np.float32(100)
    reverse_u=inputs['normalized_packets'].reshape(15,2)*stats['ACTION_STD']+stats['ACTION_MEAN']
    result=dict(schema=NAME,role=branch.name,seed=SEED,frame_order=FRAMES,context_frames=[0,5,10],target_frame=15,
        frames=frame_paths,context_proprio_path=f'{branch.name}/consumer/context_proprio.npy',
        normalized_packets_path=f'{branch.name}/consumer/normalized_packets.npy',
        controls_packets_path=f'{branch.name}/consumer/controls_packets.npy',arms=ARMS,
        action_packet_layout='shape3x10, packet row-major five steps xy; row0[0:5],row1[5:10],row2[10:15]',
        file_hashes={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(dest.iterdir())},
        native_stats={k:v.tolist() for k,v in stats.items()},
        precision=dict(dataset_projection_env_u_max_abs=float(np.max(np.abs(projected_u.astype(np.float64)-inputs['raw_env_commands']))),
            inverse_normalization_env_u_max_abs=float(np.max(np.abs(reverse_u.astype(np.float64)-inputs['raw_env_commands'])))),
        target_scope='frame15 successor label only; future consumer must not use target state/proprio/RGB to select schedule or predictor history',
        controls_not_executed_physically=True,neural_executed=False,action_fidelity_claim=False,P5_admission=False)
    base.save_json(dest/'inputs_manifest.json',result)
    return dict(path=f'{branch.name}/consumer/inputs_manifest.json',sha256=sha(dest/'inputs_manifest.json'))


def run(root,manifest,manifest_path,expected_sha,out):
    if sha(manifest_path)!=expected_sha:raise ValueError('explicit manifest SHA rejected')
    verify(root,manifest);base=helpers(root)
    if out.exists():raise FileExistsError('exclusive new output directory required')
    if out.parent.resolve()!=root/f'results/{NAME}':raise ValueError('output must be direct nonzero result child')
    base.mkdir(out);journal=base.Journal(out/'journal.jsonl');calls=None;profile_owned=False;collected={};consumer={};problem=None
    rows=[dict(frame=i,status='not_run') for i in range(16)];source_after=False
    stats=native_stats(root);inputs=action_inputs(stats)
    try:
        # Entire fixed action array is durable BEFORE loading native runtime or
        # constructing/resetting either env. Never derive it from observations.
        base.save_array(out/'schedule_env_u.npy',inputs['raw_env_commands'])
        journal.append(dict(phase='preflight_intent',protocol=protocol(),manifest_sha256=expected_sha,schedule_file_sha256=sha(out/'schedule_env_u.npy')))
        runtime,parent_manifest=base.load_runtime(root)
        probe=runtime.probe(root,parent_manifest,root/f'results/{base.RUNTIME}/manifest_frozen.json')
        base.save_json(out/'runtime_probe.json',probe)
        if probe['status']!='native_import_ready_physics_not_executed':raise ValueError('strict native runtime probe rejected')
        source=root/runtime.ENV_REL
        spec=importlib.util.spec_from_file_location(NAME+'_official_native',source)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        if Path(module.PushTEnv.__init__.__code__.co_filename).resolve()!=source:raise ValueError('native constructor provenance rejected')
        import pymunk  # Main-only path; no import in prepare/tests.
        calls=base.NativeCalls(module.PushTEnv,pymunk);calls.start();profile_owned=True
        try:
            for role in ('a','b'):
                collected[role]=collect_trajectory(base,module.PushTEnv,role,out,journal,calls,inputs)
                consumer[role]=export_inputs(base,out/role,collected[role],inputs,stats)
                journal.append(dict(phase='consumer_assets_durable',role=role,consumer=consumer[role]))
        finally:calls.stop();profile_owned=False
        rows=base.compare_records(collected['a'],collected['b'])
        counts=calls.receipt()['python_call_entries']
        if any(counts.get(k,0)!=v for k,v in protocol()['expected'].items()):raise ValueError('native call counts differ')
        if any(e['dt']!=.01 for e in calls.space_steps):raise ValueError('physics dt differs')
    except Exception as exc:problem=type(exc).__name__+': '+str(exc)
    finally:
        if calls is not None and profile_owned:calls.stop()
        try:verify(root,manifest);source_after=True
        except Exception as exc:problem=(problem+'; ' if problem else '')+str(exc)
    exact=problem is None and source_after and all(x.get('all_exact',False) for x in rows)
    result=dict(schema=NAME,status='fresh_native_nonzero_factual_recorded' if exact else 'fresh_native_nonzero_factual_failed',
        protocol=protocol(),manifest_file_sha256=expected_sha,source_hashes_unchanged=source_after,error=problem,rows=rows,
        consumer_inputs=consumer,trajectory_frames={k:len(v) for k,v in collected.items()},
        native_call_receipts=calls.receipt() if calls else None,physics_executed=bool(calls and calls.counts.get('space_step',0)),
        neural_executed=False,action_fidelity_claim=False,control_benefit_claim=False,P5_admission=False,
        evidence_scope='fixed nonzero factual data plus exact independent replay identity; model action fidelity and counterfactual benefit NOT evaluated')
    journal.append(dict(phase='terminal',status=result['status'],summary_sha256=hashlib.sha256(base.canonical(result)).hexdigest()))
    journal.close();result['journal_sha256']=sha(out/'journal.jsonl');base.save_json(out/'report.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','run'],nargs='?',default='prepare')
    p.add_argument('--manifest',type=Path,default=ROOT/f'results/{NAME}/manifest_frozen.json')
    p.add_argument('--manifest-sha256');p.add_argument('--confirm-physics',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/f'results/{NAME}/factual_main_v1')
    args=p.parse_args()
    if args.command=='prepare':value=prepare(ROOT)
    else:
        if not args.confirm_physics or not args.manifest_sha256:p.error('main-only run requires confirm-physics and manifest-sha256')
        value=run(ROOT,json.loads(args.manifest.read_text()),args.manifest,args.manifest_sha256,args.output)
    print(json.dumps(value,sort_keys=True,indent=2,allow_nan=False))
    if value['status']=='fresh_native_nonzero_factual_failed':raise SystemExit(2)


if __name__=='__main__':main()
