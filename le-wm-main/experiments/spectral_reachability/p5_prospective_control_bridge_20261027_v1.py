"""Prepare/freeze-only C1 architecture bridge; no neural or physics entry point.

Never imports native/torch code. Main supplies independently sealed goal asset,
CUDA compiler and native executor. Existing frame15 is diagnostic, not a goal.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p4_feedback_admission_20261027_v1 as saved
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal

TAG='p5_prospective_control_bridge_20261027_v1'
REPO=Path(__file__).resolve().parents[2]
FACTUAL=REPO/'results/p5_jepa_factual_nonzero_20261027_v2/factual_main_v1'
V11=REPO/'results/p5_jepa_fresh_fidelity_main_20261027_v11/cuda_main_v1/report.json'
ENV=REPO/'external/jepa-wms/evals/simu_env_planning/envs/pusht_env/pusht_env.py'
SEED=2026102701
ARMS=('compiled','nominal','factual_reference')
NAMES=('rgb.npy','state.npy','proprio.npy','full_public_state.json','transition.json')
require=saved.require


def sha(value):
    return hashlib.sha256(value).hexdigest()


def digest(value):
    return sha(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())


def array(inputs,path,shape,dtype):
    a=np.load(io.BytesIO(inputs.raw(path,2*1024*1024)),allow_pickle=False)
    require(a.shape==shape and a.dtype==np.dtype(dtype) and np.isfinite(a).all(),'asset shape/dtype/finite differs')
    return a


def packet(value):
    a=np.asarray(value)
    require(a.shape==(5,2) and a.dtype.kind in 'fi' and np.isfinite(a).all() and np.all(np.abs(a)<=1),
        'finite native-u packet within predeclared bridge support [-1,1] required')
    return a.astype(np.float64).tolist()


def contract():
    return dict(seed=SEED,context_frames=[0,5,10],decision_frame=10,terminal_frame=15,
        prefix_action_interval=[0,10],suffix_action_interval=[10,15],suffix_commands=5,
        input_layout='three actual RGB/proprio frames; two between packets; new outgoing packet appended',
        goal='separately supplied native fixed-object-goal scene, never factual frame15 or its encoding',
        nominal='five native env u=[0,0]; NOT normalized zero and NOT None',
        native_action_support=[-1,1],support_scope='bridge policy, not native advertised absolute-position Box',
        compiler=dict(backend='host_cuda_only',coordinates='ten native normalized outgoing action coordinates',
            derivative='ten sequential native JVP standard basis columns; unchanged frozen native model',
            nominal_env_packet=np.zeros((5,2)).tolist(),damping=.01,
            solve='unchanged v8 ridge solve G=J.T@J/D, b=J.T@(independent_goal_feature-nominal)/D; delta=solve(G+.01I,b)',
            candidates=['nominal','ridge'],ridge='one full ridge step around native-zero nominal; no clipping or retuning; out-of-support fails entire compile',
            selection='nonlinear goal-feature global MSE argmin; stable nominal-first tie; no physical outcome selection',
            replay='exact CUDA replay selected packet before freeze',
            expected_completed=dict(official_encoder=1,official_rgb_frames=4,proprio_encoder=1,
                predictor=13,jvp_columns=10,autograd_grad=20),
            predictor_phases=dict(nominal=1,jvp=10,ridge_nonlinear=1,selected_replay=1)),
        execution=dict(arms=list(ARMS),order=list(ARMS),independent_envs=3,
            root='construct each env separately; explicit same fixed seed BEFORE reset; replay original prefix10',
            forbid='no public-snapshot restoration, live branch reuse, outcome-dependent action edits, reselection or root replacement',
            prefix_gate='all eleven frames0..10 RGB/state/proprio/full-public-state/transition file bytes must match factual role a',
            before_dispatch='compiler result + selected packet + all three branch manifests journaled and fsynced before first branch construction',
            expected=dict(env_construct=3,seed=6,reset=3,control_step=45,space_step=453,
                native_render=48,native_get_obs=48,close=3),
            partial='retain every intent/return/asset; guard_abort, integrity_failure and environment termination distinct; no endpoint padding'),
        evaluation=dict(agent='separate agent displacement/velocity diagnostics; no agent goal invented',
            object='native body-origin XY and wrapped angle errors versus fixed native goal; not agent/velocity-mixed state norm',
            coverage='unmodified native final/max coverage; source threshold .95; native done is hardcoded False',
            image_metric='global patch goal MSE is a mixed-scene proxy, not independently validated object progress',
            observed_frame15='already opened factual diagnostic only; never an optimization target'),
        C1_status='architecture bridge only, pending actual compilation and independent execution',
        C2_status='not_established: no same-root non-greedy/reachable-capability/closed-loop-benefit conjunction',
        P1_admission=False,P5_admission=False,generalization=False)


def diagnostic(public):
    """Existing factual motion, no thresholds fitted and no future packet chosen."""
    require(len(public)==16,'all sixteen saved factual frames required')
    rows=[]
    for start,end in ((0,5),(5,10),(10,15)):
        a,b=public[start],public[end]
        position=lambda f,body: np.asarray(f['bodies'][body]['position'],float)
        ang=lambda f: float(f['bodies']['block']['angle'])
        goal=np.asarray(a['env']['goal_pose']['values'],float)
        require(goal.shape==(3,) and all(v['env']['goal_pose']==a['env']['goal_pose'] for v in public), 'fixed native goal differs')
        unchanged=all(v['bodies']['block']['position']==a['bodies']['block']['position'] and
            v['bodies']['block']['angle']==ang(a) for v in public[start:end+1])
        dtheta=ang(b)-ang(a)
        rows.append(dict(start=start,end=end,agent_displacement=(position(b,'agent')-position(a,'agent')).tolist(),
            object_origin_displacement=(position(b,'block')-position(a,'block')).tolist(),
            object_rotation_principal=float(np.arctan2(np.sin(dtheta),np.cos(dtheta))),
            object_pose_exactly_unchanged_at_all_saved_steps=unchanged,
            object_xy_goal_error=float(np.linalg.norm(position(b,'block')-goal[:2])),
            object_angle_goal_error=float(abs(np.arctan2(np.sin(ang(b)-goal[2]),np.cos(ang(b)-goal[2])))),
            final_native_coverage=b['env']['coverage_arr'][-1],max_native_coverage=max(b['env']['coverage_arr'])))
    return dict(segments=rows,source='saved native float64 public body origin/angle and native coverage, not force inference',
        C2_status=contract()['C2_status'],role='diagnostic only, excluded from compiler bundle')


def factual(inputs):
    manifest=inputs.json(FACTUAL.parent/'manifest_frozen.json')
    report=inputs.json(FACTUAL/'report.json')
    require(manifest['schema']=='p5_jepa_factual_nonzero_20261027_v2' and manifest['seals'] and
        report['status']=='fresh_native_nonzero_factual_recorded' and report['error'] is None and
        report['source_hashes_unchanged'] is True and report['protocol']==manifest['protocol'] and
        report['manifest_file_sha256']==inputs.hash(FACTUAL.parent/'manifest_frozen.json'), 'bound complete factual source required')
    require(report['protocol']['seed']==SEED and report['trajectory_frames']==dict(a=16,b=16) and
        len(report['rows'])==16 and all(r['frame']==i and r['all_exact'] is True and all(r['exact'].values())
        for i,r in enumerate(report['rows'])), 'two complete exact same-seed trajectories required')
    for path,h in manifest['seals'].items():
        require(inputs.hash(path)==h,'factual source seal drift')
    raw=inputs.raw(FACTUAL/'journal.jsonl',32*1024*1024)
    require(sha(raw)==report['journal_sha256'],'factual journal binding differs')
    previous=None;records={}
    for i,line in enumerate(raw.splitlines()):
        e=saved.strict_json(line)
        require(e['sequence']==i and e['previous_sha256']==previous and
            digest({k:v for k,v in e.items() if k!='entry_sha256'})==e['entry_sha256'],'factual hash chain differs')
        previous=e['entry_sha256'];event=e['event']
        if event.get('phase')=='assets_durable':
            key=event['role'],event['record']['frame']
            require(key not in records,'duplicate factual asset receipt')
            records[key]=event['record']
    require(set(records)=={(r,i) for r in ('a','b') for i in range(16)} and
        e['event']['phase']=='terminal' and e['event']['summary_sha256']==digest({k:v for k,v in report.items() if k!='journal_sha256'}),
        'complete terminal factual evidence required')
    for frame in range(16):
        for name in NAMES:
            hashes=[]
            for role in ('a','b'):
                path=FACTUAL/role/f'frame_{frame:02d}'/name
                h=inputs.hash(path);hashes.append(h)
                require(records[role,frame]['files'][name]['sha256']==h,'durable factual asset differs')
            require(hashes[0]==hashes[1],'factual repeat identity differs')
    commands=[]
    for i in range(1,16):
        a=inputs.json(FACTUAL/'a'/f'command_{i:02d}.json')
        require(a['frame']==i and a['action_index']==i-1,'factual action time differs')
        commands.append(a['raw_env_argument'])
    require(np.array_equal(commands,np.asarray(report['protocol']['schedule']).reshape(15,2)), 'predeclared factual schedule differs')
    public=[inputs.json(FACTUAL/'a'/f'frame_{i:02d}'/'full_public_state.json') for i in range(16)]
    return commands,public


def goal_asset(inputs,spec_path,forbidden):
    spec=inputs.json(spec_path)
    require(set(spec)=={'schema','rgb_path','rgb_sha256','object_goal_pose','agent_goal_role','origin','generation_receipt','generation_receipt_sha256'},
        'explicit independent goal asset schema required')
    require(spec['schema']==TAG+'.goal' and spec['origin']=='independently_declared_native_goal_scene' and
        spec['agent_goal_role']=='scene nuisance; no agent success claim' and
        spec['object_goal_pose']==[256.,256.,float(np.pi/4)], 'fixed native object goal/provenance required')
    path=Path(spec['rgb_path']).resolve()
    require(not path.is_relative_to(FACTUAL.parent) and path.name!='frame15.npy','factual output cannot supply goalRGB')
    image=array(inputs,path,(224,224,3),np.uint8)
    require(inputs.hash(path)==spec['rgb_sha256'] and sha(image.tobytes()) not in forbidden,
        'goal cannot be any saved factual frame or renamed factual target')
    receipt=inputs.json(spec['generation_receipt'])
    require(inputs.hash(spec['generation_receipt'])==spec['generation_receipt_sha256'] and
        receipt['schema']==TAG+'.goal_generation' and receipt['goal_rgb_sha256']==spec['rgb_sha256'] and
        receipt['object_goal_pose']==spec['object_goal_pose'] and receipt['used_factual_future'] is False and
        receipt['method']=='separate native goal scene; no branch execution' and receipt['source_bindings'],
        'reviewable independent native goal generation receipt required')
    require(all(inputs.hash(p)==h for p,h in receipt['source_bindings'].items()),'goal generation sources drift')
    return spec


def prepare(goal_spec=None,compiler_source=None,executor_source=None):
    inputs=saved.Inputs();commands,public=factual(inputs)
    v11=inputs.json(V11)
    require(v11['status']=='fresh_native_fidelity_recorded' and v11['model_unchanged'] and
        v11['counts_match'] and v11['sources_unchanged'] and not v11['finalization_errors'] and
        not v11['source_check_errors'] and v11['model_before']==v11['model_after'], 'complete actual v11 identity prerequisite required')
    # Only model identity is carried onward, never v11 costs/targets/predictions.
    frames=[];allowed={};all_frame_pixels=[]
    for i in range(16):
        path=FACTUAL/'a'/f'frame_{i:02d}'/'rgb.npy'
        all_frame_pixels.append(sha(array(inputs,path,(224,224,3),np.uint8).tobytes()))
    for i in (0,5,10):
        row=dict(frame=i)
        for name,shape,dtype in (('rgb',(224,224,3),np.uint8),('proprio',(4,),np.float32)):
            path=FACTUAL/'a'/f'frame_{i:02d}'/(name+'.npy')
            array(inputs,path,shape,dtype)
            row[name]=dict(path=str(path),sha256=inputs.hash(path));allowed[str(path)]=inputs.hash(path)
        frames.append(row)
    blockers=[];goal=None;runners={}
    if goal_spec is None:
        blockers.append('separately generated and provenance-bound native goalRGB required; frame15 forbidden')
    else:
        goal=goal_asset(inputs,goal_spec,all_frame_pixels)
        allowed[str(Path(goal['rgb_path']).resolve())]=goal['rgb_sha256']
    for role,path in (('compiler',compiler_source),('executor',executor_source)):
        if path is None:
            blockers.append('reviewed new '+role+' source required before ready protocol freeze')
        else:
            path=Path(path).resolve()
            require(path.is_relative_to(REPO) and path.suffix=='.py','explicit local runner source required')
            runners[role]=dict(path=str(path),sha256=inputs.hash(path))
    bundle=dict(frames=frames,history_env_packets=np.asarray(commands[:10]).reshape(2,5,2).tolist(),
        goal_rgb=None if goal is None else dict(path=goal['rgb_path'],sha256=goal['rgb_sha256']),
        allowed_data_files=allowed,forbidden='no frame15/proprio15/target features, physical evaluator state, v11 predictions or branch outcomes')
    prefix_identity=[dict(frame=i,files={name:dict(path=str(FACTUAL/'a'/f'frame_{i:02d}'/name),
        sha256=inputs.hash(FACTUAL/'a'/f'frame_{i:02d}'/name)) for name in NAMES}) for i in range(11)]
    inputs.stable()
    return dict(schema=TAG,status='ready_for_main_CUDA_compile' if not blockers else 'blocked_pre_execution_inputs',
        contract=contract(),blockers=blockers,compiler_inputs=bundle,compiler_inputs_sha256=digest(bundle),
        goal=goal,runner_sources=runners,model_identity=v11['model_after'],prefix_env_commands=commands[:10],
        factual_reference_packet=packet(commands[10:]),prefix_identity=prefix_identity,
        root_identity_sha256=digest(dict(seed=SEED,root=prefix_identity[0])),
        native_constructor={**public[0]['env']['settings'],'reset_to_state':None},
        native_environment_source=dict(path=str(ENV),sha256=inputs.hash(ENV)),
        factual_diagnostic=diagnostic(public),bindings=inputs.hashes,
        new_neural_calls=0,new_physics_steps=0,P5_admission=False,C2_established=False)


def compiler_bundle(prepared,expected_protocol_sha256):
    """Only allowed numeric compiler inputs; no evaluator/diagnostic fields."""
    require(prepared['status']=='ready_for_main_CUDA_compile' and not prepared['blockers'], 'prepare is blocked')
    require(digest(prepared)==expected_protocol_sha256,'independently retained protocol SHA differs')
    require(digest(prepared['compiler_inputs'])==prepared['compiler_inputs_sha256'],'compiler input binding differs')
    inputs=saved.Inputs();b=prepared['compiler_inputs']
    require(all(inputs.hash(p)==h for p,h in b['allowed_data_files'].items()),'compiler allowed input drift')
    return dict(rgb_context=np.stack([array(inputs,f['rgb']['path'],(224,224,3),np.uint8) for f in b['frames']]),
        raw_proprio_context=np.stack([array(inputs,f['proprio']['path'],(4,),np.float32) for f in b['frames']]),
        history_env_packets=np.asarray(b['history_env_packets'],np.float64),
        goal_rgb=array(inputs,b['goal_rgb']['path'],(224,224,3),np.uint8))


def frozen_branches(prepared,receipt):
    """Validate a main-generated CUDA receipt, not its derivative/MSE arrays."""
    require(prepared['status']=='ready_for_main_CUDA_compile' and not prepared['blockers'],'prepare is blocked')
    require(receipt['schema']==TAG+'.compiler' and receipt['status']=='selected_packet_replayed' and
        receipt['prepared_sha256']==digest(prepared) and receipt['compiler_inputs_sha256']==prepared['compiler_inputs_sha256'] and
        receipt['compiler_data_bindings']==prepared['compiler_inputs']['allowed_data_files'] and
        receipt['goal_rgb_sha256']==prepared['goal']['rgb_sha256'] and
        receipt['compiler_source']==prepared['runner_sources']['compiler'] and receipt['backend']=='host_cuda_only' and
        receipt['model_before']==receipt['model_after']==prepared['model_identity'] and
        receipt['used_observed_frame15'] is False and receipt['physical_calls']==0 and
        receipt['completed']==contract()['compiler']['expected_completed'] and
        receipt['attempted']==receipt['completed'] and receipt['selected_replay_exact'] is True,
        'complete source/model/input-bound CUDA compiler receipt required')
    require(receipt['artifact_bindings'] and receipt['partial_work_unknown'] is False,'compiler artifacts/accounting missing')
    roles=('response_columns','goal_features','nominal_successor','ridge_successor','selected_replay','gram','rhs','ridge_delta')
    require(set(receipt['artifact_roles'])==set(roles) and all(
        receipt['artifact_roles'][key] in receipt['artifact_bindings'] for key in roles), 'complete derivative/prediction artifact roles required')
    candidates=receipt['candidates']
    require([c['id'] for c in candidates]==['nominal','ridge'],'fixed nominal/ridge candidates required')
    packets=[packet(c['env_packet']) for c in candidates]
    require(packets[0]==np.zeros((5,2)).tolist(),'nominal must be native env zero')
    scores=np.asarray([c['independent_goal_mse'] for c in candidates])
    require(scores.shape==(2,) and scores.dtype.kind in 'fi' and np.isfinite(scores).all() and np.all(scores>=0), 'finite goal scores required')
    index=int(np.argmin(scores))
    require(receipt['selected_id']==candidates[index]['id'] and receipt['selected_env_packet']==packets[index], 'frozen nonlinear argmin binding differs')
    by=dict(compiled=packets[index],nominal=packets[0],factual_reference=prepared['factual_reference_packet'])
    return [dict(arm=arm,seed=SEED,prefix_env_commands=prepared['prefix_env_commands'],
        suffix_env_packet=by[arm],compiler_receipt_sha256=digest(receipt),
        root_identity_sha256=prepared['root_identity_sha256'],
        status='planned_not_constructed',expected_terminal_frame=15) for arm in ARMS]


def output_dir(path):
    path=Path(path).resolve()
    require(path.parent==REPO/'results' and path.name.startswith(TAG+'_') and not path.exists(),'new exclusive bridge output required')
    path.mkdir(exist_ok=False)
    return path


def write(path,value):
    with Path(path).open('xb') as stream:
        stream.write((json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())
        stream.flush();os.fsync(stream.fileno())
    fd=os.open(Path(path).parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def prepare_output(output,goal_spec=None,compiler_source=None,executor_source=None):
    out=output_dir(output)
    own=[Path(__file__).resolve(),REPO/'tests'/('test_'+TAG+'.py'),REPO/'innerdocs'/(TAG+'.md'),
        Path(saved.__file__),REPO/'experiments/spectral_reachability/d2p_prerun_source_seal.py']
    seal=create_seal(files=own,args_json=json.dumps(dict(goal_spec=str(goal_spec),compiler_source=str(compiler_source),
        executor_source=str(executor_source))),output_dir=out/'source_seal')
    result=prepare(goal_spec,compiler_source,executor_source)
    result['audit_source_seal']=seal
    write(out/'protocol.json',result)
    verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
    print(json.dumps(dict(protocol=str(out/'protocol.json'),status=result['status'],blockers=result['blockers'])))
    return result


def freeze_output(prepared_path,receipt_path,output):
    inputs=saved.Inputs();prepared=inputs.json(prepared_path);receipt=inputs.json(receipt_path)
    require(prepared['schema']==TAG and prepared['contract']==contract(),'unsupported prepared protocol')
    inputs.seal(prepared['audit_source_seal'])
    args=inputs.json(Path(prepared['audit_source_seal']['output_dir'])/'args.json')
    values=[None if args[k]=='None' else Path(args[k]) for k in ('goal_spec','compiler_source','executor_source')]
    rebuilt=prepare(*values);rebuilt['audit_source_seal']=prepared['audit_source_seal']
    require(rebuilt==prepared,'prepared goal/source/input protocol does not reproduce')
    require(all(inputs.hash(p)==h for p,h in prepared['bindings'].items()),'prepared input/source drift')
    branches=frozen_branches(prepared,receipt)
    require(all(inputs.hash(p)==h for p,h in receipt['artifact_bindings'].items()),'compiler artifact/source binding drift')
    out=output_dir(output)
    write(out/'selected_packet.json',dict(packet=receipt['selected_env_packet'],selection_sha256=digest(receipt)))
    write(out/'all_branches.json',dict(branches=branches,prepared_file_sha256=inputs.hash(prepared_path),
        prepared_path=str(Path(prepared_path).resolve()),compiler_receipt_path=str(Path(receipt_path).resolve()),
        compiler_receipt_file_sha256=inputs.hash(receipt_path),execution_authority='main only; no execution in this module'))
    inputs.stable()
    # All three immutable branch slots are durable before main may construct ANY env.
    write(out/'freeze_receipt.json',dict(schema=TAG+'.freeze',status='all_three_branches_durable_not_executed',
        census={a:'planned_not_constructed' for a in ARMS},selected_packet_sha256=inputs.hash(out/'selected_packet.json'),
        all_branches_sha256=inputs.hash(out/'all_branches.json'),source_seal=prepared['audit_source_seal'],
        new_neural_calls=0,new_physics_steps=0,P5_admission=False,C2_established=False))
    inputs.stable()
    return branches


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--goal-spec',type=Path);p.add_argument('--compiler-source',type=Path);p.add_argument('--executor-source',type=Path)
    f=sub.add_parser('freeze');f.add_argument('--prepared',type=Path,required=True)
    f.add_argument('--compiler-receipt',type=Path,required=True);f.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare':prepare_output(args.output,args.goal_spec,args.compiler_source,args.executor_source)
    else:freeze_output(args.prepared,args.compiler_receipt,args.output)
