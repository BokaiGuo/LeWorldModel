"""Independent native goal scene: preparation is model/physics-free; run is main-only.

No factual trajectory, frame15, v11 features, outcomes or candidate selection.
One fixed scene, one reset (including its native .01s physics), zero controls.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

from experiments.spectral_reachability import p5_jepa_factual_pilot_20261027_v1 as base
from experiments.spectral_reachability import p4_feedback_admission_20261027_v1 as saved
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal

TAG='p5_native_goal_scene_20261027_v1'
BRIDGE='p5_prospective_control_bridge_20261027_v1'
REPO=Path(__file__).resolve().parents[2]
ENV=REPO/'external/jepa-wms/evals/simu_env_planning/envs/pusht_env/pusht_env.py'
RUNTIME=REPO/'results/p5_jepa_native_physics_runtime_20261027_v1/manifest_frozen.json'
BASE_SHA='1f67254e90bd7e231c7433e0b8891e3955f24c3f8864eb69f3b49403b3116632'
GOAL=[256.,256.,float(np.pi/4)]
AGENT=[64.,448.]
GOAL_SEED=2026102799
ATOL=1e-10  # Float64 geometry identity precision, never fitted from outcomes.
require=saved.require


def constructor():
    return dict(legacy=False,block_cog=None,damping=None,render_action=False,render_size=224,
        reset_to_state=AGENT+GOAL+[0.,0.],relative=True,action_scale=100,
        with_velocity=True,with_target=True,shape='T',color='LightSlateGray')


def protocol():
    return dict(schema=TAG,object_goal_pose=GOAL,agent_nuisance=AGENT,agent_velocity=[0.,0.],
        seed=GOAL_SEED,constructor=constructor(),selection='one prior fixed native scene; no image/pose/root filtering or retry',
        agent_prior='fixed lower-left canonical scene location64,448; not sampled from factual root or future',
        object_prior='unchanged native fixed goal256,256,pi/4; actual T shape/COG retained',
        expected=dict(env_construct=1,seed=2,reset=1,control_step=0,space_step=1,
            native_render=1,native_get_obs=1,close=1),
        extra_geometry_work='one official _get_goal_pose_body call after reset, no physics step/render',
        reset_boundary='native reset includes exactly one .01s space.step; NOT a physics-free rendering claim',
        geometry='all actual native Poly vertices transformed by live block and official goal body must agree, atol1e-10 rtol0',
        output='native returned RGB only, no compositing/resizing/camera edits; agent is nuisance not success target',
        forbidden_inputs=['factual trajectories','frame15','v11 predictions/features/costs','physical branch outcomes'],
        scope='goal asset generation only; not a control episode, held-out root, coverage measurement or C1/P5 success',
        neural_calls=0,P5_admission=False,C2_established=False)


def prepare():
    inputs=saved.Inputs()
    require(inputs.hash(Path(base.__file__))==BASE_SHA,'sealed capture/helper source differs')
    runtime=inputs.json(RUNTIME)
    require(runtime['seals'],'nonempty prior native runtime source manifest required')
    for path,h in runtime['seals'].items():
        require(inputs.hash(path)==h,'native runtime source drift: '+path)
    source=inputs.raw(ENV).decode()
    require('self.space.step(1.0 / self.sim_hz)' in source and
        'self.goal_pose = np.array([256, 256, np.pi / 4])' in source,'unreviewed reset/native goal contract')
    for path in (Path(__file__),REPO/'tests'/('test_'+TAG+'.py'),REPO/'innerdocs'/(TAG+'.md'),
        Path(saved.__file__),REPO/'experiments/spectral_reachability/d2p_prerun_source_seal.py'):
        inputs.hash(path)
    inputs.stable()
    return dict(schema=TAG,status='prepared_goal_not_generated',protocol=protocol(),bindings=inputs.hashes,
        runtime_manifest_sha256=inputs.hash(RUNTIME),native_source_sha256=inputs.hash(ENV),
        physics_executed=False,neural_executed=False,goal_rgb_available=False,P5_admission=False)


def geometry_witness(env):
    """Live native transforms, not a guessed center-distance or fabricated T."""
    require(np.allclose(np.asarray(env.goal_pose),GOAL,rtol=0,atol=ATOL), 'native goal pose differs')
    require(np.allclose(np.asarray(env.agent.position),AGENT,rtol=0,atol=ATOL),'prior agent nuisance shifted')
    require(np.allclose(np.asarray(env.block.position),GOAL[:2],rtol=0,atol=ATOL) and
        abs(float(env.block.angle)-GOAL[2])<=ATOL,'post-reset native object pose shifted')
    for body in (env.agent,env.block):
        require(np.array_equal(np.asarray(body.velocity),[0.,0.]) and float(body.angular_velocity)==0.,
            'post-reset scene has unplanned velocity')
    goal=env._get_goal_pose_body(np.asarray(GOAL))
    rows=[]
    for shape in env.block.shapes:
        require(type(shape).__name__=='Poly','official T must expose real polygon vertices')
        local=np.asarray([list(v) for v in shape.get_vertices()],np.float64)
        actual=np.asarray([list(env.block.local_to_world(v)) for v in shape.get_vertices()],np.float64)
        target=np.asarray([list(goal.local_to_world(v)) for v in shape.get_vertices()],np.float64)
        require(local.ndim==2 and local.shape[1]==2 and local.shape[0]>=3 and
            np.isfinite(local).all() and np.isfinite(actual).all() and np.isfinite(target).all(), 'finite official polygons required')
        require(np.allclose(actual,target,rtol=0,atol=ATOL),'T polygon does not align with official goal transform')
        rows.append(dict(local_vertices=local.tolist(),actual_world_vertices=actual.tolist(),
            official_goal_world_vertices=target.tolist(),shape_radius=float(shape.radius),
            max_abs_vertex_error=float(np.max(np.abs(actual-target)))))
    require(rows,'missing native block polygons')
    rows.sort(key=lambda r:json.dumps(r,sort_keys=True))
    return dict(object_goal_pose=GOAL,agent_nuisance=AGENT,actual_block_cog=list(env.block.center_of_gravity),
        official_goal_cog=list(goal.center_of_gravity),polygons=rows,atol=ATOL,rtol=0,
        coverage_measured=False,force_contact_claim=False)


def capture_scene(cls,out,journal,calls):
    """Main calls with authenticated native class; tests use metadata-only fakes."""
    env=None;result=None;errors=[]
    try:
        env=base.operation(journal,calls,'goal','construct',dict(config=constructor()),lambda:cls(**constructor()))
        base.save_json(out/'constructor_rng.json',base.rng_state(env))
        base.operation(journal,calls,'goal','seed',dict(seed=GOAL_SEED),lambda:env.seed(GOAL_SEED))
        base.save_json(out/'pre_reset_rng.json',base.rng_state(env))
        observation,state=base.operation(journal,calls,'goal','reset',dict(expected_space_steps=1,dt=.01),env.reset)
        # Retain actual native return assets BEFORE validating any pose/geometry.
        record=base.capture(env,observation,state,dict(kind='independent_goal_scene_reset'),out/'frame_00')
        journal.append(dict(phase='assets_durable',record=record))
        witness=base.operation(journal,calls,'goal','geometry_check',dict(no_advance=True),lambda:geometry_witness(env))
        base.save_json(out/'geometry.json',witness)
        result=dict(record=record,geometry=witness)
    except Exception as exc:
        errors.append(dict(phase='capture_scene',type=type(exc).__name__,message=str(exc)))
    finally:
        if env is not None:
            try:base.operation(journal,calls,'goal','close',{},env.close)
            except Exception as exc:errors.append(dict(phase='close',type=type(exc).__name__,message=str(exc)))
    return result,errors


def directory(output):
    path=Path(output).resolve()
    require(path.parent==REPO/'results' and path.name.startswith(TAG+'_') and not path.exists(), 'exclusive new goal output required')
    base.mkdir(path)
    return path


def prepare_output(output):
    out=directory(output);plan=prepare()
    files=[Path(__file__),REPO/'tests'/('test_'+TAG+'.py'),REPO/'innerdocs'/(TAG+'.md'),
        Path(base.__file__),Path(saved.__file__),ENV,
        REPO/'experiments/spectral_reachability/d2p_prerun_source_seal.py']
    seal=create_seal(files=files,args_json=json.dumps(protocol()),output_dir=out/'source_seal')
    result=dict(plan=plan,source_seal=seal)
    base.save_json(out/'protocol.json',result)
    print(json.dumps(dict(protocol=str(out/'protocol.json'),status=plan['status'],physics_executed=False)))
    return result


def run_main(prepared_path,expected_sha256,output,main_authorized=False):
    require(main_authorized,'explicit main authorization required; this agent never executes run')
    require(os.environ.get('PYTHONPATH')==str(REPO/'.jepa-physics-site') and
        os.environ.get('PYTHONNOUSERSITE')=='1','isolated native physics overlay required')
    require(not any(k=='torch' or k.startswith('torch.') for k in sys.modules),'no neural process for goal generation')
    inputs=saved.Inputs();manifest=inputs.json(prepared_path)
    require(inputs.hash(prepared_path)==expected_sha256 and manifest['plan']==prepare(), 'fixed prior goal plan/source drift')
    inputs.seal(manifest['source_seal'])
    out=directory(output);journal=base.Journal(out/'journal.jsonl');calls=None;owned=False;errors=[];assets=None
    try:
        journal.append(dict(phase='fixed_goal_intent',prepared_sha256=expected_sha256,protocol=protocol()))
        runtime,parent=base.load_runtime(REPO)
        probe=runtime.probe(REPO,parent,RUNTIME);base.save_json(out/'runtime_probe.json',probe)
        require(probe['status']=='native_import_ready_physics_not_executed','native runtime probe not ready')
        spec=importlib.util.spec_from_file_location(TAG+'_official_native',ENV)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        require(Path(module.PushTEnv.__init__.__code__.co_filename).resolve()==ENV,'native constructor provenance differs')
        import pymunk  # Main-only, never imported by prepare/tests.
        require(not any(k=='torch' or k.startswith('torch.') for k in sys.modules),
            'native dependency import unexpectedly introduced neural libraries; no construction authorized')
        calls=base.NativeCalls(module.PushTEnv,pymunk);calls.start();owned=True
        assets,errors=capture_scene(module.PushTEnv,out,journal,calls)
        calls.stop();owned=False
        require(not errors,'goal scene capture failed; retain original failures')
        observed=calls.receipt()['python_call_entries']
        require(all(observed.get(k,0)==v for k,v in protocol()['expected'].items()), 'goal native operation census differs')
        require(len(calls.space_steps)==1 and calls.space_steps[0]['dt']==.01, 'unexpected goal physics advancement')
    except Exception as exc:
        errors.append(dict(phase='run_main',type=type(exc).__name__,message=str(exc)))
    finally:
        if owned:calls.stop()
        try:
            require(manifest['plan']==prepare(),'goal plan/source changed after generation')
            verify_seal(output_dir=manifest['source_seal']['output_dir'],expected_manifest_sha256=manifest['source_seal']['manifest_sha256'])
            inputs.stable()
        except Exception as exc:errors.append(dict(phase='final_integrity',type=type(exc).__name__,message=str(exc)))
    result=dict(schema=TAG,status='native_goal_scene_recorded' if not errors else 'native_goal_scene_failed',
        protocol=protocol(),prepared_sha256=expected_sha256,errors=errors,assets=assets,
        native_calls=None if calls is None else calls.receipt(),new_neural_calls=0,
        physics_executed=bool(calls and calls.counts.get('space_step',0)),P5_admission=False)
    if not errors:
        result['goal_spec_path']=str(out/'goal_spec.json')
    journal.append(dict(phase='terminal',status=result['status'],summary_sha256=hashlib.sha256(base.canonical(result)).hexdigest()))
    journal.close();result['journal_sha256']=base.sha(out/'journal.jsonl');base.save_json(out/'report.json',result)
    # Publish the bridge-usable spec LAST, after durable terminal/report evidence.
    # If publication is interrupted, a missing spec cannot authorize compilation.
    if not errors:
        rgb=out/'frame_00/rgb.npy'
        bindings=dict(manifest['plan']['bindings']);bindings[str(Path(prepared_path).resolve())]=expected_sha256
        generation=dict(schema=BRIDGE+'.goal_generation',status=result['status'],goal_rgb_sha256=base.sha(rgb),
            object_goal_pose=GOAL,agent_nuisance=AGENT,used_factual_future=False,
            method='separate native goal scene; no branch execution',source_bindings=bindings,
            geometry_sha256=base.sha(out/'geometry.json'),native_calls=result['native_calls'],
            execution_report=str(out/'report.json'),execution_report_sha256=base.sha(out/'report.json'),
            journal_sha256=result['journal_sha256'])
        base.save_json(out/'generation.json',generation)
        goal_spec=dict(schema=BRIDGE+'.goal',rgb_path=str(rgb),rgb_sha256=base.sha(rgb),object_goal_pose=GOAL,
            agent_goal_role='scene nuisance; no agent success claim',origin='independently_declared_native_goal_scene',
            generation_receipt=str(out/'generation.json'),generation_receipt_sha256=base.sha(out/'generation.json'))
        base.save_json(out/'goal_spec.json',goal_spec)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',type=Path,required=True)
    r=sub.add_parser('run');r.add_argument('--prepared',type=Path,required=True);r.add_argument('--expected-sha256',required=True)
    r.add_argument('--output',type=Path,required=True);r.add_argument('--main-authorized',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare':prepare_output(args.output)
    else:
        result=run_main(args.prepared,args.expected_sha256,args.output,args.main_authorized)
        print(json.dumps(dict(status=result['status'],errors=result['errors'])))
        if result['errors']:raise SystemExit(2)
