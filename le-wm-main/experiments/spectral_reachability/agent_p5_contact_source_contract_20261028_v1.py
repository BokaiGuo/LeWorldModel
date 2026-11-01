"""Saved-only OOD contact-source contract; no simulator/model runner or release.

Native reset_to_state is a supported API but changes the old random-root contract.
Geometry is a candidate precondition, never an observed contact/action witness.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

TAG = 'agent_p5_contact_source_contract_20261028_v1'
ROOT = Path(__file__).resolve().parents[2]
BASE = 'p5_jepa_factual_pilot_20261027_v1'
GEOMETRY = 'p5_native_contact_geometry_main_20261027_v1'
GOAL = 'p5_native_goal_scene_20261027_v1'
PRODUCER = 'agent_p5_native_action_words_20261028_v2'
ARMS = ('push_x','raw_zero')
SEED = 2026102801
STATE = [180.,271.,256.,256.,0.,0.,0.]
MIB = 1024**2
PINS = {
    f'experiments/spectral_reachability/{BASE}.py':'1f67254e90bd7e231c7433e0b8891e3955f24c3f8864eb69f3b49403b3116632',
    f'experiments/spectral_reachability/{GEOMETRY}.py':'e14cf437c7c8963ec191f54d66117b75309d993354c86c44b764ee70a35ddd4a',
    f'experiments/spectral_reachability/{GOAL}.py':'ea2e77e43a6e46f9f9c8aafd6c728eeb999cbfefd0284656f5492178c8ece16d',
    f'experiments/spectral_reachability/{PRODUCER}.py':'5dc5d158ff817d9f39dca0f82ea9eb562f0f9566cc3e74a2abfd2be1af4d74ce',
    'external/jepa-wms/evals/simu_env_planning/envs/pusht_env/pusht_env.py':'6ba34790b6c2ce017a76322c2f1c205ef97d4d3b08a620f844a57075c608a723',
    'results/p5_jepa_native_physics_runtime_20261027_v1/manifest_frozen.json':'76953de72838fcce618862f175f4242a18b3030ec11c6342196f88d919df48b4',
    'results/p5_jepa_factual_panel_20261027_v3/factual_main_v1/seed_2026102702/a/frame_00/full_public_state.json':'5ae584a1f6766bea6ae5a58e14f068bd88e32a2dec7a8701693fbf14001d9757',
    f'results/{GOAL}_scene_main_v1/goal_spec.json':'45824c9abd99cccf279eabe21a0f9ec5c4a8662f9d40ebe7845249cce4819fd1',
    f'results/{GOAL}_scene_main_v1/generation.json':'a1336e95a3d764d1693844e557951cc925e40751d70d010746d88faf8d5c2f4f',
    f'results/{GOAL}_scene_main_v1/report.json':'eb3776d2d0e6851eec0e40b56c0a9bed3e33bdebdddca8d1fbe3d2ef866e76e2',
    f'results/{PRODUCER}_prepare_MAIN01/manifest.json':'1515cb228c6ab0a4fe826074324fd566781b06812a1e5163c41652492518a905',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def sha(path):
    path = Path(path)
    require(path.is_file() and not any(p.is_symlink() for p in (path,*path.parents)),'regular nonsymlink file: '+str(path))
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(MIB),b''):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    sha(path)
    require(Path(path).stat().st_size<=8*MIB,'JSON cap')
    def pairs(items):
        out = {}
        for k,v in items:
            require(k not in out,'duplicate JSON key')
            out[k] = v
        return out
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,
        parse_constant=lambda s:require(False,'nonfinite JSON '+s))


def load(name):
    rel = f'experiments/spectral_reachability/{name}.py'
    require(sha(ROOT/rel)==PINS[rel],'bound helper source')
    spec = importlib.util.spec_from_file_location(TAG+'_'+name,ROOT/rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def constructor(original):
    require(original['reset_to_state'] is None and original['relative'] is True
            and original['action_scale']==100 and original['with_velocity'] is True,'original reset contract')
    return {**original,'reset_to_state':list(STATE)}


def reference_queues():
    push = np.tile(np.array([.1,0.],np.float64),(10,1))
    return dict(push_x=push,raw_zero=np.zeros((10,2),np.float64))


def plans(original):
    return [dict(arm=arm,seed=SEED,constructor=constructor(original),
                 prefix_raw_u=np.zeros((10,2),np.float64).tolist(),
                 reference_suffix_raw_u=queue.tolist()) for arm,queue in reference_queues().items()]


def geometry_candidate(geometry, saved_shape):
    """Reuse sealed actual native local shape metadata, not old body placement."""
    shape_only = dict(shapes=saved_shape['shapes'],
                      bodies=dict(block=dict(position=STATE[2:4],angle=STATE[4])))
    polygons,radius = geometry.geometry(shape_only)
    start = np.asarray(STATE[:2],np.float64)
    first_target = start+100*reference_queues()['push_x'][0]
    initial = geometry.gap(start,polygons,radius)
    target = geometry.gap(first_target,polygons,radius)
    require(initial==1. and target==-9.,'fixed side-gap geometry; no outcome adjustment')
    return dict(agent_radius=radius,block_local_origin=STATE[2:4],block_angle=STATE[4],
                world_polygons=[p.tolist() for p in polygons],agent_initial_center=start.tolist(),
                initial_circle_polygon_gap=initial,first_push_absolute_target=first_target.tolist(),
                first_target_static_geometry_gap=target,
                status='static_contact_candidate_not_observed',actual_contacts=None,
                limitation='static target overlaps circle clearance; no swept-path/PD/native contact proof')


def blind_schema():
    # No source plan, seed, pose, witness, reference suffix or private pointer.
    return dict(schema=TAG+'.blind_schema',status='schema_only_not_planner_ready',
        ready=False,context_frames=[0,5,10],
        files=dict(context_rgb_00=['uint8',[224,224,3]],context_rgb_05=['uint8',[224,224,3]],
                   context_rgb_10=['uint8',[224,224,3]],context_proprio=['float32',[3,4]],
                   history_raw_u=['float64',[10,2]],history_normalized=['float32',[2,10]],
                   goal_rgb=['uint8',[224,224,3]]),
        publication='separate public directory of exact allowlisted numeric arrays only',
        release='requires separate MAIN-reviewed native source generator and actual strict consumer; this module cannot release')


def contract(original, candidate, stats, goal_rgb_sha):
    expected = dict(env_construct=2,seed=4,reset=2,control_step=40,space_step=402,
                    native_render=42,native_get_obs=42,close=2,setup=2,set_state=2,
                    space_construct=2,native_goal_body=82,body_construct=88)
    return dict(schema=TAG,status='saved_contact_source_proposal_not_generated',
        calibration='deliberately constructed OOD source, NOT original v12 random roots, fresh cohort or D2P contribution',
        constructor_change=dict(field='reset_to_state',old=None,new=list(STATE),
            API='unmodified native constructor+reset; no post-reset setter or saved-state restore'),
        generator_private=dict(plans=plans(original),geometry=candidate,
            fixed_order=list(ARMS),denominator=2,actual_attempts=0,
            flow='fresh constructor->explicit seed->native reset once->prefix10->suffix10->close for each arm',
            native_reset='one .01s Space.step in _set_state, authoritative initial state is POST reset',
            prefix='ten raw64 zero controls, real physics, not pause/warm-start from old trace',
            identity='both independent prefix frames0..10: all five native files byte-exact before reference suffix',
            state7='agent_x,agent_y,block_x,block_y,block_angle_mod2pi,agent_vx,agent_vy',
            native_goal='unchanged [256,256,pi/4], with_target=True, no setter/extra render',
            expected_native_call_entries=expected,
            accounting='each axis separate; actual Python call entries plus durable native operation intent/return, failure not refunded',
            failure='retain both denominator slots and all partial assets; no retry/replacement/cap top-up',
            required_witness=['actual reset geometry/zero velocities/goal/native state validation',
                'all11frame prefix byte gates between independently rebuilt arms',
                'all suffix10 frames native block/agent pose/velocity/coverage/contact/RGB and command receipts',
                'push callback contacts vs raw_zero, block native response vs prefix and paired control',
                'source/report/asset SHA closure, attempt-return census, close and hook cleanup'],
            missing_actual_witness=None),
        frozen_native_action_stats=stats,independent_goal_rgb_sha256=goal_rgb_sha,
        budget=dict(output_cap_bytes=128*MIB,payload_cap_bytes=96*MIB,final_reserve_bytes=8*MIB,
                    max_nodes=2048,overhead_per_node=8192,free_floor_bytes=200*MIB,wall_seconds=900),
        missing=['new exclusive native generator/strict source consumer not implemented by this contract',
                 'actual source contacts and object response not observed',
                 'actual post-reset/prefix RGB/proprio public context not generated',
                 'MAIN native execution approval and actual source consumer report SHA absent'],
        scientific_admission=False,task_success=None,action_witness_established=False,
        source_witness_to_planner=False,neural_executed=False,physics_executed=False,
        deployment=blind_schema())


def prepare():
    bindings = {}
    for rel,h in PINS.items():
        require(sha(ROOT/rel)==h,'existing source/asset drift: '+rel)
        bindings[str(ROOT/rel)] = h
    base,geometry = load(BASE),load(GEOMETRY)
    # Checks sealed runtime/native metadata only: never runtime.probe or import env.
    path,runtime = base.runtime_manifest(ROOT)
    bindings.update(runtime['seals'])
    sample = ROOT/'results/p5_jepa_factual_panel_20261027_v3/factual_main_v1/seed_2026102702/a/frame_00/full_public_state.json'
    candidate = geometry_candidate(geometry,read_json(sample))
    pm = read_json(ROOT/f'results/{PRODUCER}_prepare_MAIN01/manifest.json')
    stats = {k:pm['roots'][0]['stats'][k] for k in ('ACTION_MEAN','ACTION_STD')}
    require(all({k:r['stats'][k] for k in stats}==stats for r in pm['roots']),'frozen shared action stats')
    goal = read_json(ROOT/f'results/{GOAL}_scene_main_v1/goal_spec.json')
    generation = read_json(goal['generation_receipt'])
    require(sha(goal['generation_receipt'])==goal['generation_receipt_sha256']
            and generation['status']=='native_goal_scene_recorded'
            and generation['used_factual_future'] is False
            and sha(goal['rgb_path'])==goal['rgb_sha256']==generation['goal_rgb_sha256'],'existing independent goal asset')
    bindings[goal['rgb_path']] = goal['rgb_sha256']
    result = contract(base.CTOR,candidate,stats,goal['rgb_sha256'])
    for p in (Path(__file__).resolve(),ROOT/'tests'/f'test_{TAG}.py',ROOT/'innerdocs'/f'{TAG}.md'):
        bindings[str(p)] = sha(p)
    require(all(sha(p)==h for p,h in bindings.items()),'prepare input drift')
    return dict(**result,bindings=bindings)


def project_blind(public_dir, stats, goal_rgb_sha):
    """Allowlist projection of separately published arrays, NEVER source release.

    Even perfect numeric files are not certified native captures. No source
    receipt, geometry, contact label or reference suffix can cross this ABI.
    """
    public_dir = Path(public_dir)
    require(public_dir.is_dir() and not any(p.is_symlink() for p in (public_dir,*public_dir.parents)), 'public directory nonsymlink')
    schema = blind_schema()
    wanted = {k+'.npy' for k in schema['files']}
    require({p.name for p in public_dir.iterdir()}==wanted,'public directory exact allowlist; no witness/source extras')
    arrays,records = {},{}
    for key,(dtype,shape) in schema['files'].items():
        p = public_dir/(key+'.npy')
        h = sha(p)
        require(p.stat().st_size<=200000,'public array cap')
        a = np.load(p,allow_pickle=False,max_header_size=1024)
        require(a.dtype==np.dtype(dtype) and a.shape==tuple(shape) and np.isfinite(a).all(),'public array ABI')
        arrays[key] = a
        records[key] = dict(path=str(p.resolve()),sha256=h,bytes=p.stat().st_size,dtype=dtype,shape=shape)
    require(np.array_equal(arrays['history_raw_u'],np.zeros((10,2),np.float64)), 'exact fixed prefix raw history only')
    producer = load(PRODUCER)
    normalized = producer.normalize(arrays['history_raw_u'].reshape(2,5,2),stats)
    require(np.array_equal(arrays['history_normalized'],normalized),'native historical normalization; raw zero != normalized zero')
    require(records['goal_rgb']['sha256']==goal_rgb_sha,'copy original independent goal RGB, no new scene')
    require(all(sha(rec['path'])==rec['sha256'] for rec in records.values()),'publication input drift')
    return dict(schema=TAG+'.blind_payload_projection',status='unreleased_numeric_projection',ready=False,
        inputs=records,source_witness_included=False,
        missing='actual separately reviewed native source consumer approval; numeric ABI is not a capture certificate')


def write_new(path, value):
    path = Path(path)
    require(path.parent.is_dir() and not any(p.is_symlink() for p in (path,*path.parents)),'new nonsymlink output parent')
    data = canonical(value)+b'\n'
    require(len(data)<=2*MIB,'output cap')
    with path.open('xb') as f:
        require(f.write(data)==len(data),'short write')
        f.flush()
        os.fsync(f.fileno())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','verify','project-blind'])
    p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--contract-sha256')
    p.add_argument('--public-dir',type=Path)
    p.add_argument('--output',type=Path)
    a = p.parse_args(argv)
    if a.command=='prepare':
        require(not a.contract.exists(),'new contract only')
        value = prepare()
        write_new(a.contract,value)
        result = dict(status=value['status'],path=str(a.contract.resolve()),sha256=sha(a.contract))
    else:
        require(a.contract_sha256 and sha(a.contract)==a.contract_sha256,'actual contract FILE SHA')
        value = read_json(a.contract)
        require(value==prepare(),'contract/source drift')
        if a.command=='verify':
            result = dict(status='saved_contract_verified_not_generated',ready=False)
        else:
            require(a.public_dir is not None and a.output is not None,'actual public-dir and new output required')
            result = project_blind(a.public_dir,value['frozen_native_action_stats'],value['independent_goal_rgb_sha256'])
            write_new(a.output,result)
    print(json.dumps(result,sort_keys=True,allow_nan=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
