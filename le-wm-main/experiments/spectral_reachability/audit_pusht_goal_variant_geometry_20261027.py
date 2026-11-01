"""Capture actual simulator state AT goal rendering, no neural execution."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def geometry(requested, observed):
    a,b=np.asarray(requested,float),np.asarray(observed,float)
    if a.shape!=(7,) or b.shape!=(7,) or not np.isfinite([a,b]).all():
        raise ValueError('finite seven-state geometry required')
    angle=float(abs(np.arctan2(np.sin(a[4]-b[4]),np.cos(a[4]-b[4]))))
    xy=float(np.linalg.norm(a[2:4]-b[2:4]))
    return dict(object_xy_displacement=xy,object_angle_displacement=angle,
        agent_xy_displacement=float(np.linalg.norm(a[:2]-b[:2])),
        object_geometry_unchanged=xy<=1e-9 and angle<=1e-9)


def run(intervention, output):
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal,verify_seal
    intervention,output=map(Path,(intervention,output))
    source=json.loads(intervention.read_text())
    if source['status']!='completed' or not source['gates'] or not all(v is True for v in source['gates'].values()):
        raise ValueError('complete original goal intervention required')
    bank_path=Path(source['protocol']['bank_path'])
    bank=json.loads(bank_path.read_text())
    upstream=base.physical._load_preflight(Path(bank['upstream_path']))
    bindings={str(p.resolve()):base.digest(p) for p in (intervention,bank_path,Path(bank['upstream_path']))}
    if bindings[str(bank_path.resolve())]!=source['protocol']['bank_sha256']:
        raise ValueError('original goal bank changed')
    output.mkdir(exist_ok=False)
    env=base.gym.make('swm/PushT-v1',render_mode='rgb_array')
    unit=env.unwrapped
    original=unit.render
    captures=[]
    def capture():
        state=np.asarray(unit._get_obs(),float).copy()
        rgb=np.asarray(original(),np.uint8).copy()
        captures.append(dict(state=state.tolist(),rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
        return rgb
    had_override='render' in vars(unit);old_override=vars(unit).get('render')
    protocol=dict(bindings=bindings,offsets=source['protocol']['goal_agent_xy_offsets'],
        scope='actual state at original 20 target renders; no planning, no model, no new future branches',
        physical_state_is_diagnostic_only=True,P1_admission=False)
    seal=create_seal(files=[Path(__file__),Path(base.__file__),Path(base.physical.__file__),
        Path(original.__func__.__code__.co_filename)],args_json=json.dumps(protocol),output_dir=output/'source_seal')
    base.write_new(output/'protocol.json',protocol)
    rows=[]
    try:
        roots=base.collect_nonterminal_boundary_contact_roots(env,count=4,
            seed=int(upstream['protocol']['root_seed']),oversample_factor=4)
        references={r['root_sha256']:r for r in source['rows']}
        if len(roots)!=4 or {base.root_fingerprint(r) for r in roots}!=set(references):
            raise ValueError('same four roots required')
        unit.render=capture
        for index,root in enumerate(roots):
            key=base.root_fingerprint(root);saved=references[key];variants=[]
            for variant in saved['variants']:
                goal=np.asarray(root['goal'],float).copy();goal[:2]+=variant['offset']
                captures.clear()
                _,info=env.reset(seed=0,options={'state':np.asarray(root['state']).copy(),'goal_state':goal})
                if not captures:raise ValueError('goal render hook was not invoked')
                first=captures[0]
                target_hash=hashlib.sha256(np.asarray(info['goal'],np.uint8).tobytes()).hexdigest()
                if first['rgb_sha256']!=target_hash or target_hash!=variant['goal_rgb_sha256']:
                    raise ValueError('actual captured goal render differs from sealed image')
                variants.append(dict(offset=variant['offset'],requested_goal=goal.tolist(),
                    actual_goal_render_state=first['state'],goal_rgb_sha256=target_hash,
                    **geometry(goal,first['state'])))
            rows.append(dict(root_index=index,root_sha256=key,variants=variants))
    finally:
        if had_override:unit.render=old_override
        elif 'render' in vars(unit):delattr(unit,'render')
        env.close()
    unchanged=all(base.digest(Path(p))==sha for p,sha in bindings.items())
    verification=verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
    flat=[v for r in rows for v in r['variants']]
    gates=dict(all20_goal_renders=len(flat)==20,all_goal_images_match_original=True,inputs_unchanged=unchanged)
    base.write_new(output/'report.json',dict(schema='pusht_goal_variant_actual_geometry_20261027.v1',
        status='geometry_audit_completed' if all(gates.values()) else 'integrity_fail',
        protocol=protocol,rows=rows,gates=gates,source_seal=seal,source_verification=verification,
        changed_object_geometries=sum(not v['object_geometry_unchanged'] for v in flat),
        new_predict_samples=0,new_encoder_samples=0,P1_admission=False))
    print(json.dumps(dict(goal_renders=len(flat),changed_object_geometries=sum(not v['object_geometry_unchanged'] for v in flat))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('intervention','output'):p.add_argument('--'+name,type=Path,required=True)
    run(**vars(p.parse_args()))
