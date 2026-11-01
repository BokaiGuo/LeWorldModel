"""Saved-only setup input join; no physics, model call or full strict replay.

The caller pins the MAIN resolver preparation and separately authenticated
control5 body. Original paired source loader verifies its own frozen inputs.
"""
import hashlib
import io
import numpy as np
from PIL import Image
from experiments.spectral_reachability import agent_e3_gd_host_cuda_20261029_v1 as host
from experiments.spectral_reachability import p1_moving_overlay_recompile_physical_20261027_v1 as source

need=host.need


def assemble(prepared, data, cell, body):
    r=host.numerical.resolver
    need(type(cell) is int and 0<=cell<16 and prepared['schema']==r.NAME+'.prepare'
         and prepared['ready']==16 and len(prepared['cells'])==16, 'fixed complete saved resolver')
    slot=prepared['cells'][cell]; fixed=r.roster()[cell]
    need(all(slot[k]==fixed[k] for k in ('index','cell_id','root','goal_index','horizon_packets','domain'))
         and slot['status']=='input_ready_NOT_run_authorized' and slot['task']['body']==body,
         'control5 to exact resolver cell body')
    root=slot['root'];prov=prepared['root_provenance'][root]
    need(prov['root']==root and prov['status']=='qualified_saved_prefix', 'qualified fixed root')
    evidence=prov['provenance']; native=data['roots'][root]
    need(native['root_sha256']==body['root_sha256']==evidence['original_task']['binding']['root_sha256'],
         'original source root identity')
    branches=evidence['root_native_branches'];need(len(branches)==2, 'both original native branches')
    a,b=branches
    for k in ('state','snapshot','pixel_sha256'):
        need(a['c0'][k]==b['c0'][k] and a['c5'][k]==b['c5'][k], 'both same actual root and cp5')
    warm=native['warm'];packet=native['packets']['off']
    need(all(k in warm for k in (0,5,10)), 'actual original warm snapshots')
    states=np.asarray([warm[k]['state'] for k in (0,5,10)],dtype=np.float64)
    history=np.asarray(packet['rgb_history']);raw=np.asarray(packet['raw_action_history'])
    need(states.shape==(3,7) and np.isfinite(states).all() and history.shape==(3,224,224,3)
         and history.dtype==np.uint8 and raw.shape==(10,2) and raw.dtype==np.float32 and np.isfinite(raw).all(),
         'original actual warm state/RGB/action ABI')
    need(states[-1].astype('<f8').tobytes()==np.asarray(a['c0']['state'],dtype='<f8').tobytes()
         and warm[10]['snapshot']==a['c0']['snapshot'], 'original warm endpoint matches control0')
    prefix=np.asarray(evidence['prefix5'],dtype=np.float32)
    need(prefix.shape==(5,2) and np.isfinite(prefix).all() and (np.abs(prefix)<=1).all(), 'literal original prefix')
    goal=slot['goal_evidence'];ev=slot['evaluator']
    need(goal['native_goal']==ev['native_goal'] and goal['goal_pixel_sha256']==ev['goal_pixel_sha256']
         and goal['latent']==body['goal'], 'same fixed goal across physical and planner')
    if slot['goal_index']==0:
        rgb=np.asarray(packet['goal_rgb'])
    else:
        refs=goal['RGB_files'];need(type(refs) is list and len(refs)==2, 'both saved actual G1 RGB refs')
        ref=refs[0];data_bytes=host.bounded(ref['path'],r.CAPS['PNG'])
        need(hashlib.sha256(data_bytes).hexdigest()==ref['sha256']
             and prepared['bindings'].get(ref['path'])==ref['sha256'], 'G1 pinned PNG bytes')
        with Image.open(io.BytesIO(data_bytes)) as im:
            need(im.format=='PNG' and im.mode=='RGB' and im.size==(224,224),'G1 original RGB')
            rgb=np.asarray(im).copy()
    need(rgb.dtype==np.uint8 and rgb.shape==(224,224,3)
         and hashlib.sha256(rgb.tobytes()).hexdigest()==goal['goal_pixel_sha256'], 'exact original goal RGB pixels')
    cp=a['c5'];need(type(cp['step']) is int and cp['step']==5, 'actual control5')
    ref=dict(slot=native['slot'],raw_history=raw,history=history,states=states,
             dynamics=[warm[k]['snapshot'] for k in (0,5,10)],goal=native['goal'])
    return dict(ref=ref,source_plan=data['source_plan'],original_root_sha=native['root_sha256'],
                new_goal=np.asarray(goal['native_goal'],dtype=np.float64),new_rgb=rgb,
                expected_goal=np.asarray(ev['native_goal'],dtype=np.float64),
                expected_goal_pixel=goal['goal_pixel_sha256'],prefix5=prefix,expected_cp5=cp)


def resolve(prepared_path,prepared_sha,cell,body):
    prepared=host.read(prepared_path,prepared_sha,host.numerical.resolver.CAPS['prepared'])
    data=source.source_material()
    value=assemble(prepared,data,cell,body)
    host.source.rehash(data['bindings'])
    need(host.sha(prepared_path)==prepared_sha,'resolver stable across join')
    value['source_bindings']=dict(data['bindings'])
    return value
