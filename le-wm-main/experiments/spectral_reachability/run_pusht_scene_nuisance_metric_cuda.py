"""Fixed scene-conditioned RGB nuisance Q attempt; no control or P1 claim."""
import argparse
import inspect
import json
from pathlib import Path
import numpy as np


def run(goal_metric,probes,policy,output):
    import torch
    from PIL import Image
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability import pusht_scene_nuisance_metric as method
    from experiments.spectral_reachability.pusht_fixed_residual_response import central_difference
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal,verify_seal
    goal_metric,probes,policy,output=map(Path,(goal_metric,probes,policy,output))
    previous=json.loads(goal_metric.read_text())
    if previous['status']!='construction_completed' or not previous['gates'] or not all(v is True for v in previous['gates'].values()):raise ValueError('completed goal Q required')
    for s in previous['source_seals']:verify_seal(output_dir=s['output_dir'],expected_manifest_sha256=s['manifest_sha256'])
    bindings=dict(previous['protocol']['bindings']);bindings.update(previous['metric_hashes'])
    bindings[str(goal_metric.resolve())]=base.digest(goal_metric)
    if bindings.get(str(probes.resolve()))!=base.digest(probes):raise ValueError('same complete saved probes required')
    if not all(base.digest(Path(p))==d for p,d in bindings.items()):raise ValueError('input changed')
    original=next(Path(p) for p,d in bindings.items() if d==checked.INPUT_SHA)
    _,_,config=checked.validate_inputs(original,policy)
    originals=json.loads(original.read_text())
    reports=[json.loads(Path(p).read_text()) for p in originals['protocol']['bindings'] if Path(p).name=='report.json']
    intervention=next(r for r in reports if 'bank_path' in r.get('protocol',{}))
    bank=json.loads(Path(intervention['protocol']['bank_path']).read_text())
    upstream=base.physical._load_preflight(Path(bank['upstream_path']))
    prior={}
    for p in previous['metric_hashes']:
        v=json.loads(Path(p).read_text());key=v['root_sha256']
        if key in prior:raise ValueError('duplicate metric root')
        prior[key]=v
    if len(prior)!=4:raise ValueError('four causal root-goal pairs required')
    output.mkdir(exist_ok=False)
    protocol=dict(bindings=bindings,agent_grid=list(method.POSITIONS),expected_encoder_calls=68,
        uses_current_and_goal_RGB=True,object_basis='same six goal augmentations as prior run',
        physical_state_or_probe_outcomes_used_to_construct_Q=False,all_metrics_saved_before_probe_diagnostics=True,
        known_calibration=True,approximate_image_edits_not_physical_counterfactuals=True,P1_admission=False)
    files=checked.sources_before_allocate(originals,config)
    files += [Path(__file__),Path(method.__file__),Path(inspect.getfile(central_difference)),Path(base.__file__),Path(base.physical.__file__),
              Path(inspect.getfile(base.collect_nonterminal_boundary_contact_roots))]
    files=sorted(set(p.resolve() for p in files))
    seals=[create_seal(files=files[i:i+50],args_json=json.dumps(protocol),output_dir=output/f'source_seal_{i//50}') for i in range(0,len(files),50)]
    base.write_new(output/'protocol.json',protocol)
    host=base.require_cuda_allocation('cuda');device=torch.device('cuda')
    model=base.load_compatible_pretrained(policy,device=device).eval().requires_grad_(False)
    before=base.p0._parameter_hash(model);buffers={k:v.clone() for k,v in model.named_buffers()}
    actual_encodes=0;original_predict=model.predict
    def forbid(*args,**kwargs):raise RuntimeError('metric construction permits encode only')
    model.predict=forbid
    env=base.gym.make('swm/PushT-v1',render_mode='rgb_array');transform=base.p0.image_transform();metrics={}
    def encode(rgb):
        nonlocal actual_encodes
        with torch.no_grad():v=model.encode({'pixels':base.p0._image_tensor(rgb,transform=transform,device=device)})['emb'][0,0]
        if v.device.type!='cuda' or not torch.isfinite(v).all():raise ValueError('CUDA encoding required')
        actual_encodes+=1;return v.cpu().numpy().astype(float)
    try:
        roots=base.collect_nonterminal_boundary_contact_roots(env,count=4,seed=int(upstream['protocol']['root_seed']),oversample_factor=4)
        if len(roots)!=4 or {base.root_fingerprint(r) for r in roots}!=set(prior):raise ValueError('root set changed')
        refs={r['root_sha256']:r for r in upstream['rows']}
        for index,root in enumerate(roots):
            key=base.root_fingerprint(root);p=prior[key]
            frames,_,goal=base.physical._reset_and_warmup(env,root,history_len=3)
            if (base.physical._sha256(goal)!=refs[key]['goal_rgb_sha256'] or
                base.physical._sha256(np.stack(frames))!=refs[key]['history_rgb_sha256']):raise ValueError('RGB binding changed')
            current=frames[-1];current_vector=encode(current);differences=[];images=[]
            for scene,original_rgb,base_vector in [('current',current,current_vector),('goal',goal,np.asarray(p['goal_vector']))]:
                for v in method.agent_variants(original_rgb):
                    path=output/f'root_{index}.{scene}.{v["id"]}.png';Image.fromarray(v['rgb']).save(path)
                    differences.append(encode(v['rgb'])-base_vector)
                    images.append(dict(scene=scene,id=v['id'],path=str(path.resolve()),sha256=base.digest(path)))
            value=dict(root_sha256=key,root_index=index,images=images,extra_nuisance_differences=np.array(differences).tolist(),
                metric=method.compile_metric(p['original_nuisance_goal_vectors'],p['object_vectors'],differences))
            base.write_new(output/f'metric_{index}.json',value);metrics[key]=value
            print(f'root {index}: nuisance rank {value["metric"]["nuisance_rank"]}, task rank {value["metric"]["task_rank"]}',flush=True)
    finally:env.close();model.predict=original_predict
    hashes={str((output/f'metric_{v["root_index"]}.json').resolve()):base.digest(output/f'metric_{v["root_index"]}.json') for v in metrics.values()}
    base.write_new(output/'metrics_manifest.json',hashes)
    physical=json.loads(probes.read_text())
    if physical['status']!='diagnostic_completed' or not physical['gates'] or not all(v is True for v in physical['gates'].values()):raise ValueError('complete probes required')
    if len(physical['rows'])!=4 or {r['root_sha256'] for r in physical['rows']}!=set(metrics):raise ValueError('probe root mismatch')
    old_rows={r['root_sha256']:r for r in originals['rows']};diagnostics=[]
    for row in physical['rows']:
        key=row['root_sha256'];new=np.asarray(metrics[key]['metric']['projection'])
        old=np.asarray(old_rows[key]['compiled']['scoring']['metric']['projection']);values=[]
        for b in row['contexts']:
            try:
                j=central_difference(b['probes'],physical['protocol']['epsilon_atanh'])
                olde=float(np.sum((old@j)**2));newe=float(np.sum((new@j)**2))
                values.append(dict(candidate_id=b['candidate_id'],status='compared',old_energy=olde,new_energy=newe,
                    retained_fraction=newe/olde if olde else None,object_pose_invariant=len({tuple(p['state'][2:5]) for p in b['probes']})==1))
            except ValueError as exc:values.append(dict(candidate_id=b['candidate_id'],status='unavailable',reason=str(exc)))
        diagnostics.append(dict(root_index=row['root_index'],root_sha256=key,contexts=values))
    gates=dict(four_metrics=len(metrics)==4,actual_encoder68=actual_encodes==68,parameters_unchanged=before==base.p0._parameter_hash(model),
        buffers_unchanged=all(torch.equal(buffers[k],v) for k,v in model.named_buffers()),frozen_eval=all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters()),
        metrics_unchanged=all(base.digest(Path(p))==d for p,d in hashes.items()),inputs_unchanged=all(base.digest(Path(p))==d for p,d in bindings.items()))
    verified=[verify_seal(output_dir=s['output_dir'],expected_manifest_sha256=s['manifest_sha256']) for s in seals]
    base.write_new(output/'report.json',dict(schema='pusht_scene_nuisance_metric_cuda.v1',status='construction_completed' if all(gates.values()) else 'integrity_fail',
        protocol=protocol,host=host,gates=gates,metric_hashes=hashes,actual_encodes=actual_encodes,diagnostics=diagnostics,source_seals=seals,source_verification=verified,
        P1_admission=False,warning='finite augmentation invariance is not task alignment or physical efficacy'))
    if not all(gates.values()):raise RuntimeError('scene nuisance construction integrity failed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('goal_metric','probes','policy','output'):p.add_argument('--'+name.replace('_','-'),dest=name,type=Path,required=True)
    run(**vars(p.parse_args()))
