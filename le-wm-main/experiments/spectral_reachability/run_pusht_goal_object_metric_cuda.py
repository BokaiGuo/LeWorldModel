"""Prospective RGB augmentation Q construction, then saved-probe diagnostics.

Only encoding on host CUDA. No training, no physical-label-driven metric choice,
no control efficacy or stage admission. Uses known calibration roots.
"""
import argparse
import inspect
import json
from pathlib import Path
import numpy as np


def run(calibration,probes,policy,output):
    import torch
    from PIL import Image
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability import pusht_goal_object_tangent_metric as metric
    from experiments.spectral_reachability.pusht_fixed_residual_response import central_difference
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal,verify_seal
    calibration,probes,policy,output=map(Path,(calibration,probes,policy,output))
    source=json.loads(calibration.read_text())
    if source['status']!='calibration_completed' or not source['gates'] or not all(v is True for v in source['gates'].values()):
        raise ValueError('complete calibration required')
    for s in source['source_seals']:verify_seal(output_dir=s['output_dir'],expected_manifest_sha256=s['manifest_sha256'])
    bindings=dict(source['protocol']['bindings']);bindings.update(source['plan_hashes'])
    bindings[str(calibration.resolve())]=base.digest(calibration);bindings[str(probes.resolve())]=base.digest(probes)
    if not all(base.digest(Path(p))==d for p,d in bindings.items()):raise ValueError('bound inputs changed')
    original=next(Path(p) for p,d in bindings.items() if d==checked.INPUT_SHA)
    _,_,config=checked.validate_inputs(original,policy)
    reports=[json.loads(Path(p).read_text()) for p in source['protocol']['bindings'] if Path(p).name=='report.json']
    intervention=next(r for r in reports if 'bank_path' in r.get('protocol',{}))
    bank=json.loads(Path(intervention['protocol']['bank_path']).read_text())
    upstream=base.physical._load_preflight(Path(bank['upstream_path']))
    plans={}
    for p in source['plan_hashes']:
        plan=json.loads(Path(p).read_text());key=plan['root_sha256']
        if key in plans:raise ValueError('duplicate root')
        plans[key]=plan
    if len(plans)!=4:raise ValueError('four known calibration roots required')
    protocol=dict(bindings=bindings,transforms=['x-4','x+4','y-4','y+4','angle-5deg','angle+5deg'],
        canonical_object_rgb=[119,136,153],canonical_agent_rgb=[65,105,225],
        physical_states_used_for_metric=False,additional_training=False,
        expected_encoder28=True,all_metrics_saved_before_probe_labels_loaded=True,
        image_edits_are_approximate_not_simulator_states=True,P1_admission=False)
    output.mkdir(exist_ok=False)
    files=checked.sources_before_allocate(json.loads(original.read_text()),config)
    files += [Path(__file__),Path(metric.__file__),Path(inspect.getfile(central_difference)),
        Path(base.__file__),Path(base.physical.__file__),Path(inspect.getfile(base.collect_nonterminal_boundary_contact_roots))]
    files=sorted(set(p.resolve() for p in files))
    seals=[create_seal(files=files[i:i+50],args_json=json.dumps(protocol),output_dir=output/f'source_seal_{i//50}') for i in range(0,len(files),50)]
    base.write_new(output/'protocol.json',protocol)
    host=base.require_cuda_allocation('cuda');device=torch.device('cuda')
    model=base.load_compatible_pretrained(policy,device=device).eval().requires_grad_(False)
    before=base.p0._parameter_hash(model);buffers={k:v.clone() for k,v in model.named_buffers()}
    original_predict=model.predict
    def forbid(*args,**kwargs):raise RuntimeError('prediction forbidden; metric construction only encodes goals')
    model.predict=forbid
    env=base.gym.make('swm/PushT-v1',render_mode='rgb_array');transform=base.p0.image_transform()
    metrics={};encodes=0
    try:
        roots=base.collect_nonterminal_boundary_contact_roots(env,count=4,seed=int(upstream['protocol']['root_seed']),oversample_factor=4)
        if len(roots)!=4 or {base.root_fingerprint(r) for r in roots}!=set(plans):raise ValueError('root mismatch')
        refs={r['root_sha256']:r for r in upstream['rows']}
        for index,root in enumerate(roots):
            key=base.root_fingerprint(root);c=plans[key]['compiled']
            _,_,goal=base.physical._reset_and_warmup(env,root,history_len=3)
            if base.physical._sha256(goal)!=refs[key]['goal_rgb_sha256']:raise ValueError('goal RGB changed')
            images=[dict(id='original_goal',rgb=goal),*metric.variants(goal)];vectors=[];image_receipts=[]
            for v in images:
                path=output/f'root_{index}.{v["id"]}.png';Image.fromarray(v['rgb']).save(path)
                with torch.no_grad():embedding=model.encode({'pixels':base.p0._image_tensor(v['rgb'],transform=transform,device=device)})['emb'][0,0]
                if embedding.device.type!='cuda' or not torch.isfinite(embedding).all():raise ValueError('invalid CUDA embedding')
                encodes+=1;vectors.append(embedding.cpu().tolist())
                image_receipts.append(dict(id=v['id'],path=str(path.resolve()),sha256=base.digest(path)))
            np.testing.assert_allclose(vectors[0],c['goal_vectors'][0],rtol=1e-5,atol=1e-5)
            result=metric.compile_metric(c['goal_vectors'],vectors[1:])
            value=dict(root_sha256=key,root_index=index,goal_vector=vectors[0],object_vectors=vectors[1:],
                original_nuisance_goal_vectors=c['goal_vectors'],metric=result,images=image_receipts)
            base.write_new(output/f'metric_{index}.json',value);metrics[key]=value
            print(f'root {index}: six RGB object augmentations, task rank {result["task_rank"]}',flush=True)
    finally:env.close();model.predict=original_predict
    metric_hashes={str((output/f'metric_{v["root_index"]}.json').resolve()):base.digest(output/f'metric_{v["root_index"]}.json') for v in metrics.values()}
    base.write_new(output/'metrics_manifest.json',metric_hashes)
    # Metrics are now immutable. Old calibration probe outcomes cannot tune Q.
    physical=json.loads(probes.read_text())
    if physical['status']!='diagnostic_completed' or not physical['gates'] or not all(v is True for v in physical['gates'].values()):
        raise ValueError('complete saved probe diagnostic required')
    for s in physical['source_seals']:verify_seal(output_dir=s['output_dir'],expected_manifest_sha256=s['manifest_sha256'])
    if len(physical['rows'])!=4 or {r['root_sha256'] for r in physical['rows']}!=set(metrics):raise ValueError('probe roots differ')
    diagnostics=[]
    for row in physical['rows']:
        key=row['root_sha256'];new=np.array(metrics[key]['metric']['projection'])
        old=np.array(plans[key]['compiled']['scoring']['metric']['projection']);contexts=[]
        for b in row['contexts']:
            try:
                j=central_difference(b['probes'],physical['protocol']['epsilon_atanh'])
                oldenergy=float(np.sum((old@j)**2));newenergy=float(np.sum((new@j)**2))
                contexts.append(dict(candidate_id=b['candidate_id'],status='compared',old_energy=oldenergy,new_energy=newenergy,
                    retained_fraction=newenergy/oldenergy if oldenergy else None,
                    object_pose_invariant=len({tuple(p['state'][2:5]) for p in b['probes']})==1))
            except ValueError as exc:contexts.append(dict(candidate_id=b['candidate_id'],status='unavailable',reason=str(exc)))
        diagnostics.append(dict(root_sha256=key,root_index=row['root_index'],contexts=contexts))
    gates=dict(four_metrics=len(metrics)==4,actual_encoder28=encodes==28,
        parameters_unchanged=before==base.p0._parameter_hash(model),buffers_unchanged=all(torch.equal(buffers[k],v) for k,v in model.named_buffers()),
        frozen_eval=all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters()),
        metrics_unchanged=all(base.digest(Path(p))==d for p,d in metric_hashes.items()),
        inputs_unchanged=all(base.digest(Path(p))==d for p,d in bindings.items()))
    verified=[verify_seal(output_dir=s['output_dir'],expected_manifest_sha256=s['manifest_sha256']) for s in seals]
    base.write_new(output/'report.json',dict(schema='pusht_goal_object_metric_cuda.v1',status='construction_completed' if all(gates.values()) else 'integrity_fail',
        protocol=protocol,host=host,gates=gates,metric_hashes=metric_hashes,diagnostics=diagnostics,actual_encodes=encodes,
        source_seals=seals,source_verification=verified,P1_admission=False,
        warning='rank reduction suppresses responses trivially; task selectivity/physical benefit not established'))
    if not all(gates.values()):raise RuntimeError('metric construction integrity failed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('calibration','probes','policy','output'):p.add_argument('--'+name,type=Path,required=True)
    run(**vars(p.parse_args()))
