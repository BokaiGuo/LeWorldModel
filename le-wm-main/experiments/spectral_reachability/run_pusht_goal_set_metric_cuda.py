"""Encode fixed goal nuisance catalog, then diagnose unchanged saved outcomes.

Host CUDA, no training/control, no outcome-based candidate or threshold choice.
"""
import argparse
import inspect
import json
from pathlib import Path
import numpy as np


def run(goal_metric,calibration,probes,policy,output):
    import torch
    from PIL import Image
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability import pusht_goal_set_metric as metric
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal,verify_seal
    goal_metric,calibration,probes,policy,output=map(Path,(goal_metric,calibration,probes,policy,output))
    previous=json.loads(goal_metric.read_text())
    if previous['status']!='construction_completed' or not previous['gates'] or not all(v is True for v in previous['gates'].values()):raise ValueError('complete goal metric construction required')
    for s in previous['source_seals']:verify_seal(output_dir=s['output_dir'],expected_manifest_sha256=s['manifest_sha256'])
    bindings=dict(previous['protocol']['bindings']);bindings.update(previous['metric_hashes'])
    bindings[str(goal_metric.resolve())]=base.digest(goal_metric)
    for path in (calibration,probes):
        if bindings.get(str(path.resolve()))!=base.digest(path):raise ValueError('same saved calibration/probes required')
    entries={}
    for path in previous['metric_hashes']:
        entry=json.loads(Path(path).read_text());key=entry['root_sha256']
        if key in entries:raise ValueError('duplicate root')
        image=entry['images'][0]
        if image['id']!='original_goal':raise ValueError('actual supplied goal required')
        bindings[image['path']]=image['sha256'];entries[key]=entry
    if len(entries)!=4 or not all(base.digest(Path(p))==d for p,d in bindings.items()):raise ValueError('four unchanged goals required')
    original=next(Path(p) for p,d in bindings.items() if d==checked.INPUT_SHA)
    original_report,_,config=checked.validate_inputs(original,policy)
    output.mkdir(exist_ok=False)
    protocol=dict(bindings=bindings,grid_side=16,grid_min=16,grid_max=208,batch_size=16,
        generated_images_per_goal=256,existing_equivalent_goals_per_root=5,expected_encoder_samples=1024,
        objective='minimum latent MSE over complete finite goal catalog',
        goal_catalogs_saved_before_loading_probe_and_control_outcomes=True,additional_training=False,
        RGB_edits_are_approximate_not_certified_simulator_goals=True,P1_admission=False)
    files=checked.sources_before_allocate(original_report,config)+[Path(__file__),Path(metric.__file__),Path(base.__file__)]
    files=sorted(set(p.resolve() for p in files))
    seals=[create_seal(files=files[i:i+50],args_json=json.dumps(protocol),output_dir=output/f'source_seal_{i//50}') for i in range(0,len(files),50)]
    base.write_new(output/'protocol.json',protocol)
    host=base.require_cuda_allocation('cuda');device=torch.device('cuda')
    model=base.load_compatible_pretrained(policy,device=device).eval().requires_grad_(False)
    before=base.p0._parameter_hash(model);buffers={k:v.clone() for k,v in model.named_buffers()}
    original_predict=model.predict
    def forbid(*args,**kwargs):raise RuntimeError('goal-set construction allows encode only')
    model.predict=forbid
    catalogs={};calls=samples=0;transform=base.p0.image_transform()
    try:
        for key,entry in entries.items():
            rgb=np.array(Image.open(entry['images'][0]['path']).convert('RGB'))
            images=list(metric.catalog_images(rgb));generated=[]
            for start in range(0,len(images),16):
                group=images[start:start+16]
                tensor=torch.cat([base.p0._image_tensor(x['rgb'],transform=transform,device=device) for x in group],dim=0)
                with torch.no_grad():vectors=model.encode({'pixels':tensor})['emb'][:,0]
                if vectors.device.type!='cuda' or vectors.shape!=(len(group),192) or not torch.isfinite(vectors).all():raise ValueError('finite CUDA catalog embeddings required')
                calls+=1;samples+=len(group);generated.extend(vectors.cpu().tolist())
            vectors=entry['original_nuisance_goal_vectors']+generated
            if len(vectors)!=261:raise ValueError('all five source goals and all256 augmentations required')
            catalog=dict(root_sha256=key,root_index=entry['root_index'],vectors=vectors,
                ids=[f'source_goal_{i}' for i in range(5)]+[x['id'] for x in images],
                source_goal_rgb_path=entry['images'][0]['path'],source_goal_rgb_sha256=entry['images'][0]['sha256'])
            path=output/f'catalog_{entry["root_index"]}.json';base.write_new(path,catalog);catalogs[key]=catalog
            print(f'root {entry["root_index"]}: all261 goal vectors saved, 256 actual CUDA encodes',flush=True)
    finally:model.predict=original_predict
    hashes={str((output/f'catalog_{v["root_index"]}.json').resolve()):base.digest(output/f'catalog_{v["root_index"]}.json') for v in catalogs.values()}
    base.write_new(output/'catalogs_manifest.json',hashes)
    physical=json.loads(probes.read_text());control=json.loads(calibration.read_text())
    for report,status in ((physical,'diagnostic_completed'),(control,'calibration_completed')):
        if report['status']!=status or not report['gates'] or not all(v is True for v in report['gates'].values()):raise ValueError('complete saved outcomes required')
        if len(report['rows'])!=4 or {r['root_sha256'] for r in report['rows']}!=set(catalogs):raise ValueError('outcome root mismatch')
        for s in report['source_seals']:verify_seal(output_dir=s['output_dir'],expected_manifest_sha256=s['manifest_sha256'])
    # No control is selected from these measurements. All branches retained.
    probe_diagnostics=[];control_diagnostics=[]
    plans={}
    for path in control['plan_hashes']:
        if base.digest(Path(path))!=control['plan_hashes'][path]:raise ValueError('control plan changed')
        p=json.loads(Path(path).read_text());plans[p['root_sha256']]=p
    for row in physical['rows']:
        key=row['root_sha256'];q=np.asarray(plans[key]['compiled']['scoring']['metric']['projection'])
        goal=np.asarray(entries[key]['goal_vector']);values=[]
        for b in row['contexts']:
            vectors=np.asarray([p['vector'] for p in b['probes']]);old=np.mean(((vectors-goal)@q.T)**2,axis=1)
            new=metric.distances(vectors,catalogs[key]['vectors'])
            values.append(dict(candidate_id=b['candidate_id'],statuses=[p['status'] for p in b['probes']],
                old_Q_costs=old.tolist(),goal_set_costs=new['costs'],goal_indices=new['goal_indices'],
                old_Q_cost_range=float(np.ptp(old)),goal_set_cost_range=float(np.ptp(new['costs'])),
                object_pose_invariant=len({tuple(p['state'][2:5]) for p in b['probes']})==1))
        probe_diagnostics.append(dict(root_sha256=key,root_index=row['root_index'],contexts=values))
    for row in control['rows']:
        key=row['root_sha256'];vectors=np.asarray([b['final_vector'] for b in row['branches']])
        q=np.asarray(plans[key]['compiled']['scoring']['metric']['projection']);goal=np.asarray(entries[key]['goal_vector'])
        old=np.mean(((vectors-goal)@q.T)**2,axis=1);new=metric.distances(vectors,catalogs[key]['vectors'])
        control_diagnostics.append(dict(root_sha256=key,root_index=row['root_index'],branches=[dict(
            candidate_id=b['candidate_id'],status=b['status'],old_Q_cost=float(old[i]),goal_set_cost=new['costs'][i],
            nearest_goal_index=new['goal_indices'][i],object_pose_cost=b['pose']['pose_cost']) for i,b in enumerate(row['branches'])]))
    gates=dict(four_complete_catalogs=len(catalogs)==4 and all(len(c['vectors'])==261 for c in catalogs.values()),
        actual_encoder_samples1024=samples==1024,actual_encoder_calls64=calls==64,
        parameters_unchanged=before==base.p0._parameter_hash(model),buffers_unchanged=all(torch.equal(buffers[k],v) for k,v in model.named_buffers()),
        frozen_eval=all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters()),
        catalogs_unchanged=all(base.digest(Path(p))==d for p,d in hashes.items()),inputs_unchanged=all(base.digest(Path(p))==d for p,d in bindings.items()))
    verified=[verify_seal(output_dir=s['output_dir'],expected_manifest_sha256=s['manifest_sha256']) for s in seals]
    base.write_new(output/'report.json',dict(schema='pusht_goal_set_metric_cuda.v1',status='construction_completed' if all(gates.values()) else 'integrity_fail',
        protocol=protocol,host=host,gates=gates,catalog_hashes=hashes,actual_encoder_calls=calls,actual_encoder_samples=samples,
        probe_diagnostics=probe_diagnostics,control_diagnostics=control_diagnostics,source_seals=seals,source_verification=verified,
        P1_admission=False,control_selection_changed=False,physical_benefit_verified=False))
    if not all(gates.values()):raise RuntimeError('goal set construction integrity failed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('goal_metric','calibration','probes','policy','output'):p.add_argument('--'+name.replace('_','-'),dest=name,type=Path,required=True)
    run(**vars(p.parse_args()))
