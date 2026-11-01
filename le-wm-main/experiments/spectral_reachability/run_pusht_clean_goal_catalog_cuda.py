"""Correct full-sprite preprocessing, rerun known-root goal-set diagnostic."""
import argparse
import json
from pathlib import Path
import numpy as np


def run(previous,calibration,probes,policy,output):
    import torch
    from PIL import Image
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability import pusht_clean_goal_sprite_catalog as clean
    from experiments.spectral_reachability.pusht_goal_set_metric import distances
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal,verify_seal
    previous,calibration,probes,policy,output=map(Path,(previous,calibration,probes,policy,output))
    old=json.loads(previous.read_text())
    if old['status']!='construction_completed' or not old['gates'] or not all(v is True for v in old['gates'].values()):raise ValueError('complete prior goal-set run required')
    for s in old['source_seals']:verify_seal(output_dir=s['output_dir'],expected_manifest_sha256=s['manifest_sha256'])
    bindings=dict(old['protocol']['bindings']);bindings.update(old['catalog_hashes']);bindings[str(previous.resolve())]=base.digest(previous)
    for path in (calibration,probes):
        if bindings.get(str(path.resolve()))!=base.digest(path):raise ValueError('same saved physical calibration required')
    if not all(base.digest(Path(p))==d for p,d in bindings.items()):raise ValueError('inputs changed')
    original=next(Path(p) for p,d in bindings.items() if d==checked.INPUT_SHA)
    orig,_,config=checked.validate_inputs(original,policy)
    # Freeze all eligibility from RGB before neural work or outcome diagnostics.
    plans={};planned={}
    for path in old['catalog_hashes']:
        c=json.loads(Path(path).read_text());key=c['root_sha256']
        if key in plans or base.digest(Path(c['source_goal_rgb_path']))!=c['source_goal_rgb_sha256']:raise ValueError('goal source mismatch')
        rgb=np.asarray(Image.open(c['source_goal_rgb_path']).convert('RGB'));plan=clean.build(rgb)
        plan['original_goals']=c['vectors'][:5];plan['root_index']=c['root_index'];plans[key]=plan
        planned[key]=dict(root_index=c['root_index'],valid_ids=[v['id'] for v in plan['valid']],invalid=plan['invalid'],
            source_bbox=plan['source_bbox'],source_nonwhite_pixels=plan['source_nonwhite_pixels'],
            old_pure_mask_residual_pixels=plan['old_pure_mask_residual_pixels'])
    if len(plans)!=4:raise ValueError('four complete goals required')
    expected=sum(len(p['valid']) for p in plans.values())
    output.mkdir(exist_ok=False)
    protocol=dict(bindings=bindings,planned=planned,expected_encoder_samples=expected,batch_size=16,
        method='complete source sprite translated only to wholly white visible target patches',
        goal_eligibility_uses_RGB_only=True,physical_outcomes_used_for_goal_eligibility=False,
        preserve_all_original_five_goal_vectors=True,additional_training=False,P1_admission=False)
    files=checked.sources_before_allocate(orig,config)+[Path(__file__),Path(clean.__file__),Path(distances.__code__.co_filename),Path(base.__file__)]
    files=sorted(set(p.resolve() for p in files))
    seals=[create_seal(files=files[i:i+50],args_json=json.dumps(protocol),output_dir=output/f'source_seal_{i//50}') for i in range(0,len(files),50)]
    base.write_new(output/'protocol.json',protocol)
    host=base.require_cuda_allocation('cuda');device=torch.device('cuda')
    model=base.load_compatible_pretrained(policy,device=device).eval().requires_grad_(False)
    before=base.p0._parameter_hash(model);buffers={k:v.clone() for k,v in model.named_buffers()}
    original_predict=model.predict
    def forbid(*args,**kwargs):raise RuntimeError('only goal encoding permitted')
    model.predict=forbid
    catalogs={};calls=samples=0;transform=base.p0.image_transform()
    try:
        for key,plan in plans.items():
            generated=[];images=plan['valid']
            for start in range(0,len(images),16):
                group=images[start:start+16]
                x=torch.cat([base.p0._image_tensor(v['rgb'],transform=transform,device=device) for v in group],dim=0)
                with torch.no_grad():vectors=model.encode({'pixels':x})['emb'][:,0]
                if vectors.device.type!='cuda' or vectors.shape!=(len(group),192) or not torch.isfinite(vectors).all():raise ValueError('CUDA goal embeddings required')
                generated.extend(vectors.cpu().tolist());calls+=1;samples+=len(group)
            catalog=dict(root_sha256=key,root_index=plan['root_index'],vectors=plan['original_goals']+generated,
                ids=[f'source_goal_{i}' for i in range(5)]+[v['id'] for v in images])
            catalogs[key]=catalog;base.write_new(output/f'catalog_{plan["root_index"]}.json',catalog)
            Image.fromarray(plan['sprite']).save(output/f'root_{plan["root_index"]}.full_sprite.png')
            Image.fromarray(plan['source_removed_rgb']).save(output/f'root_{plan["root_index"]}.source_removed.png')
            print(f'root {plan["root_index"]}: complete sprite; {len(images)} RGB-eligible positions; old residual {plan["old_pure_mask_residual_pixels"]} pixels',flush=True)
    finally:model.predict=original_predict
    hashes={str((output/f'catalog_{v["root_index"]}.json').resolve()):base.digest(output/f'catalog_{v["root_index"]}.json') for v in catalogs.values()}
    base.write_new(output/'catalogs_manifest.json',hashes)
    physical=json.loads(probes.read_text());control=json.loads(calibration.read_text())
    for r,status in ((physical,'diagnostic_completed'),(control,'calibration_completed')):
        if r['status']!=status or not r['gates'] or not all(v is True for v in r['gates'].values()):raise ValueError('completed same physical evidence required')
        if len(r['rows'])!=4 or {row['root_sha256'] for row in r['rows']}!=set(catalogs):raise ValueError('physical roots differ')
    diagnostics=[];control_diagnostics=[]
    prior={r['root_sha256']:r for r in old['probe_diagnostics']}
    for row in physical['rows']:
        key=row['root_sha256'];values=[];oldbranches={b['candidate_id']:b for b in prior[key]['contexts']}
        for b in row['contexts']:
            result=distances([p['vector'] for p in b['probes']],catalogs[key]['vectors'])
            values.append(dict(candidate_id=b['candidate_id'],statuses=[p['status'] for p in b['probes']],
                old_Q_cost_range=oldbranches[b['candidate_id']]['old_Q_cost_range'],
                old_partial_sprite_range=oldbranches[b['candidate_id']]['goal_set_cost_range'],
                clean_sprite_range=float(np.ptp(result['costs'])),clean_goal_set_costs=result['costs'],goal_indices=result['goal_indices'],
                object_pose_invariant=len({tuple(p['state'][2:5]) for p in b['probes']})==1))
        diagnostics.append(dict(root_index=row['root_index'],root_sha256=key,contexts=values))
    for row in control['rows']:
        result=distances([b['final_vector'] for b in row['branches']],catalogs[row['root_sha256']]['vectors'])
        control_diagnostics.append(dict(root_index=row['root_index'],root_sha256=row['root_sha256'],branches=[dict(candidate_id=b['candidate_id'],status=b['status'],
            object_pose_cost=b['pose']['pose_cost'],clean_goal_set_cost=result['costs'][i],goal_index=result['goal_indices'][i]) for i,b in enumerate(row['branches'])]))
    gates=dict(four_catalogs=len(catalogs)==4,all_RGB_planned_samples_encoded=samples==expected,
        all_grid_positions_accounted=all(len(p['valid'])+len(p['invalid'])==256 for p in plans.values()),
        parameters_unchanged=before==base.p0._parameter_hash(model),buffers_unchanged=all(torch.equal(buffers[k],v) for k,v in model.named_buffers()),
        frozen_eval=all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters()),
        catalogs_unchanged=all(base.digest(Path(p))==d for p,d in hashes.items()),inputs_unchanged=all(base.digest(Path(p))==d for p,d in bindings.items()))
    verified=[verify_seal(output_dir=s['output_dir'],expected_manifest_sha256=s['manifest_sha256']) for s in seals]
    base.write_new(output/'report.json',dict(schema='pusht_clean_goal_catalog_cuda.v1',status='construction_completed' if all(gates.values()) else 'integrity_fail',
        protocol=protocol,host=host,gates=gates,catalog_hashes=hashes,actual_encoder_calls=calls,actual_encoder_samples=samples,
        probe_diagnostics=diagnostics,control_diagnostics=control_diagnostics,source_seals=seals,source_verification=verified,
        P1_admission=False,control_selection_changed=False,physical_benefit_verified=False))
    if not all(gates.values()):raise RuntimeError('complete sprite construction integrity failed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('previous','calibration','probes','policy','output'):p.add_argument('--'+name,type=Path,required=True)
    run(**vars(p.parse_args()))
