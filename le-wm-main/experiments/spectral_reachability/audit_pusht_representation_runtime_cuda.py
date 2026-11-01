"""Independent legacy encoder and official rollout interop, host CUDA only."""
import argparse
import inspect
import json
from pathlib import Path
import numpy as np


def errors(actual,reference):
    a,b=np.asarray(actual,float),np.asarray(reference,float)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('finite aligned reference arrays required')
    delta=a-b;den=float(np.linalg.norm(b))
    return dict(max_absolute=float(np.max(np.abs(delta))),mse=float(np.mean(delta**2)),
        relative_l2=float(np.linalg.norm(delta))/den if den else None,
        tolerance_match=bool(np.allclose(a,b,atol=1e-4,rtol=1e-4)))


def run(compilation,physical_fd,goal_metric,policy,output):
    import torch
    from PIL import Image
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability import preflight_pusht_d2p_p1_reconfiguration as p1
    from experiments.spectral_reachability import lewm_legacy_vit_reference as reference
    from experiments.spectral_reachability.d2p_model_call_instrumentation import PredictInstrumentation
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal,verify_seal
    compilation,physical_fd,goal_metric,policy,output=map(Path,(compilation,physical_fd,goal_metric,policy,output))
    source,bindings,config=checked.validate_inputs(compilation,policy)
    fd=json.loads(physical_fd.read_text());goals=json.loads(goal_metric.read_text())
    for r,status in ((fd,'completed'),(goals,'construction_completed')):
        if r['status']!=status or not r['gates'] or not all(v is True for v in r['gates'].values()):raise ValueError('complete image sources required')
    images=[]
    for row in fd['rows']:
        for branch in row['branches']:
            zero=next(p for p in branch['probes'] if p['id']=='zero')
            images.append(dict(id=f'physical_{row["root_index"]}_{branch["candidate_id"]}',path=zero['rgb_path'],sha256=zero['rgb_file_sha256']))
    for path,sha in goals['metric_hashes'].items():
        if base.digest(Path(path))!=sha:raise ValueError('goal image metadata changed')
        entry=json.loads(Path(path).read_text());v=entry['images'][0]
        images.append(dict(id=f'goal_{entry["root_index"]}',path=v['path'],sha256=v['sha256']))
        bindings[path]=sha
    if len(images)!=12 or len({v['id'] for v in images})!=12:raise ValueError('eight physical and four goal images required')
    for v in images:bindings[v['path']]=v['sha256']
    for p in (physical_fd,goal_metric):bindings[str(p.resolve())]=base.digest(p)
    if not all(base.digest(Path(p))==d for p,d in bindings.items()):raise ValueError('image inputs changed')
    protocol=dict(bindings=bindings,images=images,legacy_config=dict(hidden=192,layers=12,heads=3,patch=14,layernorm_eps=1e-12,batchnorm_eps=1e-5),
        atol=1e-4,rtol=1e-4,original_checkpoint_keys_not_remapped_for_reference=True,
        rollout_methods=['direct_root_anchored_P1_2tokens','official_cached_rollout_outgoing_history_2tokens','legacy_incoming_history_rollout_3predictions'],
        expected_predict_samples=448,physical_actions_changed=False,P1_admission=False)
    output.mkdir(exist_ok=False)
    files=checked.sources_before_allocate(source,config)
    files += [Path(__file__),Path(reference.__file__),Path(p1.__file__),Path(base.__file__),Path(inspect.getfile(base.load_compatible_pretrained))]
    files=sorted(set(p.resolve() for p in files))
    seals=[create_seal(files=files[i:i+50],args_json=json.dumps(protocol),output_dir=output/f'source_seal_{i//50}') for i in range(0,len(files),50)]
    base.write_new(output/'protocol.json',protocol)
    host=base.require_cuda_allocation('cuda');device=torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32=False
    model=base.load_compatible_pretrained(policy,device=device).eval().requires_grad_(False)
    raw=torch.load(policy/'weights.pt',map_location=device,weights_only=True)
    before=base.p0._parameter_hash(model);buffers={k:v.clone() for k,v in model.named_buffers()}
    transform=base.p0.image_transform();encoding=[]
    with torch.no_grad():
        for entry in images:
            rgb=np.asarray(Image.open(entry['path']).convert('RGB'));x=base.p0._image_tensor(rgb,transform=transform,device=device)[:,0]
            refcls=reference.cls_reference(x,raw);refemb=reference.projected_reference(refcls,raw)
            livecls=model.encoder(x,interpolate_pos_encoding=True).last_hidden_state[:,0]
            liveemb=model.encode({'pixels':x[:,None]})['emb'][:,0]
            value=dict(id=entry['id'],cls=errors(livecls.cpu(),refcls.cpu()),projected=errors(liveemb.cpu(),refemb.cpu()),
                current_cls=livecls.cpu().tolist(),reference_cls=refcls.cpu().tolist(),
                current_projected=liveemb.cpu().tolist(),reference_projected=refemb.cpu().tolist())
            encoding.append(value)
            print(f'image {entry["id"]}: legacy/current projected max error {value["projected"]["max_absolute"]:.3g}',flush=True)
    rows=[];projection_checks=[]
    # Cached rollout uses dummy pixels for shape ONLY; all three methods use
    # the exact saved root latent context. No image encoding is hidden here.
    dummy=torch.zeros((1,1,3,3,224,224),dtype=torch.float32,device=device)
    with PredictInstrumentation(model) as counter:
        with counter.phase('evaluation',method='runtime_interop_diagnostic_max_horizon3',horizon_tokens=3):
            with torch.no_grad():
                for row in source['rows']:
                    c=row['compiled'];emb=torch.tensor(c['input_context']['emb'],dtype=torch.float32,device=device)
                    history=torch.tensor(c['input_context']['actions'],dtype=torch.float32,device=device);values=[]
                    # Functional pred_proj check on actual predictor features.
                    features=model.predictor(emb,model.action_encoder(history))
                    manual=reference.projected_reference(features.reshape(-1,192),raw,prefix='pred_proj').reshape_as(features)
                    projected=model.pred_proj(features.reshape(-1,192)).reshape_as(features)
                    projection_checks.append(errors(projected.cpu(),manual.cpu()))
                    for receipt in c['receipts']:
                        params=torch.tensor(receipt['executed_program_params'],dtype=torch.float32,device=device)[None]
                        future=p1._batched_live_action_tokens(params,action_block=5)
                        direct=p1._suffix_terminal(model,emb,history,params)[0]
                        correct=torch.cat([history[:,1:],future],dim=1)[:,None]
                        legacy=torch.cat([history,future],dim=1)[:,None]
                        a=model.rollout({'pixels':dummy,'emb':emb[:,None]},correct,history_size=3)['predicted_emb']
                        b=model.rollout({'pixels':dummy,'emb':emb[:,None]},legacy,history_size=3)['predicted_emb']
                        goal=torch.tensor(c['goal_vectors'][0],dtype=torch.float32,device=device)[None,None,None]
                        official_cost=model.criterion({'predicted_emb':a,'goal_emb':goal})[0,0]
                        expected=torch.mean((a[0,0,-1]-goal[0,0,0])**2)*192
                        values.append(dict(candidate_id=receipt['candidate_id'],
                            direct_vs_correct_official=errors(direct.cpu(),a[0,0,-1].cpu()),
                            direct_vs_saved=errors(direct.cpu(),receipt['predicted_terminal_vector']),
                            direct_vs_legacy_extra_step=errors(direct.cpu(),b[0,0,-1].cpu()),
                            correct_predicted_frames=int(a.shape[2]),legacy_predicted_frames=int(b.shape[2]),
                            official_SSE_equals_192_MSE=bool(torch.allclose(official_cost,expected,atol=1e-4,rtol=1e-4))))
                    rows.append(dict(root_index=row['root_index'],root_sha256=row['root_sha256'],programs=values))
                    print(f'root {row["root_index"]}: all16 direct/official/legacy predictions compared',flush=True)
        records=counter.records
    allprograms=[p for r in rows for p in r['programs']]
    checks=dict(legacy_encoder_sample_equivalence=all(v['cls']['tolerance_match'] and v['projected']['tolerance_match'] for v in encoding),
        pred_proj_functional_equivalence=all(v['tolerance_match'] for v in projection_checks),
        direct_equals_correct_official=all(v['direct_vs_correct_official']['tolerance_match'] for v in allprograms),
        saved_forecasts_reproduced=all(v['direct_vs_saved']['tolerance_match'] for v in allprograms),
        criterion_normalization_understood=all(v['official_SSE_equals_192_MSE'] for v in allprograms))
    gates=dict(twelve_real_images=len(encoding)==12,all64_programs=len(allprograms)==64,
        actual_predict_samples448=sum(r['predict_samples'] for r in records)==448 and all(r['completed'] for r in records),
        parameters_unchanged=before==base.p0._parameter_hash(model),buffers_unchanged=all(torch.equal(buffers[k],v) for k,v in model.named_buffers()),
        frozen_eval=all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters()),
        inputs_unchanged=all(base.digest(Path(p))==d for p,d in bindings.items()))
    verified=[verify_seal(output_dir=s['output_dir'],expected_manifest_sha256=s['manifest_sha256']) for s in seals]
    base.write_new(output/'report.json',dict(schema='pusht_representation_runtime_cuda.v1',
        status='audit_completed' if all(gates.values()) else 'integrity_fail',protocol=protocol,host=host,gates=gates,checks=checks,
        encoding=encoding,pred_proj_checks=projection_checks,rows=rows,predict=records,
        runtime=dict(encoder_config=model.encoder.config.to_dict(),matmul_tf32=False),
        source_seals=seals,source_verification=verified,P1_admission=False))
    if not all(gates.values()):raise RuntimeError('runtime audit integrity failed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('compilation','physical_fd','goal_metric','policy','output'):p.add_argument('--'+name.replace('_','-'),dest=name,type=Path,required=True)
    run(**vars(p.parse_args()))
