"""Fixed selected/zero spectral prefixes, five physical probes each, host CUDA.

Known calibration roots, no new selection or P1 admission. Physical FD at .05
is a finite-scale response, not proof of an infinitesimal derivative/contact law.
"""
import argparse
import inspect
import json
from pathlib import Path
import numpy as np


def run(calibration,policy,output):
    import torch
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability import pusht_fixed_residual_response as fixed
    from experiments.spectral_reachability.pusht_visible_command_executor import execute
    from experiments.spectral_reachability.pusht_rgb_observability_guard import RgbObservabilityGuard
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal,verify_seal
    calibration,policy,output=map(Path,(calibration,policy,output))
    source=json.loads(calibration.read_text())
    if source['status']!='calibration_completed' or not source['gates'] or not all(v is True for v in source['gates'].values()):
        raise ValueError('complete spectral calibration required')
    for s in source['source_seals']:
        verify_seal(output_dir=s['output_dir'],expected_manifest_sha256=s['manifest_sha256'])
    bindings=dict(source['protocol']['bindings'])
    bindings[str(calibration.resolve())]=base.digest(calibration)
    bindings.update(source['plan_hashes'])
    if not all(base.digest(Path(p))==d for p,d in bindings.items()):raise ValueError('input changed')
    original=next(Path(p) for p in bindings if base.digest(Path(p))==checked.INPUT_SHA)
    _,_,config=checked.validate_inputs(original,policy)
    candidates=[json.loads(Path(p).read_text()) for p in bindings if Path(p).name=='report.json']
    intervention=next(r for r in candidates if 'bank_path' in r.get('protocol',{}))
    bank=json.loads(Path(intervention['protocol']['bank_path']).read_text())
    upstream=base.physical._load_preflight(Path(bank['upstream_path']))
    plans={}
    for p in source['plan_hashes']:
        plan=json.loads(Path(p).read_text()); key=plan['root_sha256']
        if key in plans:raise ValueError('duplicate root plan')
        plans[key]=plan
    if len(source['rows'])!=4 or len(plans)!=4 or set(plans)!={r['root_sha256'] for r in source['rows']}:
        raise ValueError('four exact matching roots required')
    epsilon=.05
    probes=[dict(axis=None,sign=0,params=[0.,0.])]
    for a in range(2):
        for sign in (-1,1):
            v=[0.,0.];v[a]=sign*epsilon;probes.append(dict(axis=a,sign=sign,params=v))
    protocol=dict(bindings=bindings,epsilon_atanh=epsilon,probes=probes,contexts=8,planned_probes=40,
        selection='unchanged sealed spectral selected and zero prefix only',
        fixed_residual='goal[0] minus saved zero-prefix nominal suffix terminal, common within root',
        neural_work='only terminal encodes on host CUDA; no model predictions',
        physical_state_used_for_selection=False,P1_admission=False)
    output.mkdir(exist_ok=False)
    files=[Path(__file__),Path(fixed.__file__),Path(inspect.getfile(execute)),Path(inspect.getfile(RgbObservabilityGuard)),
        Path(base.__file__),Path(base.physical.__file__),Path(base.p0.__file__),Path(base.p0.packets.__file__),
        Path(inspect.getfile(base.load_compatible_pretrained)),Path(inspect.getfile(base.collect_nonterminal_boundary_contact_roots))]
    for seal in source['source_seals']:
        files += [Path(f['source']) for f in json.loads((Path(seal['output_dir'])/'manifest.json').read_text())['files']]
    files=sorted(set(p.resolve() for p in files))
    seals=[create_seal(files=files[i:i+50],args_json=json.dumps(protocol),output_dir=output/f'source_seal_{i//50}') for i in range(0,len(files),50)]
    base.write_new(output/'protocol.json',protocol)
    host=base.require_cuda_allocation('cuda');device=torch.device('cuda')
    model=base.load_compatible_pretrained(policy,device=device).eval().requires_grad_(False)
    before=base.p0._parameter_hash(model);buffers={k:v.clone() for k,v in model.named_buffers()}
    original_predict=model.predict
    def forbid(*args,**kwargs):raise RuntimeError('neural prediction forbidden in physical response probe')
    model.predict=forbid
    env=base.gym.make('swm/PushT-v1',render_mode='rgb_array');transform=base.p0.image_transform()
    rows=[];encodes=0
    try:
        roots=base.collect_nonterminal_boundary_contact_roots(env,count=4,seed=int(upstream['protocol']['root_seed']),oversample_factor=4)
        if len(roots)!=4 or {base.root_fingerprint(r) for r in roots}!=set(plans):raise ValueError('roots changed')
        if tuple(env.unwrapped.variation_space['agent']['color'].value.tolist())!=(65,105,225):raise ValueError('renderer changed')
        actual={r['root_sha256']:r for r in source['rows']};refs={r['root_sha256']:r for r in upstream['rows']}
        for index,root in enumerate(roots):
            key=base.root_fingerprint(root);plan=plans[key];c=plan['compiled']
            receipts={r['candidate_id']:r for r in c['receipts']}
            names=[c['selected_candidate_id'],'prefix_0']
            if len(set(names))!=2:raise ValueError('distinct selected and zero required')
            residual=np.asarray(c['goal_vectors'][0])-np.asarray(receipts['prefix_0']['nominal_terminal'])
            q=np.asarray(c['scoring']['metric']['projection']);contexts=[]
            saved={r['candidate_id']:r for r in actual[key]['branches']}
            for name in names:
                prefix=np.asarray(receipts[name]['program_live_commands'][0]); values=[]
                for probe_index,probe in enumerate(probes):
                    frames,_,goalrgb=base.physical._reset_and_warmup(env,root,history_len=3)
                    if (base.physical._sha256(np.stack(frames))!=refs[key]['history_rgb_sha256'] or
                        base.physical._sha256(goalrgb)!=refs[key]['goal_rgb_sha256']):raise ValueError('root RGB changed')
                    first=execute(env,np.repeat(prefix[None],5,axis=0),root_id=key,agent_rgb=(65,105,225))
                    rgb=first.pop('terminal_rgb')
                    prefix_trace=saved[name]['trace'][:5]
                    if (first['status']!='completed' or len(prefix_trace)!=5 or
                        base.physical._sha256(rgb)!=prefix_trace[-1]['rgb_sha256'] or
                        not np.array_equal(base.p0.packets.state_of(env),prefix_trace[-1]['state'])):
                        raise ValueError('exact visible prefix replay required')
                    params=torch.tensor(probe['params'],dtype=torch.float32,device=device)
                    live=torch.tanh(params).cpu().numpy()
                    result=execute(env,np.repeat(live[None],5,axis=0),root_id=key,agent_rgb=(65,105,225))
                    rgb=result.pop('terminal_rgb')
                    with torch.no_grad():vector=model.encode({'pixels':base.p0._image_tensor(rgb,transform=transform,device=device)})['emb'][0,0]
                    encodes+=1
                    if vector.device.type!='cuda' or not torch.isfinite(vector).all():raise ValueError('CUDA encoding invalid')
                    value=dict(**probe,**result,executed_params=params.cpu().tolist(),raw_command=live.tolist(),
                        prefix_replay_exact=True,vector=vector.cpu().tolist(),rgb_sha256=base.physical._sha256(rgb),
                        state=base.p0.packets.state_of(env).tolist())
                    values.append(value);base.write_new(output/f'root_{index}.{name}.probe_{probe_index}.json',value)
                model_response=fixed.response(receipts[name]['jacobian'],residual,q)
                try:
                    physical_j=fixed.central_difference(values,epsilon)
                    comparison=dict(status='compared',physical_jacobian=physical_j.tolist(),
                        fixed_model=response_copy(model_response),fixed_physical=fixed.response(physical_j,residual,q))
                except ValueError as exc:comparison=dict(status='unavailable',reason=str(exc))
                contexts.append(dict(candidate_id=name,probes=values,model_response=model_response,comparison=comparison))
            row=dict(root_index=index,root_sha256=key,fixed_residual=residual.tolist(),contexts=contexts)
            rows.append(row);base.write_new(output/f'root_{index}.json',row)
            print(f'root {index}: 10 physical probes retained',flush=True)
    finally:env.close();model.predict=original_predict
    gates=dict(four_roots=len(rows)==4,all40_retained=sum(len(b['probes']) for r in rows for b in r['contexts'])==40,
        actual_encoder40=encodes==40,no_execution_errors=all(p['status'] in ('completed','observation_unavailable','environment_terminated') for r in rows for b in r['contexts'] for p in b['probes']),
        parameters_unchanged=before==base.p0._parameter_hash(model),buffers_unchanged=all(torch.equal(buffers[k],v) for k,v in model.named_buffers()),
        frozen_eval=all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters()),
        inputs_unchanged=all(base.digest(Path(p))==d for p,d in bindings.items()))
    verified=[verify_seal(output_dir=s['output_dir'],expected_manifest_sha256=s['manifest_sha256']) for s in seals]
    base.write_new(output/'report.json',dict(schema='pusht_spectral_response_physical_cuda.v1',status='diagnostic_completed' if all(gates.values()) else 'integrity_fail',
        protocol=protocol,host=host,gates=gates,rows=rows,actual_encodes=encodes,source_seals=seals,source_verification=verified,P1_admission=False))
    if not all(gates.values()):raise RuntimeError('physical response integrity failed')


def response_copy(value):
    return dict(value)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('calibration','policy','output'):p.add_argument('--'+name,type=Path,required=True)
    run(**vars(p.parse_args()))
