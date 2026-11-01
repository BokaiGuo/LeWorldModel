"""Offline finite-probe task/latent response audit; no policy selection."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.spectral_reachability.pusht_fixed_residual_response import central_difference
from experiments.spectral_reachability.run_pusht_d2p_budget_guard_cuda import digest


def state_response(probes,epsilon):
    values=[]
    for p in probes:
        vector=np.asarray(p['state'])
        if vector.shape!=(7,) or vector.dtype.kind not in 'fi' or not np.isfinite(vector).all():
            raise ValueError('finite official state required for evaluator-only audit')
        values.append(dict(p,vector=vector.tolist()))
    jac=central_difference(values,epsilon)
    # Replace angular subtraction with its local circular displacement.
    for axis in (0,1):
        pair=[next(p for p in probes if p['axis']==axis and p['sign']==s) for s in (-1,1)]
        angular=pair[1]['state'][4]-pair[0]['state'][4]
        jac[4,axis]=np.arctan2(np.sin(angular),np.cos(angular))/(2*float(np.float32(epsilon)))
    return dict(agent_xy_jacobian_norm=float(np.linalg.norm(jac[:2])),
        object_xy_jacobian_norm=float(np.linalg.norm(jac[2:4])),object_angle_jacobian_norm=float(np.linalg.norm(jac[4])),
        all_five_object_poses_exact=len({tuple(p['state'][2:5]) for p in probes})==1,
        state_jacobian=jac.tolist())


def analyze(report_path):
    report_path=Path(report_path);r=json.loads(report_path.read_text())
    if r['status']!='diagnostic_completed' or not r['gates'] or not all(v is True for v in r['gates'].values()):
        raise ValueError('complete probe run required')
    rows=[]
    for row in r['rows']:
        contexts=[]
        for b in row['contexts']:
            result=dict(candidate_id=b['candidate_id'],probe_count=len(b['probes']),status=b['comparison']['status'])
            if len(b['probes'])!=5:raise ValueError('all five planned probes required')
            if b['comparison']['status']=='compared':
                result.update(state_response(b['probes'],r['protocol']['epsilon_atanh']))
                c=b['comparison']
                result.update(model_fixed_residual_improvement=c['fixed_model']['action_improvement'],
                    physical_fixed_residual_improvement=c['fixed_physical']['action_improvement'],
                    physical_projected_latent_jacobian_norm=c['fixed_physical']['jacobian_frobenius'])
            contexts.append(result)
        if len(contexts)!=2 or contexts[1]['candidate_id']!='prefix_0':raise ValueError('sealed selected/zero order required')
        selected,zero=contexts
        delta=(selected['physical_fixed_residual_improvement']-zero['physical_fixed_residual_improvement']
               if all(c['status']=='compared' for c in contexts) else None)
        rows.append(dict(root_index=row['root_index'],root_sha256=row['root_sha256'],contexts=contexts,
            physical_fixed_residual_selected_minus_zero=delta))
    return dict(schema='pusht_spectral_task_response_audit.v1',input_path=str(report_path.resolve()),
        input_sha256=digest(report_path),rows=rows,model_inference=False,physical_states_evaluator_only=True,
        P1_admission=False,scope='known-root fixed-scale diagnostic, not new selection or causal efficacy',
        warning='same latent direction may respond to agent motion even when object pose stays exactly fixed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); result=analyze(args.report)
    with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    for row in result['rows']:
        print(row['root_index'],row['physical_fixed_residual_selected_minus_zero'],
              'selected object poses invariant',row['contexts'][0].get('all_five_object_poses_exact'))
