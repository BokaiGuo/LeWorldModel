"""Saved model-only audit: full10 coordinates do not imply full10 proposal span."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import p1_moving_overlay_recompile_cuda_20261027_v1 as source

ROOT=source.ROOT
DIRECTORY=ROOT/'results/p1_moving_overlay_recompile_cuda_20261027_v1_cuda_MAIN01'
REPORT_SHA='2b2e5064f74f8aca712ae0d0415fcbfb64a2c8e6325bdffe24ae4fdc3b373381'
require=source.require


def coverage(params,jacobian):
    p=np.asarray(params,np.float64);j=np.asarray(jacobian,np.float64)
    require(p.shape==(10,10) and j.shape==(192,10) and np.isfinite(p).all() and np.isfinite(j).all(),'original10/full Jacobian')
    _,s,v=np.linalg.svd(p,full_matrices=True)
    tolerance=float(max(p.shape)*np.finfo(np.float64).eps*s[0]) if s[0]>0 else 0.
    rank=int(np.sum(s>tolerance));basis=v[:rank]
    projection=basis.T@basis;omitted=j@(np.eye(10)-projection)
    total=float(np.linalg.norm(j)**2)
    return dict(logit_dimension=10,proposal_span_rank=rank,omitted_logit_dimensions=10-rank,
        proposal_singular_values=s.tolist(),rank_tolerance=tolerance,
        root_jacobian_rank=int(np.linalg.matrix_rank(j)),
        omitted_J_frobenius_energy_ratio=float(np.linalg.norm(omitted)**2/total) if total else None,
        caveat='Logit proposal span only; nonlinear tanh images need not form a linear raw-action subspace.')


def run(output):
    print('Audit all four original D2P banks, not outcome-selected roots; no model or physics calls.',flush=True)
    plans=source.consume(DIRECTORY,REPORT_SHA)
    report=source.read(DIRECTORY/'report.json',4*1024**2)
    bindings={str(DIRECTORY/'report.json'):REPORT_SHA,str(Path(__file__).resolve()):source.sha(__file__)}
    rows=[]
    for index in (1,3,5,7):
        row=report['rows'][index];plan=plans[index]
        rp=DIRECTORY/row['result_file'];result=source.read(rp,16*1024**2);bindings[str(rp)]=source.sha(rp)
        tp=DIRECTORY/plan['task_file'];export=source.read(tp,4*1024**2);bindings[str(tp)]=source.sha(tp)
        task=source.task_interface(export)['gn_task']
        goal=np.asarray(task['goal'],np.float64)
        current=np.asarray(task['root_context']['latent_history'][-1],np.float64)
        root_Q=float(np.mean((current-goal)**2))
        candidates=result['candidates']
        require(len(candidates)==10 and all(c['completed'] is True for c in candidates),'all original candidates')
        winner=min(candidates,key=lambda c:(c['objective'],c['index']))
        greedy=min(candidates,key=lambda c:(c['stage_cost'],c['index']))
        require(winner==result['winner'],'same original selection')
        item={k:plan[k] for k in ('root_slot','condition','arm','task_id','selection_digest')}
        item.update(coverage(result['bank']['params'],result['root_jacobian']))
        item.update(bank_kinds=result['bank']['kinds'],root_current_Q=root_Q,
            winner_index=winner['index'],greedy_index=greedy['index'],
            winner_stage_minus_current=winner['stage_cost']-root_Q,
            winner_stage_minus_greedy=winner['stage_cost']-greedy['stage_cost'],
            candidates=[dict(index=c['index'],stage_minus_current=c['stage_cost']-root_Q,
                stage_cost=c['stage_cost'],own_reach_value=c['own_reach_value'],
                objective=c['objective'],nonlinear_terminal_cost=c['nonlinear_terminal_cost']) for c in candidates])
        rows.append(item)
    result=dict(schema='audit_prefix_span_MAIN_20261028_v1.report',status='all_four_saved_prefix_banks_audited',
        rows=rows,bindings=bindings,known_roots=2,fresh_roots=0,new_neural_calls=0,new_physics_calls=0,P1_admission=False,
        interpretation='A deliberately small library can restrict search despite full10 parameter coordinates; not a proven wiring bug or a theory negative.',
        next_hypothesis='Prospective all-mode prefix expansion can test this restriction; no new candidates are scored or executed by this audit.',
        limitations=['No physical labels were read to select roots or modes.',
                     'Saved-only span and model-Q diagnostics do not prove omitted directions improve real control.'])
    for f,h in bindings.items():require(source.sha(f)==h,'final audit input/source seal')
    data=json.dumps(result,sort_keys=True,allow_nan=False).encode()
    with Path(output).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(status=result['status'],sha256=source.sha(output),
        ranks=[r['proposal_span_rank'] for r in rows],omitted_energy=[r['omitted_J_frobenius_energy_ratio'] for r in rows])))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    run(p.parse_args().output)
