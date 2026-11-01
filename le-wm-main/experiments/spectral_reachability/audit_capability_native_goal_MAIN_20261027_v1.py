"""All40 saved probes: official joint-position target versus object-only direction."""
import argparse
import hashlib
import json
from pathlib import Path
import os
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FILES = dict(
    probe=('moving_overlay_capability_probe_physical_20261027_v1_MAIN01/report.json', 'e075d3b75c7b5a16c47d8c76c846eaa2ac4d8ebd202abd18037a7242792b076a'),
    original=('p1_moving_overlay_recompile_physical_20261027_v1_MAIN01/report.json', '56962e9370fcf249a2ed88f004a400414b491f503a4c3a9db1bd951cf984a521'),
    latent=('consume_capability_direction_MAIN_20261027_v1_MAIN01.json', 'f564c6771c1940dc3f6925999647569670a9dbbb4381bb6da5a37a5636a0c5bd'))


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(goal, state):
    goal, state = np.asarray(goal,np.float64), np.asarray(state,np.float64)
    require(goal.shape == state.shape == (7,) and np.isfinite(goal).all() and np.isfinite(state).all(), 'native state7')
    delta = goal-state
    joint = float(np.linalg.norm(delta[:4]))
    raw_angle = float(min(abs(delta[4]),2*np.pi-abs(delta[4])))
    wrapped = float(abs(np.arctan2(np.sin(delta[4]),np.cos(delta[4]))))
    return dict(allxy_l2=joint,object_xy_l2=float(np.linalg.norm(delta[2:4])),agent_xy_l2=float(np.linalg.norm(delta[:2])),
                state7_distance=float(np.linalg.norm(delta)),velocity_l2=float(np.linalg.norm(delta[5:])),
                native_angle_term=raw_angle,wrapped_angle=wrapped,
                native_success=bool(joint<20 and raw_angle<np.pi/9),wrapped_success=bool(joint<20 and wrapped<np.pi/9),
                negative_native_angle=raw_angle<0)


def run(output):
    docs={}; bindings={}
    for k,(name,h) in FILES.items():
        path=ROOT/'results'/name
        require(sha(path)==h,'frozen input SHA'); docs[k]=json.loads(path.read_bytes()); bindings[str(path)]=h
    probe,original,latent=(docs[k] for k in ('probe','original','latent'))
    require(len(probe['rows'])==len(latent['rows'])==40 and len(original['rows'])==8,'fixed denominator')
    rows=[]
    for i,(p,l) in enumerate(zip(probe['rows'],latent['rows'],strict=True)):
        o=original['rows'][i//5]; zero=probe['rows'][i//5*5]
        require(p['index']==l['index']==i and p['status']=='probe_completed' and p['start_verified'] is True and
                all(p[k]==l[k]==o[k] for k in ('root_slot','condition','arm','selection_digest','task_id')) and
                p['probe']==l['probe'],'same branch and original goal')
        final=metrics(o['goal'],p['responses'][-1]['state'])
        baseline=metrics(o['goal'],zero['responses'][-1]['state'])
        require(final['native_success'] is p['responses'][-1]['terminated'],'native flag formula preserved')
        delta=final['allxy_l2']-baseline['allxy_l2']
        row={k:p[k] for k in ('index','root_slot','condition','arm','probe','selection_digest')}
        row.update(native=final,zero_native=baseline,allxy_minus_zero=delta,
                   actual_Q_minus_zero=l['actual_minus_zero_Q'],
                   object_direction_projection=l['native_object_task_projection'],
                   actual_Q_allxy_sign_agree=bool(np.sign(delta)==np.sign(l['actual_minus_zero_Q'])))
        for path in ('teacher','open'):
            q=l[path]['predicted_minus_zero_Q']
            row[path]=dict(predicted_Q_minus_zero=q,allxy_sign_agree=bool(np.sign(q)==np.sign(delta)))
        rows.append(row)
    nonzero=[r for r in rows if r['probe']!='zero']
    result=dict(status='saved_native_goal_axes_verified',rows=rows,input_bindings=bindings,
                fixed_branches=40,known_roots=2,fresh_roots=0,nonzero_comparisons=32,
                actual_Q_allxy_agreement=sum(r['actual_Q_allxy_sign_agree'] for r in nonzero),
                prediction_allxy_agreement={p:sum(r[p]['allxy_sign_agree'] for r in nonzero) for p in ('teacher','open')},
                native_success_count=sum(r['native']['native_success'] for r in rows),
                new_neural_calls=0,new_physics_calls=0,P1_admission=False,
                interpretation='Official position success includes agent+object; zero object progress alone is not official task failure.',
                limitations=['Known roots, fixed small probes; not optimized closed-loop benefit.',
                             'Raw native angle behavior preserved; wrapped diagnostic does not redefine success.'])
    for path,h in bindings.items(): require(sha(path)==h,'final source drift')
    with Path(output).open('xb') as f:
        f.write(json.dumps(result,sort_keys=True,allow_nan=False).encode());f.flush();os.fsync(f.fileno())
    print(json.dumps({k:result[k] for k in ('status','actual_Q_allxy_agreement','prediction_allxy_agreement','native_success_count')}|dict(sha256=sha(output))))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);run(p.parse_args().output)
