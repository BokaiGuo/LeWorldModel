"""Independent all-pair trajectory/schedule extension; no producer import or worker."""
from __future__ import annotations
import argparse
import itertools
import os
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_p3_commitment_numeric_recompute_20261029_v1 as numeric

NAME='agent_p3_commitment_trajectory_recompute_20261029_v1'
require=numeric.require


def raw(value):
    a=np.asarray(value)
    require(a.ndim==2 and a.shape[1]==2 and a.dtype.kind in 'fi' and np.isfinite(a).all() and
        np.array_equal(a,a.astype(np.float32)) and (abs(a)<=1).all(),'literal finite float32 raw actions')
    return a.astype(np.float32)


def bits(value):return __import__('hashlib').sha256(raw(value).tobytes()).hexdigest()


def trace(node):
    req=node['request'];r=node['receipt'];decision=r['decision'];bank=r['bank']
    t=req['task'];goal=numeric.vector(t['goal'],192);projection=np.asarray(t['projection'],np.float64)
    require(projection.shape==(192,192) and np.array_equal(projection,np.eye(192)), 'original identity192 Q')
    diag=r['diagnostic'];rows=[]
    for i,c in enumerate(r['candidates']):
        require(c['index']==i and c['status']=='returned' and c['whole']==c['segmented'],
            'all full candidate identities retained')
        action=raw(c['actions']);end=numeric.vector(c['whole'][-1]['latent_history'][-1],192)
        residual=projection@(goal-end);q=float(np.mean(residual*residual))
        require(q==c['cost'] and np.array_equal(action,raw(bank['actions'][i])),
            'full terminal Q independently recomputed on corresponding actions')
        rows.append(dict(index=i,fraction=bank['fractions'][i],raw_sha256=bits(action),
            literal_raw=action.tolist(),terminal_Q=q))
    if rows:
        require(decision is not None and len(rows)==decision['candidates'] and
            all(len(x['literal_raw'])==5*decision['horizon'] for x in rows),'entire declared candidate support')
        winner=min(rows,key=lambda c:(c['terminal_Q'],c['index']))
        require(r['winner']==r['candidates'][winner['index']] and
            raw(r['commands']).tobytes()==raw(winner['literal_raw']).tobytes(), 'stable native witness winner')
    else:winner=None
    if req['commitment'] is not None:
        require(r['status']=='observed_commitment' and decision is None and not rows and r['commands'] is None,
            'active full commitment pays signals but does not schedule/reselect')
    return dict(episode_id=node['episode_id'],boundary=node['boundary'],status=node['status'],
        context_sha256=numeric.digest(t['root_context']),actual_raw10_sha256=bits(req['raw10']),
        goal_sha256=numeric.digest(t['goal']),decision=decision,pending=req['commitment'],
        chi=diag['spectra'][0]['short_nominal_reference']['chi'],signed_gain=diag['signed_gain'],
        scheduling_gain=diag['scheduling_gain'],candidates=rows,winner=winner,
        paid_predict_attempted=node['resource_receipt']['attempted']['predict_API'],
        paid_predict_completed=node['resource_receipt']['completed']['predict_API'])


def observations(output,events,eid):
    entries=[e['receipt'] for e in events if e['kind']=='fresh_observation' and e.get('episode_id')==eid]
    result={}
    for e in entries:
        key=(e['phase'],e['step']);require(key not in result,'unique actual phase/step observation')
        path=Path(e['path']);require(path.parent==output and path.resolve()==path and
            path.stat().st_size<=1024**2 and numeric.sha(path)==e['file_sha256'],'original bounded actual PNG bytes')
        result[key]=dict(state=numeric.vector(e['state'],7).tobytes().hex(),snapshot=e['snapshot'],
            goal=numeric.vector(e['goal'],7).tobytes().hex(),goal_pose=np.asarray(e['goal_pose'],np.float64).tobytes().hex(),
            goal_rgb_sha256=e['goal_rgb_sha256'],pixel_sha256=e['pixel_sha256'],
            png_sha256=numeric.sha(path),png_bytes=path.read_bytes())
    return result


def candidate_support(left,right):
    """Exact raw-action support, not an assumed nested 4-of-8 grid."""
    a=left['candidates'];b=right['candidates'];common=[]
    for x,y in itertools.product(a,b):
        if raw(x['literal_raw']).tobytes()==raw(y['literal_raw']).tobytes():
            common.append(dict(left_index=x['index'],right_index=y['index'],left_fraction=x['fraction'],
                right_fraction=y['fraction'],same_Q=x['terminal_Q']==y['terminal_Q']))
    wa,wb=left['winner'],right['winner']
    return dict(left_count=len(a),right_count=len(b),all_shared_raw_pairs=common,
        selected_raw_equal=(raw(wa['literal_raw']).tobytes()==raw(wb['literal_raw']).tobytes())
            if wa is not None and wb is not None else None,
        selected_Q_equal=(wa['terminal_Q']==wb['terminal_Q']) if wa is not None and wb is not None else None,
        same_context=left['context_sha256']==right['context_sha256'],
        same_goal=left['goal_sha256']==right['goal_sha256'])


def analyze(output,report_sha,prepared,prepare_sha,strict,strict_sha):
    # This is our standalone independent implementation, never producer.consume or producer metrics.
    numbers=numeric.analyze(output,report_sha,prepared,prepare_sha,strict,strict_sha)
    require(numbers['complete'] is True,'full trace interpretation requires actual complete strict/numeric evidence')
    output=Path(output).absolute();r=numeric.read(output/'report.json',expected=report_sha)
    events,tail=numeric.journal(output/'journal.jsonl');require(tail==r['journal_chain_tail'],'same actual journal')
    traces=[trace(n) for n in r['nodes'] if n['status'] in ('sealed','observed_commitment','refused')]
    pairs=[]
    for slot in (0,1):
        for left,right in itertools.combinations(numeric.MODES,2):
            a=next(x for x in r['rows'] if (x['source_slot'],x['mode'])==(slot,left))
            b=next(x for x in r['rows'] if (x['source_slot'],x['mode'])==(slot,right))
            aa,bb=observations(output,events,a['episode_id']),observations(output,events,b['episode_id'])
            keys=sorted(set(aa)|set(bb));rows=[]
            for key in keys:
                x,y=aa.get(key),bb.get(key)
                rows.append(dict(phase=key[0],step=key[1],both_available=x is not None and y is not None,
                    equal_fields={f:x[f]==y[f] for f in ('state','snapshot','goal','goal_pose','goal_rgb_sha256',
                        'pixel_sha256','png_sha256','png_bytes')} if x is not None and y is not None else None))
            by_boundary=[]
            for boundary in (5,10,15):
                ta=next((t for t in traces if t['episode_id']==a['episode_id'] and t['boundary']==boundary),None)
                tb=next((t for t in traces if t['episode_id']==b['episode_id'] and t['boundary']==boundary),None)
                by_boundary.append(dict(boundary=boundary,left_status=ta['status'] if ta else 'not_reached',
                    right_status=tb['status'] if tb else 'not_reached',
                    support=candidate_support(ta,tb) if ta is not None and tb is not None else None))
            ca=a['commitments'];cb=b['commitments']
            for row in (a,b):
                executed=raw(row['executed_raw']);declared=list(row['prefix']['commands'])
                next_start=5
                for commitment in row['commitments']:
                    require(commitment['start']==next_start and commitment['end']==commitment['start']+
                        len(commitment['commands']),'ordered full commitment intervals')
                    declared+=commitment['commands'];next_start=commitment['end']
                require(executed.tobytes()==raw(declared)[:len(executed)].tobytes(),
                    'literal raw20 actually follows prefix and sealed full commitments')
            ap=numbers['paid']['native'][a['episode_id']];bp=numbers['paid']['native'][b['episode_id']]
            pairs.append(dict(source_slot=slot,left=left,right=right,
                left_actual_steps=len(a['executed_raw']),right_actual_steps=len(b['executed_raw']),
                exact_raw_control_equal=raw(a['executed_raw']).tobytes()==raw(b['executed_raw']).tobytes(),
                both_full20=len(a['executed_raw'])==len(b['executed_raw'])==20,
                trajectory=rows,all_observed_trajectory_exact=bool(rows) and all(x['both_available'] and
                    all(x['equal_fields'].values()) for x in rows),
                left_commitments=[{k:c[k] for k in ('start','end','seal_sha256')} for c in ca],
                right_commitments=[{k:c[k] for k in ('start','end','seal_sha256')} for c in cb],
                left_predict_completed=ap['completed']['predict_API'],right_predict_completed=bp['completed']['predict_API'],
                left_VJP_completed=ap['completed']['VJP_directions'],right_VJP_completed=bp['completed']['VJP_directions'],
                boundary_support=by_boundary,budget_match='notmatched',scheduling_bug_inferred=False,
                conclusion_scope='observed literal action/trajectory identity does not imply equal planning work or schedule bug'))
    require(numeric.sha(output/'report.json')==report_sha,'actual report unchanged')
    return dict(schema=NAME+'.analysis',report_sha256=report_sha,strict_sha256=strict_sha,prepare_sha256=prepare_sha,
        complete=True,rows=numbers['rows'],all_six_pairings=pairs,schedule_trace=traces,
        fixed_denominator=6,node_denominator=18,budget_match='notmatched',equal_FLOPs=False,
        known_roots=2,independent_confirmation=False,P1_admission=False,P3_admission=False,
        scientific_admission=False,freshness=False,consumer_source_sha256=numeric.sha(__file__),
        independent_numeric_source_sha256=numeric.sha(numeric.__file__))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('output','report-sha','prepared','prepare-sha','strict-receipt','strict-sha','receipt'):
        p.add_argument('--'+key,required=True)
    a=p.parse_args(argv);result=analyze(a.output,a.report_sha,a.prepared,a.prepare_sha,a.strict_receipt,a.strict_sha)
    target=Path(a.receipt).absolute()
    require(target.parent==numeric.ROOT/'results' and target.name.startswith(NAME) and target.resolve()==target
        and not target.exists(),'exclusive new independent trajectory output')
    with target.open('xb') as f:
        payload=numeric.canonical(result)+b'\n';require(f.write(payload)==len(payload),'whole receipt');f.flush();os.fsync(f.fileno())
    fd=os.open(target.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    print(numeric.canonical(dict(receipt=str(target),sha256=numeric.sha(target),complete=True)).decode())


if __name__=='__main__':main()
