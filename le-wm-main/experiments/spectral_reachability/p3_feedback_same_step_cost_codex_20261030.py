"""Read-only, receipt-bound matched-order analysis. No model or environment calls."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from experiments.spectral_reachability import p3_matched_order_feedback_host_v2_codex_20261029 as host
from experiments.spectral_reachability import p3_matched_order_feedback_host_saved_v2_codex_20261029 as saved

NAME = 'p3_feedback_same_step_cost_codex_20261030'
ROSTER = host.life.seam.ROSTER
CHECKPOINTS = (0, 5, 10, 15, 20)
need = host.need


def metrics(state, goal):
    a, b = np.asarray(state, dtype=np.float64), np.asarray(goal, dtype=np.float64)
    need(a.shape == b.shape == (7,) and np.isfinite(a).all() and np.isfinite(b).all(), 'finite state7 and goal7')
    d = a-b
    return dict(native_distance7=float(np.linalg.norm(d)), joint_xy=float(np.linalg.norm(d[:4])),
                object_xy=float(np.linalg.norm(d[2:4])),
                wrapped_angle=float(abs(np.arctan2(np.sin(d[4]), np.cos(d[4])))))


def summarize_slot(row, events, wrapped):
    slot = row['slot']; episode = f'matched_slot_{slot:02}'
    bounds = [e for e in events if e['resource'] in ('matched_slot_attempt','matched_slot_return') and e['detail'].get('slot') == slot]
    need(len(bounds) == 2 and bounds[0]['resource'] == 'matched_slot_attempt' and bounds[1]['resource'] == 'matched_slot_return', 'slot bounds')
    es = [e for e in events if bounds[0]['index'] < e['index'] < bounds[1]['index']]
    obs = [e for e in es if e['resource']=='native_fresh_observation' and e['detail'].get('episode_id') == episode and e['detail']['receipt']['phase']=='plan']
    ret = [e for e in es if e['resource']=='native_control_return' and e['detail'].get('episode_id') == episode]
    need([e['detail']['receipt']['step'] for e in obs] == list(range(21)) and
         [e['detail']['step'] for e in ret] == list(range(1,21)), 'complete same-step observations')
    goal = obs[0]['detail']['receipt']['goal']
    points = []
    for step,e in enumerate(obs):
        o=e['detail']['receipt'];need(o['goal']==goal, 'constant saved goal')
        m=metrics(o['state'],goal)
        if step:
            d=ret[step-1]['detail']; reward=d.get('reward')
            need(type(reward) in (float,int) and math.isfinite(reward) and
                 math.isclose(float(reward),-m['native_distance7'],rel_tol=1e-12,abs_tol=1e-10), 'saved reward/state consistency')
            need(d['terminated'] is False and d['truncated'] is False, 'complete unstopped trajectory')
        if step in CHECKPOINTS:
            points.append(dict(step=step,metrics=m,state=o['state']))
    finals=[w['payload']['result'] for w in wrapped if w['payload']['kind']=='slot_final']
    need(len(finals)==1 and host.life.seam.core.digest(finals[0])==row['result_sha256'], 'result identity')
    final=finals[0]; ledger=final['resource_receipt'];host.life.seam.core.audit_ledger(ledger)
    need(ledger['attempted']==ledger['completed']==host.life.seam.design.expected(), 'complete paid axes')
    counts={}
    for e in es:
        if e['origin']=='physical' and e['resource'].startswith('native_meter/') and e['kind']=='intent':
            n=e['resource'].split('/',1)[1];counts[n]=counts.get(n,0)+1
    return dict(slot=slot,root=row['root'],policy=row['policy'],goal=goal,checkpoints=points,
                horizons=[b['horizon'] for b in final['blocks']],
                numerical_attempted=ledger['attempted'],numerical_completed=ledger['completed'],
                native_meter_attempted=counts,encoding_boundaries=4,
                ledger_wall_ns=ledger['wall_ns'],inclusive_operation_wall_ns=ledger['inclusive_operation_wall_ns'],
                cuda_ns=None,FLOPs=None)


def compare(slots):
    need(len(slots)==4 and [(s['slot'],s['root'],s['policy']) for s in slots]==ROSTER, 'fixed roster')
    pairs=[]
    for root in range(2):
        fixed,signal=slots[2*root:2*root+2]
        need(fixed['goal']==signal['goal'] and fixed['checkpoints'][0]['state']==signal['checkpoints'][0]['state'], 'matched initial state and goal')
        points=[]
        for f,s in zip(fixed['checkpoints'],signal['checkpoints'],strict=True):
            need(f['step']==s['step'], 'same step')
            points.append(dict(step=f['step'],fixed=f['metrics'],signal=s['metrics'],
                               signal_minus_fixed={k:s['metrics'][k]-f['metrics'][k] for k in f['metrics']}))
        pairs.append(dict(root=root,checkpoints=points,numerical_axes_equal=fixed['numerical_completed']==signal['numerical_completed'],
                          native_meter_attempts_equal=fixed['native_meter_attempted']==signal['native_meter_attempted']))
    return pairs


def analyze(output, receipt_path, receipt_sha):
    read=host.source.host.design.read_pinned
    receipt=read(Path(receipt_path),receipt_sha)
    need(receipt.get('schema')==host.NAME+'.consume' and receipt.get('status')=='four_saved_feedback_engineering_verified_known_roots' and
         receipt.get('full_saved_strict') is True and receipt.get('scientific_pass') is False, 'complete saved receipt required')
    report_sha,completion_sha=receipt['report_sha256'],receipt['completion_sha256']
    reconstructed=saved.consume(output,report_sha,completion_sha)
    need(reconstructed==receipt, 'independent saved consume equals pinned receipt')
    directory=Path(output);report=read(directory/'report.json',report_sha)
    journal=host.storage.inspect_journal(directory/'events.jsonl',report['journal_sha256'])
    events=journal['events'];slots=[]
    for row in report['rows']:
        wrapped,_=saved._events_with_positions(events,row['slot'])
        slots.append(summarize_slot(row,events,wrapped))
    return dict(schema=NAME+'.analysis',receipt_sha256=receipt_sha,report_sha256=report_sha,
        completion_sha256=completion_sha,journal_sha256=report['journal_sha256'],slots=slots,pairs=compare(slots),
        job_elapsed_seconds=report['elapsed_seconds'],same_time_definition='same public control step, not same elapsed wall time',
        limitations=['two known roots, not independent fresh roots','saved source-trusted engineering evidence, not attestation',
          'no inferred native success field','inclusive operation times overlap and must not be summed',
          'ledger wall includes interleaved physical and encoding work; not pure model compute',
          'no per-slot total wall or CUDA time or FLOPs; counted axes do not prove equal total compute',
          'post hoc descriptive comparison, not scientific pass'],scientific_pass=False,equal_FLOPs=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--receipt',required=True);p.add_argument('--receipt-sha256',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output).absolute();need(out.parent==host.ROOT/'results' and out.name.startswith(NAME) and not out.exists() and out.resolve()==out, 'exclusive analysis output')
    result=analyze(a.run,a.receipt,a.receipt_sha256)
    digest=host.write_new(out,result)
    print(json.dumps(dict(output=str(out),sha256=digest,scientific_pass=False),sort_keys=True))


if __name__=='__main__':main()
