"""Analyze ridge gain on the existing joint panel; no model or physical execution."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from experiments.spectral_reachability import agent_p1_joint_vreach_saved_diff_20261029_v1 as saved
from experiments.spectral_reachability import p1_joint_suffix_fd_host_20261029 as source

ROOT = source.ROOT


def gain(goal, nominal, reach):
    g, n = np.asarray(goal, np.float64), np.asarray(nominal, np.float64)
    if g.shape != (192,) or n.shape != (192,) or not np.isfinite(g).all() or not np.isfinite(n).all() or not np.isfinite(reach):
        raise ValueError('finite aligned goal and nominal required')
    val = float(np.mean((g-n)**2)-reach)
    if val < -1e-9:
        raise ValueError('ridge score exceeds zero feasible action')
    return val


def build():
    report = saved.read_verified(source.REPORT, source.REPORT_SHA)
    receipt = saved.read_verified(source.RECEIPT, source.RECEIPT_SHA)
    if receipt.get('report_sha256') != source.REPORT_SHA:
        raise ValueError('receipt binding')
    prep = saved.read_verified(report['prepared'], report['prepare_sha256'])
    saved.analyze(report, receipt, prep)
    rows = []
    for arm in report['rows']:
        result = arm['result']; goal = prep['requests'][arm['slot']]['task']['goal']
        cs = result['candidates']
        baseline = gain(goal, cs[0]['signal']['nominal_context']['latent_history'][-1],cs[0]['own_reach_value'])
        for c in cs:
            value = gain(goal,c['signal']['nominal_context']['latent_history'][-1],c['own_reach_value'])
            rows.append(dict(slot=arm['slot'],arm=arm['arm'],start=c['start'],iteration=c['iteration'],
                gain=value,baseline_gain=baseline,gain_better=value>baseline,
                stage_uphill=c['stage_uphill'],terminal_better=c['terminal_better'],
                original_reach_better=c['reach_better']))
    if len(rows) != 160:
        raise ValueError('fixed occurrence denominator')
    return dict(schema='p1_joint_gain_panel_20261029',source_report_sha256=source.REPORT_SHA,
                source_receipt_sha256=source.RECEIPT_SHA,rows=rows,
                summary=dict(candidates=160,gain_better=sum(r['gain_better'] for r in rows),
                    uphill_and_gain=sum(r['stage_uphill'] and r['gain_better'] for r in rows),
                    uphill_and_terminal=sum(r['stage_uphill'] and r['terminal_better'] for r in rows),
                    all_three=sum(r['stage_uphill'] and r['gain_better'] and r['terminal_better'] for r in rows)),
                P1_admission=False,scope='posthoc model proxy on known roots, repeated candidates, not new gate or physical outcome')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not a.output.is_absolute() or a.output.parent != ROOT/'results' or a.output.exists():
        raise ValueError('exclusive absolute results file required')
    print('Recompute model ridge gain for all fixed 160 occurrences; no model or physics.',flush=True)
    value=build(); data=json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()
    with a.output.open('xb') as f: f.write(data)
    print(json.dumps(dict(path=str(a.output),sha256=hashlib.sha256(data).hexdigest(),summary=value['summary'])))


if __name__ == '__main__': main()
