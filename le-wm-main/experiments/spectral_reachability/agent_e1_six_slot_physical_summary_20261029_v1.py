"""Fixed-denominator saved E1 native-goal summary; no physics or neural execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

NAME = 'agent_e1_six_slot_physical_summary_20261029_v1'
PHYSICAL = 'agent_e1_six_slot_physical_20261029_v1'
PHYSICAL_SHA = '47e238e9d5188a5495476c429ac5bbadf120338cec6f5565360a2e177d339a19'
ARMS = ('arbitrary_latent_waypoint', 'existing_learned_prior', 'dynamics_compiled_witness')
SLOTS = tuple((root, arm) for root in (0, 1) for arm in ARMS)
STEPS = (0, 5, 10)
CAP = 32 * 1024 * 1024
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'experiments/spectral_reachability' / (PHYSICAL + '.py')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def read_bound(path, expected):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file() and not path.is_symlink(), 'absolute regular file')
    require(isinstance(expected, str) and len(expected) == 64 and all(c in '0123456789abcdef' for c in expected), 'external SHA')
    require(path.stat().st_size <= CAP, 'input byte cap')
    data = path.read_bytes()
    require(len(data) <= CAP and digest(data) == expected, 'input SHA')
    def unique(pairs):
        obj = {}
        for k, v in pairs:
            require(k not in obj, 'duplicate JSON key')
            obj[k] = v
        return obj
    obj = json.loads(data, object_pairs_hook=unique, parse_constant=lambda _: require(False, 'nonfinite JSON'))
    canonical(obj)
    return obj


def vector(value):
    require(isinstance(value, list) and len(value) == 7 and all(type(x) in (float, int) and math.isfinite(x) for x in value), 'finite native state7')
    return [float(x) for x in value]


def metrics(state, goal):
    state, goal = vector(state), vector(goal)
    d = [x-y for x, y in zip(state, goal, strict=True)]
    sq = [x*x for x in d]
    xy = math.sqrt(math.fsum(sq[:4])); a = abs(d[4]); angle = min(a, 2*math.pi-a)
    return dict(state=state, goal=goal, delta=d, axis_squared=sq,
                agent_xy=math.sqrt(math.fsum(sq[:2])), object_xy=math.sqrt(math.fsum(sq[2:4])),
                all_xy=xy, native7=math.sqrt(math.fsum(sq)), velocity_squared=math.fsum(sq[5:7]),
                native_angle_term=angle, wrapped_object_angle_error=abs(math.atan2(math.sin(d[4]), math.cos(d[4]))),
                native_success=xy < 20 and angle < math.pi/9,
                metric_note='independent stdlib fsum; native success uses original xy and angle rule')


def summarize(receipt, report):
    require(receipt.get('schema') == PHYSICAL+'.consume' and receipt.get('status') == 'six_slot_native_recorded' and
            receipt.get('fixed_branch_denominator') == 6 and receipt.get('scientific_admission') is False and
            receipt.get('P1_admission') is False, 'completed strict saved receipt required')
    require(report.get('schema') == PHYSICAL+'.report' and report.get('status') == receipt['status'] and
            report.get('fixed_branch_denominator') == 6 and report.get('scientific_admission') is False and
            report.get('P1_admission') is False and receipt['rows'] == report['rows'] and
            receipt['counts'] == report['counts'], 'receipt/report exact rows and counts')
    rows = receipt['rows']; require(len(rows) == 6 and [(r['root_slot'], r['arm']) for r in rows] == list(SLOTS), 'ordered fixed six slots')
    flattened = []
    for i, row in enumerate(rows):
        require(row['branch_index'] == i and row['selected_plan']['root_slot'] == row['root_slot'] and
                row['selected_plan']['arm'] == row['arm'] and row['selection_digest'] == row['selected_plan']['selection_digest'], 'plan identity')
        for cp in row['checkpoints']:
            flattened.append(dict(branch=i, root_slot=row['root_slot'], arm=row['arm'], checkpoint=cp))
    require(receipt['checkpoints'] == flattened, 'receipt checkpoint projection')
    result=[]; indexed={}
    for i, row in enumerate(rows):
        cps=row['checkpoints']; steps=[c['step'] for c in cps]
        require(steps == sorted(set(steps)) and all(s in STEPS for s in steps), 'unique fixed checkpoint steps')
        require(type(row['own_returned']) is int and row['own_returned'] in range(11) and
                row['status'] in ('horizon_completed','native_success','native_stopped') and
                steps == [s for s in STEPS if s <= row['own_returned']], 'real available checkpoint slots')
        goal=vector(row['goal']); observed={cp['step']:cp for cp in cps}
        cells=[]
        for step in STEPS:
            cp=observed.get(step)
            if cp is None:
                cell=dict(step=step, status='missing', reason='native stopped before checkpoint', value=None, png=None)
            else:
                cell=dict(step=step, status='observed', reason=None, value=metrics(cp['state'],goal),
                          png=dict(name=cp['png'],sha256=cp['file_sha256'],pixel_sha256=cp['pixel_sha256']))
            cells.append(cell); indexed[(row['root_slot'],row['arm'],step)] = cell
        result.append(dict(branch=i,root=row['root_slot'],arm=row['arm'],status=row['status'],
                           selection_digest=row['selection_digest'],own_returned=row['own_returned'],cells=cells))
    pairs=[]
    for root in (0,1):
        for a,b in ((ARMS[0],ARMS[1]),(ARMS[0],ARMS[2]),(ARMS[1],ARMS[2])):
            for step in STEPS:
                x=indexed[(root,a,step)]['value']; y=indexed[(root,b,step)]['value']
                reason = 'missing checkpoint' if x is None or y is None else ('different native goals' if canonical(x['goal']) != canonical(y['goal']) else None)
                pairs.append(dict(root=root,arms=[a,b],step=step,status='unknown' if reason else 'observed',reason=reason,
                                  native7_difference=None if reason else x['native7']-y['native7'],
                                  object_xy_difference=None if reason else x['object_xy']-y['object_xy']))
    return dict(rows=result,pairs=pairs,slot_denominator=6,checkpoint_denominator=18,pair_denominator=18,
                distinct_intermediate_target_quality_comparison=False,scientific_admission=False,
                boundary='common native goal only; within-known-root descriptive contrasts, not causal or independent confirmation')


def analyze(receipt_path, receipt_sha, report_path, report_sha):
    require(digest(SOURCE.read_bytes()) == PHYSICAL_SHA, 'frozen physical source SHA')
    receipt=read_bound(receipt_path,receipt_sha); report=read_bound(report_path,report_sha)
    require(receipt['report_sha256']==report_sha, 'receipt binds external report SHA')
    files=(Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'))
    source_hashes={str(path):digest(path.read_bytes()) for path in files}
    return dict(schema=NAME+'.analysis',receipt_path=str(receipt_path),receipt_sha256=receipt_sha,
                report_path=str(report_path),report_sha256=report_sha,physical_source_sha256=PHYSICAL_SHA,
                source_hashes=source_hashes,
                trust_boundary='externally SHA-pinned physical strict receipt; does not rerun physical fullstrict',
                result=summarize(receipt,report))


def publish(path, value):
    path=Path(path); require(path.is_absolute() and path.resolve()==path and not path.exists(), 'new absolute output')
    raw=canonical(value)
    with path.open('xb') as f:
        require(f.write(raw)==len(raw),'short write'); f.flush(); os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return digest(raw)


def main():
    ap=argparse.ArgumentParser(description=__doc__); sub=ap.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('analyze')
    for name in ('receipt','receipt-sha','report','report-sha','output'):a.add_argument('--'+name,required=True)
    v=sub.add_parser('verify');v.add_argument('--analysis',required=True);v.add_argument('--analysis-sha',required=True)
    args=ap.parse_args()
    if args.cmd=='analyze':
        value=analyze(args.receipt,args.receipt_sha,args.report,args.report_sha)
        print(json.dumps(dict(output=args.output,sha256=publish(args.output,value)),sort_keys=True))
    else:
        value=read_bound(args.analysis,args.analysis_sha)
        require(value==analyze(value['receipt_path'],value['receipt_sha256'],value['report_path'],value['report_sha256']), 'analysis recomputation')
        print(json.dumps(dict(status='verified',sha256=args.analysis_sha),sort_keys=True))


if __name__=='__main__': main()
