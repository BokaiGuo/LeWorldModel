"""Saved-only E1 cross-panel same-root/goal/time physical joins; no new execution."""
import argparse
import math
import os
import struct
from pathlib import Path

from experiments.spectral_reachability import agent_e1_six_slot_physical_summary_20261029_v1 as six
from experiments.spectral_reachability import agent_e1_forced_candidate_paired_summary_20261029_v1 as forced

NAME = 'agent_e1_cross_panel_physical_pair_20261029_v1'
ROOT = Path(__file__).resolve().parents[2]
SIX = ROOT/'results/agent_e1_six_slot_physical_summary_20261029_v1_MAIN01.json'
FORCED = ROOT/'results/agent_e1_forced_candidate_paired_summary_20261029_v1_MAIN01.json'
SIX_SHA = '64796746b7956fcade47e5919597887ff3b3fc72c1a1fb28a2a6e1e5c7a74a11'
FORCED_SHA = '9d15df249e17ecc35a99ad57db38ef6fc581ca0f1f7e5e7a40bdf406a6431efd'
SOURCE_PINS = {str(Path(six.__file__).resolve()):'afeeebbc8b50689ded8b09102d33e1b0a9aa9bf353e5ef373528fcbcce2bfbc1',
               str(Path(forced.__file__).resolve()):'2fecb769b0eb6a26a6cd9e6348f615677fc613684e49153780b8b2783d3cdbda'}
need = six.require


def raw_bits(commands):
    need(type(commands) is list and len(commands)==10, 'ten raw commands')
    need(all(type(x) is list and len(x)==2 and all(type(v) in (int,float) and math.isfinite(v) for v in x) for x in commands), 'finite raw commands')
    return [struct.pack('<2f',*x) for x in commands]


def source_summary():
    for p,h in SOURCE_PINS.items(): need(six.digest(Path(p).read_bytes()) == h, 'summary source SHA')
    a=six.read_bound(SIX,SIX_SHA);b=forced.read_bound(FORCED,FORCED_SHA)
    aa=six.analyze(a['receipt_path'],a['receipt_sha256'],a['report_path'],a['report_sha256'])
    bb=forced.analyze(b['receipt_path'],b['receipt_sha256'],b['report_path'],b['report_sha256'],b['physical_source_sha256'])
    need(a==aa and b==bb, 'full existing summary reconstruction')
    r1=six.read_bound(a['report_path'],a['report_sha256']);r2=forced.read_bound(b['report_path'],b['report_sha256'])
    return a,b,r1,r2


def compare(rows_a, rows_b):
    need(len(rows_a)==len(rows_b)==6, 'six slots per panel')
    need([(x['root_slot'],x['arm']) for x in rows_a]==list(six.SLOTS), 'six-panel fixed order')
    need([(x['root_slot'],x['arm']) for x in rows_b]==list(forced.SLOTS), 'forced-panel fixed order')
    pairs=[]; joins=[]
    for root in (0,1):
        left=rows_a[root*3:root*3+3];right=rows_b[root*3:root*3+3]
        base=left[0]['selected_plan']
        keys=('root_sha256','task_id','task_sha256','task_contract_sha256','blind_sha256')
        for r in left+right:
            need(all(r['selected_plan'][k]==base[k] for k in keys), 'root/task provenance mismatch')
        cps0=[[cp for cp in r['checkpoints'] if cp['step']==0] for r in left+right]
        need(all(len(c)<=1 for c in cps0), 'unique initial observation')
        available=all(cps0)
        if available:
            ref=cps0[0][0];goal=left[0]['goal']
            for row,cs in zip(left+right,cps0,strict=True):
                cp=cs[0]
                need(row['goal']==goal and all(cp[k]==ref[k] for k in ('state','snapshot','pixel_sha256','file_sha256')), 'root/native goal/RGB mismatch')
        joins.append(dict(root=root,provenance={k:base[k] for k in keys},initial_exact=available,
                          missing_reason=None if available else 'missing cp0; paired values blocked'))
        for a in left:
            for b in right:
                plan_a=raw_bits(a['selected_plan']['commands']);plan_b=raw_bits(b['selected_plan']['commands'])
                for step in (0,5,10):
                    xa=[c for c in a['checkpoints'] if c['step']==step];xb=[c for c in b['checkpoints'] if c['step']==step]
                    need(len(xa)<=1 and len(xb)<=1, 'unique checkpoint')
                    va=six.metrics(xa[0]['state'],a['goal']) if xa else None
                    vb=six.metrics(xb[0]['state'],b['goal']) if xb else None
                    if va is not None and vb is not None:need(va['goal']==vb['goal'], 'same-time native goal mismatch')
                    observed=available and va is not None and vb is not None
                    fields=('native7','all_xy','agent_xy','object_xy','native_angle_term','wrapped_object_angle_error','velocity_squared')
                    delta={k:vb[k]-va[k] for k in fields} if observed else None
                    pairs.append(dict(root=root,step=step,subgoal_arm=a['arm'],forced_arm=b['arm'],
                        direction='forced minus subgoal',status='observed' if observed else 'missing',
                        missing=[name for name,val in (('root_join',available),('subgoal_checkpoint',bool(xa)),('forced_checkpoint',bool(xb))) if not val],
                        subgoal=va,forced=vb,delta=delta,
                        same_first5=plan_a[:5]==plan_b[:5],same_raw10=plan_a==plan_b,
                        subgoal_branch=dict(status=a['status'],own_returned=a['own_returned'],terminated=a['terminated'],truncated=a['truncated']),
                        forced_branch=dict(status=b['status'],own_returned=b['own_returned'],terminated=b['terminated'],truncated=b['truncated'])))
    return dict(joins=joins,pairs=pairs,denominator=54,observed=sum(p['status']=='observed' for p in pairs))


def build():
    a,b,r1,r2=source_summary();result=compare(r1['rows'],r2['rows'])
    for p,h in SOURCE_PINS.items():need(six.digest(Path(p).read_bytes())==h,'stable source SHA')
    return dict(schema=NAME+'.analysis',sources={'six_summary':dict(path=str(SIX),sha256=SIX_SHA),
        'forced_summary':dict(path=str(FORCED),sha256=FORCED_SHA),'source_pins':SOURCE_PINS},
        result=result, panels=dict(subgoal=dict(errors=r1['errors'],counts=r1['counts']),forced=dict(errors=r2['errors'],counts=r2['counts'])),
        scope='post-hoc cross-panel descriptive comparison; common task goal not common intermediate target',
        missing_policy='missing values stay null; no padding; missing cp0 blocks pair',
        trust='reconstructed pinned saved summaries and their receipt/report joins; inherits upstream physical strict, no fresh fullstrict',
        paired_not_independent=True,causal_claim=False,fresh_roots=0,scientific_admission=False,
        resource_caveat='panel counts recorded separately; generation costs and FLOPs not matched or inferred',
        new_calls=dict(physics=0,neural=0,gpu=0,training=0))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ('build','verify'):
        x=sub.add_parser(name);x.add_argument('--output',required=True)
        if name=='verify':x.add_argument('--sha256',required=True)
    args=p.parse_args();value=build();path=Path(args.output)
    if args.command=='build':
        need(path.is_absolute() and path.resolve()==path and path.parent==ROOT/'results' and path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink(), 'exclusive output')
        data=six.canonical(value)
        with path.open('xb') as f:need(f.write(data)==len(data),'complete write');f.flush();os.fsync(f.fileno())
        fd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        sha=six.digest(data)
    else:
        need(six.read_bound(path,args.sha256)==value,'whole reconstruction');sha=args.sha256
    print(six.canonical(dict(path=str(path),sha256=sha,verified=args.command=='verify')).decode(),end='')


if __name__=='__main__':main()
