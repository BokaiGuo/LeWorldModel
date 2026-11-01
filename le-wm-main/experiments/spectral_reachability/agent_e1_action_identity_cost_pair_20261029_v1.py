"""Saved-only action identities, same-time observations and separately scoped costs."""
import argparse
import hashlib
import os
from pathlib import Path

from experiments.spectral_reachability import agent_e1_cross_panel_physical_pair_20261029_v1 as cross

NAME='agent_e1_action_identity_cost_pair_20261029_v1'
ROOT=cross.ROOT
need=cross.need
six=cross.six
RESULT=ROOT/'results'
CROSS=RESULT/'agent_e1_cross_panel_physical_pair_20261029_v1_MAIN01.json'
PINS={
 str(CROSS):'e9024bb6a0f9c604e65dd04488e81fed05cfe940ed85a2b5d76e1d25f1602c08',
 str(Path(cross.__file__).resolve()):'ee43c36dc6d11585c9e160f431fae760b29751db5453c46c2413233fb562e511',
 str(RESULT/'agent_e1_six_slot_common_cuda_20261029_v2_consume_MAIN01.json'):'1ca8e949cc4b219b9b03adb3d34ee859e924e0078f9f81c519be2c03ed77be3d',
 str(RESULT/'agent_e1_forced_candidate_replay_cuda_20261029_v1_consume_MAIN01.json'):'0605d651ffcb7b4edafed61c12604a89453594e7df5b573b3a944a048c548dc5',
 str(RESULT/'agent_e1_six_slot_physical_20261029_v1_prepare_MAIN01.json'):'8c6426b249fadc3ad730f1ca3bad9aa3b805badbe5538ef11e5132be021dc922',
 str(RESULT/'agent_e1_forced_candidate_physical_20261029_v1_prepare_MAIN01.json'):'7d7400cbc3c74756c984bc1d5beaf3dd76f54119b12700ef0716015edc867b5b',
 str(RESULT/'agent_e1_forced_candidate_replay_cuda_20261029_v1_cuda_MAIN01/report.json'):'ba2df80d87c825452cdbf3050813e6720f074cd69ba70aea4f716b9007d5717e',
}


def digest(data):return hashlib.sha256(data).hexdigest()


def identity(row,panel):
    plan=row['selected_plan'];bits=cross.raw_bits(plan['commands'])
    n=row['own_returned'];need(type(n) is int and 0<=n<=10,'actual returned count')
    cps={c['step']:c for c in row['checkpoints']}
    need(len(cps)==len(row['checkpoints']) and all(k in (0,5,10) and k<=n for k in cps),'checkpoint count and time')
    return dict(panel=panel,root=row['root_slot'],arm=row['arm'],selection_digest=row['selection_digest'],
        plan_raw5_sha256=digest(b''.join(bits[:5])),plan_raw10_sha256=digest(b''.join(bits)),
        executed_prefix_sha256=digest(b''.join(bits[:n])),returned=n,attempted=row['own_attempted'],
        status=row['status'],terminated=row['terminated'],truncated=row['truncated'],
        first_error=row['first_error'],secondary_errors=row['secondary_errors'],
        checkpoints={str(k):dict(state=v['state'],snapshot=v['snapshot'],pixel_sha256=v['pixel_sha256']) for k,v in cps.items()})


def group_identities(rows):
    groups={}
    for row in rows:
        key=(row['root'],row['plan_raw10_sha256'])
        groups.setdefault(key,[]).append(dict(panel=row['panel'],arm=row['arm']))
    return [dict(root=root,plan_raw10_sha256=h,members=members) for (root,h),members in groups.items()]


def classify(a,b,step):
    need(type(step) is int and step in (0,5,10),'common time')
    a_cp=a['checkpoints'].get(str(step));b_cp=b['checkpoints'].get(str(step))
    if a_cp is None or b_cp is None:return dict(status='missing',same_observation=None,same_state=None,same_snapshot=None,same_pixels=None)
    eq_state=a_cp['state']==b_cp['state'];eq_snapshot=a_cp['snapshot']==b_cp['snapshot'];eq_pixels=a_cp['pixel_sha256']==b_cp['pixel_sha256']
    return dict(status='observed',same_state=eq_state,same_snapshot=eq_snapshot,same_pixels=eq_pixels,
                same_observation=eq_state and eq_snapshot and eq_pixels)


def slot_costs(subplan,ctrl,forced_report,forced_receipt,prep_a,prep_b):
    need(prep_a['receipt_sha256']==PINS[str(RESULT/'agent_e1_six_slot_common_cuda_20261029_v2_consume_MAIN01.json')], 'subgoal receipt binding')
    need(prep_b['receipt_sha256']==PINS[str(RESULT/'agent_e1_forced_candidate_replay_cuda_20261029_v1_consume_MAIN01.json')], 'forced receipt binding')
    need(prep_b['report_sha256']==forced_receipt['report_sha256']==PINS[str(RESULT/'agent_e1_forced_candidate_replay_cuda_20261029_v1_cuda_MAIN01/report.json')], 'forced replay report binding')
    need(len(ctrl['rows'])==len(forced_report['rows'])==len(subplan)==6,'six per-slot costs')
    need([(x['root'],x['arm']) for x in ctrl['rows']]==list(six.SLOTS),'controller slots')
    need([(x['root'],x['method']) for x in forced_report['rows']]==list(cross.forced.SLOTS),'replay slots')
    out=[]
    for plan,c in zip(subplan,ctrl['rows'],strict=True):
        need(plan['root_slot']==c['root'] and plan['arm']==c['arm'] and plan['compiler_selection_sha256']==c['selection_sha256'],'controller selection join')
        out.append(dict(panel='subgoal',root=c['root'],arm=c['arm'],controller=c['controller_resources'],replay=c['replay_resources'],
                        target_generation=None,physical_per_slot=None,total_cost=None))
    for r in forced_report['rows']:
        out.append(dict(panel='forced',root=r['root'],arm=r['method'],controller=None,replay=r['resources'],
                        target_generation=None,physical_per_slot=None,total_cost=None))
    return out


def build():
    docs={p:six.read_bound(p,h) for p,h in PINS.items() if p.endswith('.json')}
    for p,h in PINS.items():need(six.digest(Path(p).read_bytes())==h,'stable source SHA')
    prior=docs[str(CROSS)];need(prior==cross.build(),'whole previous comparison reconstruction')
    _,_,r1,r2=cross.source_summary()
    aa=[identity(r,'subgoal') for r in r1['rows']];bb=[identity(r,'forced') for r in r2['rows']]
    indexed={(x['panel'],x['root'],x['arm']):x for x in aa+bb}
    comparisons=[]
    for p in prior['result']['pairs']:
        a=indexed['subgoal',p['root'],p['subgoal_arm']];b=indexed['forced',p['root'],p['forced_arm']]
        obs=classify(a,b,p['step'])
        if p['status']=='observed':need(obs['status']=='observed','same-time observation join')
        comparisons.append(dict(root=p['root'],step=p['step'],subgoal_arm=p['subgoal_arm'],forced_arm=p['forced_arm'],
            same_raw5=p['same_first5'],same_raw10=p['same_raw10'],observation=obs,delta=p['delta'],missing=p['missing'],
            identical_actions_observation_diverged=(not obs['same_observation']) if p['same_raw10'] and obs['status']=='observed' else None))
    prep_a=docs[str(RESULT/'agent_e1_six_slot_physical_20261029_v1_prepare_MAIN01.json')]
    prep_b=docs[str(RESULT/'agent_e1_forced_candidate_physical_20261029_v1_prepare_MAIN01.json')]
    ctrl=docs[str(RESULT/'agent_e1_six_slot_common_cuda_20261029_v2_consume_MAIN01.json')]
    frc=docs[str(RESULT/'agent_e1_forced_candidate_replay_cuda_20261029_v1_consume_MAIN01.json')]
    fr=docs[str(RESULT/'agent_e1_forced_candidate_replay_cuda_20261029_v1_cuda_MAIN01/report.json')]
    costs=slot_costs(prep_a['plans'],ctrl,fr,frc,prep_a,prep_b)
    for p,h in PINS.items():need(six.digest(Path(p).read_bytes())==h,'source changed')
    return dict(schema=NAME+'.analysis',sources=PINS,arms=aa+bb,action_groups=group_identities(aa+bb),
        common_time_pairs=comparisons,paired_denominator=54,slot_costs=costs,
        panel_physical_counts=prior['panels'],total_cost_comparable=False,FLOPs=None,
        cost_boundary='model module axes are nested; replay is additional; forced generation/inherited controller cost and target generation not inferred; physical counts panel-level only',
        scientific_admission=False,causal_claim=False,fresh_roots=0,new_calls=dict(physics=0,neural=0,gpu=0,training=0),
        trust='hash-pinned saved receipts and prior recomputation; inherits upstream physics strict; no neural/physical replay')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for command in ('build','verify'):
        q=sub.add_parser(command);q.add_argument('--output',required=True)
        if command=='verify':q.add_argument('--sha256',required=True)
    x=p.parse_args();value=build();out=Path(x.output)
    if x.command=='build':
        need(out.is_absolute() and out.resolve()==out and out.parent==RESULT and out.name.startswith(NAME+'_') and not out.exists() and not out.is_symlink(),'exclusive output')
        raw=six.canonical(value)
        with out.open('xb') as f:need(f.write(raw)==len(raw),'complete output');f.flush();os.fsync(f.fileno())
        fd=os.open(out.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        h=digest(raw)
    else:
        need(six.read_bound(out,x.sha256)==value,'whole saved reconstruction');h=x.sha256
    print(six.canonical(dict(path=str(out),sha256=h,verified=x.command=='verify')).decode(),end='')


if __name__=='__main__':main()
