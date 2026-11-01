"""Paired saved-catalog check on IDENTICAL goal IDs, no neural/control work."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.spectral_reachability.pusht_goal_set_metric import distances
from experiments.spectral_reachability.run_pusht_d2p_budget_guard_cuda import digest


def compare(old,new,vectors):
    if old['root_sha256']!=new['root_sha256']:raise ValueError('catalog root mismatch')
    for c in (old,new):
        if len(c['ids'])!=len(c['vectors']) or len(c['ids'])!=len(set(c['ids'])):raise ValueError('unique aligned catalog IDs required')
    oldindex={identity:i for i,identity in enumerate(old['ids'])}
    if not set(new['ids'])<=set(oldindex):raise ValueError('common predeclared goal IDs missing')
    restricted=[old['vectors'][oldindex[i]] for i in new['ids']]
    a=distances(vectors,restricted);b=distances(vectors,new['vectors'])
    return dict(common_goal_ids=len(new['ids']),old_common_costs=a['costs'],clean_costs=b['costs'],
        old_common_cost_range=float(np.ptp(a['costs'])),clean_cost_range=float(np.ptp(b['costs'])))


def run(old_report,new_report,probes):
    paths=list(map(Path,(old_report,new_report,probes)));oldr,newr,physical=[json.loads(p.read_text()) for p in paths]
    for r,status in ((oldr,'construction_completed'),(newr,'construction_completed'),(physical,'diagnostic_completed')):
        if r['status']!=status or not r['gates'] or not all(v is True for v in r['gates'].values()):raise ValueError('complete artifacts required')
    def load_catalogs(report):
        result={}
        for path,sha in report['catalog_hashes'].items():
            if digest(Path(path))!=sha:raise ValueError('catalog changed')
            c=json.loads(Path(path).read_text())
            if c['root_sha256'] in result:raise ValueError('duplicate catalog root')
            result[c['root_sha256']]=c
        return result
    old,new=map(load_catalogs,(oldr,newr))
    if len(physical['rows'])!=4 or len(old)!=4 or set(old)!=set(new) or {r['root_sha256'] for r in physical['rows']}!=set(old):raise ValueError('four matching roots required')
    rows=[]
    for row in physical['rows']:
        key=row['root_sha256'];values=[]
        for b in row['contexts']:
            if len(b['probes'])!=5:raise ValueError('all five probes required')
            values.append(dict(candidate_id=b['candidate_id'],statuses=[p['status'] for p in b['probes']],
                object_pose_invariant=len({tuple(p['state'][2:5]) for p in b['probes']})==1,
                comparison=compare(old[key],new[key],[p['vector'] for p in b['probes']])))
        rows.append(dict(root_sha256=key,root_index=row['root_index'],contexts=values))
    return dict(schema='pusht_goal_sprite_pair_audit.v1',inputs={str(p.resolve()):digest(p) for p in paths},
        rows=rows,common_goal_eligibility_controlled=True,neural_work=False,control_selection_changed=False,
        P1_admission=False,scope='known-root metric sensitivity only; not physical control benefit')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('old_report','new_report','probes','output'):p.add_argument('--'+name.replace('_','-'),dest=name,type=Path,required=True)
    args=vars(p.parse_args());output=args.pop('output');r=run(**args)
    with output.open('x') as stream:json.dump(r,stream,indent=2,allow_nan=False)
    for row in r['rows']:
        print(row['root_index'],[(b['candidate_id'],b['comparison']['old_common_cost_range'],b['comparison']['clean_cost_range']) for b in row['contexts']])
