"""Match goal grid identities before comparing saved frozen embeddings."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability.pusht_goal_set_metric import distances


def common_catalogs(edited, simulator):
    if edited['root_sha256'] != simulator['root_sha256']:
        raise ValueError('stable root identity mismatch')
    lookup=[]
    for catalog in (edited,simulator):
        ids,vec=catalog['ids'],np.asarray(catalog['vectors'],float)
        if len(ids)!=len(set(ids)) or vec.ndim!=2 or len(vec)!=len(ids) or not np.isfinite(vec).all():
            raise ValueError('unique IDs and complete finite embedding rows required')
        lookup.append(dict(zip(ids,vec)))
    ids=sorted(set(lookup[0]) & set(lookup[1]))
    if not ids or any(not i.startswith('agent_') for i in ids):
        raise ValueError('nonempty matching grid identities required; no source goals')
    a,b=(np.asarray([table[i] for i in ids]) for table in lookup)
    if a.shape!=b.shape:raise ValueError('same embedding dimensions required')
    return ids,a,b


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(edited_report, simulator_report, probes, control, output):
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal,verify_seal
    paths=list(map(Path,(edited_report,simulator_report,probes,control)))
    edited_report,simulator_report,probes,control=paths
    old,new,physical,controls=[json.loads(p.read_text()) for p in paths]
    for r,status in zip((old,new,physical,controls),('construction_completed','diagnostic_completed','diagnostic_completed','calibration_completed')):
        if r['status']!=status or not r['gates'] or not all(v is True for v in r['gates'].values()):
            raise ValueError('complete source reports required')
        for s in r['source_seals']:verify_seal(output_dir=s['output_dir'],expected_manifest_sha256=s['manifest_sha256'])
    prefix='pusht_simulator_goal_catalog_20261027'
    protocol_path=simulator_report.parent/f'{prefix}_protocol.json'
    protocol=json.loads(protocol_path.read_text())
    from experiments.spectral_reachability.d2p_composition_witness_v2 import digest
    if digest(protocol)!=new['protocol_sha256']:raise ValueError('simulator protocol changed')
    bindings={str(p.resolve()):sha(p) for p in paths+[protocol_path]}
    edited={}
    for p,d in old['catalog_hashes'].items():
        if sha(p)!=d:raise ValueError('edited catalog changed')
        c=json.loads(Path(p).read_text());edited[c['root_sha256']]=c;bindings[p]=d
    simulator={}
    for row in protocol['prepared']:
        p=simulator_report.parent/f'{prefix}_catalog_{row["root_index"]}.json'
        c=json.loads(p.read_text());simulator[c['root_sha256']]=c;bindings[str(p.resolve())]=sha(p)
        if c['ids'] != [e['id'] for e in row['entries'] if e['valid']]:
            raise ValueError('catalog eligible identity sequence changed')
    roots=set(edited)
    if (len(roots)!=4 or set(simulator)!=roots or len(physical['rows'])!=4 or len(controls['rows'])!=4
            or {r['root_sha256'] for r in physical['rows']}!=roots
            or {r['root_sha256'] for r in controls['rows']}!=roots):
        raise ValueError('all four stable roots required')
    rows=[]
    for key in sorted(roots):
        ids,a,b=common_catalogs(edited[key],simulator[key])
        query=next(r for r in physical['rows'] if r['root_sha256']==key)
        contexts=[]
        for branch in query['contexts']:
            v=[p['vector'] for p in branch['probes']]
            if len(v)!=5:raise ValueError('all five real probes required')
            da,db=distances(v,a),distances(v,b)
            contexts.append(dict(candidate_id=branch['candidate_id'],
                object_pose_invariant=len({tuple(p['state'][2:5]) for p in branch['probes']})==1,
                edited_costs=da['costs'],simulator_costs=db['costs'],
                edited_range=float(np.ptp(da['costs'])),simulator_range=float(np.ptp(db['costs'])),
                statuses=[p['status'] for p in branch['probes']]))
        control_row=next(r for r in controls['rows'] if r['root_sha256']==key)
        da,db=(distances([b['final_vector'] for b in control_row['branches']],goals) for goals in (a,b))
        rows.append(dict(root_sha256=key,root_index=edited[key]['root_index'],common_grid_ids=ids,
            original_edited_size=len(edited[key]['ids']),original_simulator_size=len(simulator[key]['ids']),
            mean_same_id_embedding_MSE=float(np.mean((a-b)**2)),max_same_id_embedding_MSE=float(np.max(np.mean((a-b)**2,axis=1))),
            contexts=contexts,controls=[dict(candidate_id=c['candidate_id'],status=c['status'],
                object_pose_cost=c['pose']['pose_cost'],edited_cost=da['costs'][i],simulator_cost=db['costs'][i])
                for i,c in enumerate(control_row['branches'])]))
    output=Path(output);output.mkdir(exist_ok=False)
    seal=create_seal(files=[Path(__file__),Path(distances.__code__.co_filename)],args_json=json.dumps(bindings),output_dir=output/'source_seal')
    if any(sha(p)!=d for p,d in bindings.items()):raise ValueError('saved inputs changed')
    result=dict(status='matched_identity_diagnostic_completed',inputs=bindings,rows=rows,
        source_seal=seal,source_verification=verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256']),
        scope='saved encoding comparisons under identical grid identities; eligibility/scenes remain different; no control change',
        P1_admission=False,physical_efficacy_verified=False,new_neural_calls=0)
    with (output/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(status=result['status'],common_ids=[len(r['common_grid_ids']) for r in rows])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('edited_report','simulator_report','probes','control','output'):
        p.add_argument('--'+name.replace('_','-'),dest=name,type=Path,required=True)
    run(**vars(p.parse_args()))
