"""Candidate future bridge receipt ABI; byte and semantic join, never authority issuance."""
import argparse
import hashlib
import json
from pathlib import Path

from experiments.spectral_reachability import agent_p3_cp10_saved_source_closure_20261029_v1 as files
from experiments.spectral_reachability import agent_p3_future_cp10_task_projection_20261029_v1 as projection

NAME = 'agent_p3_future_cp10_receipt_join_20261029_v1'
PRODUCER = 'agent_p3_future_cp10_bridge_20261029_v1'
OLD_SOURCE = '40d51e4d5ad5fce5db9417fecc639e0288f286b678a5e7b0eb4e22adea392b77'
need = files.need
raw = files.stage.raw


def read(ref, cap=96*1024*1024):
    need(type(ref) is dict and set(ref)=={'path','sha256'}, 'exact descriptor')
    body=files.data(ref['path'],ref['sha256'],cap)
    def pairs(items):
        out={}
        for k,v in items:
            need(k not in out, 'duplicate key')
            out[k]=v
        return out
    obj=json.loads(body,object_pairs_hook=pairs,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite')))
    raw(obj)
    return obj


def join(refs):
    need(type(refs) is dict and set(refs)=={'prepare','report','completion','receipt','source'}, 'five external references')
    source=refs['source']; need(type(source) is dict and set(source)=={'path','sha256'}, 'source descriptor')
    files.data(source['path'],source['sha256'])
    need(source['sha256'] != OLD_SOURCE, 'historical bridge cannot be future')
    p,r,c,s=(read(refs[k]) for k in ('prepare','report','completion','receipt'))
    for obj,suffix in ((p,'prepare'),(r,'report'),(c,'completion'),(s,'consume')):
        need(obj['schema']==PRODUCER+'.'+suffix, 'future producer schema')
    need(p['producer_source']==r['producer_source']==s['producer_source']==source, 'source byte join')
    need(r['prepared']==c['prepared']==refs['prepare']['path'] and
         r['prepare_sha256']==c['prepare_sha256']==refs['prepare']['sha256'], 'prepare join')
    need(c['report_sha256']==s['report_sha256']==refs['report']['sha256'] and
         s['completion_sha256']==refs['completion']['sha256'] and s['completion']==c, 'report/receipt join')
    need(c['status']=='published_within_cap' and r['status']==s['status']=='all5_sealed_not_admitted'
         and r['errors']==r['blocked']==s['errors']==s['blocked']==[], 'complete only')
    for obj in (r,c,s):
        for key in ('execution_authorized','scientific_admission','freshness','independent_confirmation'):
            need(obj[key] is False, 'no admission '+key)
    need(r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'], 'same model hash')
    files.stage.core.hash_value(p['expected_model_hash'])
    need(len(p['rows'])==len(r['rows'])==len(s['tasks'])==5 and r['fixed_denominator']==s['fixed_denominator']==5, 'five slots')
    need(r['counts']==s['counts']=={'encoder':5,'projector':5,'predict':0,'physics':0,'training':0}, 'recorded resource counts')
    need(s['bindings'][refs['prepare']['path']]==refs['prepare']['sha256'] and
         s['bindings'][refs['report']['path']]==refs['report']['sha256'] and
         s['bindings'][refs['completion']['path']]==refs['completion']['sha256'] and
         s['bindings'][source['path']]==source['sha256'], 'receipt-bound references')
    output=Path(refs['report']['path']).parent
    latents=[]
    for i,(spec,encoded,item) in enumerate(zip(p['rows'],r['rows'],s['tasks'],strict=True)):
        need((spec['root_slot'],spec['role']) == projection.ORDER[i] and spec['index']==i, 'fixed order')
        need(spec['cp5_task']['binding']['model_sha256']==p['expected_model_hash'], 'task model')
        pr=read(spec['source_report']); ps=read(spec['source_receipt'])
        need(ps['physical_report_sha256']==spec['source_report']['sha256'] and
             ps['bindings'][spec['source_report']['path']]==spec['source_report']['sha256'] and
             raw(ps['rows'])==raw(pr['rows']) and ps['errors']==pr['errors']==[], 'physical receipt joins')
        need(s['bindings'][spec['source_report']['path']]==spec['source_report']['sha256'] and
             s['bindings'][spec['source_receipt']['path']]==spec['source_receipt']['sha256'], 'physical authority bindings')
        for d in (encoded['pixels'],encoded['latent']):
            path=str(output/d['file'])
            need(s['bindings'][path]==d['sha256'] and c['artifact_sha256'][d['file']]==d['sha256'], 'saved artifact closure')
        latents.append(files.verify_row(spec,encoded,item,ps['rows'][spec['source_slot']],output))
    result=projection.project(p['rows'],latents,[x['latent'] for x in r['rows']])
    need(all(raw(a)==raw(b) for a,b in zip(result['endpoints'],s['tasks'],strict=True)), 'exact projected tasks')
    return dict(schema=NAME+'.join',status='candidate_receipt_join_only_not_execution_authority',
        source_class='future_declared_not_freshness_proven',producer_source=source,slots=5,history_frames=15,
        task_projection=result,inputs=refs,physical_fullstrict_inherited_not_rebuilt=True,
        producer_fullstrict_inherited_not_rebuilt=True,source_authority_verified=False,
        trusted_timestamp_proven=False,fresh_roots_certified=False,execution_authorized=False,scientific_admission=False)


def main():
    a=argparse.ArgumentParser()
    for k in ('prepare','report','completion','receipt','source'):
        a.add_argument('--'+k,required=True); a.add_argument('--'+k+'-sha',required=True)
    p=a.parse_args(); refs={k:dict(path=getattr(p,k),sha256=getattr(p,k+'_sha')) for k in ('prepare','report','completion','receipt','source')}
    result=join(refs)
    print(json.dumps({'schema':result['schema'],'status':result['status'],'sha256_of_compact_join':hashlib.sha256(raw(result)).hexdigest(),
                      'slots':result['slots'],'execution_authorized':False},sort_keys=True))


if __name__=='__main__':main()
