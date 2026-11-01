"""Saved comparison using MAIN's real bounded-reader receipt, without fullstrict."""
import argparse
from pathlib import Path
from experiments.spectral_reachability import agent_p3_equal_cap_comparison_20261029_v1 as v1

NAME='agent_p3_equal_cap_comparison_20261029_v2'
READER='main_p3_equal_cap_saved_consumer_20261029_v1'
PRODUCER_SHA='ea5b8266ed1c0f26c20db91c7cfbb9648a589a3291a5d10480ec0ab3bacc52d1'
V1_SHA='47a10c014a2bcd04d954b77d434789ee4c3daacf457c547adcb13a8fd6d4d3d4'
ROOT=v1.ROOT
require,digest,sha,canonical=v1.require,v1.digest,v1.sha,v1.canonical


def validate_new(p,r,s,refs):
    require(p['schema']==v1.NEW+'.prepare' and r['schema']==v1.NEW+'.report'
        and s['schema']==READER+'.consume','actual producer and MAIN reader schemas')
    require(s['status'] in ('saved_equal_cap_verified_not_admitted','saved_equal_cap_partial_verified_not_admitted'),
        'strict actual reader status')
    require(r['prepared']==s['prepared']==refs['prepare']['path'] and
        r['prepare_sha256']==s['prepare_sha256']==refs['prepare']['sha256'] and
        s['report_sha256']==refs['report']['sha256'],'external SHA prepare/report/receipt join')
    require(s['compatibility']==dict(original_reader_record_cap=64*1024**2,
        record_cap=p['profile']['final_reserve_bytes'],total_cap=p['profile']['output_bytes'],
        original_producer_unchanged=True,original_full_math_resource_timing_validation=True,
        new_neural_calls=0,new_physics_calls=0),'exact disclosed reader ABI; no rerun')
    require(s['inherited_producer_sha256']==PRODUCER_SHA,'frozen original producer identity')
    expected={str(ROOT/'experiments/spectral_reachability'/(READER+'.py')),
        str(ROOT/'tests'/('test_'+READER+'.py')),str(ROOT/'innerdocs'/(READER+'.md'))}
    require(set(s['consumer_sources'])==expected,'actual reader source/tests/doc closure')
    require(all(sha(path)==h for path,h in s['consumer_sources'].items()),'reader files exact receipt SHAs')
    require(s['common_budget']==p['profile'] and s['equal_cap_contract'] is True and s['matching_axes']==[]
        and s['total_FLOPs'] is None,'common caps not actual matching')
    require(len(p['tasks'])==2 and len(r['rows'])==4,'known2 fixed4')
    require([(x['slot'],x['root'],x['mode']) for x in r['rows']]==
        [(i,i//2,('fixed','adaptive')[i%2]) for i in range(4)],'fixed slot identities')
    require(not r['errors'] and r['sources_verified'] is True,'certified report closure')
    for k in ('P1_admission','P3_admission','freshness','equal_FLOPs','equal_actual_resources','equal_total_compute','equal_wall'):
        require(r[k] is False and s[k] is False,'no science/equal-actual claim')
    require(s['fixed_denominator']==r['fixed_denominator']==p['fixed_denominator']==
        dict(arms=4,requests_per_arm=144,total_requests=576),'all576 receipt/report/prepare')
    indexed={a['slot']:a for a in s['arms']}
    require(len(indexed)==len(s['arms']) and set(indexed)==
        {x['slot'] for x in r['rows'] if x['result'] is not None},'observed core receipt census')
    for row in r['rows']:
        rr=row['result']
        if rr is None:continue
        a=indexed[row['slot']]
        require((a['root'],a['mode'],a['status'])==(row['root'],row['mode'],rr['status']),'arm identity/status')
        require(digest(a['resources'])==digest(rr['resource_receipt']) and
            a['resources']['caps']==p['profile']['resource_caps'][row['mode']],'full27 resource join')
        require(a['plan_available'] is row['plan_available'] and digest(a['timing'])==digest(row['timing'])
            and a['backend_kind']==rr['backend_kind'],'real reader release/timing/backend')
        require(all(a[k]==rr[k] for k in v1.UNKNOWN),'unknowns not filled')
    return [v1.arm(x,True) for x in r['rows']]


def analyze(inputs):
    require(sha(v1.__file__)==V1_SHA,'frozen v1 helper')
    require(set(inputs)=={'old','new'} and all(set(side)=={'prepare','report','receipt'} for side in inputs.values()),'six explicit input SHA bindings')
    data={side:[v1.load(inputs[side][kind]) for kind in ('prepare','report','receipt')] for side in ('old','new')}
    p0,r0,s0=data['old'];p1,r1,s1=data['new']
    require(digest(p0['tasks'])==digest(p1['tasks']),'exact full task/context/goal/raw identity')
    require(p0['expected_model_hash']==p1['expected_model_hash'] and r0['model_before']==r1['model_before'],'same model')
    require(p1['profile']['resource_caps']['fixed']==p1['profile']['resource_caps']['adaptive'],'new common cap')
    old=v1.validate_bundle(p0,r0,s0,False,inputs['old']);new=validate_new(p1,r1,s1,inputs['new'])
    out=dict(schema=NAME+'.summary',inputs=inputs,source_sha256=sha(__file__),helper_sha256=V1_SHA,
        reader_authority=dict(schema=s1['schema'],consumer_sources=s1['consumer_sources'],compatibility=s1['compatibility']),
        denominator=dict(known_roots=2,paired_arms=4,request_slots_per_arm=144,paired_request_slots=576),
        comparison='new_minus_old descriptive; ordinal slots are not guaranteed same-parent action witnesses',
        profiles=dict(old=p0['profile'],new=p1['profile']),
        wall=dict(old_whole=r0['wall_seconds'],new_whole=r1['wall_seconds'],
            old_arm_scope='not independently synchronized per-arm in old report; resource wall_ns is core scope',
            new_arm_scope=p1['profile']['timer_scope']),arms=v1.compare(old,new),**v1.FLAGS)
    require(all(sha(ref['path'])==ref['sha256'] for side in inputs.values() for ref in side.values()) and
        all(sha(f)==h for f,h in s1['consumer_sources'].items()),'all input/reader bytes stable')
    return out


def write(path,obj):
    p=Path(path).absolute();body=canonical(obj)
    require(p.parent==ROOT/'results' and p.name.startswith(NAME) and p.resolve()==p and not p.exists(),'exclusive new v2 output')
    require(len(body)<=32*1024**2,'summary cap')
    with p.open('xb') as f:
        require(f.write(body)==len(body),'whole write');f.flush();v1.os.fsync(f.fileno())
    return dict(path=str(p),sha256=sha(p))


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    q=sub.add_parser('analyze')
    for side in ('old','new'):
        for kind in ('prepare','report','receipt'):
            q.add_argument('--'+side+'-'+kind,required=True);q.add_argument('--'+side+'-'+kind+'-sha',required=True)
    q.add_argument('--output',required=True)
    q=sub.add_parser('verify');q.add_argument('--summary',required=True);q.add_argument('--summary-sha',required=True)
    a=ap.parse_args(argv)
    if a.command=='analyze':
        inputs={side:{kind:dict(path=str(Path(getattr(a,side+'_'+kind)).absolute()),sha256=getattr(a,side+'_'+kind+'_sha'))
            for kind in ('prepare','report','receipt')} for side in ('old','new')}
        out=write(a.output,analyze(inputs))
    else:
        saved=v1.load(dict(path=str(Path(a.summary).absolute()),sha256=a.summary_sha))
        require(canonical(saved)==canonical(analyze(saved['inputs'])),'entire v2 recomputation')
        out=dict(verified=True,sha256=a.summary_sha)
    print(canonical(out).decode())


if __name__=='__main__':main()
