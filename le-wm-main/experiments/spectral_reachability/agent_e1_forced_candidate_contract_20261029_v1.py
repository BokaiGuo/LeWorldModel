"""Saved-only fixed forced-candidate contract; no new inference or physics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from experiments.spectral_reachability import agent_e1_same_target_host_cuda_20261029_v1 as host

old=host.old
ROOT=host.ROOT
NAME='agent_e1_forced_candidate_contract_20261029_v1'
REPORT=ROOT/'results/agent_e1_same_target_host_cuda_20261029_v1_cuda_MAIN01/report.json'
REPORT_SHA='92a8c6d5c9a6fa60bb1738cfd5ed9abd70d5cc889ada04d62886b14c1ca5f3b7'
RECEIPT=ROOT/'results/agent_e1_same_target_host_cuda_20261029_v1_consume_MAIN01.json'
RECEIPT_SHA='f385e1de76ca391a7ef4d03c198211cd303eb2e4d2bc3bdd69537146e8330eba'
PREPARE_SHA='958c2fa2c73ddf0b66c0d9dd6d5fd04c610d0f679894940664261d868d83b762'
METHODS=('bank_winner','zero_extra','witness_extra')


def plans(authority, saved):
    old.require(len(authority['rows'])==2 and len(saved['choices'])==len(saved['results'])==4,'complete two roots/four slots')
    out=[]
    for root in (0,1):
        original, witnessed=saved['choices'][2*root:2*root+2]
        results=saved['results'][2*root:2*root+2]
        old.require(original['arm']=='zero_extra' and witnessed['arm']=='witness_extra','arm order')
        old.require(old.canonical(results[0]['candidates'])==old.canonical(results[1]['candidates']), 'exact original paired banks')
        # We retain the existing bank winner, defined by original Q/tie rule,
        # not the selected extra. This rule is fixed even for future changes.
        bank=results[0]['candidates']
        old.require(len(bank)==72 and all(c['index']==i for i,c in enumerate(bank)), '72 ordered bank')
        index=min(range(72),key=lambda i:(bank[i]['cost'],i))
        prefix=authority['rows'][root]['witness_raw5']
        extras=host.stage.four.spec.extras(prefix)
        old.require(old.canonical(original['extra'])==old.canonical(extras['zero_extra']) and
                    old.canonical(witnessed['extra'])==old.canonical(extras['witness_extra']), 'literal original extra actions')
        sources=(bank[index]['commands'],extras['zero_extra'],extras['witness_extra'])
        for kind,raw in zip(METHODS,sources,strict=True):
            raw=host.stage.four.spec.raw_actions(raw,10)
            out.append(dict(root=root,method=kind,task_id=authority['rows'][root]['task_id'],
                task_sha256=authority['rows'][root]['task_sha256'],target_sha256=authority['rows'][root]['target_sha256'],
                raw10=raw,raw_sha256=old.sha(old.canonical(raw)),bank_winner_index=index,
                inherited_witness_selection_sha256=authority['rows'][root]['compiled_selection_sha256']))
    return out


def build():
    report=old.parse_canonical(old.read_bound(REPORT,REPORT_SHA,old.RESERVE))
    receipt=old.parse_canonical(old.read_bound(RECEIPT,RECEIPT_SHA,old.RESERVE))
    old.require(receipt['status']=='four_saved_controller_replay_verified' and receipt['report_sha256']==REPORT_SHA
                and receipt['verification']['verified'] is True and receipt['verification']['slots']==4
                and receipt['scientific_admission'] is False and report['status']=='all_four_sealed'
                and report['errors']==[] and report['prepare_sha256']==PREPARE_SHA,'complete inherited receipt/report')
    prepared=old.parse_canonical(old.read_bound(Path(report['prepared']),PREPARE_SHA,16*1024*1024))
    execution_sources=[ROOT/'experiments/spectral_reachability'/(name+'.py') for name in host.LOCAL]
    for path in execution_sources:
        old.require(host.stream_sha(path)==prepared['closure'][str(path)], 'parent execution source drift')
    authority,_=host.authorities()
    payload,events=old.read_store(REPORT.parent,report)
    del events
    rows=plans(authority,payload['result.z'])
    paths=[Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')] + execution_sources
    return dict(schema=NAME+'.prepare',report_sha256=REPORT_SHA,receipt_sha256=RECEIPT_SHA,
                parent_prepare_sha256=PREPARE_SHA,
                source_hashes={str(p):host.stream_sha(p) for p in paths},authority=authority,slots=rows,
                protocol=dict(roots=[0,1],methods=list(METHODS),slots=6,public_controls_per_slot=10,
                    forced_actions=True,raw_clip=False,reselection=False,post_outcome_selection=False,
                    model_replay_per_slot=4,model_replay_axis_total=24,
                    native_ceiling_hypothesis=dict(public_steps=120,physics_steps=1224,explicit_png=132,
                        constructor=6,close=6,reset=12,set_state=24,goal_render=12,goal_bind=6),
                    native_ceiling_qualification='old six-branch recipe conditional on no extra calls; new hook census required',
                    full_output_before_scoring=True,early_stop='retain missing, never pad',
                    arrival_threshold=None,generation_cost_equal=False,FLOPs=None),
                receipt_scope='inherited saved verification, not rerun fullstrict',production_authorized=False,
                scientific_admission=False,physical_reachability_authenticated=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify'))
    parser.add_argument('--output');parser.add_argument('--manifest');parser.add_argument('--manifest-sha')
    args=parser.parse_args()
    if args.command=='prepare':
        path=Path(args.output);old.require(path.is_absolute(),'absolute output')
        result=dict(path=str(path),sha256=old.put_new(path.parent,path.name,old.canonical(build())))
    else:
        saved=old.parse_canonical(old.read_bound(Path(args.manifest),args.manifest_sha,16*1024*1024))
        old.require(saved==build(),'complete source-bound contract')
        result=dict(status='contract_verified_not_production_authorized')
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
