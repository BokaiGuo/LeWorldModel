"""K2 six-slot saved resource summary and non-dispatching physical contract."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from experiments.spectral_reachability import agent_p3_c3_k2_host_cuda_20261029_v2 as host

NAME = 'agent_p3_c3_k2_saved_handoff_20261029_v1'
ROOT = host.ROOT
CAP = 16*1024**2
need = host.need
canon = host.canonical


def pinned(path, expected):
    host.durable.key(expected)
    p = host.durable.regular(path, CAP)
    need(host.codec.file_sha(p) == expected, 'external pinned bytes')
    return host.durable.read_json(p, CAP)


def summarize_tree(identity, row, tree, arm):
    need(all(row[k] == arm[k] == identity[k] for k in ('slot','root','mode','K')),
         'fixed slot and strict receipt identity')
    need(tree['schema'] == host.k2.NAME+'.tree' and tree['backend_kind'] == 'HOST_CUDA'
         and tree['mode'] == identity['mode'] and tree['K'] == 2
         and tree['status'] == row['status'] == arm['status'], 'tree identity')
    rr = tree['resource_receipt']; saved = arm['resources']; axes = host.core.AXES
    need(rr['caps'] == saved['caps'] == host.k2.caps(), 'exact common K2 cap')
    out = {}
    for key in ('caps','attempted','completed'):
        need(set(rr[key]) == set(axes) and rr[key] == saved[key], 'all axes match strict receipt')
    for a in axes:
        c, x, y = (rr[k][a] for k in ('caps','attempted','completed'))
        need(all(type(v) is int and v >= 0 for v in (c,x,y)) and y <= x <= c, 'bounded resource axis')
        out[a] = dict(cap=c,attempted=x,completed=y,unspent=c-x,attempted_not_completed=x-y)
    need(saved['events_count'] == len(rr['events']) and
         saved['events_semantic_sha256'] == host.codec.semantic(rr['events']), 'full saved ledger identity')
    jobs = tree['jobs']; need(len(jobs) == 144 and [j['slot'] for j in jobs] == list(range(144)), 'all request slots')
    need(all(j['status'] in ('not_requested','queued','returned_eligible','returned_ineligible','blocked_after_failure')
             and (j['status'] == 'not_requested') == (j['request'] is None) for j in jobs), 'job status/request identity')
    need(all(q['status'] in ('queued','stopped_by_original_child_policy','expansion_cap_before_node_queue',
        'refused_unsupported_original_type','resource_cap_before_node_queue') for q in tree['queues']), 'known queue status')
    assigned = [canon(j['request']) for j in jobs if j['request'] is not None]
    queued = [canon(req) for q in tree['queues'] if q['status'] == 'queued' for req in q['requests']]
    need(Counter(assigned) == Counter(queued), 'atomic queue assigned identities')
    refused = sum(len(q['requests']) for q in tree['queues'] if q['status'] not in
                  ('queued','stopped_by_original_child_policy'))
    unused = sum(j['status'] == 'not_requested' for j in jobs)
    need(refused <= unused, 'refused requests overlap unassigned identities')
    plan = tree['model_plan']; available = arm['plan_available']
    need(type(available) is bool and available == (plan is not None), 'strict witness availability')
    first = None; plan_sha = None
    if available:
        need(tree['status'] == 'tree_complete_not_admitted', 'only completed model plan')
        actions = plan['actions']; need(type(actions) is list and len(actions) >= 5 and len(actions) % 5 == 0,
                                     'full packet actions')
        need(all(type(pair) is list and len(pair) == 2 and
             all(type(v) in (float,int) and math.isfinite(v) and -1 <= v <= 1 for v in pair)
             for pair in actions), 'literal finite raw action domain')
        replay = tree['replays']; need(len(replay) == 2 and
            [x['mode'] for x in replay] == ['whole','segmented'] and
            all(x['status'] == 'returned' and x['path'] == plan['path'] for x in replay), 'both actual saved replays')
        first = actions[:5]; plan_sha = hashlib.sha256(canon(plan)).hexdigest()
    else:
        need(tree['status'] != 'tree_complete_not_admitted', 'no missing plan in completed tree')
    return dict(**identity,status=tree['status'],axes=out,job_denominator=144,
        jobs_by_status=dict(sorted(Counter(j['status'] for j in jobs).items())),
        assigned=len(assigned),refused_unassigned=refused,unassigned=unused,
        queues=[dict(status=q['status'],generated=len(q['requests']),minimum=q['minimum'],
                     verification_reserve=q['verification_reserve']) for q in tree['queues']],
        model_plan_available=available,model_plan_semantic_sha256=plan_sha,
        candidate_first5_raw=first,physical_dispatch_ready=False,
        physical_missing=['authenticated native reset/root and goal binding',
                          'exact model-history to native state observation guard',
                          'declared action mapping and roundtrip witness',
                          'preapproved fixed six-slot physical resource and stop contract'],
        physical_outcome=None)


def build(receipt_path,receipt_sha,output):
    receipt = pinned(receipt_path,receipt_sha)
    need(receipt.get('schema') == host.NAME+'.consume' and
         receipt.get('status') == 'saved6_k2_common_cap_engineering_verified' and
         receipt.get('denominator') == 6 and receipt.get('total_request_denominator') == 864,
         'actual K2 strict HOST receipt')
    need(all(receipt.get(k) is False for k in host.FLAGS), 'no scientific release')
    prepared = pinned(receipt['prepared'],receipt['prepare_sha256'])
    need(prepared.get('schema') == host.NAME+'.prepare' and prepared['roster'] == host.k2.roster(), 'actual prepared roster')
    reader = host.Reader(output,receipt['manifest_sha256'],receipt['completion_sha256'])
    try:
        need(host.codec.file_sha(Path(output)/'report.json') == receipt['report_sha256'], 'actual report SHA')
        report = reader.report; saved_p, rows = reader.execution()
        need(canon(saved_p) == canon(prepared) and canon(rows) == canon(report['rows'])
             and report['prepared'] == receipt['prepared'] and report['prepare_sha256'] == receipt['prepare_sha256']
             and report['status'] == 'six_slots_settled_not_admitted' and report['errors'] == [], 'saved input/report binding')
        arms = receipt['arms']; need(len(rows) == len(arms) == 6, 'six strict slots')
        result = [summarize_tree(i,r,reader.result(i['slot'],r['result']),a)
                  for i,r,a in zip(host.k2.roster(),rows,arms,strict=True)]
    finally: reader.close()
    files = (Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'))
    return dict(schema=NAME+'.analysis',sources={str(p):host.codec.file_sha(p) for p in files},
        receipt_path=str(Path(receipt_path).absolute()),receipt_sha256=receipt_sha,
        output_path=str(Path(output).absolute()),report_sha256=receipt['report_sha256'],
        manifest_sha256=receipt['manifest_sha256'],completion_sha256=receipt['completion_sha256'],
        prepared=receipt['prepared'],prepare_sha256=receipt['prepare_sha256'],rows=result,
        denominator=6,request_denominator=864,same_caps=True,same_actual_resources=False,
        equal_FLOPs=False,physical_dispatch_authorized=False,scientific_admission=False,
        selection='exploratory K2 after K1',trust='inherits externally pinned MAIN strict receipt; no fullstrict replay')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('analyze','verify'))
    for name in ('receipt','receipt-sha','output','analysis'):ap.add_argument('--'+name,required=True)
    ap.add_argument('--analysis-sha');a=ap.parse_args(); value=build(a.receipt,a.receipt_sha,a.output)
    path=Path(a.analysis).absolute();need(path.resolve()==path,'absolute output path')
    if a.command == 'analyze':
        need(a.analysis_sha is None and not path.exists() and path.parent == ROOT/'results' and path.name.startswith(NAME), 'new own result')
        raw=canon(value)
        need(len(raw) <= CAP, 'bounded output')
        with path.open('xb') as stream:
            need(stream.write(raw)==len(raw),'complete write');stream.flush();os.fsync(stream.fileno())
        host.durable.fsync_dir(path.parent)
        print(json.dumps(dict(path=str(path),sha256=host.codec.file_sha(path))))
    else:
        need(canon(pinned(path,a.analysis_sha)) == canon(value),'complete analysis rebuild')
        print(json.dumps(dict(verified=True,sha256=a.analysis_sha)))


if __name__ == '__main__': main()
