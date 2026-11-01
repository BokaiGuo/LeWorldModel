"""Saved actual-cp5 R gate decomposition; shared MAIN scalar math, no producer replay."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

from experiments.spectral_reachability import main_p3_R_gate_decomposition_20261029_v1 as arithmetic

ROOT = Path(__file__).resolve().parents[2]
NAME = 'agent_p3_actual_cp5_R_gate_decomposition_20261029_v1'
PRODUCER = 'agent_p3_actual_cp5_r_delivery_20261029_v1'
ROLE = 'actual_observed_abs5_10_15_control_minus5_0_5'
AUTH = {
    'prepared': (PRODUCER+'_prepare_MAIN01.json', 'b4b2f9d3ca1867a86edbb84c753cef3e52d757582299b6f851c3e8c9244a042d'),
    'report': (PRODUCER+'_cuda_MAIN01/report.json', 'deb69d78579b4cdc0ffb3ae4c671dd71f9ecd9b9e02729484085cb3d0c0dd19d'),
    'receipt': (PRODUCER+'_consume_MAIN01.json', '8d55a55e108aca1695a550861d125a0b0baac2b225cba914d0af4b47d30159a5'),
}
CP5 = ROOT/'results/agent_p3_actual_cp5_task_builder_20261029_v1_prepare_MAIN01.json'
CP5_SHA = 'ce6670d84f6f0f712bdfe5e1b7471be65b0441338c18010f79b3967ef28397c7'
PROTOCOL = ROOT/'innerdocs/main_p3_actual_cp5_R_delivery_protocol_20261029_v1.md'
PROTOCOL_SHA = 'deed88fd70073c1aaca96d32c81968b7f02a7a3841e3d8a7be00d4e4d573275f'
PINS = {
    ROOT/'experiments/spectral_reachability'/(PRODUCER+'.py'): '09af91981549547c5f770ad3d0ffa42ad92b48e37daefd612191d18ec9755a26',
    ROOT/'tests'/('test_'+PRODUCER+'.py'): '9cd129b701d5f7d87cfaa74a10ea8c709bf140bbc12687582d087537ab93c70f',
    ROOT/'innerdocs'/(PRODUCER+'.md'): '79dc91fc2f1304d9d6c526614f99fba900c7b61ad05e579c280d04a4df9e31d1',
    Path(arithmetic.__file__).resolve(): '910b3186748bb13f4a574dac58c3bfb30418fbffb7c79b9e0b539f423c1b57ef',
    PROTOCOL: PROTOCOL_SHA,
}
FLAGS = ('P1_admission', 'P3_admission', 'full_C3_demonstrated', 'physical_release',
         'execution_authorized', 'scientific_admission', 'freshness', 'equal_FLOPs',
         'equal_cap_tree', 'full_policy_executed', 'full_tree_executed',
         'original_R_equivalence', 'policy_tree_executed')
CAP = 256*1024**2
need, sha = arithmetic.need, arithmetic.sha


def canonical(value):
    """Producer compact JSON bytes (no newline); never substitute arithmetic.encode."""
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def unique(pairs):
    out = {}
    for k, v in pairs:
        need(k not in out, 'duplicate JSON key')
        out[k] = v
    return out


def read_pinned(path, expected):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= CAP, 'bounded regular saved JSON')
    raw = path.read_bytes()
    need(len(raw) <= CAP and hashlib.sha256(raw).hexdigest() == expected, 'exact saved SHA')
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: need(False, 'nonfinite JSON'))


def false_flags(doc):
    need(all(doc.get(k) is False for k in FLAGS), 'actual false admission flags including full_C3_demonstrated')


def context(c):
    need(type(c) is dict, 'full context')
    need(len(c['latent_history']) == 3 and len(c['action_history']) == 2, 'full3 latent/two token context')
    for v in c['latent_history']:
        arithmetic.vector(v, 192)
    for v in c['action_history']:
        arithmetic.vector(v, 10)


def task_pair(task, pro, root):
    old = pro['original_task']
    context(task['root_context']); context(old['root_context'])
    arithmetic.vector(task['goal'], 192)
    need(task['projection'] == [[float(i == j) for j in range(192)] for i in range(192)], 'identity192 Q')
    need(pro['source_slot'] == root and pro['local_task_sha256'] == digest(task)
         and pro['original_task_sha256'] == digest(old), 'root/local task double identity')
    need(pro['actual_cp5_absolute_steps'] == [5, 10, 15] and pro['actual_cp5_control_steps'] == [-5, 0, 5]
         and pro['original_observed_absolute_steps'] == [0, 5, 10]
         and pro['original_control_steps'] == [-10, -5, 0], 'actual cp5 not control0 relabel')
    need(len(pro['branch_witnesses']) == 4 and len({x['episode_id'] for x in pro['branch_witnesses']}) == 4,
         'four original branches retained')
    need(len(pro['raw10']) == 10 and len(pro['prefix5']) == 5 and pro['raw10'][5:] == pro['prefix5'], 'actual raw10/prefix5')
    for v in pro['raw10']:
        need(all(abs(x) <= 1 for x in arithmetic.vector(v, 2)), 'raw domain')
    need(task['root_context']['latent_history'][:2] == old['root_context']['latent_history'][1:], 'actual older2 shift')
    for k in ('goal', 'projection', 'action_mean', 'action_std'):
        need(canonical(task[k]) == canonical(old[k]), 'original frozen '+k)
    arithmetic.vector(task['action_mean'], 2)
    need(all(x > 0 for x in arithmetic.vector(task['action_std'], 2)), 'positive normalizer')
    for k in ('root_sha256', 'blind_sha256', 'model_sha256', 'goal_sha256', 'projection_sha256',
              'normalizer_sha256', 'action_domain_sha256'):
        need(task['binding'][k] == old['binding'][k], 'original binding '+k)
    need(task['binding']['task_id'] != old['binding']['task_id'], 'distinct actual task')


def component(row, goal):
    need(row['status'] == 'complete', 'complete native item')
    context(row['postcontext'])
    arithmetic.vector(row['signal']['nominal'], 192)
    need(len(row['signal']['jacobian']) == 192, 'full192x10 J')
    for v in row['signal']['jacobian']:
        arithmetic.vector(v, 10)
    return arithmetic.scalar(row, goal)  # Deliberate shared arithmetic, NOT second independent math.


def decompose(group, goal):
    need(group['status'] in ('no_witness', 'complete') and group['control']['index'] == 'zero', 'complete zero+8 group')
    need([x['index'] for x in group['rows']] == list(range(8)), 'all eight in original order')
    zero = component(group['control'], goal)
    items = [dict(index='zero', role='control', scalar=zero, delta={k: 0. for k in zero},
                  nongreedy=None, better_reach=None, eligible=None, rejection_factors=[], gate_applies=False)]
    for row in group['rows']:
        s = component(row, goal)
        delta = {k: s[k]-zero[k] for k in s}
        arithmetic.close(delta['reach'], delta['immediate']+delta['nominal_drift']-delta['bounded_linear_gain'])
        # Never use recomputed/toleranced values for the original strict qualification.
        nongreedy = row['immediate'] > group['control']['immediate']
        reach = row['reach'] < group['control']['reach']
        need(row['eligible'] is (nongreedy and reach), 'saved exact gate')
        factors = ([] if nongreedy else ['immediate_not_strictly_above_zero'])
        factors += [] if reach else ['reach_not_strictly_below_zero']
        items.append(dict(index=row['index'], role='candidate', scalar=s, delta=delta,
                          nongreedy=nongreedy, better_reach=reach, eligible=row['eligible'],
                          rejection_factors=factors, gate_applies=True,
                          gain_improved=delta['bounded_linear_gain'] > 0,
                          drift_increased=delta['nominal_drift'] > 0,
                          saved_immediate=row['immediate'], saved_reach=row['reach']))
    candidates = items[1:]
    eligible = [x['index'] for x in candidates if x['eligible']]
    need(group['eligible_indices'] == eligible and (bool(eligible) or group['status'] == 'no_witness'), 'entire eligible list/status')
    return dict(items=items, eligible_indices=eligible,
                counts={k: sum(x[k] for x in candidates) for k in ('nongreedy', 'better_reach', 'eligible')},
                rejection_counts={k: sum(k in x['rejection_factors'] for x in candidates)
                                  for k in ('immediate_not_strictly_above_zero', 'reach_not_strictly_below_zero')})


def explain(p, report, receipt):
    for doc, suffix in ((p, 'prepare'), (report, 'report'), (receipt, 'consume')):
        need(doc['schema'] == PRODUCER+'.'+suffix, 'actual own schema '+suffix)
        false_flags(doc)
    need(report['status'] == 'four_slots_settled_not_admitted' and report['errors'] == []
         and receipt['status'] == 'actual_cp5_R_model_deliveries_verified_NOT_physical'
         and receipt['denominator'] == 4, 'complete actual saved status')
    need(report['prepared'] == receipt['prepared'] == str(ROOT/'results'/AUTH['prepared'][0])
         and report['prepare_sha256'] == receipt['prepare_sha256'] == AUTH['prepared'][1]
         and receipt['report_sha256'] == AUTH['report'][1], 'three saved authority joins')
    cp5ref = dict(path=str(CP5), sha256=CP5_SHA)
    need(p['cp5_tasks'] == receipt['cp5_tasks'] == cp5ref and receipt['protocol_sha256'] == PROTOCOL_SHA
         and receipt['root_context_role'] == ROLE and p['rules']['root_context_role'] == ROLE
         and p['rules']['actual_cp5_start'] is True, 'actual history protocol authority')
    need(report['sources_verified'] is True and report['source_manifest_sha256'] == digest(p['sources']), 'full source manifest join')
    need(report['model_before'] == report['model_after']
         and report['model_before']['original_state_dict'] == p['expected_model_hash'], 'frozen model')
    need(report['classes_before'] == report['classes_after'] and report['classes_before'], 'installed classes')
    for c in report['classes_before']:
        need(p['sources'].get(c['path']) == c['sha256'], 'installed class source')
    roster = [dict(slot=i, root=r, horizon=h) for i, (r, h) in enumerate(((0, 1), (0, 3), (1, 1), (1, 3)))]
    need(p['roster'] == roster and len(p['tasks']) == len(p['provenance']) == 2
         and len(report['rows']) == len(receipt['arms']) == 4, 'all fixed four tasks/slots')
    groups = []
    for identity, row, arm in zip(roster, report['rows'], receipt['arms'], strict=True):
        need(all(type(row[k]) is int and row[k] == v and arm[k] == v for k, v in identity.items()), 'ordered slot identity')
        root = row['root']; task, pro = p['tasks'][root], p['provenance'][root]
        task_pair(task, pro, root)
        r = row['result']; false_flags(r); false_flags(arm)
        need(r['schema'] == PRODUCER+'.group' and r['backend_kind'] == 'HOST_CUDA'
             and r['root_context_role'] == ROLE and r['actual_cp5_start'] is True
             and r['horizon'] == row['horizon'] and r['errors'] == [], 'actual settled group')
        need(r['task_sha256'] == digest(task) and r['provenance_sha256'] == digest(pro)
             and r['source_closure_sha256'] == digest(p['sources']), 'full group identity closure')
        need(len(r['R_groups']) == 1 and arm['complete'] is True
             and arm['resources'] == r['resource_receipt'], 'complete paid group and receipt')
        out = decompose(r['R_groups'][0], task['goal'])
        order = sorted(out['eligible_indices'], key=lambda i: (r['R_groups'][0]['rows'][i]['immediate'], i))
        need(r['eligible_order'] == arm['eligible_order'] == order, 'all eligible original modelQ order')
        state = 'model_delivery_verified' if order else 'no_witness'
        need(row['status'] == r['status'] == arm['status'] == state
             and r['model_delivery_verified'] is bool(order) and arm['model_delivery_verified'] is bool(order)
             and arm['no_witness'] is (not bool(order)) and arm['delivery'] == r['delivery'], 'real witness state')
        need(bool(order) or r['delivery'] is None, 'no witness no fallback delivery')
        pointer = '/rows/'+str(identity['slot'])+'/result/R_groups/0'
        for item in out['items']:
            item['report_origin_pointer'] = pointer+('/control' if item['role'] == 'control' else '/rows/'+str(item['index']))
        groups.append(dict(**identity, status=state, task=deepcopy(task), provenance=deepcopy(pro),
                           task_sha256=digest(task), provenance_sha256=digest(pro),
                           source_closure_sha256=r['source_closure_sha256'],
                           resources=deepcopy(arm['resources']), **out))
    return dict(schema=NAME+'.analysis', groups=groups, groups_count=4, items_count=36, candidates_count=32,
                arithmetic_reuse='MAIN scalar reused; NOT a second independent mathematical implementation',
                exact_gate='saved immediate > zero.immediate AND saved reach < zero.reach; no tolerance',
                scope='fixed known2 actual-cp5 bank only; no all-space impossibility or control-benefit claim',
                source_closure=deepcopy(p['sources']), upstream_fullstrict_rebuilt_here=False,
                new_neural_calls=0, new_physics_calls=0, **dict.fromkeys(FLAGS, False))


def check_sources(sources):
    need(type(sources) is dict and sources, 'nonempty complete source closure')
    for name, expected in sources.items():
        need(type(expected) is str and len(expected) == 64 and all(c in '0123456789abcdef' for c in expected), 'source SHA')
        path = Path(name)
        need(path.is_absolute() and path.is_file() and sha(path) == expected, 'current complete source bytes '+name)


def analyze():
    for path, expected in PINS.items():
        need(sha(path) == expected, 'frozen component source '+str(path))
    docs = {k: read_pinned(ROOT/'results'/name, h) for k, (name, h) in AUTH.items()}
    p = docs['prepared']
    for path, h in PINS.items():
        if path != Path(arithmetic.__file__).resolve():
            need(p['sources'].get(str(path)) == h, 'producer closure required pin')
    need(p['sources'].get(str(CP5)) == CP5_SHA, 'cp5 source closure')
    check_sources(p['sources'])
    cp5 = read_pinned(CP5, CP5_SHA)
    need(canonical(cp5['tasks']) == canonical(p['tasks'])
         and canonical(cp5['provenance']) == canonical(p['provenance']), 'exact saved actual history tasks')
    result = explain(p, docs['report'], docs['receipt'])
    result['authorities'] = {k: dict(path=str(ROOT/'results'/n), sha256=h) for k, (n, h) in AUTH.items()}
    result['own_sources'] = {str(path): sha(path) for path in (
        Path(__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md'), *PINS)}
    # Close read/check races without replaying the upstream consumer or any model.
    check_sources(p['sources'])
    check_sources(result['own_sources'])
    for k, (name, h) in AUTH.items():
        need(sha(ROOT/'results'/name) == h, 'unchanged authority '+k)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('analyze', 'verify'))
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args(argv)
    path = a.output.absolute()
    need(path.parent == ROOT/'results' and path.parent.resolve() == path.parent
         and path.name.startswith(NAME+'_') and not path.is_symlink(), 'exclusive result path')
    result = analyze()
    raw = arithmetic.encode(result)
    need(len(raw) <= CAP, 'analysis output cap')
    if a.command == 'verify':
        need(path.is_file() and path.stat().st_size <= CAP and path.read_bytes() == raw, 'entire deterministic saved result')
    else:
        with path.open('xb') as f:
            need(f.write(raw) == len(raw), 'complete output write')
            f.flush(); os.fsync(f.fileno())
    print(json.dumps(dict(verified=True, path=str(path), sha256=sha(path))))


if __name__ == '__main__':
    main()
