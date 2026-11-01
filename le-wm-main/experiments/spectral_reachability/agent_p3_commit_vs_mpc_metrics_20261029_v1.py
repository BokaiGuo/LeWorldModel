"""Independent native metrics on fixed four commit-H3 vs MPC/H1 comparisons.

Saved-only: external full receipts are inherited, never invoked or reissued.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import numpy as np

NAME = 'agent_p3_commit_vs_mpc_metrics_20261029_v1'
PHYSICAL = 'agent_p3_h3_witness_physical_20261029_v1'
SHARED = 'agent_p3_shared_panel_20261029_v1'
ROOT = Path(__file__).resolve().parents[2]
CP = (0, 5, 10, 15, 20)
ORDER = tuple((r, b) for r in (0, 1) for b in ('endpoints', 'midpoints'))
OLD_REPORT_SHA = '668db35d79eb650419e4efa42dfc21a15cdf91725700b2f202e2a2ec975bcd6a'
OLD_STRICT_SHA = 'd23ffa48bc954306bc27a15f924da8daafffc9b4bc76a91e59aa949ab1abbafd'
DOC_SHA = '13a1be54bf3e87885563bade7302c60a1745ce4715460abad47f949443fe3ba6'
METRICS = ('joint_xy', 'agent_xy', 'object_xy', 'angle_absolute', 'angle_wrapped',
           'native_angle_term', 'velocity_error', 'native7_distance', 'native7_reward')
CAP = 256 * 1024**2


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def raw(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def same(a, b, msg):
    require(raw(a) == raw(b), msg)


def sha(path):
    p = Path(path)
    require(p.is_file() and not p.is_symlink(), 'regular non-symlink file')
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def unique(items):
    out = {}
    for k, v in items:
        require(k not in out, 'duplicate JSON key')
        out[k] = v
    return out


def absolute(path):
    p = Path(path)
    require(p.is_absolute() and str(p.resolve()) == str(p), 'canonical absolute path')
    return p


def read(path, expected):
    p = absolute(path)
    require(p.stat().st_size <= CAP and sha(p) == expected, 'bounded exact external SHA')
    obj = json.loads(p.read_bytes(), object_pairs_hook=unique)
    raw(obj)
    return obj


def numeric(x, shape, dtype=np.float64):
    require(not any(type(v) is bool for v in np.asarray(x, dtype=object).flat), 'no bool coordinates')
    a = np.asarray(x, np.float64)
    require(a.shape == shape and np.isfinite(a).all(), 'finite exact shape')
    b = a.astype(dtype)
    require(np.array_equal(a, b.astype(np.float64)), 'lossless coordinates')
    if dtype == np.float32:
        require((np.abs(b) <= 1).all(), 'original raw box without clipping')
    return b


def action_bits(x, n):
    return numeric(x, (n, 2), np.float32).tobytes()


def metrics(goal, state):
    d = numeric(state, (7,)) - numeric(goal, (7,))
    a = float(abs(d[4])); native_angle = float(min(a, 2*np.pi-a))
    distance = float(np.linalg.norm(d)); xy = float(np.linalg.norm(d[:4]))
    return dict(joint_xy=xy, agent_xy=float(np.linalg.norm(d[:2])), object_xy=float(np.linalg.norm(d[2:4])),
        angle_absolute=a, angle_wrapped=float(abs(np.arctan2(np.sin(d[4]), np.cos(d[4])))),
        native_angle_term=native_angle, velocity_error=float(np.linalg.norm(d[5:])),
        native7_distance=distance, native7_reward=-distance,
        native_success=bool(xy < 20 and native_angle < np.pi/9), delta_native7=d.tolist())


def checkpoint_map(row, old=False):
    cps = row['checkpoints']
    if old:
        cps = [c for c in cps if c['phase'] == 'plan']
    require([c['step'] for c in cps] == sorted(set(c['step'] for c in cps)) and
        all(type(c['step']) is int and c['step'] in CP for c in cps), 'unique ordered fixed checkpoints')
    result = {c['step']:c for c in cps}
    if not old:
        require(row['missing_checkpoints'] == [b for b in CP if b not in result], 'explicit missing checkpoints')
    for c in cps:
        numeric(c['state'], (7,))
        require(numeric(c['snapshot']['state'], (7,)).tobytes() == numeric(c['state'], (7,)).tobytes(), 'snapshot state exact bits')
    return result


def control(a, b):
    if a is None or b is None:
        return dict(state_bits_equal=None, snapshot_equal=None, pixels_equal=None, exact=None, status='missing')
    state = numeric(a['state'], (7,)).tobytes() == numeric(b['state'], (7,)).tobytes()
    snapshot = raw(a['snapshot']) == raw(b['snapshot'])
    pixels = a['pixel_sha256'] == b['pixel_sha256']
    return dict(state_bits_equal=state, snapshot_equal=snapshot, pixels_equal=pixels,
        exact=state and snapshot and pixels, status='exact' if state and snapshot and pixels else 'engineering_control_mismatch')


def labels(row, plan, events, operations):
    n = row['own_returned']
    require(type(n) is int and 0 <= n <= 20 and row['status'] in
        ('not_started', 'engineering_failed', 'native_stopped', 'horizon_completed'), 'retained native row disposition')
    commands = numeric(plan['commands'], (20, 2), np.float32)
    returned = [e for e in events if e['kind'] == 'returned_transition' and e.get('branch') == row['index'] and e['phase'] == 'plan']
    require([e['step'] for e in returned] == list(range(1, n+1)), 'all actual returned plan actions')
    ops = [e for e in operations if e['resource'] == 'public_step' and e.get('branch') == row['index'] and e['phase'] == 'plan']
    require([o['step'] for o in ops] == list(range(1,len(ops)+1)) and len(ops) <= 20, 'contiguous actual attempts')
    require(sum(o['status'] == 'returned' for o in ops) == n, 'operation returned count')
    for j, o in enumerate(ops):
        require(o['raw_bits'] == commands[j].tobytes().hex(), 'all dispatched raw bits')
    for e in returned:
        require(e['actual_dtype'] == np.dtype(np.float32).str and
            e['actual_bits'] == e['requested_bits'] == commands[e['step']-1].tobytes().hex(), 'actual action signed bits')
        same(e['goal_state'], plan['native_goal'], 'actual returned original goal')
        calc = metrics(plan['native_goal'], e['state'])
        require(type(e['terminated']) is bool and type(e['truncated']) is bool and
            e['terminated'] is calc['native_success'] and e['reward'] == calc['native7_reward'], 'independent native labels')
    require(all(not e['terminated'] and not e['truncated'] for e in returned[:-1]), 'no post-native-stop padding')
    require(len(row['responses']) == n, 'all response labels')
    for e, response in zip(returned, row['responses'], strict=True):
        for k in ('step', 'state', 'reward', 'terminated', 'truncated'):
            same(response[k], e[k], 'response actual return join')
        require(response['native_success'] is e['terminated'], 'response success original label')
    if returned:
        same(row['final_state'], returned[-1]['state'], 'actual final state')
        require(row['terminated'] is returned[-1]['terminated'] and row['truncated'] is returned[-1]['truncated'], 'actual final flags')
    if row['status'] == 'horizon_completed':
        require(n == 20, 'full20 is actual20')
    if row['status'] == 'native_stopped' and n:
        require(returned[-1]['terminated'] or returned[-1]['truncated'], 'native stop not arbitrary flag')
    return returned


def pair_metrics(commit, reference, goal, kind, root, bank):
    records = []
    for b in CP:
        a, c = commit.get(b), reference.get(b)
        left = None if a is None else metrics(goal, a['state'])
        right = None if c is None else metrics(goal, c['state'])
        delta = None if left is None or right is None else {k:left[k]-right[k] for k in METRICS}
        records.append(dict(step=b, commit_metrics=left, reference_metrics=right, commit_minus_reference=delta,
            signs=None if delta is None else {k:'negative' if v<0 else 'positive' if v>0 else 'zero' for k,v in delta.items()},
            success=dict(commit=None if left is None else left['native_success'], reference=None if right is None else right['native_success']),
            commit_missing=a is None, reference_missing=c is None))
    return dict(root_slot=root, bank=bank, comparison=kind, direction='commit_minus_reference', checkpoints=records)


def summarize(p, r, receipt, old, old_receipt, events):
    require(p['schema'] == PHYSICAL+'.prepare' and r['schema'] == PHYSICAL+'.report' and receipt['schema'] == PHYSICAL+'.consume', 'own physical ABI')
    require(old['schema'] == SHARED+'.report' and old_receipt['schema'] == SHARED+'.consume' and old_receipt['complete'] is True, 'original complete shared-panel authority')
    require(old_receipt['report_sha256'] == OLD_REPORT_SHA and old_receipt['status'] == 'saved_shared_panel_verified_NOT_admitted', 'original exact authority')
    require(receipt['planner_authority']['report_sha256'] == OLD_REPORT_SHA and
        receipt['planner_authority']['receipt_sha256'] == OLD_STRICT_SHA, 'same original planner authority')
    same(receipt['planner_authority'], p['planner_authority'], 'prepare authority')
    same(r['execution_binding']['planner_authority'], p['planner_authority'], 'execution authority')
    same(p['contract'], r['contract'], 'actual own contract')
    same(receipt['rows'], r['rows'], 'whole strict-authenticated own rows')
    same(receipt['plans'], p['plans'], 'whole strict-authenticated own plans')
    for k in ('status', 'counts', 'errors'):
        same(receipt[k], r[k], 'strict report '+k)
    require(r['fixed_denominator'] == receipt['fixed_denominator'] == 4 and r['known_roots'] == receipt['known_roots'] == 2,
        'fixed four/two known roots')
    require(r['fresh_roots'] == receipt['fresh_roots'] == 0, 'not fresh')
    for obj in (r, receipt, old, old_receipt):
        require(all(obj[k] is False for k in ('P1_admission','P3_admission','C3','freshness','scientific_admission')), 'no science admission')
    require(len(r['rows']) == len(p['plans']) == 4 and len(old['rows']) == 8 and len(old['nodes']) == 24, 'all fixed denominators')
    require([(x['root_slot'], x['arm']) for x in r['rows']] == list(ORDER), 'four original order')
    old_rows = {x['episode_id']:x for x in old['rows']}
    require(len(old_rows) == 8, 'unique original episodes')
    rows = []; primary = []; auxiliary = []
    for index, (row, plan, (root, bank)) in enumerate(zip(r['rows'], p['plans'], ORDER, strict=True)):
        require(plan is not None and row['index'] == plan['index'] == index and
            row['condition'] == plan['condition'] == 'off' and (plan['root_slot'],plan['arm']) == (root,bank), 'literal own plan slot')
        same(row['plan'], plan, 'whole row plan')
        for k in ('task_id','selection_digest'):
            same(row[k], plan[k], 'exact own task/selection')
        eid = f's{root}_adaptive_{bank}'; fixed_id = f's{root}_fixedH1_{bank}'
        mpc, fixed = old_rows[eid], old_rows[fixed_id]
        require((mpc['source_slot'],mpc['mode']) == (root,'adaptive_'+bank) and
            (fixed['source_slot'],fixed['mode']) == (root,'fixedH1_'+bank), 'matched original identities')
        nodes = [n for n in old['nodes'] if n['episode_id'] == eid and n['boundary'] == 5]
        require(len(nodes) == 1 and nodes[0]['receipt']['selected_horizon'] == 3, 'original selected b5 H3')
        node = nodes[0]; winner = node['receipt']['panels'][1]['winner']
        same(plan['source_witness']['winner'], winner, 'original whole H3 winner, not selected anew')
        same(plan['source_witness']['request'], node['request'], 'original whole observed request')
        commands = numeric(plan['commands'], (20,2), np.float32)
        original = numeric(mpc['executed_raw'], (mpc['control_returned'],2), np.float32)
        require(len(original) >= 10 and commands[:10].tobytes() == original[:10].tobytes(), 'original prefix and chosen FIRST5 bits')
        require(commands[5:].tobytes() == action_bits(winner['actions'],15), 'all original H3 suffix bits')
        require(commands[:5].tobytes() == action_bits(plan['prefix_commands'],5) and
            commands[5:].tobytes() == action_bits(plan['suffix_commands'],15), 'plan prefix/suffix literal bits')
        cm, mm, fm = checkpoint_map(row), checkpoint_map(mpc,True), checkpoint_map(fixed,True)
        goal = plan['native_goal'];numeric(goal,(7,))
        require(all(b in mm and b in fm for b in CP), 'original full20 checkpoints')
        for ref in (mm, fm):
            for c in ref.values():
                same(c['goal'], goal, 'same actual native goal across paired arms')
        returned = labels(row,plan,events,r['actual_operation_receipts'])
        n = row['own_returned']
        require(all(b <= n for b in cm) and (row['initial_exact'] or not cm), 'no synthetic future checkpoints')
        if row['status'] in ('horizon_completed','native_stopped'):
            require(list(cm) == [b for b in CP if row['initial_exact'] and b <= n], 'no reached checkpoint dropped')
        for b,c in cm.items():
            if b:
                same(c['state'],returned[b-1]['state'],'checkpoint exact returned state')
        controls = {str(b):control(cm.get(b),mm.get(b)) for b in (0,5,10)}
        fixed_controls = {str(b):control(cm.get(b),fm.get(b)) for b in (0,5)}
        valid = all(c['exact'] is True for c in controls.values()) and all(c['exact'] is True for c in fixed_controls.values())
        complete_action = n == 20
        terminal = None if not returned else dict(step=n,metrics=metrics(goal,returned[-1]['state']),terminated=returned[-1]['terminated'],truncated=returned[-1]['truncated'])
        rows.append(dict(index=index,root_slot=root,bank=bank,status=row['status'],own_returned=n,missing_checkpoints=deepcopy(row['missing_checkpoints']),
            exact_controls_vs_MPC=controls,exact_controls_vs_fixedH1=fixed_controls,effect_interpretable=valid,
            control_failure_not_scientific_negative=any(c['exact'] is False for c in list(controls.values())+list(fixed_controls.values())),
            complete_same_action_H3_physical_path=valid and complete_action,latent_fidelity_available=False,
            actions=dict(all_returned_bits_match_plan=True,first10_bits_match_original_MPC=True,
                first_original_MPC_difference=next((j+1 for j in range(20) if j<len(original) and commands[j].tobytes()!=original[j].tobytes()),None),
                missing_selected_actions=20-n),eventtime_terminal=terminal,
            at20=None if 20 not in cm else metrics(goal,cm[20]['state']),
            errors=dict(first=row['first_error'],secondary=deepcopy(row['secondary_errors']))))
        a=pair_metrics(cm,mm,goal,'primary_commit_vs_adaptive_MPC',root,bank)
        a['effect_interpretable']=valid;primary.append(a)
        a=pair_metrics(cm,fm,goal,'auxiliary_commit_vs_fixedH1',root,bank)
        a['effect_interpretable']=valid;auxiliary.append(a)
    return dict(schema=NAME+'.summary',fixed_denominator=4,primary_pair_denominator=4,auxiliary_pair_denominator=4,
        checkpoint_denominator_per_comparison=20,known_roots=2,fresh_roots=0,rows=rows,primary_pairs=primary,auxiliary_pairs=auxiliary,
        report_status=r['status'],errors=deepcopy(r['errors']),native_resource_counts=deepcopy(r['counts']),
        P1_admission=False,P3_admission=False,C3=False,freshness=False,scientific_admission=False,
        latent_metrics=None,new_neural_calls=0,new_physics_calls=0,full_strict_rebuilt_here=False,producer_math_imported=False,
        interpretation='fixed descriptive native axes only; missing is null; controls failure is engineering, not mechanism negative; no CI/equalFLOPs')


def analyze(report, report_sha, receipt, receipt_sha, original_report, original_strict):
    args=dict(report=report,report_sha=report_sha,receipt=receipt,receipt_sha=receipt_sha,original_report=original_report,original_strict=original_strict)
    r,s = read(report,report_sha),read(receipt,receipt_sha)
    require(s['physical_report_sha256'] == report_sha, 'actual physical report bound by external receipt')
    binding = r['execution_binding'];pp=binding['prepared'];ph=binding['prepare_sha256'];p=read(pp,ph)
    journal=absolute(report).parent/'journal.jsonl'
    require(journal.stat().st_size <= CAP and sha(journal) == r['journal_sha256'], 'actual original journal bytes')
    for path,h in ((report,report_sha),(pp,ph),(str(journal),r['journal_sha256'])):
        require(s['bindings'].get(path) == h, 'full receipt actual file binding')
    old,old_s=read(original_report,OLD_REPORT_SHA),read(original_strict,OLD_STRICT_SHA)
    events=[]
    with journal.open('rb') as f:
        for line in f:
            require(line.endswith(b'\n') and len(line) <= CAP, 'complete journal lines')
            e=json.loads(line,object_pairs_hook=unique);raw(e);events.append(e)
    out=summarize(p,r,s,old,old_s,events)
    doc=ROOT/'innerdocs/agent_p3_witness_interpretation_20261029_v1.md'
    require(sha(doc) == DOC_SHA, 'frozen interpretation rules')
    files=[Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    bindings={report:report_sha,receipt:receipt_sha,pp:ph,str(journal):r['journal_sha256'],original_report:OLD_REPORT_SHA,original_strict:OLD_STRICT_SHA,str(doc):DOC_SHA}
    bindings.update({str(f):sha(f) for f in files})
    out.update(arguments=args,bindings=bindings,external_authority=dict(physical_receipt_sha256=receipt_sha,physical_report_sha256=report_sha,
        planner_report_sha256=OLD_REPORT_SHA,planner_receipt_sha256=OLD_STRICT_SHA),upstream_bindings=deepcopy(s['bindings']))
    for path,h in bindings.items():require(sha(path)==h,'saved inputs/source unchanged')
    return out


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('analyze','verify'))
    for k in ('report','report-sha','receipt','receipt-sha','output','summary','summary-sha'):
        ap.add_argument('--'+k)
    ap.add_argument('--original-report',default=str(ROOT/'results/agent_p3_shared_panel_20261029_v1_MAIN01/report.json'))
    ap.add_argument('--original-strict',default=str(ROOT/'results/agent_p3_shared_panel_20261029_v1_consume_MAIN01.json'))
    a=ap.parse_args(argv)
    if a.command=='verify':
        saved=read(a.summary,a.summary_sha);same(saved,analyze(**saved['arguments']),'deterministic saved reconstruction')
        result=dict(verified=True,sha256=a.summary_sha)
    else:
        result=analyze(**{k:getattr(a,k) for k in ('report','report_sha','receipt','receipt_sha','original_report','original_strict')})
        dest=absolute(a.output);body=raw(result)
        require(dest.name.startswith(NAME+'_') and len(body)<=CAP,'exclusive bounded own output')
        with dest.open('xb') as f:
            require(f.write(body)==len(body),'whole output');f.flush();os.fsync(f.fileno())
        result=dict(path=str(dest),sha256=sha(dest))
    print(raw(result).decode())


if __name__=='__main__':
    main()
