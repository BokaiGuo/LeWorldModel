"""Saved-only descriptive budget responses; no driver imports or model execution."""
from __future__ import annotations
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np

NAME = 'agent_p2_wallcapped_budget_response_20261029_v1'
RUNTIME = 'agent_p2_wallcapped_cuda_20261029_v1'
ACCOUNT = 'agent_p2_compute_matched_protocol_20261029_v2.accounting'
ROOT = Path(__file__).resolve().parents[2]
DECLARATION = ROOT/'innerdocs/agent_p2_wallcapped_budget_response_analysis_20261029_v1.md'
DECLARATION_SHA = '31f2c6da78a86626b3ee1358868dea81d48291b218e66927d97705e533b7edf1'
RUNTIME_SHA = '5e733e75f56b3dd2f5598d6e9c4ae7878c1e3dd0d613c5b4777c1ecd891b9776'
ORDER = ((0, 'joint'), (0, 'composition'), (1, 'composition'), (1, 'joint'))
FIELDS = ('predict','action_encoder','predictor','pred_proj','jacobian','mathematical_VJP',
          'jacobian_elements','AD_rollout_steps','root_SVD','root_ridge','encoder','projector','physics','training')
CAPS = dict(zip(FIELDS, (1902,1902,1902,1902,85,16320,489600,255,2,1,0,0,0,0)))
WALL = 450.


def require(ok, message):
    if not ok: raise ValueError(message)


def raw(value):
    # Producer codec: compact, sorted JSON WITH its final newline.
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()+b'\n'


def digest(value): return hashlib.sha256(raw(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2), b''): h.update(b)
    return h.hexdigest()


def equal(a, b, message): require(raw(a) == raw(b), message)


def read(path, cap):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'regular non-symlink saved file')
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rb') as f: body = f.read(cap+1)
    require(len(body) <= cap, 'bounded saved JSON')
    return json.loads(body)


def finite(value, shape):
    a = np.asarray(value, dtype=np.float64)
    require(a.shape == shape and np.isfinite(a).all(), 'finite exact objective dimensions')
    return a


def objective(witness, goal):
    require(set(witness) == {'logits','contexts'} and len(witness['contexts']) == 3, 'complete full15 witness')
    u = finite(witness['logits'], (30,))
    require(np.linalg.norm(u) <= float(.75*np.sqrt(15)), 'original full30 ball')
    for c in witness['contexts']:
        finite(c['latent_history'], (3,192)); finite(c['action_history'], (2,10))
    z = finite(witness['contexts'][-1]['latent_history'][-1], (192,))
    g = finite(goal, (192,))
    return float(np.mean((z-g)**2))


def journal_commits(output):
    """Authenticate all envelopes; retain only scalar-analysis witness commands."""
    previous = '0'*64; total = 0; commits = [[] for _ in ORDER]; sealed = []
    diagnostics = [dict(arm_exceptions=[],ledger_failures=[]) for _ in ORDER]
    active = 0; pending = None; started = False; starts = {}
    with gzip.open(Path(output)/'events.jsonl.gz', 'rb') as f:
        for i, line in enumerate(iter(lambda:f.readline(16*1024**2+1), b'')):
            total += len(line)
            require(total <= 1024**3 and len(line) <= 16*1024**2 and line.endswith(b'\n'), 'bounded complete journal line')
            e = json.loads(line)
            require(set(e) == {'index','previous_sha256','payload','event_sha256'} and type(e['index']) is int
                    and e['index'] == i and e['previous_sha256'] == previous, 'journal ordinal/hash chain')
            require(digest({k:v for k,v in e.items() if k != 'event_sha256'}) == e['event_sha256']
                    and raw(e) == line, 'canonical journal bytes and digest')
            previous = e['event_sha256']; p = e['payload']; kind = p['kind']
            require(kind in {'load_intent','consumed_buffer_frozen','arm_event','arm_sealed'}, 'exact outer event kind')
            if kind in {'load_intent','consumed_buffer_frozen'}:
                require(not started, 'loader before arms only'); continue
            started = True
            require(active < 4 and p['index'] == active, 'fixed contiguous journal arms')
            if kind == 'arm_sealed':
                require(pending is None and active in starts, 'closed commit and actual init before seal')
                sealed.append(p['row_sha256']); active += 1; continue
            q = p['payload']
            if q['kind'] == 'ledger_init':
                require(active not in starts and (q['root'],q['arm']) == ORDER[active], 'actual ledger start identity')
                starts[active] = scalar(q['now'])
            if q['kind'] == 'arm_exception': diagnostics[active]['arm_exceptions'].append(deepcopy(q))
            if q['kind'] == 'ledger_failure': diagnostics[active]['ledger_failures'].append(deepcopy(q))
            if pending is not None:
                require(q['kind'] in {'ledger_return','ledger_failure'} and q['method'] == 'commit', 'adjacent commit result')
                require(active in starts, 'commit after actual ledger start')
                commits[active].append(dict(command=pending, result=q,ledger_start=starts[active])); pending = None
            elif q['kind'] == 'ledger_intent' and q['method'] == 'commit': pending = q
    require(active == 4 and len(sealed) == 4 and pending is None, 'full four journal bookends')
    return commits, sealed, diagnostics


def scalar(value):
    require(type(value) in (int,float) and math.isfinite(value), 'finite scalar response')
    return value


def classify(value):
    return 'missing' if value is None else 'better' if value < 0 else 'worse' if value > 0 else 'tie'


def difference(comp, joint): return None if comp is None or joint is None else comp-joint


def analyze_row(index, slot, row, account, plan, commands, model_hash):
    root, arm = ORDER[index]
    require(row['index'] == index and row['root_slot'] == root and row['arm'] == arm, 'fixed actual row identity')
    equal(row['accounting'], account, 'raw accounting')
    require(account['schema'] == ACCOUNT and account['root'] == root and account['arm'] == arm
            and account['sealed'] is True, 'sealed original accounting identity')
    equal(account['binding'], slot, 'original complete task binding')
    equal(account['caps'], CAPS, 'original common resource caps')
    require(account['wall_cap_seconds'] == WALL, 'original wall cap')
    elapsed = scalar(account['elapsed_seconds']); require(elapsed >= 0, 'nonnegative elapsed')
    require(type(account['wall_overrun']) is bool and account['wall_overrun'] == (elapsed > WALL), 'actual overrun flag')
    resources = {}
    require(set(account['attempted']) == set(account['completed']) == set(FIELDS), 'full resource vector')
    for k in FIELDS:
        a, c = account['attempted'][k], account['completed'][k]
        require(type(a) is int and type(c) is int and 0 <= c <= a <= CAPS[k], 'original cumulative counts')
        resources[k] = dict(cap=CAPS[k],attempted=a,completed=c,failed_or_unreturned_requested=a-c,
                            utilization=None if CAPS[k] == 0 else a/CAPS[k],
                            utilization_status='not_applicable' if CAPS[k] == 0 else 'defined')
    goal = slot['packet']['goal']; baseline = None; incumbent = None; trajectory = []; failures = []
    # account.events are strict-replayed ledger events, not host absolute times.
    unit_times = [e['elapsed'] for e in account['events'] if e['kind'] == 'unit_start']
    require(not commands or unit_times, 'commit requires started unit')
    for item in commands:
        e, ret = item['command'], item['result']
        require(e['kwargs'] == {} and len(e['args']) == 3, 'original commit positional ABI')
        w, g, tag = e['args']; equal(g, goal, 'immutable original goal')
        cost = objective(w,g)
        if ret['kind'] == 'ledger_failure':
            failures.append(dict(tag=tag,Q=cost,error=ret['error'],host_now=e['now'])); continue
        # Match each successful command to its saved scalar commit, preserving all.
        j = len(trajectory); require(j < len(account['commits']), 'no unrecorded commit')
        saved = account['commits'][j]; t = scalar(saved['elapsed'])
        now = scalar(e['now'])
        # Subtraction, rather than now == start+t, preserves producer clock arithmetic.
        require(now-scalar(item['ledger_start']) == t and 0 <= t <= elapsed, 'exact commit relative clock')
        accepted = t <= WALL and (incumbent is None or cost < incumbent['cost'])
        expected = dict(tag=tag,cost=cost,accepted=accepted,late=t>WALL,elapsed=t)
        equal(saved, expected, 'independent all-commit Q/accepted/late')
        require(ret['result_sha256'] == digest(accepted), 'original commit boolean return')
        if tag == 'nominal':
            require(baseline is None, 'single original nominal'); baseline = cost
        if accepted:
            incumbent = dict(tag=tag,cost=cost,witness=deepcopy(w),
                             raw_float32_bits=np.tanh(np.asarray(w['logits'],np.float64)).astype(np.float32).tobytes().hex(),elapsed=t)
        trajectory.append(dict(expected,Q=cost,witness_sha256=digest(w),
                               gain_from_nominal=None if baseline is None else baseline-cost))
    require(len(trajectory) == len(account['commits']), 'all commits including ties/late retained')
    equal(account['incumbent'], incumbent, 'independently rebuilt capped incumbent')
    export = row['result']['export']
    require(type(account['export_complete']) is bool and account['export_missing'] == (not account['export_complete'])
            and account['export_complete'] == (export is not None) == (plan is not None), 'export missing preserved')
    export_q = None; export_match = None
    if export is not None:
        require(incumbent is not None, 'export requires incumbent')
        equal(export,incumbent['witness'],'export exact incumbent witness')
        export_q = objective(export,goal); equal(plan['contexts'],export['contexts'],'plan full3 witness')
        for key, val in dict(root_slot=root,arm=arm,task_id=slot['task_id'],task_sha256=slot['task_sha256'],
                             root_context=slot['packet']['root_context'],goal=goal,model_hash=model_hash,
                             model_Q=export_q,raw_float32_bits=incumbent['raw_float32_bits'],warm_raw10=slot['warm_raw10']).items():
            equal(plan[key],val,'strict export '+key)
        bits = np.asarray(plan['commands'],np.float32)
        require(bits.shape == (15,2) and bits.tobytes().hex() == incumbent['raw_float32_bits'], 'export raw bits')
        require(plan['selection_digest'] == digest({k:v for k,v in plan.items() if k != 'selection_digest'}), 'plan producer digest')
        export_match = True
    finished = account['finished_units']
    require(len(finished) == account['units_completed'], 'full finished unit count')
    def category(u):
        return 'seed' if u.startswith('seed:') else 'local' if u.startswith(('joint:','H2:','H1:')) else u
    phases = {k:sum(category(u)==k for u in finished) for k in ('nominal','bank','seed','local','replay')}
    iq = None if incumbent is None else incumbent['cost']
    return dict(serial_position=index,root=root,arm=arm,task_id=slot['task_id'],task_sha256=slot['task_sha256'],
        model_hash=model_hash,strict_status='wallcapped_records_verified_not_admitted',
        stop=account['stop'],errors=deepcopy(account['errors']),aborted_unit=row['result']['aborted_unit'],
        active_unit=account['active_unit'],pending=account['pending'],finished_units=finished,
        units_completed=account['units_completed'],phase_completion=phases,elapsed_seconds=elapsed,
        wall_cap_seconds=WALL,wall_overrun=elapsed>WALL,overrun_seconds=max(0.,elapsed-WALL),
        resources=resources,original_caps=deepcopy(CAPS),
        peak_allocated_bytes=account['peak_allocated_bytes'],peak_reserved_bytes=account['peak_reserved_bytes'],
        Q_nominal=baseline,Q_capped_incumbent=iq,model_Q_gain=difference(baseline,iq),
        incumbent_tag=None if incumbent is None else incumbent['tag'],
        incumbent_elapsed=None if incumbent is None else incumbent['elapsed'],
        commits=trajectory,commit_failures=failures,Q_export=export_q,export_exact_incumbent_match=export_match,
        export_complete=account['export_complete'],export_missing=account['export_missing'],plan_presence=plan is not None)


def paired(rows):
    pairs = []
    for root in (0,1):
        arms = {r['arm']:r for r in rows if r['root'] == root}
        require(set(arms) == {'joint','composition'}, 'fixed two arm pair')
        c,j = arms['composition'],arms['joint']; values = {}
        for key in ('elapsed_seconds','overrun_seconds','Q_capped_incumbent','Q_export','model_Q_gain'):
            delta = difference(c[key],j[key]); values[key] = dict(composition=c[key],joint=j[key],delta=delta)
            if key.startswith('Q_'): values[key]['ordering'] = classify(delta)
        vector = {kind:{k:c['resources'][k][kind]-j['resources'][k][kind] for k in FIELDS}
                  for kind in ('attempted','completed','failed_or_unreturned_requested')}
        pairs.append(dict(root=root,direction='composition - joint',values=values,resource_delta=vector,
            phase_completion={a:r['phase_completion'] for a,r in arms.items()},stops={a:r['stop'] for a,r in arms.items()},
            cap_deviation=c['wall_overrun'] or j['wall_overrun'],
            export_pair_available=c['plan_presence'] and j['plan_presence']))
    return pairs


def assemble(prepared, report, receipt, rows, commit_sets, arm_diagnostics=None):
    require(report['schema'] == RUNTIME+'.report' and report['status'] == 'recorded_not_admitted'
            and report['errors'] == [] and report['fixed_denominator'] == 4, 'analysis_not_admitted_integrity_unresolved: report')
    require(receipt['schema'] == RUNTIME+'.consume' and receipt['status'] == 'wallcapped_records_verified_not_admitted'
            and receipt['fixed_denominator'] == 4 and len(receipt['accounts']) == len(receipt['plans']) == len(rows) == len(commit_sets) == 4,
            'analysis_not_admitted_integrity_unresolved: full strict receipt')
    equal(prepared['root_arm_order'],[list(v) for v in ORDER],'fixed complete original order')
    require(report['model_unchanged'] is True and report['model_before'] == report['model_after'] == prepared['expected_model_hash'], 'frozen model identity')
    panels = []
    if arm_diagnostics is None: arm_diagnostics = [dict(arm_exceptions=[],ledger_failures=[]) for _ in ORDER]
    require(len(arm_diagnostics) == 4, 'all journal failure panels')
    for i,(root,arm) in enumerate(ORDER):
        equal(report['rows'][i],dict(index=i,root_slot=root,arm=arm,file=f'arm{i}.json.gz'),'all actual rows')
        summary = {k:v for k,v in rows[i]['accounting'].items() if k != 'events'}
        summary['ledger_events_sha256'] = digest(rows[i]['accounting']['events'])
        equal(receipt['accounts'][i],summary,'strict full accounting matches original row')
        panels.append(analyze_row(i,prepared['slots'][root],rows[i],rows[i]['accounting'],receipt['plans'][i],
                                  commit_sets[i],prepared['expected_model_hash']))
        panels[-1]['journal_failures'] = deepcopy(arm_diagnostics[i])
    equal(report['arm_stops'],[r['stop'] for r in panels],'producer stop sequence')
    pairs = paired(panels)
    counts = {key:{label:sum(p['values'][key]['ordering']==label for p in pairs)
                   for label in ('better','tie','worse','missing')} for key in ('Q_capped_incumbent','Q_export')}
    return dict(schema=NAME+'.analysis',status='strict_bound_descriptive_not_admitted',rows=panels,pairs=pairs,
        fixed_denominator=4,paired_denominator=2,known_roots=2,fresh_roots=0,
        export_pairs_available=sum(p['export_pair_available'] for p in pairs),export_pairs_fixed_denominator=2,
        cap_deviation_roots=sum(p['cap_deviation'] for p in pairs),paired_ordering_counts=counts,
        producer_status=report['status'],producer_errors=report['errors'],producer_wall_seconds=report['wall_seconds'],
        original_caps=deepcopy(CAPS),wall_cap_seconds=WALL,allocated_cap_bytes=4*1024**3,reserved_cap_bytes=6*1024**3,
        total_FLOPs=None,engine_internal_backward_calls=None,equal_FLOPs=False,equal_total_compute=False,
        new_neural_calls=0,new_physics_calls=0,P1_admission=False,P2_admission=False,P2_mechanism=False,
        physical_release=False,selection=False,significance=False)


def consume(output, report_sha, strict, strict_sha):
    output = Path(output).resolve(); strict = Path(strict).absolute(); report_path = output/'report.json'
    require(sha(report_path) == report_sha and sha(strict) == strict_sha, 'actual external report/strict SHA')
    r = read(report_path,4*1024**2); s = read(strict,128*1024**2)
    require(s.get('report_sha256') == report_sha, 'strict binds actual report')
    require(s.get('schema') == RUNTIME+'.consume' and s.get('status') == 'wallcapped_records_verified_not_admitted',
            'analysis_not_admitted_integrity_unresolved: strict status')
    for plan in s['plans']:
        if plan is not None:
            require(plan['schema'] == RUNTIME+'.plan' and plan['report_sha256'] == report_sha, 'actual export report binding')
    bindings = s['bindings']
    require(bindings.get(str(report_path)) == report_sha, 'strict report path binding')
    ppath = Path(r['prepared']).resolve()
    require(bindings.get(str(ppath)) == r['prepare_sha256'], 'strict prepare path/SHA binding')
    runtime_path = ROOT/'experiments/spectral_reachability'/f'{RUNTIME}.py'
    require(bindings.get(str(runtime_path)) == RUNTIME_SHA and sha(runtime_path) == RUNTIME_SHA, 'frozen runtime source binding')
    require(sha(DECLARATION) == DECLARATION_SHA, 'frozen result-blind declaration')
    for path,h in bindings.items(): require(sha(path) == h, 'strict bound file unchanged: '+path)
    p = read(ppath,4*1024**2)
    require(sha(ppath) == r['prepare_sha256'] and p['schema'] == RUNTIME+'.prepare', 'actual original prepare')
    equal(p['resource_caps_per_root_arm'],CAPS,'sealed original caps')
    require(p['wall_cap_seconds_per_root_arm'] == WALL, 'sealed original wall')
    names = set(r['artifact_sha256'])
    require({f.name for f in output.iterdir()} == names|{'report.json'}, 'exact original artifact census')
    for name,h in r['artifact_sha256'].items():
        require(Path(name).name == name and bindings.get(str(output/name)) == h and sha(output/name) == h, 'every artifact strict binding')
    commits,sealed,diagnostics = journal_commits(output)
    rows = [read(output/f'arm{i}.json.gz',512*1024**2) for i in range(4)]
    for row,h in zip(rows,sealed): require(digest(row) == h, 'sealed whole arm row')
    result = assemble(p,r,s,rows,commits,diagnostics)
    result['authority'] = dict(report=str(report_path),report_sha256=report_sha,strict=str(strict),strict_sha256=strict_sha,
        prepared=str(ppath),prepare_sha256=r['prepare_sha256'],runtime_source_sha256=RUNTIME_SHA,
        declaration=str(DECLARATION),declaration_sha256=DECLARATION_SHA,
        model_hash=p['expected_model_hash'],source_closure=deepcopy(bindings))
    result['own_files_sha256'] = {str(f):sha(f) for f in (Path(__file__).resolve(),ROOT/'tests'/f'test_{NAME}.py',ROOT/'innerdocs'/f'{NAME}.md')}
    require(sha(strict) == strict_sha and sha(report_path) == report_sha, 'authorities stable after analysis')
    for path,h in bindings.items(): require(sha(path) == h, 'bound files stable after analysis')
    return result


def write_new(path, value):
    path = Path(path)
    require(path.is_absolute() and path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink(), 'exclusive new derivative path')
    b = raw(value); require(len(b) <= 16*1024**2, 'analysis16MiB cap')
    with path.open('xb') as f: require(f.write(b) == len(b), 'complete derivative'); f.flush(); os.fsync(f.fileno())
    return dict(path=str(path),sha256=sha(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=('analyze','verify'))
    for key in ('output','report-sha','strict','strict-sha'): ap.add_argument('--'+key,required=True)
    ap.add_argument('--analysis'); ap.add_argument('--analysis-sha'); a = ap.parse_args()
    result = consume(a.output,a.report_sha,a.strict,a.strict_sha)
    if a.command == 'analyze': summary = write_new(a.analysis,result)
    else:
        require(sha(a.analysis) == a.analysis_sha,'external derivative SHA')
        equal(read(a.analysis,16*1024**2),result,'full deterministic saved analysis rebuild')
        summary = dict(verified=True,sha256=a.analysis_sha)
    print(json.dumps(summary,sort_keys=True,allow_nan=False))


if __name__ == '__main__': main()
