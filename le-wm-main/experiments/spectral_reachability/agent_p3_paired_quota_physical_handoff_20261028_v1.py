"""Paired-quota physical handoff, MAIN native execution, full saved consumer.

No neural loader. Exact old evidence reuse precedes outcomes. New physics runs
only under explicit MAIN authorization, never by prepare/verify/consume.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from experiments.spectral_reachability import agent_p3_paired_quota_runtime_20261028_v1 as runtime
from experiments.spectral_reachability import agent_p3_budget_physical_handoff_20261028_v1 as bridge

NAME='agent_p3_paired_quota_physical_handoff_20261028_v1'
ROOT,core,contract=runtime.ROOT,runtime.core,runtime.contract
saved,physical=bridge.saved,bridge.physical
old,compiler=physical.old,physical.compiler
require,sha,digest,canonical=runtime.require,runtime.sha,runtime.digest,runtime.canonical
READ=runtime.base.read
Store,Meter,execute=physical.Store,physical.Meter,physical.execute
new_row,ledger,png=physical.new_row,physical.ledger,physical.png
same_frame,raw,metric,exact=physical.same_frame,physical.raw,physical.metric,physical.exact
AUTH='MAIN_APPROVED_PAIRED_QUOTA_NATIVE_PHYSICS'
RUNTIME_REPORT=ROOT/'results/agent_p3_paired_quota_runtime_20261028_v1_cuda_MAIN01/report.json'
RUNTIME_REPORT_SHA='c8030ebc8c48f74688f4c2ab732c0b2adb184b49a30a4e5ab4bdce5f0bbfbf83'
FLAGS=dict(runtime.FLAGS,**{k:v for k,v in bridge.FLAGS.items() if k not in runtime.FLAGS})
PINS=dict(bridge.PINS)
PINS.update(dict(zip(map(str,runtime.own_files()),(
    '8251bad724159486e4603813fa058d2c8c6e012338743e7757be86c59a3972c1',
    '2caae1fec2ccc833a91dd670743c8169b95c185904dfda8d18744a116139f5b8',
    '7ebece1d7bd4cdc36c16467af0bf3d50f83adf8f50fdd69ac0fdf45c79f315ac',
    'dd3aae29275fb4094243980297971b895b1baad71987d93ca27fe156d7100be1',
    '0dbf4b7db95b3d1f99e0b6ba00f6828a74006755aa8d032d6bf93d181180c5c1',
    'a15eca3a458cd1ac288675738afc74c6212ffa6758af0a4a1130636f88cb49fd'),strict=True)))
PINS.update(dict(zip(map(str,bridge.own_files()),(
    '6a325fc5a24204db553c11dffd211ef298b2d54566ed8ae5a7848b2761e48c5c',
    'f353da1df1fd2bb0099ac60095b6c208fe48ac72d55159db2060fe712bc04025',
    '56ad626df0367275b93acb53aedd51743d37dc4d14b2add3aee9a79e95ccdc0e'),strict=True)))

def own_files():
    return [Path(__file__).resolve(),ROOT/f'tests/{NAME}_test.py',ROOT/f'innerdocs/{NAME}.md']

def inputs(report,report_sha,receipt,receipt_sha):
    for h in (report_sha,receipt_sha):core.hash_value(h)
    return dict(runtime_report=str(contract.absolute(report)),report_sha256=report_sha,
        strict_receipt=str(contract.absolute(receipt)),receipt_sha256=receipt_sha)

def load_runtime(a):
    core.keys(a,('runtime_report','report_sha256','strict_receipt','receipt_sha256'))
    require(a==inputs(a['runtime_report'],a['report_sha256'],a['strict_receipt'],a['receipt_sha256']),'canonical authority')
    require(a['runtime_report']==str(RUNTIME_REPORT) and a['report_sha256']==RUNTIME_REPORT_SHA,'fixed actual paired report, no cohort replacement')
    report=contract.absolute(a['runtime_report']);receipt=contract.absolute(a['strict_receipt'])
    require(report.name=='report.json' and report.is_file() and receipt.is_file() and
        not report.is_symlink() and not receipt.is_symlink(),'actual terminal report/full strict required; no future SHA')
    require(sha(report)==a['report_sha256'] and sha(receipt)==a['receipt_sha256'],'external actual runtime/strict SHA BEFORE consumers')
    contract.check_sources(PINS)
    # Full producer fullarray/gate/chooser consumer, not summary fields or prepare.
    actual=runtime.consume(str(report.parent),a['report_sha256'])
    stored=READ(receipt,256*1024**2)
    require(canonical(stored)==canonical(actual),'complete actual paired runtime.consume exact persisted receipt')
    require(actual['schema']==runtime.NAME+'.consume' and actual['status']=='paired_predict_quota_fullarrays_verified_not_admitted' and
        actual['matched_predict_API_samples'] is True and all(actual[k] is False for k in runtime.FLAGS),'full paired strict with closed scientific gates')
    r=READ(report,256*1024**2);path=contract.absolute(r['prepared'])
    require(sha(path)==r['prepare_sha256'],'runtime real prepare binding')
    p=READ(path,64*1024**2);contract.check_sources(p['sources'])
    require(all(p['sources'].get(str(f))==PINS[str(f)] for f in runtime.own_files()),'pinned reviewed runtime six-file closure')
    require(len(actual['full_rows'])==48 and len(p['wiring']['children'])==24 and
        actual['full_rows']==r['session']['rows'],'complete paired fixed24/adaptive24 exact actual rows')
    require(sha(report)==a['report_sha256'] and sha(receipt)==a['receipt_sha256'],'runtime and full strict unchanged')
    return p,r,actual

def load_physical():
    # Frozen helper fully invokes original verify/P4 closure AND repaired native
    # consumer; no prefiltered success subset, no physical execution.
    return bridge.load_physical('strict_saved')

def raw_identity(value,count):
    a=physical.raw(value,count)
    header=dict(dtype=a.dtype.str,shape=list(a.shape))
    return dict(**header,bytes_hex=a.tobytes().hex(),sha256=hashlib.sha256(canonical(header)+b'\0'+a.tobytes()).hexdigest())


def start_identity(plan,manifest_sha):
    child=plan['child'];frames=plan['source_frames']
    require(child['status']=='observed_child' and frames and len(frames)==12+child['boundary'],
            'whole original warmup/prefix native restoration frames')
    for f in frames:physical.native_frame(f)
    require(frames[-1]['file_sha256']==child['observation']['png_sha256'] and
        frames[-1]['pixel_sha256']==child['observation']['pixel_sha256'],'same native PNG to encoded child context')
    return dict(physical_source_report_sha256=child['ancestry']['source_report_sha256'],
        source_manifest_sha256=manifest_sha,child=deepcopy(child),
        prefix_raw=raw_identity(child['executed_prefix'],child['boundary']),
        history_raw=raw_identity(child['raw_history10'],10),
        # Includes ALL snapshots/body float64 bits, goal, pose, image hashes and
        # source action ordinals, not merely seven wrapped state numbers.
        whole_source_frames=deepcopy(frames),source_returns=deepcopy(plan['source_returns']),
        context_sha256=child['context_sha256'],task_binding_sha256=child['ancestry']['task_binding_sha256'],
        goal_sha256=digest(frames[-1]['goal']),goal_rgb_sha256=frames[-1]['goal_rgb_sha256'])


def new_budget(rows):
    pending=[r for r in rows if r['route']=='pending_new_physics'];n=len(pending)
    prefix=sum(r['runner_plan']['child']['boundary'] for r in pending);public=15*n+prefix
    return dict(execute_slots=n,recovery_public_cap=10*n+prefix,own_public_cap=5*n,
        caps=dict(constructor=n,close=n,reset=2*n,set_state=4*n,goal_render=2*n,goal_setter=3*n,
            goal_bind=n,public_step=public,physics_step=4*n+10*public,explicit_render=17*n+prefix),
        native_earlystop_no_padding=True,nested_resources_not_added=True,new_actual_resources=None,
        historical993public_is_not_new_payment=True)


def project(p,r,oldp,oldr,oldstrict,reuse_policy,manifest_sha):
    require(len(r['session']['rows'])==48 and len(p['wiring']['children'])==24,'complete paired24/48 before projection')
    require(len(oldp['rows'])==len(oldr['rows'])==48,'original native physical48 denominator')
    require(reuse_policy!='strict_saved' or (oldstrict is not None and oldstrict['complete'] is True and
        oldstrict['rows']==oldr['rows']),'strict old consumer required BEFORE choosing any reuse')
    require(reuse_policy=='strict_saved','mandatory full old consumer, no force-rerun or outcome fallback')
    rows=[]
    for i,hostrow in enumerate(r['session']['rows']):
        child=p['wiring']['children'][i//2];mode=contract.MODES[i%2];template=oldp['rows'][i]
        contract.validate_child(child,p['wiring']['tasks'])
        require((hostrow['child_id'],hostrow['mode'])==(child['child_id'],mode),'same paired child/method ordinal')
        require(canonical(template['child'])==canonical(child) and template['mode']==mode and
            child['ancestry']['source_report_sha256']==physical.P4_SHA,'exact original same-slot native ancestry/context, never observed4')
        plan=deepcopy(template);plan['child']=deepcopy(child);plan['selected']=None
        row=dict(ordinal=i,child_id=child['child_id'],mode=mode,route='not_reached',
            paid_status=hostrow['status'],decision=deepcopy(hostrow['decision']),
            execution_quota=deepcopy(hostrow['execution_quota']),pair_seal_sha256=hostrow['pair_seal_sha256'],runner_plan=plan,
            start_sha256=None,command=None,goal_sha256=None,selection_digest=None,reuse_evidence=None,
            freshness=False,new_actual_resources=None)
        if child['status']=='not_reached':
            require(hostrow['status']=='not_reached','original unknown pair retained');plan['route']='not_reached'
        else:
            identity=start_identity(template,manifest_sha);row['start_sha256']=digest(identity)
            row['goal_sha256']=identity['goal_sha256']
            if hostrow['status']!='compiled':
                require(hostrow['status']=='paired_paid_refusal' and hostrow['compiler'] is None,'paid refused cannot issue raw5');row['route']='paid_refused'
                plan['route']='paid_refused'
            else:
                require(hostrow['compiler']['status']==runtime.compiler.FINAL and hostrow['compiler']['error'] is None and
                    hostrow['compiler']['selected']['parent_context_sha256']==child['context_sha256'] and
                    hostrow['compiler']['branch_sha256']==digest(runtime.pair.branch_for(child,mode,hostrow)),
                    'complete actual same-child selected prefix with separate original decision/quota')
                selected=hostrow['compiler']['selected'];plan['selected']=deepcopy(selected)
                row.update(route='pending_new_physics',command=raw_identity(selected['raw5'],5),
                    selection_digest=selected['selection_digest']);plan['route']='execute'
                # Decide ONLY on exact source/start/actions, before reading any
                # old reward/success/earlystop field. No cross-slot trajectory search.
                same_command=template['selected'] is not None and row['command']==raw_identity(template['selected']['raw5'],5)
                if reuse_policy=='strict_saved' and same_command:
                    historical=oldstrict['rows'][i]
                    require(historical['ordinal']==i and historical['plan_sha256']==digest(template) and
                        historical['status'] in ('returned5','native_stopped') and historical['error'] is None,
                        'strict complete same-slot old branch')
                    physical.same_frame(historical['boundary'],template['source_frames'][-1])
                    require(digest(start_identity(plan,manifest_sha))==row['start_sha256'],
                            'whole source/fullstate bits/goal/context equality before reuse')
                    row['route']='reused_saved';plan['route']='reused_saved'
                    row['reuse_evidence']=dict(report_sha256=saved.ACTUAL_REPORT_SHA,prepare_sha256=saved.ACTUAL_PREPARE_SHA,
                        original_ordinal=i,original_plan_sha256=digest(template),full_old_row=deepcopy(historical),
                        start_sha256=row['start_sha256'],command_sha256=row['command']['sha256'],
                        strict_consumer=saved.NAME,fresh_execution=False,
                        identity_only_selection=True,new_physics_calls=0)
        rows.append(row)
    census={k:sum(row['route']==k for row in rows) for k in ('pending_new_physics','reused_saved','paid_refused','not_reached')}
    require(sum(census.values())==48,'all48 handoff routes without selection/padding')
    return dict(rows=rows,census=census,new_budget=new_budget(rows),fixed_denominator=48,child_denominator=24,
        independent_roots=2,new_neural_calls=0,physical_execution_started=False,
        runner_entry=dict(module=physical.NAME,function='execute',schema=NAME+'.runner_plan',
            accepts='runner_plan of pending_new_physics slots only; original refs/source_plan from source_manifest',
            old_CLI_prepare_compatible=False,requires='MAIN reviewed authorized runner and saved consumer for this NEW handoff schema'),
        historical_resources=None if oldstrict is None else dict(resources=oldstrict['resources'],
            scope='old full panel including repeated trajectories, NOT new fees or independent confirmations'),**FLAGS)


def validate_saved(p, report, events, output):
    closed_journal(p,report,events)
    require(report['profile'] == p['profile'] and report['fixed_denominator'] == 48 and report['child_denominator'] == 24 and
            len(report['rows']) == len(p['rows']) == 48, 'whole physical denominator/profile')
    require(all(report[k] is False for k in FLAGS), 'scientific gates remain false')
    counts, pending = ledger(events, p['profile']['caps'])
    require(counts == report['resources'] and pending == report['pending'], 'independent full ledger reconciliation')
    complete = report['status'] == 'recorded' and not report['errors'] and not pending and not report['secondary_errors']
    require(type(report['wall_seconds']) in (int, float) and np.isfinite(report['wall_seconds']) and
            report['wall_seconds'] >= 0 and (not complete or report['wall_seconds'] < 600), 'actual execution wall scope')
    allframes = [e for e in events if e['kind'] == 'fresh_observation']
    require(len({e['receipt']['path'] for e in allframes}) == len(allframes), 'no duplicate/borrowed physical PNG')
    for e in allframes:
        require(e['receipt']['observed_event_index'] == e['index'], 'actual observation event ordinal')
        png(e['receipt'], output)
    for i, (plan, row) in enumerate(zip(p['rows'], report['rows'], strict=True)):
        expected = new_row(i, plan)
        for k in ('ordinal', 'episode_id', 'plan_sha256', 'route'):
            require(row[k] == expected[k], 'fixed physical plan/order identity')
        es = [e for e in events if e.get('episode_id') == row['episode_id']]
        if plan['route'] != 'execute':
            require(row == expected and not es, 'refused/not reached retained unknown, never executed'); continue
        fs = [e['receipt'] for e in es if e['kind'] == 'fresh_observation']
        require(len(fs) == row['PNG'], 'every physical RGB accounted')
        if not complete and row['status'] not in ('returned5', 'native_stopped'): continue
        starts = [e for e in es if e['kind'] == 'branch_started']; finishes = [e for e in es if e['kind'] == 'branch_finished']
        require(len(starts) == len(finishes) == 1 and starts[0]['ordinal'] == i and
                starts[0]['plan_sha256'] == digest(plan) and finishes[0]['row'] == row, 'durable exact branch start/terminal report row')
        require(row['error'] is None and not row['secondary_errors'] and row['initial_exact'] is True and
                row['warmup_returned'] == 10 and row['prefix_returned'] == plan['child']['boundary'], 'complete original recovery')
        b = plan['child']['boundary']; n = row['own_returned']
        require(type(n) is int and 1 <= n <= 5 and len(fs) == 12+b+n, 'native raw5 earlystop no padding')
        phases = [('warmup', j) for j in range(11)] + [('plan', 0)] + [('prefix', j) for j in range(1, b+1)] + [('own', j) for j in range(1, n+1)]
        require([(f['phase'], f['step']) for f in fs] == phases, 'full physical observation time order')
        for f, source in zip(fs[:12+b], plan['source_frames'], strict=True): same_frame(f, source)
        seals = [e for e in es if e['kind'] == 'child_boundary_sealed']
        require(len(seals) == 1 and seals[0]['boundary'] == row['boundary'] == fs[11+b] and
                seals[0]['context_sha256'] == plan['child']['context_sha256'] and
                seals[0]['raw_history10'] == plan['child']['raw_history10'] and
                seals[0]['selected_sha256'] == digest(plan['selected']), 'same actual native child and raw/context seal')
        ds = [e for e in es if e['kind'] == 'authorized_dispatch']; rs = [e for e in es if e['kind'] == 'step_return']
        commands = np.concatenate([raw(plan['child']['executed_prefix'], b), raw(plan['selected']['raw5'], 5)[:n]])
        require(len(ds) == len(rs) == b+n and row['executed_raw'] == commands.tolist(), 'exact prefix+selected raw5 actual denominator')
        exact(raw(row['executed_raw'],b+n),commands,'returned raw bits including signed zero')
        for j, (d, r, command) in enumerate(zip(ds, rs, commands, strict=True)):
            phase, stepnum = ('prefix', j+1) if j < b else ('own', j-b+1)
            require((d['phase'], d['step']) == (r['phase'], r['step']) == (phase, stepnum) and
                    d['index'] < r['index'] < fs[12+j]['observed_event_index'], 'dispatch -> return -> current observed state')
            calls = [e for e in es if e['kind'] == 'operation_attempted' and e['resource'] == 'public_step' and
                     d['index'] < e['index'] < r['index']]
            require(len(calls) == 1, 'one literal authorized public API call per selected/prefix ordinal')
            returned = next(e for e in es if e['kind'] == 'operation_completed' and e['operation_id'] == calls[0]['operation_id'])
            require(calls[0]['region'] == ('prefix' if phase == 'prefix' else 'suffix') and returned['index'] < r['index'] and
                    sum(e['kind'] == 'operation_attempted' and e.get('resource') == 'physics_step' and
                        calls[0]['index'] < e['index'] < returned['index'] for e in es) == 10, 'ten actual nested native integrations per public return')
            if phase == 'own': require(fs[11+b]['observed_event_index'] < seals[0]['index'] < d['index'], 'whole native boundary sealed BEFORE selected raw5 dispatch')
            exact(raw([d['command']], 1), command[None], 'literal dispatched selected/raw prefix')
            exact(raw([r['command']], 1), command[None], 'literal returned selected/raw prefix')
            f = fs[12+j]; require(r['state'] == f['state'], 'actual returned native state')
            m = metric(f['goal'], f['state'])
            require(r['reward'] == m['native7_reward'] and r['terminated'] == m['native_success'] and
                    type(r['truncated']) is bool, 'native jointXY+angle termination, native7 reward separately')
            if j < b+n-1: require(not r['terminated'] and not r['truncated'], 'no dispatch after native stop')
        goal = plan['source_frames'][-1]
        for f in fs[12+b:]:
            require(all(canonical(f[k]) == canonical(goal[k]) for k in ('goal', 'goal_rgb_sha256', 'goal_pose')), 'own task goal/context not replaced')
        last = rs[-1]
        require(row['last_observation'] == fs[-1] and row['final_metrics'] == metric(fs[-1]['goal'], fs[-1]['state']) and
                row['terminated'] == last['terminated'] and row['truncated'] == last['truncated'] and
                (row['status'] == 'native_stopped') == bool(last['terminated'] or last['truncated']) and
                (n == 5 or row['status'] == 'native_stopped'), 'native earlystop terminal receipt')
        # Every public call includes setup warmup; all actual returns stay on the branch ledger.
        pub = [e for e in es if e['kind'] == 'operation_completed' and e['resource'] == 'public_step']
        require(len(pub) == 10+b+n, 'warmup+executed prefix+own raw5 full actual public cost')
        want = dict(constructor=1, reset=2, set_state=4, goal_render=2, goal_setter=3, goal_bind=1,
                    public_step=10+b+n, physics_step=4+10*(10+b+n), explicit_render=12+b+n)
        for resource, count in want.items():
            for kind in ('operation_attempted', 'operation_completed'):
                require(sum(e['kind'] == kind and e['resource'] == resource for e in es if 'resource' in e) == count,
                        'exact real native branch/reset/nested resource census: '+resource)
        require(sum(e['kind'] == 'cleanup_attempted' for e in es) == sum(e['kind'] == 'cleanup_completed' for e in es) == 1,
                'actual branch cleanup')
        warm = [e for e in es if e['kind'] == 'warmup_return']
        require([e['step'] for e in warm] == list(range(1, 11)) and all(not e['terminated'] and not e['truncated'] for e in warm),
                'all nonzero warmup returns, no reset/padding after stop')
        for e, command in zip(warm, old.kernel.moving.warmup_commands(), strict=True):
            exact(raw([e['command']], 1), command[None], 'original literal nonzero warmup history')
    require(not complete or all((r['status'] in ('returned5','native_stopped') if plan['route']=='execute' else
        r==new_row(i,plan)) for i,(plan,r) in enumerate(zip(p['rows'],report['rows'],strict=True))),
            'complete cannot promote any engineering failure/unstarted branch')
    return dict(schema=NAME+'.consume', status='native_children_verified_not_admitted' if complete else 'partial_preserved_not_verified',
                fixed_denominator=48, child_denominator=24, routes={route:sum(plan['route']==route for plan in p['rows'])
                    for route in ('execute','reused_saved','paid_refused','not_reached')},
                independent_roots=2, rows=report['rows'], resources=counts, complete=complete,
                restoration_cost_included=True, new_neural_calls=0, **FLAGS)
def _execute_verified(p, prepared, expected, output, authorization):
    """Internal implementation; CLI must pass run authorization. RAM tests inject no native backend."""
    require(authorization==AUTH,'MAIN authorization at native runner entry')
    started = time.monotonic()
    m = old.read(p['source_manifest'])
    data = old.kernel.load_source(m['sourcepack'], old.SOURCE_SHA, m['source_prepared'], m['source_prepare_sha256'])
    store = Store(output); meter = Meter(store, p['profile']['caps'])
    meter.job_check = lambda: require(time.monotonic()-started < 600, 'execute entry600s deadline')
    rows = [new_row(i, r) for i, r in enumerate(p['rows'])]; errors = []
    store.put('binding.json', dict(prepared=str(prepared), prepare_sha256=expected))
    store.emit('run_started', profile=p['profile'])
    for plan, row in zip(p['rows'], rows, strict=True):
        if plan['route'] != 'execute' or errors: continue
        env = None; meter.episode = row['episode_id']; meter.region = 'setup'
        store.emit('branch_started', episode_id=row['episode_id'], ordinal=row['ordinal'], plan_sha256=row['plan_sha256'])
        try:
            env = meter.call('constructor', lambda: old.native_factory(meter))
            execute(env, plan, data['references'][plan['child']['root_slot']], data['plan'], row, meter, store)
        except Exception as exc:
            row['status'] = 'engineering_failed'; row['error'] = dict(type=type(exc).__name__, message=str(exc))
            errors.append(dict(ordinal=row['ordinal'], **row['error']))
        finally:
            if env is not None:
                # Cleanup must still happen after exhausted wall/output. Count honestly without re-dispatching physics.
                meter.counts['close']['attempted'] += 1
                if not store.broken:
                    try: store.emit('cleanup_attempted', episode_id=row['episode_id'])
                    except Exception as exc: errors.append(dict(stage='cleanup_journal', message=str(exc)))
                try:
                    env.close(); meter.counts['close']['completed'] += 1
                    if not store.broken:
                        try: store.emit('cleanup_completed', episode_id=row['episode_id'])
                        except Exception as exc: errors.append(dict(stage='cleanup_journal', message=str(exc)))
                except Exception as exc:
                    row['secondary_errors'].append(dict(stage='close', message=str(exc)))
                    errors.append(dict(ordinal=row['ordinal'], stage='close', message=str(exc)))
            if not store.broken:
                try: store.emit('branch_finished', episode_id=row['episode_id'], row=row, cleanup_counts=meter.counts['close'])
                except Exception as exc: errors.append(dict(stage='journal', message=str(exc)))
    if time.monotonic()-started >= 600: errors.append(dict(stage='wall', message='execution600s exceeded'))
    try: compiler.contract.check_sources(p['sources'])
    except Exception as exc: errors.append(dict(stage='final_sources', message=str(exc)))
    report = dict(schema=NAME+'.report', prepared=str(prepared), prepare_sha256=expected, rows=rows,
                  resources=meter.counts, pending=list(meter.pending.values()), secondary_errors=meter.secondary,
                  errors=errors, fixed_denominator=48, child_denominator=24, profile=p['profile'],
                  wall_seconds=time.monotonic()-started, journal_sha256=sha(store.path/'journal.jsonl'),
                  status='recorded' if not errors else 'engineering_failed', **FLAGS)
    store.write('report.json', canonical(report), final=True)
    return dict(output=str(store.path), report_sha256=sha(store.path/'report.json'), status=report['status'], errors=errors)


def profile(handoff):
    b=handoff['new_budget']
    return dict(caps=deepcopy(b['caps']),recovery_public_cap=b['recovery_public_cap'],
        own_public_cap=b['own_public_cap'],output_bytes=physical.CAP,final_reserve_bytes=physical.FINAL,
        journal_bytes=physical.JOURNAL_CAP,png_bytes=physical.PNG_CAP,disk_floor_bytes=physical.FLOOR,
        wall_seconds=600,wall_scope='execute entry after verify; through cleanup, before final report write',
        neural_calls=0,no_padding=True,no_retry=True,independent_roots=2,nested_resources_not_added=True,
        reset_physics_included=True,reused_physics_is_not_new_payment=True)


def preflight(a):
    core.keys(a,('runtime_report','report_sha256','strict_receipt','receipt_sha256'))
    require(a==inputs(a['runtime_report'],a['report_sha256'],a['strict_receipt'],a['receipt_sha256']),'literal authority')
    require(a['runtime_report']==str(RUNTIME_REPORT) and a['report_sha256']==RUNTIME_REPORT_SHA,'fixed actual paired report')
    for path,h in ((a['runtime_report'],a['report_sha256']),(a['strict_receipt'],a['receipt_sha256']),
        (saved.ACTUAL_OUTPUT/'report.json',saved.ACTUAL_REPORT_SHA),(physical.P4_OUTPUT/'report.json',physical.P4_SHA),
        (physical.P4_PREPARE,physical.P4_PREPARE_SHA)):
        path=contract.absolute(path)
        require(path.is_file() and not path.is_symlink() and sha(path)==h,'actual external saved boundary BEFORE consumers: '+str(path))
    contract.check_sources(PINS)


def material(a):
    preflight(a)
    p,r,strict=load_runtime(a)
    oldp,oldr,oldstrict=load_physical()
    manifest=contract.absolute(oldp['source_manifest']);manifest_sha=sha(manifest)
    require(p['sources'].get(str(manifest))==manifest_sha and
        p['sources'].get(str(physical.P4_OUTPUT/'report.json'))==physical.P4_SHA,
        'original task/seed/placement/warmup manifest shared with new paired source')
    handoff=project(p,r,oldp,oldr,oldstrict,'strict_saved',manifest_sha)
    handoff['runner_entry']=dict(module=NAME,function='run',physical_implementation=physical.NAME+'.execute',
        authorization=AUTH,only_pending_new_physics=True,new_neural_calls=0)
    sources=deepcopy(p['sources']);contract.merge(sources,oldp['sources']);contract.merge(sources,PINS)
    contract.merge(sources,{str(f):sha(f) for f in own_files()})
    extra=[Path(a['runtime_report']),Path(a['strict_receipt']),Path(r['prepared']),
        Path(a['runtime_report']).parent/'binding.json',Path(a['runtime_report']).parent/runtime.paid.JOURNAL,
        saved.ACTUAL_OUTPUT/'report.json',saved.ACTUAL_OUTPUT/'journal.jsonl',saved.ACTUAL_OUTPUT/'binding.json',
        Path(oldr['prepared']),manifest]
    extra.extend(f for f in saved.ACTUAL_OUTPUT.iterdir() if f.suffix=='.png')
    contract.merge(sources,{str(f):sha(f) for f in extra});contract.check_sources(sources)
    preflight(a)
    return dict(schema=NAME+'.prepare',authority=deepcopy(a),source_manifest=str(manifest),sources=sources,
        handoff=handoff,rows=[deepcopy(r['runner_plan']) for r in handoff['rows']],profile=profile(handoff),
        runtime_actual_resources=deepcopy(strict['actual_resources']),fixed_denominator=48,child_denominator=24,
        independent_roots=2,full_strict_recomputed=True,physical_execution_started=False,**FLAGS)


def verify(prepared,expected):
    path=contract.absolute(prepared);core.hash_value(expected)
    require(sha(path)==expected,'external actual new physical prepare SHA')
    p=READ(path,64*1024**2);contract.check_sources(p['sources'])
    require(canonical(p)==canonical(material(p['authority'])) and sha(path)==expected,
        'complete paired full strict/old native consumer/source/identity/routes rebuild')
    return p


def exclusive(path):
    path=contract.absolute(path)
    require(path.parent==ROOT/'results' and path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink(),
        'NEW exclusive own result path')
    return path


def run(prepared,expected,output,authorization=None):
    require(authorization==AUTH,'MAIN authorization BEFORE verify/native factory')
    output=exclusive(output);p=verify(prepared,expected)
    return _execute_verified(p,prepared,expected,output,authorization)


def closed_journal(p,r,events):
    allowed={'run_started','branch_started','operation_attempted','operation_completed','fresh_observation',
        'guard_passed','warmup_return','authorized_dispatch','step_return','child_boundary_sealed',
        'cleanup_attempted','cleanup_completed','branch_finished'}
    require(events and events[0]['kind']=='run_started' and events[0]['profile']==p['profile'] and
        sum(e['kind']=='run_started' for e in events)==1 and all(e['kind'] in allowed for e in events),
        'closed native journal and exact engineering scope')
    own={new_row(i,plan)['episode_id'] for i,plan in enumerate(p['rows']) if plan['route']=='execute'}
    require(all(e.get('episode_id') in own for e in events[1:]),'no execution records for reuse/refusal/missing or foreign child')
    if r['status']=='recorded' and not r['errors']:
        starts=[e for e in events if e['kind']=='branch_started'];ends=[e for e in events if e['kind']=='branch_finished']
        expected=[i for i,plan in enumerate(p['rows']) if plan['route']=='execute']
        require([e['ordinal'] for e in starts]==expected and len(ends)==len(starts),'all pending branches in fixed order')
        for j,(a,b) in enumerate(zip(starts,ends,strict=True)):
            require(a['episode_id']==b['episode_id'] and a['index']<b['index'] and
                (j==0 or ends[j-1]['index']<a['index']),'native serial lifecycle, no overlap or restart')
            es=[e for e in events if e.get('episode_id')==a['episode_id']]
            require(all(a['index']<=e['index']<=b['index'] for e in es),'every event belongs to open native branch')
            frames=[e for e in es if e['kind']=='fresh_observation'];guards=[e for e in es if e['kind']=='guard_passed']
            require(len(frames)==len(guards) and all(f['index']<g['index'] and
                (f['receipt']['phase'],f['receipt']['step'])==(g['phase'],g['step'])
                for f,g in zip(frames,guards,strict=True)),'every saved PNG has native guard in same phase/order')


def results(p,r):
    require(len(p['handoff']['rows'])==len(r['rows'])==48,'full logical48 result denominator')
    rows=[]
    for i,(h,actual) in enumerate(zip(p['handoff']['rows'],r['rows'],strict=True)):
        route=h['route'];result=None;fresh=False
        if route=='reused_saved':result=deepcopy(h['reuse_evidence']['full_old_row'])
        elif route=='pending_new_physics':
            require(actual['status'] in ('returned5','native_stopped'),'new branch requires complete native result')
            result=deepcopy(actual);fresh=True
        else:require(route in ('paid_refused','not_reached'),'known full roster route')
        rows.append(dict(ordinal=i,child_id=h['child_id'],mode=h['mode'],route=route,
            decision=deepcopy(h['decision']),execution_quota=deepcopy(h['execution_quota']),
            pair_seal_sha256=h['pair_seal_sha256'],start_sha256=h['start_sha256'],command=deepcopy(h['command']),
            goal_sha256=h['goal_sha256'],selection_digest=h['selection_digest'],
            physical_result=result,reuse_evidence=deepcopy(h['reuse_evidence']),fresh_execution=fresh,
            scientific_freshness=False,missing_reason=route if result is None else None))
    return rows


def consume(output,expected):
    output=contract.absolute(output);core.hash_value(expected)
    require(sha(output/'report.json')==expected,'external actual new physical report SHA')
    r=READ(output/'report.json',physical.FINAL)
    require(r['schema']==NAME+'.report','own paired physical schema, no old whitelist bypass')
    p=verify(r['prepared'],r['prepare_sha256'])
    require(READ(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),
        'new execution/prepare binding')
    require(sha(output/'journal.jsonl')==r['journal_sha256'],'new full native journal SHA')
    es=physical.journal(output/'journal.jsonl',physical.JOURNAL_CAP)
    expected_files={'report.json','binding.json','journal.jsonl'}|{Path(e['receipt']['path']).name for e in es if e['kind']=='fresh_observation'}
    require({f.name for f in output.iterdir()}==expected_files and all(f.is_file() and not f.is_symlink() for f in output.iterdir()) and
        sum(f.stat().st_size for f in output.iterdir())<=physical.CAP and (output/'report.json').stat().st_size<=physical.FINAL and
        sum(f.stat().st_size for f in output.iterdir() if f.name!='report.json')<=physical.CAP-physical.FINAL,
        'exact flat native artifact set and actual storage partitions')
    native=validate_saved(p,r,es,output)
    require(native['complete'] is True,'engineering failure/partial preserved, never strict-released')
    rows=results(p,r);contract.check_sources(p['sources']);contract.check_sources(PINS)
    require(sha(output/'report.json')==expected and sha(output/'journal.jsonl')==r['journal_sha256'],'final native artifact drift')
    return dict(schema=NAME+'.consume',status='paired_quota_native_or_exact_saved_verified_not_admitted',
        report_sha256=expected,prepare_sha256=r['prepare_sha256'],runtime_authority=p['authority'],
        fixed_denominator=48,child_denominator=24,independent_roots=2,census=p['handoff']['census'],rows=rows,
        new_physical_resources=native['resources'],historical_resources=p['handoff']['historical_resources'],
        runtime_actual_resources=p['runtime_actual_resources'],neural_calls=0,
        restoration_cost_included=True,full_old_consumer_required=True,complete=True,**FLAGS)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('prepare')
    for flag in ('runtime-report','report-sha256','strict-receipt','strict-sha256','output'):a.add_argument('--'+flag,required=True)
    for command in ('verify','run'):
        a=sub.add_parser(command);a.add_argument('--prepared',required=True);a.add_argument('--prepare-sha256',required=True)
        if command=='run':a.add_argument('--output',required=True);a.add_argument('--authorization',required=True)
    a=sub.add_parser('consume');a.add_argument('--output',required=True);a.add_argument('--report-sha256',required=True);a.add_argument('--receipt',required=True)
    args=parser.parse_args(argv)
    if args.cmd=='prepare':
        out=exclusive(args.output)
        a=inputs(args.runtime_report,args.report_sha256,args.strict_receipt,args.strict_sha256)
        result=physical.write_new(out,material(a))
    elif args.cmd=='verify':
        p=verify(args.prepared,args.prepare_sha256);result=dict(status='strict_saved_handoff_verified_NOT_executed',census=p['handoff']['census'])
    elif args.cmd=='run':result=run(args.prepared,args.prepare_sha256,args.output,args.authorization)
    else:
        out=exclusive(args.receipt);result=physical.write_new(out,consume(args.output,args.report_sha256),physical.FINAL)
    print(json.dumps(result,sort_keys=True,allow_nan=False))
    return 1 if args.cmd=='run' and result['status']!='recorded' else 0


if __name__=='__main__':raise SystemExit(main())

