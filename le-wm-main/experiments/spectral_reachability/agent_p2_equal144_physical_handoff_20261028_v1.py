"""Four equal144 selected plans, exact moving-native replay; MAIN physics only.

Saved prepare/verify/consume do not load a model or create an environment.
Four fresh executions; no old endpoint/result substitution or outcome selection.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import moving_prediction_matched_sampling_physical_20261027_v1 as primitives
from experiments.spectral_reachability import agent_p2_equal144_baseline_cuda_20261028_v1 as producer
from experiments.spectral_reachability import main_p2_equal144_saved_20261028_v1 as saved

NAME='agent_p2_equal144_physical_handoff_20261028_v1'
AUTH='MAIN_EQUAL144_PHYSICS_ONLY'
ROOT,kernel,helper,source=primitives.ROOT,primitives.kernel,primitives.helper,primitives.source
require,sha,raw,merge,absolute,hash_literal=primitives.require,primitives.sha,primitives.raw,primitives.merge,primitives.absolute,primitives.hash_literal
ARMS=list(producer.ARMS)
CAP,FLOOR,FINAL=primitives.CAP,primitives.FLOOR,primitives.FINAL
PREPARE_CAP=2*1024**2
SOURCE_SHA=primitives.SOURCE_SHA
REPORT=ROOT/'results/agent_p2_equal144_baseline_cuda_20261028_v1_cuda_MAIN01/report.json'
REPORT_SHA='7eeaa059d58895545340458455ddcaabceda4a813a41051e1a2c60f22bc10228'
RECEIPT=ROOT/'results/main_p2_equal144_saved_20261028_v1_MAIN01.json'
RECEIPT_SHA='d8ded78d152938a3c6e5c244729f120d040c42666b44d9ed89bf2e0beca029dc'
SEALED={
 str(Path(primitives.__file__).resolve()):'50552d50ce3d97c9a25f938ef2b47e99bc66c88fa1644659611b5bb32843884a',
 str(Path(producer.__file__).resolve()):'f592a485d657c849f6a5036b8e485a0287cb234f00476cf5f9ca262bb5e5a841',
 str(Path(saved.__file__).resolve()):'3de410d8ce86df7e3ee88652280fe1fbefdef9ff03aab3c991e1eee33bab5070'}
SOURCE_INPUTS=dict(
 sourcepack=str(ROOT/'results/p1_constructive_moving_history_source_20261027_v1_main01/sourcepack.json'),
 sourcepack_sha256=SOURCE_SHA,
 source_prepared=str(ROOT/'results/p1_constructive_moving_history_source_20261027_v1_prepare01.json'),
 source_prepare_sha256='c1292f0ebfb1827e5da92b32ab13d2e4a12c025c4741277ab4c4f478d6ee8cb9')
Store,Ops,close_env=primitives.Store,primitives.Ops,primitives.close_env


def read(path,cap=PREPARE_CAP):
    return primitives.read(path,cap)


def bits(commands):
    a=kernel.commands(commands)
    require(a.dtype==np.float32,'native float32 commands')
    return a.tobytes().hex()


def ownconsume():
    for path,h in SEALED.items(): require(sha(path)==h,'frozen executable source')
    require(sha(REPORT)==REPORT_SHA and sha(RECEIPT)==RECEIPT_SHA,'actual equal144 report and MAIN persisted strict SHA')
    persisted=read(RECEIPT,CAP)
    exact=saved.analyze()  # full producer.consume, saved NumPy only
    require(raw(persisted)==raw(exact),'entire MAIN wrapper exact return, not envelope as native receipt')
    c=persisted['native_strict_receipt']
    require(c['schema']==producer.NAME+'.consume' and c['status']=='new_equal144_all_four_verified'
        and c['errors']==[] and c['predict_API']==c['predictor_samples']==576
        and c['AD_forward']==4 and c['J']==2 and c['VJP']==384 and c['SVD']==4 and c['bounded_solve']==2
        and c['P1_admission'] is c['P2_admission'] is c['equal_FLOPs'] is False,'full actual equal144 native receipt')
    r=read(REPORT,CAP); p=producer.verify(r['prepared'],r['prepare_sha256'])
    rows=deepcopy(c['plans'])
    require([(x['source_slot'],x['arm']) for x in rows]==[(s,a) for s in (0,1) for a in ARMS],'all four original slot/arm plans')
    bindings={};merge(bindings,p['bindings_sha256']);merge(bindings,SEALED)
    merge(bindings,{str(REPORT):REPORT_SHA,str(RECEIPT):RECEIPT_SHA,r['prepared']:r['prepare_sha256']})
    for name,h in r['artifact_sha256'].items():
        require(Path(name).name==name and name not in ('','.','..'),'flat producer artifact')
        merge(bindings,{str(REPORT.parent/name):h})
    for row in rows:
        slot=p['slots'][row['source_slot']]
        require(type(row['source_slot']) is int and row['schema']==producer.NAME+'.plan'
            and row['whole_run_report_sha256']==REPORT_SHA and row['with_target'] is False
            and all(row[k]==slot[k] for k in ('source_slot','task_id','blind_sha256','task_export','task_sha256')),'exact task/source/goal/context identity')
        bare={k:v for k,v in row.items() if k not in ('selection_sha256','whole_run_report_sha256')}
        require(producer.core.digest(bare)==row['selection_sha256'],'original selected plan digest')
        a=np.asarray(row['commands'])
        require(np.array_equal(a,a.astype(np.float32).astype(a.dtype)),'raw list exactly represents original float32')
        bits(row['commands'])
    for path,h in bindings.items():require(sha(path)==h,'full producer saved closure')
    return rows,bindings,c


def task_bridge(row,ref,bindings):
    # helper without a schema checks exact RGB, goal, nonzero incoming history,
    # identity192 and original packet cap; actual raw provenance is established
    # by equal144 full native consume, not a new NumPy tanh reconstruction.
    bridge={k:v for k,v in row.items() if k!='schema'}
    helper.task_bridge(bridge,ref,bindings)
    from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as moving_encoder
    value=moving_encoder.load_task(row['task_export'],row['task_sha256'])
    return value['gn_task']['task_contract_sha256']


def contract(wall=600):
    c=primitives.contract(wall)
    c.update(arms=list(ARMS),prepare_cap_bytes=PREPARE_CAP,
        selection_scope='actual equal144 four winners; two KNOWN moving calibration tasks x beam/spectral',
        matched_claim='inherited per-group144 predict API AND batch1 predictor samples; not FLOPs/AD/wall',
        inherited_prediction_cost='576 predict API/samples; AE/projector overlap; ADforward4 included, J2 VJP384 SVD4 solve2 separate',
        result_reuse=False,old_endpoint_substitution=False,scientific_admission=False,P2_admission=False)
    return c


def material():
    require(sha(helper.__file__)==primitives.HELPER_SHA,'sealed physical helper')
    for f,h in helper.SEALED_HELPERS.items():require(sha(f)==h,'sealed lower physical helper')
    data=kernel.load_source(SOURCE_INPUTS['sourcepack'],SOURCE_INPUTS['sourcepack_sha256'],
        SOURCE_INPUTS['source_prepared'],SOURCE_INPUTS['source_prepare_sha256'])
    rows,bindings,receipt=ownconsume(); plans=[]
    require(len(data['references'])==2,'exact two source roots')
    for row in rows:
        s=row['source_slot'];ref=data['references'][s]
        tc=task_bridge(row,ref,bindings)
        plan=dict(root_slot=s,root_sha256=ref['root_sha256'],arm=row['arm'],commands=kernel.commands(row['commands']).tolist(),
            raw_float32_bits=bits(row['commands']),task_id=row['task_id'],blind_sha256=row['blind_sha256'],
            task_export=row['task_export'],task_sha256=row['task_sha256'],task_contract_sha256=tc,
            compiler_authority=producer.NAME,compiler_report=str(REPORT),compiler_report_sha256=REPORT_SHA,
            compiler_selection_sha256=row['selection_sha256'],own_plan_sha256=source.digest(row))
        plan['selection_digest']=source.digest(plan);plans.append(plan)
    merge(bindings,data['bindings'])
    native,h=helper.native_constructor_contract();merge(bindings,{native:h})
    for f in primitives.physical_files()+[Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]:
        merge(bindings,{str(f.resolve()):sha(f)})
    for f,h in bindings.items():require(sha(f)==h,'final source/model/task seals')
    return dict(schema=NAME+'.prepare',source_inputs=deepcopy(SOURCE_INPUTS),plans=plans,bindings=bindings,
        contract=contract(),sourcepack_sha256=SOURCE_SHA,source_prepare_id=source.digest(data['plan']),
        planner_report_sha256=REPORT_SHA,planner_receipt_sha256=RECEIPT_SHA,
        inherited_resources={k:deepcopy(receipt[k]) for k in
            ('predict_API','predictor_samples','AD_forward','J','VJP','SVD','bounded_solve','inherited_encoding_once')})


def prepare(output):
    p=material();payload=raw(p);path=Path(output).absolute()
    absolute(str(path))
    require(len(payload)<=PREPARE_CAP and not path.exists() and not path.is_symlink()
        and shutil.disk_usage(path.parent).free>=FLOOR+len(payload),'exclusive2MiB prepare+floor')
    with path.open('xb') as f:require(f.write(payload)==len(payload),'short prepare');f.flush();os.fsync(f.fileno())
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(payload))


def verify(path,expected):
    path=Path(absolute(str(path)));hash_literal(expected)
    require(sha(path)==expected,'actual own prepare SHA')
    p=read(path)
    require(p==material() and sha(path)==expected,'entire actual source/context/raw closure')
    return p


def execute(p, data, store, factory, *, authorization, execution_binding=None):
    require(authorization == AUTH, 'MAIN physics authorization required')
    require(p['contract'] == contract(p['contract']['wall_seconds']) and len(data['references']) == 2 and
            [(x['root_slot'], x['arm']) for x in p['plans']] == [(s, a) for s in (0, 1) for a in ARMS], 'fixed four denominator/contract')
    for plan in p['plans']:
        require(type(plan['root_slot']) is int and plan['root_sha256'] == data['references'][plan['root_slot']]['root_sha256'] and
                plan['selection_digest'] == source.digest({k: v for k, v in plan.items() if k != 'selection_digest'}), 'sealed plan/root join')
        require(bits(plan['commands'])==plan['raw_float32_bits'],'exact selected raw float32 bits')
    rows = [dict(branch_index=i, root_slot=x['root_slot'], arm=x['arm'], selected_plan=deepcopy(x),
        selection_digest=x['selection_digest'], status='not_started', first_error=None, secondary_errors=[],
        warmup_returned=0, own_attempted=0, own_returned=0, initial_exact=False, full_horizon=False,
        native_success=None, terminated=None, truncated=None, initial_metrics=None, final_metrics=None,
        final_state=None, endpoint_scope='not_started', PNG=0, checkpoints=[], constructor_internal_work='not_attempted')
        for i, x in enumerate(p['plans'])]
    result = dict(schema=NAME+'.report', status='running', rows=rows, errors=[], contract=p['contract'],
        sourcepack_sha256=p['sourcepack_sha256'], source_prepare_id=p['source_prepare_id'], fixed_branch_denominator=4,
        prepared=None if execution_binding is None else execution_binding['prepared'],
        prepare_sha256=None if execution_binding is None else execution_binding['prepare_sha256'],
        new_neural_calls=0, training=False, new_independent_samples=0, P1_admission=False, P2_admission=False,
        P3_admission=False, scientific_admission=False, independent_confirmation=False, equal_total_compute=False,
        equal_FLOPs=False, compute_advantage=False, walltime_advantage=False)
    ops = Ops(p, store)
    def error(stage, exc): result['errors'].append(dict(stage=stage, type=type(exc).__name__, message=str(exc)))
    try:
        store.emit('run_started', prepare=p, execution_binding=execution_binding)
        for i, (plan, row) in enumerate(zip(p['plans'], rows, strict=True)):
            env = None; holder = []
            ops.where = dict(branch=i, root_slot=plan['root_slot'], arm=plan['arm'], selection_digest=plan['selection_digest'])
            def construct():
                row['constructor_internal_work'] = 'unknown'
                value = factory(); holder.append(value); return value
            try:
                env = ops.call('environment_construct', construct)
                kernel.moving.validate_env(env.unwrapped)
                body = env.unwrapped
                require(all(getattr(body, k, None) is None for k in ('space', 'render_buffer', 'latest_action')), 'pristine constructor BEFORE hooks')
                require(body.render_action is False and body.render_size == 224 and body.render_mode == 'rgb_array', 'target-free RGB224/no marker')
                row.update(status='running', constructor_internal_work='sealed_native_pristine_no_physics')
                with kernel.native_hooks(env, ops, row):
                    kernel.branch(env, data['references'][plan['root_slot']], data['plan'], plan, row, ops, store, i)
            except BaseException as exc: kernel.retain(row, exc, 'setup_or_branch')
            finally:
                if env is None and holder: env = holder[0]
                if env is not None: close_env(env, row, ops, store)
                # Preserve native entry/return even when AFTER-return durable logging fails.
                steps = [r for r in ops.receipts if r.get('branch') == i and r['resource'] == 'public_step']
                row['own_attempted'] = sum(r.get('phase') == 'plan' for r in steps)
                row['own_returned'] = sum(r.get('phase') == 'plan' and r['status'] == 'returned' for r in steps)
                row['warmup_returned'] = sum(r.get('phase') == 'warmup' and r['status'] == 'returned' for r in steps)
                row['full_horizon'] = row['own_returned'] == 10
                if row['status'] == 'failed': row['endpoint_scope'] = 'failed_partial'
                store.emit('branch_closed', receipt=row)
    except BaseException as exc: error('execution', exc)
    try:
        for f, h in p['bindings'].items(): require(sha(f) == h, 'final binding: '+f)
    except BaseException as exc: error('final_bindings', exc)
    clean = not result['errors'] and all(r['status'] in ('horizon_completed', 'native_success', 'native_stopped') for r in rows)
    if clean:
        try:
            c = ops.counts; steps = sum(r['warmup_returned']+r['own_returned'] for r in rows)
            require(all(v['attempted'] == v['completed'] for v in c.values()) and
                c['environment_construct']['completed'] == c['environment_close']['completed'] == 4 and
                c['reset']['completed'] == 8 and c['set_state']['completed'] == 16 and
                c['goal_render']['completed'] == 8 and c['goal_bind']['completed'] == 4 and
                c['public_step']['completed'] == steps and c['physics_step']['completed'] == 16+10*steps and
                c['explicit_render']['completed'] == sum(r['PNG'] for r in rows), 'actual full/earlystop census, never pad partial')
        except BaseException as exc: error('final_census', exc); clean = False
    result.update(status='equal144_physical_recorded_not_admitted' if clean else 'partial_or_failed',
        counts=ops.counts, actual_operation_receipts=ops.receipts,
        full_horizons=sum(r['full_horizon'] for r in rows), native_successes=sum(r['status'] == 'native_success' for r in rows),
        engineering_failures=sum(r['status'] == 'failed' for r in rows), not_started=sum(r['status'] == 'not_started' for r in rows),
        early_stops=sum(r['status'] in ('native_success', 'native_stopped') and not r['full_horizon'] for r in rows),
        wall_seconds=time.monotonic()-ops.started,
        total_internal_physics_scope='unknown_constructor_work' if any(r['constructor_internal_work'] == 'unknown' for r in rows)
            else 'hook_observed_native_work', resource_equality_required=False)
    try:
        store.emit('run_finished', status=result['status'], counts=ops.counts, errors=result['errors'])
        result.update(journal_sha256=sha(store.path/'journal.jsonl'), journal_chain_tail=store.previous)
    except BaseException as exc: error('final_journal', exc); result['status'] = 'partial_or_failed'
    try: store.put('report.json', result, final=True)
    except BaseException as exc: error('final_report', exc); result['status'] = 'partial_or_failed'
    return result


def run(prepared, expected, output, factory, *, authorization):
    require(authorization == AUTH, 'MAIN authorization BEFORE I/O/factory')
    p = verify(prepared, expected); m = p['source_inputs']
    data = kernel.load_source(m['sourcepack'], m['sourcepack_sha256'], m['source_prepared'], m['source_prepare_sha256'])
    store = Store(output, p['contract']['wall_seconds'])
    binding = dict(prepared=absolute(str(prepared)), prepare_sha256=expected)
    store.put('execution_binding.json', binding)
    return execute(p, data, store, factory, authorization=authorization, execution_binding=binding)


def assemble(report, events, p):
    """Pure saved-data ABI for comparison/RGB; preserves all four/partial rows.

    Missing terminal journal or dispatch-return receipts rejects consumption,
    rather than upgrading an unverifiable partial run. No source simulator.
    """
    require(report['schema'] == NAME+'.report' and report['status'] in
            ('equal144_physical_recorded_not_admitted', 'partial_or_failed') and
            report['contract'] == p['contract'] == contract(p['contract']['wall_seconds']) and
            report['sourcepack_sha256'] == p['sourcepack_sha256'] == SOURCE_SHA and
            report['source_prepare_id'] == p['source_prepare_id'] and report['fixed_branch_denominator'] == 4,
            'own complete four-row physical identity')
    for flag in ('P1_admission', 'P2_admission', 'P3_admission', 'scientific_admission',
                 'independent_confirmation', 'equal_total_compute', 'equal_FLOPs', 'training'):
        require(report[flag] is False, 'no scientific/compute upgrade')
    require(report['new_neural_calls'] == report['new_independent_samples'] == 0, 'physical only known roots')
    require(events and events[0]['kind'] == 'run_started' and events[0]['prepare'] == p and
            events[-1]['kind'] == 'run_finished' and events[-1]['status'] == report['status'] and
            events[-1]['counts'] == report['counts'] and events[-1]['errors'] == report['errors'] and
            events[-1]['event_sha256'] == report['journal_chain_tail'], 'durable physical prepare/terminal binding')
    order = [(s, a) for s in (0, 1) for a in ARMS]
    require([(r['root_slot'], r['arm']) for r in report['rows']] == order and
            [(r['root_slot'], r['arm']) for r in p['plans']] == order, 'ALL four original identities, not successes only')
    counts = {k: dict(attempted=0, completed=0) for k in p['contract']['caps']}
    intents, terminals, stack = {}, {}, []
    for event in events:
        kind = event['kind']
        if kind not in ('operation_dispatch_intent', 'operation_returned', 'operation_failed'): continue
        r = event['receipt']; i = r['operation_index']
        require(type(i) is int and r['resource'] in counts, 'literal known meter resource/index')
        if kind == 'operation_dispatch_intent':
            require(i == len(intents) and r['status'] == 'pending', 'sequential predipatch intents')
            intents[i] = r; stack.append(i)
        else:
            require(i in intents and i not in terminals and stack and stack.pop() == i and
                    r['status'] == ('returned' if kind == 'operation_returned' else 'failed'), 'actual nested terminal receipt')
            original = {k: v for k, v in r.items() if k not in ('error_type', 'message')}
            require(dict(original, status='pending') == intents[i], 'actual terminal matches original dispatch identity')
            terminals[i] = r
    require(not stack and len(terminals) == len(intents) and report['actual_operation_receipts'] ==
            [terminals[i] for i in range(len(terminals))], 'every actual entry/return independently durable')
    for r in report['actual_operation_receipts']:
        c = counts[r['resource']]; c['attempted'] += 1; c['completed'] += r['status'] == 'returned'
    require(counts == report['counts'] and all(v['attempted'] <= p['contract']['caps'][k]
            for k, v in counts.items()), 'actual meter/caps, never expected fullcounts on partial')
    checkpoints = []
    closed = [e['receipt'] for e in events if e['kind'] == 'branch_closed']
    require(closed == [r for r in report['rows'] if r['status'] != 'not_started'], 'durable branch rows/no missing denominator')
    for index, (row, plan) in enumerate(zip(report['rows'], p['plans'], strict=True)):
        require(type(row['root_slot']) is int and row['branch_index'] == index and row['selected_plan'] == plan and
                row['selection_digest'] == plan['selection_digest'] == source.digest({k: v for k, v in plan.items() if k != 'selection_digest'}),
                'exact root/task/raw/goal source-selected plan binding')
        steps = [r for r in report['actual_operation_receipts'] if r['resource'] == 'public_step' and r.get('branch') == index]
        own = [r for r in steps if r['phase'] == 'plan']; warm = [r for r in steps if r['phase'] == 'warmup']
        require(row['own_attempted'] == len(own) and row['own_returned'] == sum(r['status'] == 'returned' for r in own) and
                row['warmup_returned'] == sum(r['status'] == 'returned' for r in warm) and
                row['full_horizon'] == (row['own_returned'] == 10), 'row actualattempt-return, not durable intent counts')
        for phase, opset, commands in (('warmup', warm, p['contract']['warmup_raw']), ('plan', own, plan['commands'])):
            require([r['step'] for r in opset] == list(range(1, len(opset)+1)) and len(opset) <= 10 and
                    all(r['raw_action'] == commands[j] for j, r in enumerate(opset)), 'exact contiguous literal raw10 no replay/padding')
            require(all(np.asarray(r['raw_action'],np.float32).tobytes()==np.asarray(commands[j],np.float32).tobytes()
                for j,r in enumerate(opset)), 'actual raw float32 bits, including signed zero')
        require(bits(plan['commands'])==plan['raw_float32_bits'],'selected ten raw float32 bits')
        cpsteps = [c['step'] for c in row['checkpoints']]
        require(cpsteps == sorted(set(cpsteps)) and all(s in (0, 5, 10) for s in cpsteps), 'only actual checkpoint0/5/10')
        observations = [e['receipt'] for e in events if e['kind'] == 'fresh_observation' and e.get('branch') == index]
        transitions = [e for e in events if e['kind'] == 'returned_transition' and e.get('branch') == index and e['phase'] == 'plan']
        for c in row['checkpoints']:
            state = source.vector(c['state'])  # finite actual7, never truncated object pose
            require(c['phase'] == 'plan' and c in observations and np.array_equal(state, c['observation_state']) and
                    c['step'] <= row['own_returned'], 'checkpoint actual7 state and returned observation')
            guards = [e for e in events if e['kind'] == 'guard_receipt' and e.get('branch') == index and e['phase'] == 'plan' and e['step'] == c['step']]
            require(len(guards) == 1 and guards[0]['accepted'] is True, 'actual checkpoint guard')
            if c['step'] == 0: require(c['state'] == row['root_state'], 'actual checkpoint0 root')
            else:
                transition = [e for e in transitions if e['step'] == c['step']]
                require(len(transition) == 1 and transition[0]['state'] == c['state'], 'checkpoint bound to actual returned raw command')
            checkpoints.append(dict(branch_index=index, root_slot=row['root_slot'], arm=row['arm'],
                selection_digest=row['selection_digest'], selected_plan=deepcopy(plan), checkpoint=deepcopy(c)))
        if row['initial_exact']:
            require(source.root_hash(row['root_state'], row['goal']) == plan['root_sha256'], 'actual root+goal SHA')
            attached = [e for e in events if e['kind'] == 'goal_attached' and e.get('branch') == index]
            require(len(attached) == 1 and attached[0]['goal'] == row['goal'], 'actual offline goal identity')
            require(all(e['returned_goal_state'] == row['goal'] and e['returned_goal_rgb_sha256'] == attached[0]['goal_rgb_sha256']
                        for e in transitions), 'goal remains fixed in actual native returns')
        if row['final_state'] is not None:
            require(row['final_metrics'] == kernel.metric(row['goal'], source.vector(row['final_state'])), 'original native7 labels, no author-reported fill')
    return checkpoints


def consume(output, expected):
    """Saved-only downstream ABI; MAIN supplies the actual external physical SHA."""
    output = Path(absolute(str(output))); hash_literal(expected)
    files = list(output.iterdir())
    require(all(f.is_file() and not f.is_symlink() for f in files) and sum(f.stat().st_size for f in files) <= CAP,
            'bounded regular physical artifacts')
    require(sha(output/'report.json') == expected, 'external actual physical report SHA')
    report = read(output/'report.json', CAP)
    p = verify(report['prepared'], report['prepare_sha256'])
    binding = read(output/'execution_binding.json')
    require(binding == dict(prepared=report['prepared'], prepare_sha256=report['prepare_sha256']), 'actual physical execution binding')
    journal = output/'journal.jsonl'
    require(sha(journal) == report['journal_sha256'] and journal.stat().st_size <= CAP, 'physical journal SHA/size')
    data = journal.read_bytes(); require(len(data) <= CAP and data.endswith(b'\n'), 'complete bounded journal')
    import json
    events = [json.loads(line) for line in data.splitlines()]; previous = None
    for i, event in enumerate(events):
        require(event['index'] == i and event['previous_sha256'] == previous and event['event_sha256'] ==
                source.digest({k: v for k, v in event.items() if k != 'event_sha256'}), 'physical event hashchain')
        previous = event['event_sha256']
    require(previous == report['journal_chain_tail'], 'physical final chain tail')
    checkpoints = assemble(report, events, p)
    # Validate ALL saved actual render files, not only the downstream5/10 subset.
    from PIL import Image
    for event in events:
        if event['kind'] != 'fresh_observation': continue
        c = event['receipt']; name = c['png']
        require(Path(name).name == name and name not in ('', '.', '..') and sha(output/name) == c['file_sha256'], 'actual PNG bytes seal')
        with Image.open(output/name) as image:
            require(image.size == (224, 224) and image.mode == 'RGB', 'native RGB224 stored image')
            frame = np.asarray(image)
        require(frame.dtype == np.uint8 and kernel.moving.image_sha(frame) == c['pixel_sha256'], 'actual PNG pixel seal')
    bindings = {}; merge(bindings, p['bindings'])
    merge(bindings, {report['prepared']: report['prepare_sha256']})
    hashes = {str(f): sha(f) for f in files}; require(hashes[str(output/'report.json')] == expected, 'report final TOCTOU')
    merge(bindings, hashes)
    for path, h in bindings.items(): require(sha(path) == h, 'final saved physical closure')
    for c in checkpoints: c['png_path'] = str(output/c['checkpoint']['png'])
    return dict(schema=NAME+'.consume', status=report['status'], rows=deepcopy(report['rows']),
        checkpoints=checkpoints, counts=deepcopy(report['counts']), report=report, prepared=p,
        bindings_sha256=bindings, physical_report_sha256=expected, fixed_branch_denominator=4,
        checkpoint_complete=all([c['step'] for c in r['checkpoints']] == [0, 5, 10] for r in report['rows']),
        scientific_admission=False, P1_admission=False, independent_confirmation=False,
        equal_total_compute=False, equal_FLOPs=False, new_neural_calls=0)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','verify','run','consume'))
    for key in ('prepared','prepare-sha','output','report-sha'):p.add_argument('--'+key,type=str)
    p.add_argument('--authorize-physics',action='store_true');a=p.parse_args(argv)
    if a.command=='prepare':r=prepare(a.output)
    elif a.command=='verify':r=dict(schema=verify(a.prepared,a.prepare_sha)['schema'])
    elif a.command=='consume':r=consume(a.output,a.report_sha)
    else:
        require(a.authorize_physics,'MAIN explicit physics authorization')
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # registration only, no neural calls
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False,render_action=False)
        r=run(a.prepared,a.prepare_sha,a.output,factory,authorization=AUTH)
    print(source.canonical(r if a.command=='consume' else
        {k:v for k,v in r.items() if k not in ('rows','actual_operation_receipts','report','prepared','bindings_sha256')}))
    return 2 if a.command=='run' and r['status']!='equal144_physical_recorded_not_admitted' else 0


if __name__=='__main__':raise SystemExit(main())

