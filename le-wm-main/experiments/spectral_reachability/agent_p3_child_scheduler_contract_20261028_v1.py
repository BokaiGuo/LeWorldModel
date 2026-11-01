"""Known-child scheduling contract. Saved prepare/plan only; no model or physics loader.

plan_child is the metered callback seam for a separately reviewed child signal
adapter. The CLI never dispatches it. RAM execution requires explicit opt-in.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import inspect
import os
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import agent_p3_common_budget_20261028_v1 as common
from experiments.spectral_reachability import agent_p4_rolling_saved_consume_20261028_v1 as rolling

core, signal = common.core, common.signal
ROOT = core.ROOT
NAME = 'agent_p3_child_scheduler_contract_20261028_v1'
FLAGS = dict(core.FLAGS, P4_admission=False, scientific_admission=False, execution_authorized=False)
require, digest, canonical, sha = core.require, core.digest, core.canonical, signal.sha
STEPS = (5, 10, 15)
EPISODES = tuple(f'e{i:02}_{m}' for i in range(4) for m in ('I', 'R'))
MODES = ('fixed', 'adaptive')
CAP = 32*1024**2
consume_boundary = rolling.consume
load_paid_panels = common.load_panels


def own_files():
    return [Path(__file__).resolve(), ROOT/f'tests/{NAME}_test.py', ROOT/f'innerdocs/{NAME}.md']


def rule():
    return dict(schema=NAME+'.rule', horizons=[1, 3], signal_predict_API=8,
        signal_J=2, signal_VJP_directions=384, search_predict_cap=24,
        selected_prefix_predict_reserve=1, independent_per_method_predict_cap=33,
        allocation='unchanged core.policy/choose candidates4/8, type/horizon/depth; NO common-budget 12/4 override',
        search_reservation='two independent H-packet replays per candidate, plus selected raw5 replay',
        over_budget='retain original decision/queue; refuse whole node before generation, never shrink or pad',
        sharing=['independent', 'shared_equal'], shared_equal='one exact child signal pair; each method charged rational 1/2 of SAME actual receipt',
        no_cross_child_sharing=True, unused_budget='unspent; not refunded attempts or wasted padding calls',
        local_tree_depth=0, depth_semantics='observed child starts a new local subtree; physical boundary/5 is NOT model composition depth',
        source_denominator=8, child_slots=24, method_slots=48, independent_roots=2,
        full_C3_tree=False, outcome_input=False, physical_dispatch=False)


def signal_caps(schedules=0):
    """Exact inherited one-child H1/H3 nominal+AD profile, no neural calls here."""
    require(type(schedules) is int and schedules in (0, 1, 2), 'fixed schedule count')
    caps = {k: v//2 for k, v in signal.profile()['resource_caps'].items()}
    caps['schedule_decisions'] = schedules
    core.counters(caps)
    return caps


def validate_child(child, tasks):
    core.keys(child, ('child_id', 'episode_id', 'root_slot', 'prefix_arm', 'source_mode', 'boundary',
        'status', 'context', 'context_sha256', 'raw_history10', 'executed_prefix', 'observation',
        'original_compiler_context_sha256', 'ancestry'))
    require(child['episode_id'] in EPISODES and type(child['boundary']) is int and child['boundary'] in STEPS and
        child['child_id'] == child['episode_id']+f"_b{child['boundary']:02}", 'fixed child identity, not terminal20')
    i = EPISODES.index(child['episode_id']); slot = i//4
    require(type(child['root_slot']) is int and child['root_slot'] == slot and
        child['prefix_arm'] == rolling.ARMS[(i//2)%2] and child['source_mode'] == ('I', 'R')[i%2], 'full source layer identity')
    core.keys(child['ancestry'], ('task_binding_sha256', 'source_report_sha256', 'original_request_sha256'))
    task = core.validate_task(tasks[slot]); a = child['ancestry']
    require(a['task_binding_sha256'] == digest(task['binding']), 'same original task binding')
    core.hash_value(a['source_report_sha256'])
    if child['status'] == 'not_reached':
        require(all(child[k] is None for k in ('context', 'context_sha256', 'raw_history10', 'executed_prefix',
            'observation', 'original_compiler_context_sha256')) and a['original_request_sha256'] is None,
            'unreached slot unknown, no fabricated history/latent')
        return deepcopy(child)
    require(child['status'] == 'observed_child', 'literal observed child or unreached')
    core.hash_value(a['original_request_sha256']); core.hash_value(child['original_compiler_context_sha256'])
    c = core.context(child['context']); require(c['latent_dim'] == 192 and digest(c) == child['context_sha256'], 'exact full observed context')
    raw10 = core.finite(child['raw_history10'], (10, 2), True)
    prefix = core.finite(child['executed_prefix'], (child['boundary'], 2), True)
    require((abs(raw10)<=1).all() and (abs(prefix)<=1).all(), 'original raw action domain')
    # For b5 the other half is original warmup; later boundaries contain only
    # own actually returned control. No copying another arm or scale repair.
    take = min(child['boundary'], 10)
    rolling.exact(raw10[-take:], prefix[-take:], 'actual returned prefix raw history')
    rolling.exact(c['action_history'], rolling.normalize(raw10, task), 'original float32 subtract/divide tokens')
    if child['boundary']==5:
        rolling.exact(c['action_history'][0], task['root_context']['action_history'][-1], 'b5 first token is original warmup last packet')
    core.keys(child['observation'], ('boundary', 'png_sha256', 'pixel_sha256', 'encoding_sha256',
        'observed_index', 'encoded_index', 'observed_latent_sha256'))
    obs = child['observation']
    for k in ('png_sha256', 'pixel_sha256', 'encoding_sha256', 'observed_latent_sha256'): core.hash_value(obs[k])
    require(type(obs['boundary']) is int and obs['boundary'] == child['boundary'] and
        type(obs['observed_index']) is type(obs['encoded_index']) is int and 0<=obs['observed_index']<obs['encoded_index'] and
        obs['observed_latent_sha256'] == rolling.packet_sha(c['latent_history'][-1]), 'causal actual PNG -> encoded current latent')
    return deepcopy(child)


def task_from_export(exported, layer, source_row):
    original = exported['task']
    require(exported['task_id'] == layer['task_id'] and exported['source_slot'] == layer['source_slot'] and
        rolling.object_sha(exported['task_contract']) == original['task_contract_sha256'],
        'original export/task contract hash codec, not runtime compact codec')
    require(source_row['slot']['index']==layer['source_slot'] and
        source_row['blind_payload_sha256']==exported['blind_sha256'] and
        exported['native_root_sha256'] in (None,source_row['root_sha256']), 'explicit same-slot/blind sourcepack root edge')
    core.hash_value(source_row['root_sha256'])
    task = {k: deepcopy(original[k]) for k in ('root_context', 'goal', 'projection', 'action_mean', 'action_std')}
    task['binding'] = dict(root_sha256=source_row['root_sha256'], task_id=exported['task_id'],
        task_sha256=layer['task_sha256'], task_contract_sha256=original['task_contract_sha256'],
        blind_sha256=exported['blind_sha256'], model_sha256=original['model_sha256'],
        goal_sha256=digest(task['goal']), projection_sha256=digest(task['projection']),
        normalizer_sha256=digest([task['action_mean'], task['action_std']]),
        context_sha256=digest(task['root_context']), action_domain_sha256=digest(core.action_domain()))
    require(len(task['goal']) == 192, 'native192 signal shape')
    return core.validate_task(task)


def collect_children(reader, output, prepared, report, receipt, report_sha):
    """Project only past RGB/actions; never pass native state/reward or saved successor."""
    require([r['episode_id'] for r in receipt['rows']] == list(EPISODES), 'complete prior visual consumer')
    require([r['episode_id'] for r in report['rows']]==list(EPISODES), 'same full original episode order')
    manifest=reader.json(prepared['original_manifest'],prepared['bindings'][prepared['original_manifest']])
    pack=reader.json(manifest['sourcepack'],manifest['sourcepack_sha256'])
    require(len(pack['rows'])==2,'full two-root sourcepack')
    tasks = {}; children = []
    for index, row in enumerate(report['rows']):
        eid = EPISODES[index]; layer = prepared['layers'][index//2]; slot = index//4
        exported = reader.json(layer['task_export'], layer['task_sha256'])
        task = task_from_export(exported, layer, pack['rows'][slot])
        require(slot not in tasks or tasks[slot] == task, 'same complete task across all source arms')
        tasks[slot] = task; observed = deepcopy(task['root_context'])
        warm = rolling.array(exported['task']['root_raw_action_history'], (10, 2))
        reached = {b['boundary'] for b in receipt['rows'][index]['boundaries']}
        for boundary in STEPS:
            child = dict(child_id=eid+f'_b{boundary:02}', episode_id=eid, root_slot=slot,
                prefix_arm=layer['arm'], source_mode=row['mode'], boundary=boundary, status='not_reached',
                context=None, context_sha256=None, raw_history10=None, executed_prefix=None, observation=None,
                original_compiler_context_sha256=None, ancestry=dict(task_binding_sha256=digest(task['binding']),
                    source_report_sha256=report_sha, original_request_sha256=None))
            if boundary in reached:
                request = reader.json(output/(child['child_id']+'_compile_request.json'))
                prefix = rolling.array(row['executed_raw'][:boundary], (boundary, 2))
                raw10 = np.concatenate((warm, prefix))[-10:]
                rolling.exact(raw10, request['actualrawhistory10'], 'own physical warmup+past raw10, not future actions')
                enc = reader.json(output/(child['child_id']+'_encode/receipt.json'))
                ref = enc['latent']; payload = reader.blob(ref['path'], ref['file_sha256'], 1024**2)
                import io
                stream = io.BytesIO(payload); z = np.load(stream, allow_pickle=False)
                require(stream.tell()==len(payload) and z.dtype==np.float32 and z.shape==(192,) and
                    rolling.digest(z.tobytes()) == ref['array_sha256'], 'actual saved native latent bytes')
                observed = rolling.rolled(observed, z, raw10, task)
                if row['mode']=='R': rolling.same_context(observed, request['context3'])
                child.update(status='observed_child', context=deepcopy(observed), context_sha256=digest(observed),
                    raw_history10=raw10.tolist(), executed_prefix=prefix.tolist(), observation=deepcopy(request['observation']),
                    original_compiler_context_sha256=digest(request['context3']))
                child['ancestry']['original_request_sha256'] = rolling.packet_sha(request)
            children.append(child)
    require(list(tasks)==[0, 1] and len(children)==24, 'all two roots/eight episodes/24 slots')
    tasks = [tasks[s] for s in (0, 1)]
    return tasks, [validate_child(c, tasks) for c in children]


def jobs(children, sharing):
    require(sharing in ('independent', 'shared_equal'), 'explicit fixed signal accounting choice')
    require([(c['episode_id'], c['boundary']) for c in children] == [(e,b) for e in EPISODES for b in STEPS],
        'no dropped or reordered child slots')
    return [dict(child_id=c['child_id'], mode=m, status='not_reached' if c['status']=='not_reached' else 'pending_new_paid_child_signal',
        signal_group=c['child_id']+(':'+m if sharing=='independent' else ':shared'),
        signal_charge=dict(numerator=1, denominator=1 if sharing=='independent' else 2),
        signal_caps=signal_caps(), search_predict_cap=24, selected_prefix_predict_reserve=1,
        queue=None, commands=None, physical_result=None) for c in children for m in MODES]


def material(inputs):
    core.keys(inputs, ('p4', 'p3', 'sharing')); core.keys(inputs['p4'], ('output', 'report_sha256', 'prepared', 'prepare_sha256'))
    require(inputs['sharing'] in ('independent','shared_equal'), 'fixed pre-signal sharing protocol')
    core.keys(inputs['p3'],MODES)
    for mode in MODES:
        core.keys(inputs['p3'][mode],('output','report_sha256')); absolute(inputs['p3'][mode]['output'])
        core.hash_value(inputs['p3'][mode]['report_sha256'])
    edge = inputs['p4']; reader = rolling.Reader()
    for key in ('report_sha256', 'prepare_sha256'): core.hash_value(edge[key])
    output = absolute(edge['output']); prepared_path = absolute(edge['prepared'])
    report = reader.json(output/'report.json', edge['report_sha256']); prepared = reader.json(prepared_path, edge['prepare_sha256'])
    receipt = consume_boundary(str(output), edge['report_sha256'], str(prepared_path), edge['prepare_sha256'])
    require(receipt['full_native_or_budget_reaudit'] is False and receipt['fixed_denominator']==8 and
        receipt['compile_denominator']==24 and receipt['independent_tasks']==2, 'supplementary visual scope is NOT full native/cost audit')
    tasks, children = collect_children(reader, output, prepared, report, receipt, edge['report_sha256'])
    pp, panels, paid = load_paid_panels(inputs['p3'])
    reference = {}
    for mode in MODES:
        ledger = panels[mode]['resource_receipt']; core.audit_ledger(ledger)
        actual = [common.split_signal_resources(panels[mode], s) for s in (0,1)]
        for v in actual:
            require(all(v[k]==signal_caps()[k] for k in core.AXES if k!='schedule_decisions'), 'inherited complete paid signal profile')
        reference[mode] = dict(report_sha256=inputs['p3'][mode]['report_sha256'],
            actual_per_root_counts=actual, receipt_sha256=digest(paid[mode]),
            role='accounting reference ONLY, never a child J/context or new payment', child_signal_reused=False)
    sources = {}; merge(sources, receipt['inputs']); merge(sources, reader.inputs)
    paths = set(own_files()) | {Path(x.__file__).resolve() for x in (core, core.existing, common, signal, common.native, rolling)}
    paths.add(Path(inspect.getfile(core.FrozenSignals)).resolve())
    for mode in MODES:
        merge(sources, pp[mode]['sources'])
        r = panels[mode]; folder = absolute(inputs['p3'][mode]['output'])
        paths.update([folder/'report.json', folder/'events.jsonl', folder/'binding.json', Path(r['prepared'])])
    merge(sources, {str(p):sha(p) for p in paths})
    check_sources(sources)
    value = dict(schema=NAME+'.prepare', inputs=deepcopy(inputs), rule=rule(), tasks=tasks, children=children,
        jobs=jobs(children, inputs['sharing']), policies=[asdict(core.policy(t)) for t in tasks], sources=sources,
        paid_root_signal_reference=reference, boundary_consume_sha256=digest(receipt),
        source_scope='rolling visual seams only; MAIN native/budget audit remains separate',
        status='known_root_child_engineering_only_pending_signal_adapter', new_child_signals=None,
        inherited_RGB_encoding_cost='original P4 account, not free and not a new encoder call',
        missing_dependencies=['reviewed CUDA child signal adapter: current NativeSignal rejects non-root context',
            'native full-array signal journal and strict independent saved child consumer',
            'progress/reconfiguration compiler and independent replay from exact observed child',
            'actual selected-prefix replay, physical context bridge and causal live integration',
            'fresh cohort/P1/applicable P2/P3 admission; matched actual compute and whole-job wall'], **FLAGS)
    require(len(canonical(value))<=CAP, 'bounded complete child metadata')
    return value


def merge(target, values):
    for path,h in values.items():
        core.hash_value(h); require(path not in target or target[path]==h, 'conflicting path/SHA')
        target[path] = h


def check_sources(sources):
    for path,h in sources.items(): require(sha(absolute(path))==h, 'full source/input drift: '+path)


def absolute(path):
    p = Path(path); require(p.is_absolute() and not any(q.is_symlink() for q in (p,*p.parents)), 'absolute nonsymlink path')
    return p


def verify(path, expected):
    core.hash_value(expected); reader = rolling.Reader(); p = reader.json(absolute(path), expected)
    check_sources(p['sources'])
    require(canonical(p)==canonical(material(p['inputs'])) and sha(path)==expected, 'full prepare source/roster/budget rebuild')
    return p


def plan(prepared):
    """Offline executable handoff. Missing actual child signals never become chi=0."""
    require(prepared['schema']==NAME+'.prepare' and prepared['rule']==rule() and
        all(prepared[k] is False for k in FLAGS), 'sealed child contract, no admission')
    children = [validate_child(c, prepared['tasks']) for c in prepared['children']]
    queue = jobs(children, prepared['inputs']['sharing'])
    require(queue==prepared['jobs'], 'full48 original method denominator')
    return dict(schema=NAME+'.plan', status='blocked_missing_paid_child_signals', fixed_denominator=48,
        child_denominator=24, independent_roots=2, jobs=queue,
        refusal_census=dict(not_reached=sum(j['status']=='not_reached' for j in queue),
            missing_child_signal=sum(j['status']!='not_reached' for j in queue),
            measured_search_budget_refusals=None, all_observed_children_budget_refused=None,
            reason='no actual child signals/decisions yet; cannot infer quota refusal from root or old outcome'),
        signal_requests=[dict(child_id=j['child_id'], group=j['signal_group'], mode=j['mode'],
            parent_context_sha256=children[i//2]['context_sha256'], horizons=[1,3],
            task_binding_sha256=children[i//2]['ancestry']['task_binding_sha256'])
            for i,j in enumerate(queue) if j['status']!='not_reached'],
        shared_request_semantics='two references to one group only in shared_equal; executor must acquire exactly once',
        missing_dependencies=prepared['missing_dependencies'], neural_calls=0, physics_calls=0, **FLAGS)


def plan_child(child, tasks, backend_factory, *, sharing='independent', permit_ram=False, authorization=None):
    """Actual paid callback -> unchanged policy -> candidate requests, NOT actions.

    No producer supplied here accepts real children yet. HOST callback use needs
    MAIN authorization and a durable sink; declaration alone is not a CUDA proof.
    """
    child = validate_child(child,tasks); require(sharing in ('independent','shared_equal'), 'signal acquisition mode')
    rows = [dict(mode=m,status='not_reached',decision=None,queue=None,error=None,signal_group=None,
        signal_charge=None,commands=None) for m in MODES]
    result = dict(schema=NAME+'.child_plan', child_id=child['child_id'], child_sha256=digest(child),
        sharing=sharing, fixed_denominator=2, rows=rows, acquisitions=[], status='not_reached', **FLAGS)
    if child['status']=='not_reached': return result
    task = core.validate_task(tasks[child['root_slot']]); node = dict(context=child['context'],depth=0)
    groups = [('shared', MODES)] if sharing=='shared_equal' else [(m,(m,)) for m in MODES]
    for group, modes in groups:
        ledger = None; acquisition = dict(group=child['child_id']+':'+group,status='not_started',signal=None,resource_receipt=None,error=None)
        result['acquisitions'].append(acquisition)
        try:
            require(permit_ram is True or authorization=='MAIN_APPROVED_CHILD_SIGNAL', 'authorization before adapter factory')
            backend = backend_factory(deepcopy(child),group)
            kind = getattr(backend,'kind',None)
            acquisition['backend_kind']=kind
            require((permit_ram is True and kind=='RAM_FIXTURE') or
                (kind=='HOST_CUDA' and authorization=='MAIN_APPROVED_CHILD_SIGNAL' and callable(getattr(backend,'event_sink',None))),
                'explicit RAM fixture or MAIN reviewed HOST CUDA durable adapter; no CPU fallback')
            ledger = core.Ledger(signal_caps(len(modes)),sink=getattr(backend,'event_sink',None))
            ledger.scope = dict(child_id=child['child_id'],group=group,phase='child_signal',parent_context_sha256=child['context_sha256'])
            s, receipt = core.local_signal(task,node,backend,ledger,core.policy(task,node))
            expected_source = getattr(backend,'source_sha256',None); core.hash_value(expected_source)
            require(all(r['producer_sha256']==expected_source for r in receipt['responses']), 'exact reviewed signal adapter source')
            require(ledger.completed==signal_caps(), 'exact fully paid H1/H3 hooks/sample/J/VJP/SVD, not borrowed root signal')
            acquisition.update(status='returned',signal=receipt)
            for mode in modes:
                ledger.scope = dict(child_id=child['child_id'],group=group,phase='schedule',mode=mode,parent_context_sha256=child['context_sha256'])
                decision = ledger.call('schedule',lambda mode=mode:core.choose(task,node,s,mode))
                h = decision['horizon']; count = decision['candidates']
                allowed = 2*h*count<=24
                queue = [core.request(task,node,operation='compile_child_program_extension', **decision,
                    child_id=child['child_id'], child_sha256=digest(child),candidate_index=k,
                    prefix_actions=[], prefix_edges=[], actual_executed_prefix=deepcopy(child['executed_prefix']),
                    signal_sha256=digest(receipt), local_response=deepcopy(receipt['responses'])) for k in range(count)]
                row = rows[MODES.index(mode)]
                row.update(status='pending_reviewed_child_compiler' if allowed else 'blocked_whole_node_search_budget',
                    decision=decision,queue=queue,signal_group=acquisition['group'],
                    signal_charge=dict(numerator=1,denominator=len(modes)),
                    reserved_search_predict_API=2*h*count,search_predict_cap=24,selected_prefix_predict_reserve=1,
                    generation_dispatch_allowed=False,composition_witness=None)
        except Exception as exc:
            acquisition.update(status='failed',error=type(exc).__name__+': '+str(exc))
            for mode in modes: rows[MODES.index(mode)].update(status='failed_child_signal_or_schedule',decision=None,queue=None,error=acquisition['error'])
        finally:
            if ledger is not None:
                acquisition['resource_receipt']=ledger.receipt(); core.audit_ledger(acquisition['resource_receipt'])
    result['status']='ram_planned_not_evidence' if permit_ram is True and all(a['status']=='returned' and a.get('backend_kind')=='RAM_FIXTURE' for a in result['acquisitions']) else 'child_plan_not_admitted'
    return result


def write_new(output, value):
    path = absolute(output)
    require(path.parent==ROOT/'results' and path.name.startswith(NAME+'_') and not path.exists(), 'exclusive new own result')
    content=canonical(value); require(len(content)<=CAP, 'bounded complete output')
    with path.open('xb') as stream:
        require(stream.write(content)==len(content),'complete write'); stream.flush(); os.fsync(stream.fileno())
    signal.fsync_dir(path.parent)
    return dict(output=str(path),sha256=sha(path),bytes=len(content))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=('prepare','verify','plan'))
    p.add_argument('--input',required=True); p.add_argument('--sha256',required=True); p.add_argument('--output')
    a=p.parse_args(argv)
    if a.output:
        out=absolute(a.output); require(out.parent==ROOT/'results' and out.name.startswith(NAME+'_') and not out.exists(),'NEW output before consumption')
    if a.command=='prepare':
        require(a.output is not None,'prepare needs exclusive output')
        inputs=rolling.Reader().json(absolute(a.input),a.sha256); value=material(inputs)
        require(sha(a.input)==a.sha256,'input manifest unchanged')
    else:
        prepared=verify(a.input,a.sha256)
        value=plan(prepared) if a.command=='plan' else dict(schema=NAME+'.verified',status=prepared['status'],**FLAGS)
    print(canonical(write_new(a.output,value) if a.output else value).decode()); return 0


if __name__=='__main__': raise SystemExit(main())
