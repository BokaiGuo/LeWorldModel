"""Fresh eight-branch predict-frontier physical consumer; no P2 admission.

Prepare requires the actual complete sealed planner report. Only MAIN may use
--execute-physics/--encode-terminals; no planning, reselection or old endpoints.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil

import numpy as np

from experiments.spectral_reachability import p3_joint_horizon_goal_metric_scheduler_20261027_v1 as planner
from experiments.spectral_reachability import p2_composition_predict_frontier_cuda_20261027_v1 as frontier
from experiments.spectral_reachability import p3_horizon_residual_physical_20261027_v4 as helpers
from experiments.spectral_reachability import p3_matched_physical_20261027_v1 as durable
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal

NAME = 'p2_composition_predict_frontier_physical_20261027_v1'
PROJECT = Path(__file__).resolve().parents[2]
REFERENCES = 'results/p3_horizon_residual_physical_20261027_v4_source_prepared01'
REFERENCES_SHA = 'c5eee78ded753d1c2b955335b97e218d420671e1d2d35636aae8303a8811622b'
ARMS = frontier.ARMS
METRICS = ('source_Q',)
io, require, write, digest = planner.io, planner.require, frontier.write, planner.digest
MIB = 1024**2
KEEP_FREE = 200*MIB
RUN_RESERVE = 100*MIB
EXPECTED_REPORT_SHA = '4023a0e3896a90407188d260a45dd334e0802ba17ecd59ef62ea931603eba173'


def read(path):
    return io.decode(Path(path).read_bytes())


def contract():
    return dict(roots=4, arms=list(ARMS), branches=8, repetitions=1, horizons=[2],
        upstream=frontier.NAME, export='NEW all80 joint/all40 conditional; never old40 exporter',
        selected_path_step_cap=80, actual_planned_steps=80,
        warmup_step_cap=120, warmup_steps_per_branch=15, root_collection_step_cap=0,
        total_public_env_step_cap=200, reset_accounting='collector/warmup attempts and returns separate',
        guard='RoyalBlue before each raw command; durable attempts/returns and fresh endpoints',
        endpoint_encoder_cap=8, endpoint_existing_projector_cap=8, new_predict_calls=0,
        no_reselection=True, no_clipping=True, old_endpoints_used=False,
        total_resource_matched=False, inherited_context_cost='unknown, not zero',
        planner_validation_cost='source-bound all-slot exporter recomputation; additional CPU arithmetic, not inference',
        disk_keep_free_bytes=KEEP_FREE, initial_output_reserve_bytes=RUN_RESERVE,
        P1_admission=False, P2_admission=False, P3_admission=False, P4_admission=False, P5_admission=False)

def disk_check(path, reserve=RUN_RESERVE):
    require(shutil.disk_usage(path).free >= KEEP_FREE+reserve, 'disk reserve: keep 200 MiB final; no deletion')


def validate_plan(p):
    require(p['schema'] == NAME+'.plan' and p['method'] in ARMS and
        type(p['planned_horizon_tokens']) is int and p['planned_horizon_tokens'] == 2 and
        p['planned_commands'] == 10, 'frontier full H2 raw10 only')
    raw = planner.array(p['commands'], (10, 2))
    require(np.all(np.abs(raw) <= 1), 'bounded raw controls; no clipping')
    require(p['metric_label'] == 'source_Q' and p['schedule'] == 'predict_frontier', 'original Q/frontier contract')
    planner.context_arrays(p['predictor_context']); planner.context_arrays(p['selected_predicted_context'])
    require(digest({k:v for k,v in p.items() if k != 'selection_sha256'}) == p['selection_sha256'], 'plan digest')


def plans_from_exports(body, exported, compilation, report_sha):
    require(exported['schema'] == frontier.NAME+'.selected_actions' and exported['report_sha256'] == report_sha and
        exported['source_bound'] is True and exported['saved_data_validation'] is True and
        exported['P1_admission'] is False and exported['P2_admission'] is False and
        exported['total_resource_matched'] is False, 'NEW complete all-slot frontier export only')
    cases = io.index(body['context']['cases'], ('root_sha256',))
    source = io.index(compilation['rows'], ('root_sha256',))
    index = io.index(exported['plans'], ('root_sha256','arm'))
    require(len(cases) == 4 and set(cases) == set(source) and
        {c['root_index'] for c in cases.values()} == set(range(4)) and
        set(index) == {(r[0],a) for r in cases for a in ARMS}, 'exact four root x two arm census')
    plans = []
    for case in sorted(cases.values(), key=lambda c:c['root_index']):
        root = case['root_sha256']; c = source[root,]['compiled']
        require(case['root']['root_id'] == root and
            frontier.digest(c['goal_vectors'][0]) == case['goal_vector_sha256'] and
            frontier.digest(c['scoring']['metric']['projection']) == case['projection_sha256'] and
            c['scoring']['metric']['sha256'] == case['metric_sha256'], 'source goal/Q/metric binding')
        initial = dict(planner.META, latent_history=c['input_context']['emb'][0],
            action_history=c['input_context']['actions'][0][1:])
        require(case['root']['context'] == initial, 'complete source root context')
        order = ARMS if case['root_index'] % 2 == 0 else ARMS[::-1]
        for arm in order:
            e = index[root,arm]; w = e['witness']; receipt = w['receipt']
            require(e['root_index'] == case['root_index'] and e['root'] == case['root'] and
                e['horizon'] == 2 and e['report_sha256'] == report_sha and
                e['action_domain'] == case['action_domain'] and e['metric_sha256'] == case['metric_sha256'] and
                receipt['provenance'] == e['provenance'] and receipt['actions'] == e['raw_commands'],
                'frontier source/action/domain/provenance binding')
            terminal = receipt['terminal_context']
            planner.check_history(initial, terminal, e['raw_commands'], body['context']['normalization'])
            require([len(x['actions']) for x in w['edges']] == ([10] if arm == ARMS[0] else [5,5]),
                'joint single full-H2 edge versus conditional two H1 edges')
            require(w == frontier.legacy().saved_witness(case['root'],e['provenance'],case['action_domain'],
                [x['input_context'] for x in w['edges']]+[w['edges'][-1]['output_context']],
                [x['actions'] for x in w['edges']],w['edges'][0]['program_id'].removesuffix('_edge_0')), 'rebound selected witness')
            cost = frontier.legacy().full20.metric_for(c)(terminal)
            require(cost == e['predicted_Q'], 'selected original Q cost')
            p = dict(schema=NAME+'.plan', root_index=case['root_index'], root_sha256=root, method=arm,
                metric_label='source_Q', schedule='predict_frontier', active_metric_id=case['metric_sha256'],
                planned_horizon_tokens=2, planned_commands=10, corrected_search_horizon=2,
                commands=deepcopy(e['raw_commands']), goal=deepcopy(c['goal_vectors'][0]),
                predictor_context=deepcopy(initial), selected_predicted_context=deepcopy(terminal),
                planner_report_sha256=report_sha, planner_row_sha256=e['row_sha256'],
                candidate_bank_sha256=e['candidate_bank_sha256'], witness=deepcopy(w),
                predicted_costs=dict(source_Q=cost), exported_resources=deepcopy(e['resources']))
            p['selection_sha256'] = digest(p); validate_plan(p); plans.append(p)
    require(len(plans) == 8 and sum(p['planned_commands'] for p in plans) == 80, 'all eight H2 paths')
    return plans


def planning_material(root, output, report_sha):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(report_sha == EXPECTED_REPORT_SHA, 'externally provided actual frontier report SHA required')
    report_path = output/'report.json'
    require(io.sha(report_path.read_bytes()) == report_sha, 'external complete report bytes')
    report = frontier.read(report_path)
    _, body, sealed = frontier.load_prepared(report['prepared'], report['prepared_sha256'])
    exported = frontier.selected_actions(output)  # NEW all-slot80/40 exporter; never legacy selected_actions
    require(exported['report_sha256'] == report_sha, 'report changed during export')
    source_path = Path(body['context']['compilation'])
    compilation = frontier.read(source_path)
    plans = plans_from_exports(body, exported, compilation, report_sha)
    refs = helpers.validate_prepared(root/REFERENCES, REFERENCES_SHA)
    packets = io.index(refs['packets'], ('root_sha256',))
    cases = io.index(body['context']['cases'], ('root_sha256',))
    require(set(packets) == set(cases), 'same four physical source roots')
    models = refs['model_bindings']
    require({str(Path(body['context']['policy']).resolve()/n) for n in ('weights.pt','config.json')} == set(models),
        'same frozen endpoint model policy')
    for key, case in cases.items():
        packet = packets[key]; c = next(r['compiled'] for r in compilation['rows'] if r['root_sha256'] == key[0])
        require(packet['root_context'] == case['root']['context'] and packet['goal'] == c['goal_vectors'][0] and
            packet['projection'] == c['scoring']['metric']['projection'], 'original source context/goal/Q')
        ref = refs['references'][key[0]]
        require(case['root']['root_rgb_sha256'] == ref['history_rgb_sha256'] and
            case['root']['goal_rgb_sha256'] == ref['goal_rgb_sha256'], 'source root/goal RGB binding')
    paths = [report_path, output/'execution.json', Path(report['prepared'])/'prepared.json',
        Path(report['prepared'])/'protocol.json', source_path]
    for i in range(4):
        paths.extend(output/f'root_{i}{suffix}' for suffix in ('.json','_bank.json','_journal.jsonl'))
    bindings = {str(p.resolve()):io.sha(p.read_bytes()) for p in paths}
    for path,h in {**body['bindings'], **models}.items():
        require(io.sha(Path(path).read_bytes()) == h, 'sealed upstream/model binding: '+path); bindings[str(Path(path).resolve())] = h
    require(bindings[str(report_path)] == report_sha, 'final external report SHA')
    metrics = {r['root_sha256']:dict(goal=r['compiled']['goal_vectors'][0],
        metrics=dict(source_Q=dict(projection=r['compiled']['scoring']['metric']['projection']))) for r in compilation['rows']}
    resources = dict(actual_unique_predict_calls=report['actual_unique_predict_calls'],
        actual_unique_action_encoder_calls=report['actual_unique_action_encoder_calls'],
        completed_operations=report['completed_operations'],completed_selection_evaluations=report['completed_selection_evaluations'],
        exporter_CPU_validation_cost=exported['CPU_validation_cost'])
    return plans, refs, metrics, models, bindings, resources

def source_files(root):
    root = Path(root)
    seeds = [Path(__file__).resolve(), root/'tests'/('test_'+NAME+'.py'), root/'innerdocs'/(NAME+'.md'),
        root/'experiments/spectral_reachability/audit_pusht_d2p_goal_agent_intervention.py',
        root/'experiments/spectral_reachability/run_pusht_sampling_comparison_20261027_v2.py',
        root/'.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py',
        root/'.venv/lib/python3.12/site-packages/stable_worldmodel/envs/__init__.py']
    return planner.old.prior.physical.source_closure(root, seeds)


def prepare(root, planner_output, report_sha, output):
    root, output = Path(root).resolve(), Path(output).absolute()
    require(not output.exists(), 'exclusive NEW prepare path')
    disk_check(output.parent, reserve=20*MIB)
    plans, refs, metrics, models, bindings, resources = planning_material(root, planner_output, report_sha)
    sources = source_files(root)
    p = dict(schema=NAME+'.protocol', contract=contract(), project_root=str(root), plans=plans,
        planner_output=str(Path(planner_output).absolute()), planner_report_sha256=report_sha,
        references=refs['references'], root_seed=refs['root_seed'], root_oversample_factor=4,
        root_metrics=metrics, model_bindings=models, input_bindings=bindings,
        inherited_planner_resources=resources, planned_selected_path_steps=sum(p['planned_commands'] for p in plans),
        sources_sha256={str(s): io.sha(s.read_bytes()) for s in sources})
    estimated = len(io.canonical(p).encode())*2 + sum(s.stat().st_size for s in sources) + MIB
    require(estimated < 20*MIB, 'small prepare <20MiB; no upstream report/journal or weights copy')
    disk_check(output.parent, reserve=estimated+RUN_RESERVE)
    output.mkdir(exist_ok=False)
    write(output/'protocol.json', p)
    seals = [create_seal(files=sources[i:i+40], args_json=io.canonical(dict(protocol_id=digest(p))),
        output_dir=output/f'source_seal_{i//40}') for i in range(0, len(sources), 40)]
    receipt = dict(status='actual_planner_bound_ready', protocol_sha256=io.sha((output/'protocol.json').read_bytes()),
        protocol_id=digest(p), source_seals=seals, planned_selected_path_steps=p['planned_selected_path_steps'],
        estimated_prepare_bytes=estimated, new_neural_calls=0, new_physical_steps=0, P1_admission=False, P2_admission=False, P3_admission=False)
    write(output/'prepare.json', receipt)
    return receipt


def validate_prepared(prepared, sha):
    prepared = Path(prepared).absolute()
    require(io.sha((prepared/'prepare.json').read_bytes()) == sha, 'prepare SHA')
    r, p = read(prepared/'prepare.json'), read(prepared/'protocol.json')
    require(io.sha((prepared/'protocol.json').read_bytes()) == r['protocol_sha256'] and digest(p) == r['protocol_id']
        and p['schema'] == NAME+'.protocol' and p['contract'] == contract(), 'protocol and contract')
    for path, h in {**p['sources_sha256'], **p['input_bindings']}.items():
        require(io.sha(Path(path).read_bytes()) == h, 'source/input drift: '+path)
    for s in r['source_seals']:
        verify_seal(output_dir=s['output_dir'], expected_manifest_sha256=s['manifest_sha256'])
    require(set(p['sources_sha256']) == {str(x) for x in source_files(p['project_root'])}, 'complete source closure')
    plans, refs, metrics, models, bindings, resources = planning_material(p['project_root'], p['planner_output'], p['planner_report_sha256'])
    require(p['plans'] == plans and p['references'] == refs['references'] and p['root_seed'] == refs['root_seed']
        and p['root_metrics'] == metrics and p['model_bindings'] == models and p['input_bindings'] == bindings
        and p['inherited_planner_resources'] == resources and p['root_oversample_factor'] == 4
        and p['planned_selected_path_steps'] == sum(x['planned_commands'] for x in plans), 'upstream reconstruction')
    return p


def physical_branch(*, plan, output, reset, env, state_of, evaluate):
    """Reuse raw-step guard, not old eight-arm consumer/old endpoint artifacts."""
    validate_plan(plan)
    output = Path(output)
    stem = f'root_{plan["root_index"]}_{plan["method"]}'
    row = {k: deepcopy(plan[k]) for k in ('root_index', 'root_sha256', 'method', 'selection_sha256',
        'planned_horizon_tokens', 'planned_commands', 'metric_label', 'schedule', 'active_metric_id')}
    row.update(status='branch_failed', failures=[], trace=[], attempted_commands=0, completed_commands=0,
        physical_env_step_attempts=0, physical_env_step_returns=0, full_horizon_completed=False,
        terminal_rgb_path=None, endpoint_fresh=False, endpoint_scope='unknown_or_stale_endpoint')
    path = output/(stem+'.events.jsonl')
    try:
        with path.open('x') as stream:
            def event(value, image):
                value = deepcopy(value)
                if image is not None:
                    from PIL import Image
                    image_path = output/f'{stem}_rgb_{value.get("primitive_step", 0)}.png'
                    with image_path.open('xb') as image_stream:
                        Image.fromarray(image).save(image_stream, format='PNG')
                        image_stream.flush(); os.fsync(image_stream.fileno())
                    h = io.sha(image_path.read_bytes())
                    value.update(rgb_path=str(image_path), rgb_file_sha256=h)
                    row.update(terminal_rgb_path=str(image_path), terminal_rgb_sha256=h)
                stream.write(io.canonical(value)+'\n'); stream.flush(); os.fsync(stream.fileno())
            row['context_identity'] = reset()
            result, _ = durable.guarded_execute(env, plan['commands'], root_id=plan['root_sha256'], state_of=state_of, event=event)
            row.update(result)
            if result['failure']:
                row['failures'].append(result['failure'])
            if result['final_state'] is not None:
                row.update(evaluate(result['final_state']))
    except (Exception, KeyboardInterrupt) as exc:
        row['failures'].append(durable.failure(exc, 'branch'))
        row['status'] = 'branch_failed'
    row['journal_path'] = str(path)
    try:
        raw = path.read_bytes()
        row['journal_sha256'] = io.sha(raw)
        events = [io.decode(line) for line in raw.splitlines()]
        images = [e for e in events if 'rgb_path' in e]
        row['endpoint_fresh'] = bool(images and row['terminal_rgb_path'] == images[-1]['rgb_path'] and
            images[-1].get('primitive_step', 0) == row['physical_env_step_returns'] and
            row['physical_env_step_attempts'] == row['physical_env_step_returns'] and
            row['status'] in ('completed', 'observation_unavailable', 'environment_terminated'))
        if row['endpoint_fresh']:
            row['endpoint_scope'] = ('full_H' if row['full_horizon_completed'] else 'partial_H')+str(plan['planned_horizon_tokens'])
    except (Exception, KeyboardInterrupt) as exc:
        row['failures'].append(durable.failure(exc, 'branch_journal_finalize'))
    try:
        write(output/(stem+'.physical.json'), row)
    except (Exception, KeyboardInterrupt) as exc:
        row['failures'].append(durable.failure(exc, 'branch_write'))
    return row


def verify_branch(plan, row):
    validate_plan(plan)
    require(all(row[k] == plan[k] for k in ('root_index', 'root_sha256', 'method', 'selection_sha256',
        'planned_horizon_tokens', 'planned_commands', 'active_metric_id')), 'exact branch identity')
    a, b = (io.count(row[k]) for k in ('physical_env_step_attempts', 'physical_env_step_returns'))
    require(b <= a <= plan['planned_commands'] and a == row['attempted_commands'] and b == row['completed_commands'], 'step accounting')
    raw = Path(row['journal_path']).read_bytes()
    require(io.sha(raw) == row['journal_sha256'], 'physical journal bytes')
    events = [io.decode(line) for line in raw.splitlines()]
    attempts = [e for e in events if e['kind'] == 'attempt']
    returns = [e for e in events if e['kind'] == 'step_returned']
    observations = [e for e in events if e['kind'] == 'observation']
    require(len(attempts) == a and len(returns) == b and len(observations) == len(row['trace']), 'journal census')
    require([e['primitive_step'] for e in attempts] == list(range(1, a+1)) and
        [e['primitive_step'] for e in returns] == list(range(1, b+1)), 'ordered actual attempts/returns')
    require(all(e['command'] == plan['commands'][i] for i, e in enumerate(attempts)), 'raw actions not reselected')
    require(all(all(e.get(k) == v for k, v in t.items()) for e, t in zip(observations, row['trace'])), 'trace binding')
    for e in events:
        if 'rgb_path' in e:
            require(io.sha(Path(e['rgb_path']).read_bytes()) == e['rgb_file_sha256'], 'fresh RGB bytes')
    images = [e for e in events if 'rgb_path' in e]
    fresh = bool(images and row.get('terminal_rgb_path') == images[-1]['rgb_path'] and
        row.get('terminal_rgb_sha256') == images[-1]['rgb_file_sha256'] and
        images[-1].get('primitive_step', 0) == b and a == b and
        row['status'] in ('completed', 'observation_unavailable', 'environment_terminated'))
    require(row['endpoint_fresh'] is fresh, 'fresh endpoint recomputed from journal')
    require(row['full_horizon_completed'] == (row['status'] == 'completed' and b == plan['planned_commands']), 'full horizon gate')
    if row['full_horizon_completed']:
        require(fresh and row['endpoint_scope'] == 'full_H'+str(plan['planned_horizon_tokens']), 'full endpoint scope')


def materialize_missing(plans, rows):
    seen = {(r['root_sha256'], r['method']) for r in rows}
    for p in plans:
        if (p['root_sha256'], p['method']) not in seen:
            row = {k: deepcopy(p[k]) for k in ('root_index', 'root_sha256', 'method', 'selection_sha256',
                'planned_commands', 'planned_horizon_tokens', 'metric_label', 'schedule', 'active_metric_id')}
            row.update(status='blocked', failures=['prior setup/integrity failure'], trace=[], attempted_commands=0,
                completed_commands=0, physical_env_step_attempts=0, physical_env_step_returns=0,
                full_horizon_completed=False, endpoint_fresh=False, endpoint_scope='unknown_or_stale_endpoint', terminal_rgb_path=None)
            rows.append(row)


def comparisons(plans, rows):
    index = io.index(rows, ('root_sha256','method'))
    require(len(plans) == 8 and set(index) == {(p['root_sha256'],p['method']) for p in plans}, 'all eight frontier branches')
    for p in plans:
        require(all(index[p['root_sha256'],p['method']][k] == p[k] for k in
            ('selection_sha256','planned_horizon_tokens','planned_commands','active_metric_id')), 'paired plan binding')
    pairs = []
    for root in dict.fromkeys(p['root_sha256'] for p in plans):
        a,b = (index[root,arm] for arm in ARMS)
        same = 'context_identity' in a and 'context_identity' in b and a['context_identity'] == b['context_identity']
        full = same and all(r['full_horizon_completed'] and r['endpoint_fresh'] and not r['failures'] for r in (a,b))
        partial = same and all(r['endpoint_fresh'] for r in (a,b)) and a['physical_env_step_returns'] == b['physical_env_step_returns']
        pairs.append(dict(root_sha256=root, left=ARMS[0],right=ARMS[1],
            classification='full_H2' if full else 'partial_same_steps' if partial else 'partial_or_unmatched',
            right_minus_left_pose=b['pose']['pose_cost']-a['pose']['pose_cost'] if full else None,
            right_minus_left_official=b['official']['official_state_distance']-a['official']['official_state_distance'] if full else None,
            identical_initial_context=same,actual_steps=[a['physical_env_step_returns'],b['physical_env_step_returns']],
            scope='known-root descriptive; not equal total budget or composition admission'))
    return pairs

def optional_encodings(protocol, rows, output, setup):
    output = Path(output)
    require((output/'physical_report.json').is_file(), 'all physics receipts durable before any encoding')
    plans = io.index(protocol['plans'], ('root_sha256', 'method'))
    receipts, encode, setup_error = [], None, None
    with (output/'encoding.events.jsonl').open('x') as stream:
        def event(e):
            stream.write(io.canonical(e)+'\n'); stream.flush(); os.fsync(stream.fileno())
        if any(r['endpoint_fresh'] for r in rows):
            try:
                event(dict(kind='encoding_setup', completed=False))
                encode = setup(event)
                event(dict(kind='encoding_setup', completed=True))
            except (Exception, KeyboardInterrupt) as exc:
                setup_error = durable.failure(exc, 'encoding_setup')
        for row in rows:
            p = plans[row['root_sha256'], row['method']]
            r = dict(root_sha256=row['root_sha256'], method=row['method'], active_metric_id=p['active_metric_id'],
                selection_sha256=p['selection_sha256'], endpoint_scope=row['endpoint_scope'],
                planned_horizon_tokens=p['planned_horizon_tokens'], attempted_encode_samples=0,
                status='blocked_stale_endpoint', failure=None)
            receipts.append(r)
            try:
                if row['endpoint_fresh']:
                    if setup_error is not None:
                        r.update(status='encoding_setup_failed', failure=setup_error)
                    else:
                        path = Path(row['terminal_rgb_path'])
                        require(io.sha(path.read_bytes()) == row['terminal_rgb_sha256'], 'exact new endpoint bytes')
                        r.update(attempted_encode_samples=1, terminal_rgb_sha256=row['terminal_rgb_sha256'])
                        event(dict(kind='endpoint_encode', completed=False, **r))
                        z = planner.array(encode(path), (192,))
                        r['terminal_vector'] = z.tolist()
                        event(dict(kind='terminal_array', **r))
                        qs = protocol['root_metrics'][row['root_sha256']]
                        r['actual_costs'] = {q: planner.metric.native_cost(z, qs['goal'], qs['metrics'][q]['projection']) for q in METRICS}
                        r.update(status='completed', predicted_costs=p['predicted_costs'])
            except (Exception, KeyboardInterrupt) as exc:
                r.update(status='encoding_failed', failure=durable.failure(exc, 'endpoint_encode'))
            event(dict(kind='endpoint_result', **r))
            write(output/f'root_{p["root_index"]}_{p["method"]}.encoding.json', r)
    return receipts


class DiskGuardEnv:
    def __init__(self, env, output):
        self.env, self.output = env, output
    def render(self):
        # Guard before the next dispatch (or while observing the previous return).
        # Do NOT raise inside step before calling the actual env: that would
        # incorrectly charge a disk denial as an attempted physical env.step.
        disk_check(self.output, reserve=16*MIB)
        return self.env.render()
    def step(self, action):
        return self.env.step(action)


class PhysicalMeter:
    """Actual public env.step/reset calls, including collection and warmup.

    Collector source uses resets, no steps; any changed collector step refuses.
    Internal simulator substeps/reset-internal work are NOT public env.step.
    """
    def __init__(self, env, sink):
        self.env, self.sink = env, sink
        self.phase, self.binding = 'root_collection', {}
        self.steps = {p: dict(attempts=0, returns=0) for p in ('root_collection', 'warmup', 'selected')}
        self.resets = deepcopy(self.steps)
        self.branches, self.journal_errors = {}, []
        self.selected_limit = None
    def __getattr__(self, name):
        return getattr(self.env, name)
    def bind(self, plan):
        self.binding = {k: plan[k] for k in ('root_sha256', 'method', 'selection_sha256')}
        self.selected_limit = plan['planned_commands']
    def call(self, name, args, kwargs):
        require(self.phase in self.steps, 'explicit physical phase')
        c = (self.steps if name == 'step' else self.resets)[self.phase]
        key = (self.binding.get('root_sha256'), self.binding.get('method'), self.phase, name)
        local = self.branches.setdefault(key, dict(attempts=0, returns=0))
        if name == 'step':
            cap = dict(root_collection=0, warmup=120, selected=80)[self.phase]
            per_branch = dict(root_collection=0, warmup=15, selected=self.selected_limit)[self.phase]
            require(per_branch is not None and c['attempts'] < cap and local['attempts'] < per_branch,
                'physical phase/branch cap exceeded before actual env.step')
        event = dict(kind='env_'+name, phase=self.phase, index=c['attempts'], completed=False, **self.binding)
        if name == 'step': event['command'] = np.asarray(args[0]).tolist()
        self.sink(deepcopy(event))
        c['attempts'] += 1; local['attempts'] += 1
        failure = None
        try:
            value = getattr(self.env, name)(*args, **kwargs)
            c['returns'] += 1; local['returns'] += 1; event['completed'] = True
            return value
        except BaseException as exc:
            failure = exc; event['error'] = type(exc).__name__+': '+str(exc)
            raise
        finally:
            try: self.sink(deepcopy(event))
            except (Exception, KeyboardInterrupt) as exc:
                self.journal_errors.append(durable.failure(exc, 'physical_meter_journal'))
                if failure is None: raise
    def step(self, *a, **kw):
        return self.call('step', a, kw)
    def reset(self, *a, **kw):
        return self.call('reset', a, kw)
    def receipt(self):
        return dict(steps=deepcopy(self.steps), resets=deepcopy(self.resets), journal_errors=deepcopy(self.journal_errors),
            total_public_env_step_attempts=sum(v['attempts'] for v in self.steps.values()),
            total_public_env_step_returns=sum(v['returns'] for v in self.steps.values()),
            total_env_reset_attempts=sum(v['attempts'] for v in self.resets.values()),
            total_env_reset_returns=sum(v['returns'] for v in self.resets.values()),
            scope='public Gym step/reset calls; internal simulator substeps and reset-internal work unmeasured')


def counted_encoding(callback, name, shape, event, stats):
    """Independent nonrefundable encoder/projector entry with terminal failure event."""
    def counted(*a, **kw):
        require(a[0].device.type == 'cuda' and tuple(a[0].shape) == shape and stats[name+'_attempts'] < 8,
            'CUDA single endpoint encoder/projector cap8')
        index = stats[name+'_attempts']; stats[name+'_attempts'] += 1
        entry = dict(kind=name, index=index, completed=False)
        event(deepcopy(entry))
        failure = None
        try:
            result = callback(*a, **kw)
            stats[name+'_completed'] += 1; entry['completed'] = True
            return result
        except BaseException as exc:
            failure = exc; entry['error'] = type(exc).__name__+': '+str(exc)
            raise
        finally:
            try: event(deepcopy(entry))
            except (Exception, KeyboardInterrupt) as exc:
                stats.setdefault('journal_errors', []).append(durable.failure(exc, 'encoding_entry_journal'))
                if failure is None: raise
    return counted


def execute_physics(prepared, sha, output, encode_terminals=False):
    p = validate_prepared(prepared, sha)
    output = Path(output).absolute()
    require(not output.exists(), 'exclusive NEW execution output; never reuse endpoints')
    disk_check(output.parent)
    output.mkdir(exist_ok=False)
    write(output/'execution_protocol.json', dict(prepared=str(Path(prepared).absolute()), prepare_sha256=sha,
        planner_report_sha256=p['planner_report_sha256'], encode_terminals=encode_terminals))
    rows, failures, encodings = [], [], []
    env, model, before, checked, host = None, None, None, None, None
    originals = []
    meter = None
    budget_path = output/'physical_budget.events.jsonl'
    budget_stream = budget_path.open('x')
    def budget_event(e):
        budget_stream.write(io.canonical(e)+'\n'); budget_stream.flush(); os.fsync(budget_stream.fileno())
    stats = dict(encoder_attempts=0, encoder_completed=0, projector_attempts=0, projector_completed=0, forbidden_predict_attempts=0)
    try:
        from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
        from experiments.spectral_reachability.run_pusht_sampling_comparison_20261027_v2 import verify_rgb_context, official_checked
        env = base.gym.make('swm/PushT-v1', render_mode='rgb_array')
        evaluator = Path(env.unwrapped.eval_state.__code__.co_filename).resolve()
        require(str(evaluator) in p['sources_sha256'] and io.sha(evaluator.read_bytes()) == p['sources_sha256'][str(evaluator)], 'sealed official evaluator')
        require(tuple(env.unwrapped.variation_space['agent']['color'].value.tolist()) == (65, 105, 225), 'RoyalBlue source guard')
        meter = PhysicalMeter(env, budget_event)
        roots = base.collect_nonterminal_boundary_contact_roots(meter, count=4, seed=int(p['root_seed']), oversample_factor=4)
        rootmap = helpers.validate_roots(roots, p['references'], base.root_fingerprint)
        identities = {}
        for plan in p['plans']:
            root = rootmap[plan['root_sha256']]
            meter.bind(plan)
            def reset():
                meter.phase = 'warmup'
                require(base.root_fingerprint(root) == plan['root_sha256'], 'root fingerprint')
                prior = deepcopy(meter.steps['warmup'])
                frames, history, goal = base.physical._reset_and_warmup(meter, root, history_len=3)
                require(all(meter.steps['warmup'][k]-prior[k] == 15 for k in ('attempts', 'returns')), 'original warmup exactly15 public steps')
                identity = verify_rgb_context(frames, goal, p['references'][plan['root_sha256']])
                identity.update(helpers.validate_reset_history(history, plan['predictor_context']['action_history']),
                    state=base.p0.packets.state_of(env).tolist(), goal=np.asarray(root['goal']).tolist())
                if plan['root_sha256'] in identities:
                    require(identity == identities[plan['root_sha256']], 'two-arm full reset/context mismatch')
                identities[plan['root_sha256']] = deepcopy(identity)
                meter.phase = 'selected'
                return identity
            def evaluate(state):
                return dict(official=official_checked(env, state, root['goal']), pose=base.physical._pose_label(state, root['goal']))
            row = physical_branch(plan=plan, output=output, reset=reset, env=DiskGuardEnv(meter, output),
                state_of=lambda proxy: base.p0.packets.state_of(proxy.env), evaluate=evaluate)
            rows.append(row)
            verify_branch(plan, row)
            require(row['status'] in ('completed', 'observation_unavailable', 'environment_terminated'), 'failed dispatch blocks remaining arms')
        require(meter.steps['selected']['attempts'] == sum(r['physical_env_step_attempts'] for r in rows) and
            meter.steps['selected']['returns'] == sum(r['physical_env_step_returns'] for r in rows), 'independent selected/actual env.step census')
    except (Exception, KeyboardInterrupt) as exc:
        failures.append(durable.failure(exc, 'physics'))
    finally:
        if env is not None:
            try: env.close()
            except (Exception, KeyboardInterrupt) as exc: failures.append(durable.failure(exc, 'env_close'))
        try: budget_stream.close()
        except (Exception, KeyboardInterrupt) as exc: failures.append(durable.failure(exc, 'physical_budget_close'))
    physical_calls = meter.receipt() if meter is not None else dict(unknown=True)
    if meter is not None and meter.journal_errors:
        failures.extend(meter.journal_errors)
    budget_sha = None
    try: budget_sha = io.sha(budget_path.read_bytes())
    except (Exception, KeyboardInterrupt) as exc: failures.append(durable.failure(exc, 'physical_budget_hash'))
    materialize_missing(p['plans'], rows)
    write(output/'physical_report.json', dict(schema=NAME+'.physical_report', rows=rows, failures=deepcopy(failures),
        planner_report_sha256=p['planner_report_sha256'], consumer_prepare_sha256=sha, contract=contract(),
        physical_call_receipt=physical_calls, physical_budget_journal_sha256=budget_sha))
    physical_sha = io.sha((output/'physical_report.json').read_bytes())
    if encode_terminals:
        def setup(event):
            nonlocal model, before, checked, host
            import torch
            from PIL import Image
            from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
            from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
            from experiments.checkpoint_compat import load_compatible_pretrained
            from experiments.gpu_preflight import require_cuda_allocation
            require(io.sha((output/'physical_report.json').read_bytes()) == physical_sha, 'durable physical report bytes before CUDA')
            for path, h in p['model_bindings'].items(): require(checked.digest(path) == h, 'same endpoint model weights')
            host = require_cuda_allocation('cuda'); device = torch.device('cuda:0')
            policy = Path(next(path for path in p['model_bindings'] if Path(path).name == 'weights.pt')).parent
            model = load_compatible_pretrained(str(policy), device=device).eval().requires_grad_(False)
            before = checked.tensor_state_hash(model)
            def forbid(*a, **kw):
                stats['forbidden_predict_attempts'] += 1
                event(dict(kind='forbidden_predict'))
                raise ValueError('endpoint encoding forbids predict')
            originals.append((model, 'predict', model.predict)); model.predict = forbid
            for name, shape in (('encoder', (1, 3, 224, 224)), ('projector', (1, 192))):
                module = getattr(model, name); callback = module.forward
                originals.append((module, 'forward', callback))
                module.forward = counted_encoding(callback, name, shape, event, stats)
            transform = p0.image_transform()
            def encode(path):
                with Image.open(path) as image: rgb = np.asarray(image.convert('RGB')).copy()
                with torch.no_grad(): z = model.encode({'pixels': p0._image_tensor(rgb, transform=transform, device=device)})['emb'][0, 0]
                require(z.device.type == 'cuda' and bool(torch.isfinite(z).all()), 'finite frozen native endpoint vector')
                torch.cuda.synchronize(device)
                return z.cpu().tolist()
            return encode
        try:
            encodings = optional_encodings(p, rows, output, setup)
        except (Exception, KeyboardInterrupt) as exc:
            failures.append(durable.failure(exc, 'encoding_persistence'))
    model_ok, source_ok = True, True
    if model is not None:
        try:
            model_ok = before is not None and before == checked.tensor_state_hash(model)
            require(model_ok, 'model changed')
        except (Exception, KeyboardInterrupt) as exc:
            model_ok = False; failures.append(durable.failure(exc, 'final_model'))
    for obj, key, original in reversed(originals):
        try: setattr(obj, key, original)
        except (Exception, KeyboardInterrupt) as exc: failures.append(durable.failure(exc, 'hook_restore'))
    try: validate_prepared(prepared, sha)
    except (Exception, KeyboardInterrupt) as exc:
        source_ok = False; failures.append(durable.failure(exc, 'final_source'))
    full = all(r['full_horizon_completed'] and r['endpoint_fresh'] and not r['failures'] for r in rows)
    encoded = not encode_terminals or len(encodings) == 8 and all(e['status'] == 'completed' for e in encodings)
    encoding_exact = not encode_terminals or all(stats[k+'_attempts'] == stats[k+'_completed'] == 8 for k in ('encoder', 'projector'))
    complete = full and encoded and encoding_exact and not failures and model_ok and source_ok and not stats.get('journal_errors') and stats['forbidden_predict_attempts'] == 0
    result = dict(schema=NAME+'.report', status='guarded_calibration_recorded' if complete else 'partial_or_incomplete',
        rows=rows, comparisons=comparisons(p['plans'], rows), encodings=encodings, failures=failures,
        consumer_prepare_sha256=sha, planner_report_sha256=p['planner_report_sha256'], physical_report_sha256=physical_sha,
        planned_selected_path_steps=p['planned_selected_path_steps'],
        physical_env_step_attempts=sum(r['physical_env_step_attempts'] for r in rows),
        physical_env_step_returns=sum(r['physical_env_step_returns'] for r in rows),
        physical_call_receipt=physical_calls, physical_budget_journal_sha256=budget_sha,
        model_unchanged=model_ok if encode_terminals else None, source_verified=source_ok, encoding_stats=stats, host=host,
        full_horizon_all_branches=full, root_search_warmup_cost='separate; 80 selected +120 warmup cap; collector/reset actual counts',
        inherited_planner_resources=p['inherited_planner_resources'], contract=contract(),
        P1_admission=False, P2_admission=False, P3_admission=False, P4_admission=False, P5_admission=False)
    write(output/'report.json', result)
    require(complete, 'partial/incomplete; preserve all assets; no automatic retry')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'verify', 'execute-physics'): action.add_argument('--'+name, action='store_true')
    parser.add_argument('--project-root', type=Path, default=PROJECT)
    parser.add_argument('--planner-output', type=Path)
    parser.add_argument('--planner-report-sha256')
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--prepare-sha256')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--encode-terminals', action='store_true')
    a = parser.parse_args()
    require(a.execute_physics or not a.encode_terminals, 'encoding only explicit main physical execution')
    if a.prepare:
        require(a.output is not None and a.planner_output is not None and a.planner_report_sha256, 'actual planner report and new output')
        r = prepare(a.project_root, a.planner_output, a.planner_report_sha256, a.output)
    else:
        require(a.prepared is not None and a.prepare_sha256, 'reviewed prepare and SHA')
        if a.verify:
            validate_prepared(a.prepared, a.prepare_sha256); r = dict(status='verified_no_neural_physics')
        else:
            require(a.output is not None, 'new output required')
            r = execute_physics(a.prepared, a.prepare_sha256, a.output, a.encode_terminals)
    print(io.canonical({k: v for k, v in r.items() if k not in ('rows', 'encodings', 'source_seals')}))


if __name__ == '__main__':
    main()

