"""Fresh sixteen-branch physical consumer for joint P3, never P1 admission.

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
from experiments.spectral_reachability import p3_horizon_residual_physical_20261027_v4 as helpers
from experiments.spectral_reachability import p3_matched_physical_20261027_v1 as durable
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal

NAME = 'p3_joint_horizon_goal_metric_physical_20261027_v1'
PROJECT = Path(__file__).resolve().parents[2]
PLANNER_PREPARED = 'results/p3_joint_horizon_goal_metric_scheduler_20261027_v1_prepared01'
PLANNER_SHA = '7886406284604cf17ba97671030ef32fde9d32f6f4c18b8293db0c2129654a47'
REFERENCES = 'results/p3_horizon_residual_physical_20261027_v4_source_prepared01'
REFERENCES_SHA = 'c5eee78ded753d1c2b955335b97e218d420671e1d2d35636aae8303a8811622b'
ARMS = planner.ARMS
io, require, write, digest = planner.io, planner.require, planner.write, planner.digest
MIB = 1024**2
KEEP_FREE = 200*MIB
RUN_RESERVE = 100*MIB


def read(path):
    return io.decode(Path(path).read_bytes())


def contract():
    return dict(roots=4, arms=list(ARMS), branches=16, repetitions=1, horizons=[2, 3],
        selected_path_step_cap=240, actual_planned_steps='sum(5H) from sixteen sealed winners; 160..240',
        warmup_step_cap=240, warmup_steps_per_branch=15, root_collection_step_cap=0,
        total_public_env_step_cap=480, reset_accounting='collector and per-branch warmup reset attempts/returns separate',
        reset='fresh root/goal, RGB/state and normalized warmup identity before each arm',
        guard='RoyalBlue RGB before every raw command; durable attempts/returns and fresh endpoints',
        metric='own Q for each arm, plus both Q cross-scores on same fresh terminal vector',
        comparisons='Qnew-old within each schedule; dynamic-uniform within each Q; sameH/mixed/partial separate',
        endpoint_encoder_cap=16, endpoint_existing_projector_cap=16, new_predict_calls=0,
        new_jacobians=0, new_vjp_directions=0, new_execution_svds=0,
        model_free_planner_export='16 source-validation +32 response SVD per reconstruction; not free',
        no_reselection=True, old_endpoints_used=False, cached_history_cost='inherited unknown, not zero',
        disk_keep_free_bytes=KEEP_FREE, initial_output_reserve_bytes=RUN_RESERVE,
        disk_scope='best effort free-space check; concurrent external writers not controlled',
        total_resource_matched=False, P1_admission=False, P2_admission=False, P3_admission=False,
        P4_admission=False, P5_admission=False, efficacy_claim=False)


def disk_check(path, reserve=RUN_RESERVE):
    require(shutil.disk_usage(path).free >= KEEP_FREE+reserve, 'disk reserve: keep 200 MiB final; no deletion')


def validate_plan(p):
    h = p['planned_horizon_tokens']
    require(p['schema'] == NAME+'.plan' and p['method'] in ARMS and type(h) is int and h in (2, 3)
        and p['planned_commands'] == 5*h, 'known arm and exact H2/H3 commands')
    raw = planner.array(p['commands'], (5*h, 2))
    require(np.all(np.abs(raw) <= 1), 'bounded raw controls; no clipping')
    require(p['metric_label']+'__'+p['schedule'] == p['method'], 'Q x scheduler label')
    planner.context_arrays(p['predictor_context'])
    planner.context_arrays(p['selected_predicted_context'])
    require(digest({k: v for k, v in p.items() if k != 'selection_sha256'}) == p['selection_sha256'], 'plan digest')


def plans_from_exports(body, exports, report_sha):
    records = io.index(body['records'], ('seed_index',))
    roots = {r['packet']['root_sha256']: r for r in records.values()}
    require(len(records) == len(roots) == 4 and set(records) == {(i,) for i in range(4)}, 'four stable source roots/seeds')
    index = io.index(exports, ('root_sha256', 'arm'))
    require(set(index) == {(root, arm) for root in roots for arm in ARMS}, 'exact sixteen root-arm census')
    plans = []
    for i in range(4):
        record = records[i,]
        source = record['packet']
        for arm in body['arm_order'][i]:
            e = index[source['root_sha256'], arm]
            label, schedule = arm.split('__')
            require(e['selection_sha256'] == digest({k: v for k, v in e.items() if k != 'selection_sha256'}), 'export digest')
            require(e['report_sha256'] == report_sha and e['protocol_id'] == digest(body) and
                e['source_input_sha256'] == source['input_sha256'] and e['seed_index'] == i and
                e['root_context'] == source['root_context'] and e['goal'] == source['goal'] and
                e['active_metric_id'] == record['metrics'][label]['active_metric_id'] and
                e['projection'] == record['metrics'][label]['projection'], 'complete source/goal/action/Q binding')
            require(e['P1_admission'] is False and e['P3_admission'] is False and e['physical_ready'] is False,
                'planner export scope; this NEW consumer supplies physical binding')
            winner = e['winner']
            require(winner['completed'] is True and type(winner['horizon']) is int and winner['horizon'] in (2, 3), 'complete winner')
            planner.check_history(e['root_context'], winner['context'], winner['actions'], body['normalization'])
            terminal = planner.array(winner['context']['latent_history'], (3, 192))[-1]
            costs = {q: planner.metric.native_cost(terminal, source['goal'], record['metrics'][q]['projection'])
                for q in planner.METRICS}
            require(costs[label] == winner['cost'], 'own-Q predicted witness')
            p = dict(schema=NAME+'.plan', root_index=i, root_sha256=e['root_sha256'], method=arm,
                metric_label=label, schedule=schedule, active_metric_id=e['active_metric_id'],
                planned_horizon_tokens=winner['horizon'], planned_commands=5*winner['horizon'],
                corrected_search_horizon=3, commands=deepcopy(winner['actions']), goal=source['goal'],
                predictor_context=deepcopy(source['root_context']), selected_predicted_context=deepcopy(winner['context']),
                source_input_sha256=source['input_sha256'], planner_export_sha256=e['selection_sha256'],
                planner_report_sha256=report_sha, predicted_costs=costs)
            p['selection_sha256'] = digest(p)
            validate_plan(p)
            plans.append(p)
    require(160 <= sum(p['planned_commands'] for p in plans) <= 240, 'actual 16-arm selected path budget')
    return plans


def planning_material(root, output, report_sha):
    root, output = Path(root).resolve(), Path(output).absolute()
    require(io.sha((output/'report.json').read_bytes()) == report_sha, 'explicit actual planner report SHA')
    binding = read(output/'execution_protocol.json')
    require(binding == dict(prepared=str(root/PLANNER_PREPARED), prepare_sha256=PLANNER_SHA), 'exact frozen planner prepare')
    body = planner.validate_prepared(root/PLANNER_PREPARED, PLANNER_SHA)
    plans = plans_from_exports(body, planner.selected_actions(output), report_sha)
    refs = helpers.validate_prepared(root/REFERENCES, REFERENCES_SHA)
    original = io.index(refs['packets'], ('root_sha256',))
    require(set(original) == {(r['packet']['root_sha256'],) for r in body['records']}, 'same four reference roots')
    for r in body['records']:
        packet = original[r['packet']['root_sha256'],]
        require(packet == r['packet'], 'original root/context/goal/old-Q input binding')
    report = read(output/'report.json')
    bindings = {str(output/n): io.sha((output/n).read_bytes()) for n in ('report.json', 'execution_protocol.json')}
    for r in report['roots']:
        require(io.sha(Path(r['journal_path']).read_bytes()) == r['journal_sha256'], 'planner journal bytes')
        bindings[r['journal_path']] = r['journal_sha256']
    root_metrics = {r['packet']['root_sha256']: dict(goal=r['packet']['goal'], metrics=r['metrics']) for r in body['records']}
    return plans, refs, root_metrics, body['model_bindings'], bindings, report['resources']


def prepare(root, planner_output, report_sha, output):
    root, output = Path(root).resolve(), Path(output).absolute()
    require(not output.exists(), 'exclusive NEW prepare path')
    disk_check(output.parent, reserve=32*MIB)
    plans, refs, metrics, models, bindings, resources = planning_material(root, planner_output, report_sha)
    seeds = [Path(__file__).resolve(), root/'tests'/('test_'+NAME+'.py'), root/'innerdocs'/(NAME+'.md'),
        root/'experiments/spectral_reachability/audit_pusht_d2p_goal_agent_intervention.py',
        root/'experiments/spectral_reachability/run_pusht_sampling_comparison_20261027_v2.py',
        root/'.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py',
        root/'.venv/lib/python3.12/site-packages/stable_worldmodel/envs/__init__.py']
    sources = planner.old.prior.physical.source_closure(root, seeds)
    p = dict(schema=NAME+'.protocol', contract=contract(), project_root=str(root), plans=plans,
        planner_output=str(Path(planner_output).absolute()), planner_report_sha256=report_sha,
        references=refs['references'], root_seed=refs['root_seed'], root_oversample_factor=4,
        root_metrics=metrics, model_bindings=models, input_bindings=bindings,
        inherited_planner_resources=resources, planned_selected_path_steps=sum(p['planned_commands'] for p in plans),
        sources_sha256={str(s): io.sha(s.read_bytes()) for s in sources})
    output.mkdir(exist_ok=False)
    write(output/'protocol.json', p)
    seals = [create_seal(files=sources[i:i+40], args_json=io.canonical(dict(protocol_id=digest(p))),
        output_dir=output/f'source_seal_{i//40}') for i in range(0, len(sources), 40)]
    receipt = dict(status='actual_planner_bound_ready', protocol_sha256=io.sha((output/'protocol.json').read_bytes()),
        protocol_id=digest(p), source_seals=seals, planned_selected_path_steps=p['planned_selected_path_steps'],
        new_neural_calls=0, new_physical_steps=0, P1_admission=False, P3_admission=False)
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
    index = io.index(rows, ('root_sha256', 'method'))
    require(len(plans) == 16 and set(index) == {(p['root_sha256'], p['method']) for p in plans}, 'all sixteen physical arms')
    for p in plans:
        require(all(index[p['root_sha256'], p['method']][k] == p[k] for k in
            ('selection_sha256', 'planned_horizon_tokens', 'planned_commands', 'active_metric_id')), 'paired plan binding')
    contrasts = [('Q_effect__'+s, 'old_Q__'+s, 'goal_patch_Q__'+s) for s in ('uniform', 'dynamic')]
    contrasts += [('schedule_effect__'+q, q+'__uniform', q+'__dynamic') for q in planner.METRICS]
    pairs = []
    for root in dict.fromkeys(p['root_sha256'] for p in plans):
        for label, left, right in contrasts:
            a, b = index[root, left], index[root, right]
            hs = [a['planned_horizon_tokens'], b['planned_horizon_tokens']]
            identical = 'context_identity' in a and 'context_identity' in b and a['context_identity'] == b['context_identity']
            full = identical and all(r['full_horizon_completed'] and r['endpoint_fresh'] for r in (a, b))
            partial = identical and all(r['endpoint_fresh'] for r in (a, b)) and a['physical_env_step_returns'] == b['physical_env_step_returns']
            stratum = (f'full_H{hs[0]}' if hs[0] == hs[1] else 'full_mixed_H2_H3') if full else (
                'partial_same_steps' if partial else 'partial_or_unmatched')
            delta = None
            if full and hs[0] == hs[1] and all('pose' in r for r in (a, b)):
                delta = b['pose']['pose_cost']-a['pose']['pose_cost']
            pairs.append(dict(root_sha256=root, contrast=label, left=left, right=right, selected_horizons=hs,
                classification=stratum, identical_initial_context=identical, right_minus_left_pose=delta,
                actual_steps=[a['physical_env_step_returns'], b['physical_env_step_returns']],
                scope='known roots descriptive; no mixed-H/full/partial pooling or admission'))
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
                        r['actual_costs'] = {q: planner.metric.native_cost(z, qs['goal'], qs['metrics'][q]['projection']) for q in planner.METRICS}
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
            cap = dict(root_collection=0, warmup=240, selected=240)[self.phase]
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
        require(a[0].device.type == 'cuda' and tuple(a[0].shape) == shape and stats[name+'_attempts'] < 16,
            'CUDA single endpoint encoder/projector cap16')
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
                    require(identity == identities[plan['root_sha256']], 'four-arm full reset/context mismatch')
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
    encoded = not encode_terminals or len(encodings) == 16 and all(e['status'] == 'completed' for e in encodings)
    encoding_exact = not encode_terminals or all(stats[k+'_attempts'] == stats[k+'_completed'] == 16 for k in ('encoder', 'projector'))
    complete = full and encoded and encoding_exact and not failures and model_ok and source_ok and not stats.get('journal_errors') and stats['forbidden_predict_attempts'] == 0
    result = dict(schema=NAME+'.report', status='guarded_calibration_recorded' if complete else 'partial_or_incomplete',
        rows=rows, comparisons=comparisons(p['plans'], rows), encodings=encodings, failures=failures,
        consumer_prepare_sha256=sha, planner_report_sha256=p['planner_report_sha256'], physical_report_sha256=physical_sha,
        planned_selected_path_steps=p['planned_selected_path_steps'],
        physical_env_step_attempts=sum(r['physical_env_step_attempts'] for r in rows),
        physical_env_step_returns=sum(r['physical_env_step_returns'] for r in rows),
        physical_call_receipt=physical_calls, physical_budget_journal_sha256=budget_sha,
        model_unchanged=model_ok if encode_terminals else None, source_verified=source_ok, encoding_stats=stats, host=host,
        full_horizon_all_branches=full, root_search_warmup_cost='separate; 240 cap covers selected paths only',
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
