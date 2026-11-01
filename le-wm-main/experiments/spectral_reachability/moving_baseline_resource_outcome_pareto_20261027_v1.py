"""SHA-bound saved resource/outcome joins; no model, physics or producer calls."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import shutil

from experiments.spectral_reachability import p1_moving_baseline_budget_matrix_20261027_v1 as budget

STEM = 'moving_baseline_resource_outcome_pareto_20261027_v1'
ROOT = Path(__file__).resolve().parents[2]
MATRIX_SHA = '721d978d29516f8ecf1d1056e159736ef231723e6849490a83678bd76baa1ebc'
COMPARISON_SHA = '96cf27b143bb1a839edf7bc1e4de02f4a97e433c6475c4c731c78af76300d32e'
ARMS = budget.ARMS
COUNTERS = budget.COUNTERS + ('predict_API', 'scalar_output_VJP_directions')
SCALARS = budget.SCALARS + ('scalar_loss_reverse_sweeps',)
PHYSICAL_AXES = ('physics_step', 'public_step', 'reset', 'set_state', 'explicit_render',
                 'goal_bind', 'goal_render', 'environment_construct', 'environment_close')
PROFILES = ('CPU_complete_profile', 'native_source_resource_profile')
EXTRA = ('image_encoder', 'image_projector') + PHYSICAL_AXES + PROFILES + ('pipeline_walltime_seconds',)
AXES = COUNTERS + SCALARS + EXTRA
COUNTER_COSTS = COUNTERS + ('image_encoder', 'image_projector') + PHYSICAL_AXES
OBJECTIVES = dict(realQ_terminal10='min', native_success='max', native_distance='min', native_angle_term='min')
OVERLAP = [
    dict(axes=['AD_forward_in_predict', 'predict'], relation='subset_not_extra_forward'),
    dict(axes=['predict', 'predict_API', 'forward_samples', 'action_encoder', 'pred_projector'],
         relation='overlapping_module_API_sample_views_not_additive'),
    dict(axes=['backward', 'grad_API', 'jacobian', 'jacobian_API', 'scalar_output_VJP_directions',
               'scalar_loss_reverse_sweeps', 'mathematical_reverse_directions'],
         relation='overlapping_derivative_views; scalar_loss_sweep_is_not_output_VJP_direction'),
    dict(axes=['public_step', 'physics_step'], relation='different_nested_step_units_not_additive'),
    dict(axes=['native_source_resource_profile'] + list(COUNTERS), relation='raw_native_receipt_aliases_not_extra_charges'),
]
CAP, OUTPUT_CAP, FLOOR = 8 * 1024**2, 2 * 1024**2, 200 * 1024**2
require = budget.require


def encode(x):
    return (json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n').encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(x):
    return sha(encode(x))


def hash_value(x):
    require(type(x) is str and len(x) == 64 and all(c in '0123456789abcdef' for c in x), 'literal SHA256')
    return x


def unknown(reason):
    return dict(status='unknown', value=None, reason=reason)


def actual(value, basis):
    return dict(status='actual', value=deepcopy(value), basis=basis)


def typed(c, kind='scalar'):
    require(isinstance(c, dict) and c.get('status') in
            ('actual', 'missing', 'partial', 'unmeasured', 'unknown') and 'value' in c, 'typed evidence cell')
    if c['status'] != 'actual':
        require(c['value'] is None and bool(c.get('reason')), 'unknown is null with reason, not zero')
    else:
        require(bool(c.get('basis')), 'actual evidence basis required')
        v = c['value']
        if kind == 'counter':
            budget.count(v)
        elif kind == 'bool':
            require(type(v) is bool, 'native/full-horizon boolean required')
        elif kind == 'sha':
            hash_value(v)
        elif kind == 'profile':
            require(isinstance(v, dict) and bool(v), 'complete typed operation profile, not a scalar total')
            for key, count in v.items():
                require(type(key) is str and bool(key), 'CPU operation unit name required')
                budget.count(count)
        else:
            require(type(v) in (int, float) and math.isfinite(v), 'finite scalar, bool forbidden')
            if kind != 'signed':
                require(v >= 0, 'nonnegative cost/Q/distance')
    return deepcopy(c)


def own_plan_sha(plan):
    # Original physical adapter uses default-spaces JSON, not newline canonical.
    return sha(json.dumps(plan, sort_keys=True, allow_nan=False).encode())


def unique_rows(rows, fields, label):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in fields)
        require(key not in result, 'duplicate ' + label)
        result[key] = row
    return result


def cost_vector(counters, latency, resources):
    result = {}
    for axis in COUNTERS + SCALARS:
        c = typed(counters[axis], 'counter' if axis in COUNTERS else 'scalar')
        c.update(unit=axis, scope='one_root_one_arm_model_producer')
        result[axis] = c
    for axis in EXTRA:
        result[axis] = dict(**unknown('no complete attributable per-root pipeline receipt; no cohort division'),
                            unit=axis, scope='one_root_one_arm_complete_pipeline')
    if resources['status'] == 'actual' and resources['value'].get('raw_resources'):
        raw = typed(actual(resources['value']['raw_resources'], 'literal native-source per-root counters; aliases, not extra work'), 'profile')
        raw.update(unit='native_source_resource_profile', scope='one_root_one_arm_model_producer')
        result['native_source_resource_profile'] = raw
    # Per-root model latency, when present, still is NOT complete pipeline time.
    typed(latency)
    return result


def dominance(a, b):
    """Strict complete-vector historical Pareto, never an equal-budget test.

    Both cost phases are minimized. All costs and selected measured objectives
    must be known and share units/scopes/semantics before *any* verdict, including
    non-dominance. Missing dimensions cannot be dropped to infer a frontier.
    """
    gaps, av, bv = [], [], []
    if a['comparison_context'] != b['comparison_context']:
        gaps.append('task/root/goal/horizon/metric context mismatch')
    for axis in AXES:
        ca = a['costs'].get(axis, unknown('axis missing'))
        cb = b['costs'].get(axis, unknown('axis missing'))
        va, vb = (typed(c, 'counter' if axis in COUNTER_COSTS else
                        'profile' if axis in PROFILES else 'scalar') for c in (ca, cb))
        if va['status'] != 'actual' or vb['status'] != 'actual':
            gaps.append('cost.' + axis + ': unknown')
            continue
        if any(not ca.get(k) or ca.get(k) != cb.get(k) for k in ('unit', 'scope')):
            gaps.append('cost.' + axis + ': incompatible unit/scope')
            continue
        if axis == 'mathematical_reverse_directions' and a['derivative_semantics'] != b['derivative_semantics']:
            gaps.append('cost.' + axis + ': scalar-loss versus output-VJP semantics')
            continue
        if axis in PROFILES:
            if set(va['value']) != set(vb['value']):
                gaps.append('cost.' + axis + ': incompatible CPU operation units')
                continue
            for unit in sorted(va['value']):
                if any(c['value'][unit]['attempted'] != c['value'][unit]['completed'] for c in (va, vb)):
                    gaps.append('cost.' + axis + '.' + unit + ': partial attempts')
                for phase in ('attempted', 'completed'):
                    av.append(va['value'][unit][phase]); bv.append(vb['value'][unit][phase])
        elif axis in COUNTER_COSTS:
            if any(c['value']['attempted'] != c['value']['completed'] for c in (va, vb)):
                gaps.append('cost.' + axis + ': partial attempts')
            for phase in ('attempted', 'completed'):
                av.append(va['value'][phase]); bv.append(vb['value'][phase])
        else:
            av.append(va['value']); bv.append(vb['value'])
    for axis, direction in OBJECTIVES.items():
        ca, cb = (typed(r['outcomes'].get(axis, unknown('objective missing')),
                        'bool' if axis == 'native_success' else 'signed' if axis == 'native_angle_term' else 'scalar')
                  for r in (a, b))
        if ca['status'] != 'actual' or cb['status'] != 'actual':
            gaps.append('outcome.' + axis + ': unknown')
            continue
        sign = 1 if direction == 'min' else -1
        av.append(sign * ca['value']); bv.append(sign * cb['value'])
    if any(typed(r['outcomes'].get('full_horizon', unknown('horizon absent')), 'bool')['value'] is not True for r in (a, b)):
        gaps.append('complete physical horizon unavailable')
    if gaps:
        return dict(status='unknown', gaps=gaps)
    if av == bv:
        status = 'tie'
    elif all(x <= y for x, y in zip(av, bv)):
        status = 'a_dominates_b'
    elif all(y <= x for x, y in zip(av, bv)):
        status = 'b_dominates_a'
    else:
        status = 'nondominated_pair'
    return dict(status=status, gaps=[])


def ledger(matrix):
    """Charge entries once; per-root/totals/standalone fields are views, not refunds."""
    entries = [dict(id='shared_inherited_encoding', kind='inherited_encoding', arms=list(ARMS),
                    receipt=deepcopy(matrix['inherited_encoding_shared_once']), charge_once=True)]
    for run in matrix['model_invocations']:
        methods = [m for m in matrix['methods'] if m['source_id'] == run['source_id']]
        # Same operation across disjoint roots/arms only; no cross-axis total.
        counters = {axis: budget.total([m['two_root_totals'][axis] for m in methods], axis in COUNTERS)
                    for axis in COUNTERS + SCALARS}
        entries.append(dict(id='model:' + run['source_id'], kind='model', arms=run['arms'],
            provenance=deepcopy(run['provenance']), counters=counters,
            latency_seconds=deepcopy(run['latency_seconds']), charge_once=True))
        entries.append(dict(id='consumer_CPU:' + run['source_id'], kind='saved_consumer_verification_CPU',
            arms=run['arms'], receipt=unknown('no complete separately metered consumer CPU profile'),
            provenance=deepcopy(run['provenance']), charge_once=True))
    for kind, cohorts in (('physics', matrix['physical_cohorts']), ('RGB', matrix['RGB_cohorts'])):
        for c in cohorts:
            entries.append(dict(id=kind + ':' + c['id'], kind=kind, arms=deepcopy(c['arms']),
                provenance=deepcopy(c['provenance']), counters=deepcopy(c['counts']),
                latency_seconds=deepcopy(c['latency_seconds']), charge_once=True,
                scope='whole_cohort_once_not_per_arm_or_root'))
    require(len({e['id'] for e in entries}) == len(entries), 'duplicate ledger charge ID')
    receipt_keys = []
    for e in entries:
        if 'provenance' in e:
            ref = e['provenance'].get('consume' if e['kind'] == 'saved_consumer_verification_CPU' else 'report')
            require(ref is not None, 'ledger receipt reference required')
            receipt_keys.append((e['kind'], ref['sha256']))
    require(len(set(receipt_keys)) == len(receipt_keys), 'same receipt charged twice')
    return dict(entries=entries, overlap=deepcopy(OVERLAP), cross_unit_total=None,
        rule='No refunds/subtraction/netting; row/totals/standalone/cohort links are NON-ADDITIVE views of entries.',
        pipeline_total_status='unknown; incomplete CPU attribution and per-root walltime')


def assemble(matrix, comparison, tasks, consumes, physical, rgb):
    """RAM seam: IO separately enforces pinned SHA of every loaded metadata file."""
    require(matrix == budget.assemble(comparison, matrix['input']), 'matrix/comparison complete resource projection mismatch')
    require(matrix['input']['sha256'] == COMPARISON_SHA, 'matrix comparison SHA binding')
    require(len(tasks) == 2, 'both fixed task exports required')
    roots = []
    for slot, e in enumerate(tasks):
        require(type(e['source_slot']) is int and e['source_slot'] == slot and e['with_target'] is False, 'task slot/target')
        t = e['task']
        for h in (e['task_id'], e['blind_sha256'], t['task_contract_sha256']): hash_value(h)
        require(own_plan_sha(t) == e['task_id'], 'literal task payload SHA identity')
        require(own_plan_sha(e['task_contract']) == t['task_contract_sha256'], 'task contract digest (upstream default-spaces JSON)')
        if e.get('native_root_sha256') is not None: hash_value(e['native_root_sha256'])
        roots.append(dict(source_slot=slot, task_id=e['task_id'], task_sha256=budget.TASK_SHAS[slot],
            blind_sha256=e['blind_sha256'], task_contract_sha256=t['task_contract_sha256'],
            root_context_sha256=digest(t['root_context']), goal_sha256=digest(t['goal']),
            upstream_native_root_sha256=(actual(e['native_root_sha256'], 'literal task export field')
                if e.get('native_root_sha256') is not None else unknown('native_root_sha256 is null in task export; never fabricated'))))
    require(len({r['task_id'] for r in roots}) == len({r['blind_sha256'] for r in roots}) == 2, 'distinct roots')
    rows, native_roots = [], {}
    for mi, method in enumerate(matrix['methods']):
        sid, arm = method['source_id'], method['algorithm']
        model_run = next(r for r in comparison['runs'] if r['source_id'] == sid)
        consume = consumes[sid]
        plans = consume.get('plans', consume.get('rows'))
        plan_map = unique_rows(plans, ('source_slot', 'arm'), 'selected model plan')
        require(set(plan_map) == {(s, a) for a in model_run['arms'] for s in (0, 1)}, 'complete selected plan denominator')
        phids, rgids = method['physics_cohort_ids'], method['RGB_cohort_ids']
        require(len(phids) == len(rgids) == 1, 'one exact saved physical/RGB cohort per method')
        ph, rg = physical[phids[0]], rgb[rgids[0]]
        phmap = unique_rows(ph['rows'], ('root_slot', 'arm'), 'physical root/arm')
        rgmap = unique_rows(rg['rows'], ('source_slot', 'arm', 'role'), 'RGB root/arm/role')
        phcohort = next(c for c in matrix['physical_cohorts'] if c['id'] == phids[0])
        rgcohort = next(c for c in matrix['RGB_cohorts'] if c['id'] == rgids[0])
        require(set(phmap) == {(s, a) for s in (0, 1) for a in phcohort['arms']} and
                ph['counts'] == phcohort['counts'], 'physical complete cohort denominator/count receipt')
        require(set(rgmap) == {(s, a, role) for s in (0, 1) for a in rgcohort['arms'] for role in ('prefix5', 'terminal10')} and
                rg['resources'] == rgcohort['counts'], 'RGB complete cohort denominator/count receipt')
        for slot in (0, 1):
            index = 2 * mi + slot
            row, identity, task = comparison['rows'][index], roots[slot], tasks[slot]['task']
            plan, phys = plan_map[slot, arm], phmap[slot, arm]
            require(type(plan['source_slot']) is int and type(phys['root_slot']) is int,
                    'literal integer plan/physical slots, not bool')
            require(all(type(phys[k]) is bool for k in ('native_success', 'full_horizon')) and
                    type(phys['own_returned']) is int and phys['own_returned'] >= 0, 'literal native/horizon receipt types')
            require(row['source_id'] == sid and row['physical_source_id'] == phids[0], 'same model/physical source ID')
            for key in ('task_id', 'task_sha256', 'blind_sha256'):
                require(row[key] == plan[key] == identity[key], 'task/root/arm SHA join: ' + key)
            if 'task_contract_sha256' in plan:
                require(plan['task_contract_sha256'] == identity['task_contract_sha256'], 'plan task contract SHA')
            elif arm != ARMS[3]:
                require(False, 'task contract SHA missing outside sealed kino format')
            command_sha = typed(row['commands_sha256'], 'sha')
            require(command_sha['value'] == digest(plan['commands']), 'same literal commands SHA')
            selection = typed(row['physical_selection_digest'], 'sha')
            require(selection['value'] == phys['selection_digest'], 'same physical selection digest')
            model_q = typed(row['model_Q'])
            require(model_q['value'] == plan.get('model_terminal_cost', plan.get('terminal_cost')), 'same selected model Q')
            state_sha = digest(phys['root_state'])
            require(slot not in native_roots or native_roots[slot] == state_sha, 'same physical native root across arms')
            native_roots[slot] = state_sha
            outcomes = {key: typed(row.get(key, unknown('comparison metric absent')),
                'bool' if key in ('native_success', 'full_horizon') else 'scalar')
                for key in ('model_Q', 'model_root_Q', 'realQ_root', 'realQ_prefix5', 'realQ_terminal10',
                            'native_success', 'full_horizon', 'physical_returned_steps')}
            require(outcomes['native_success']['value'] == phys['native_success'] and
                    outcomes['full_horizon']['value'] == phys['full_horizon'] and
                    outcomes['physical_returned_steps']['value'] == phys['own_returned'], 'same literal native/horizon receipt')
            require(phys['final_metrics']['native_success'] == phys['native_success'], 'native metric boolean consistency')
            for key in ('native_distance', 'native_angle_term', 'object_position_error', 'wrapped_object_angle_error'):
                v = phys['final_metrics'].get(key)
                outcomes[key] = typed(unknown('saved native metric absent/null') if v is None else
                                      actual(v, 'literal saved physical final_metrics.' + key),
                                      'signed' if key == 'native_angle_term' else 'scalar')
            endpoints = {}
            for role in ('prefix5', 'terminal10'):
                image = rgmap[slot, arm, role]; item = image['provenance']
                require(image['status'] == 'encoded' and item['Q'] == 'identity192', 'encoded original identity metric')
                require(item['source_slot'] == item['root_slot'] == slot and item['arm'] == arm, 'RGB exact root/arm')
                for key in ('task_id', 'task_sha256', 'blind_sha256'):
                    require(item[key] == identity[key] and image[key] == identity[key], 'RGB task identity SHA')
                require(item['root'] == task['root_context']['latent_history'][-1] and item['goal'] == task['goal'], 'exact saved RGB root/goal')
                require(item['selection_digest'] == selection['value'] and item['own_plan_sha256'] == own_plan_sha(plan) and
                        item['compiler_report_sha256'] == model_run['provenance']['report']['sha256'], 'RGB exact selected model/physical plan SHA')
                require(image['metrics']['root_Q'] == outcomes['realQ_root']['value'] and
                        image['metrics']['actual_Q'] == outcomes['realQ_' + role]['value'], 'same actual RGB Q; no model/reported substitution')
                require(image['latent'] == row['RGB_latents'][role], 'same saved latent file SHA')
                hash_value(image['latent']['sha256'])
                endpoints[role] = deepcopy(image['latent'])
            context = dict(**identity, physical_root_state_sha256=state_sha,
                horizon=task['horizon'], packet_steps=task['packet_steps'], metric='identity_native192_MSE', with_target=False)
            # Upstream native-root field is provenance, including explicit unknown.
            binding = dict(**context, arm=arm, model_source_id=sid,
                full_sealed_contract_sha256=digest(row['contract']), commands_sha256=command_sha['value'],
                physical_selection_sha256=selection['value'], own_plan_sha256=own_plan_sha(plan), RGB_latents=endpoints)
            refs = ['shared_inherited_encoding', 'model:' + sid, 'consumer_CPU:' + sid,
                    'physics:' + phids[0], 'RGB:' + rgids[0]]
            rows.append(dict(binding=binding, binding_sha256=digest(binding), comparison_context=context,
                outcomes=outcomes, costs=cost_vector(method['root_receipts'][slot]['counters'], row['latency_seconds'], row['resources']),
                derivative_semantics=('scalar_loss_sweeps' if arm in (ARMS[0], ARMS[4]) else
                                      'scalar_output_VJP_directions' if arm == ARMS[5] else 'no_derivative_schedule'),
                ledger_refs=refs, charge_semantics='view_only_no_new_charge_no_refund',
                accounting_contract=deepcopy(method['contract_fields']),
                native_initial_metrics=deepcopy(phys['initial_metrics']), native_final_metrics=deepcopy(phys['final_metrics'])))
    pairwise = []
    for slot in (0, 1):
        selected = [r for r in rows if r['binding']['source_slot'] == slot]
        for a, b in itertools.combinations(selected, 2):
            pairwise.append(dict(source_slot=slot, arms=[a['binding']['arm'], b['binding']['arm']],
                binding_sha256=[a['binding_sha256'], b['binding_sha256']], **dominance(a, b)))
    accounting = ledger(matrix)
    require(all(set(r['ledger_refs']) <= {e['id'] for e in accounting['entries']} for r in rows), 'every charge view resolves')
    frontier = []
    for arm in ARMS:
        related = [p for p in pairwise if arm in p['arms']]
        frontier.append(dict(arm=arm, status='unknown' if any(p['status'] == 'unknown' for p in related) else
            'see_per_root_pairs_no_cross_root_average_frontier'))
    return dict(schema=STEM + '.pareto', status='saved_resource_outcome_join_not_admitted', rows=rows,
        pairwise=pairwise, frontier=frontier, ledger=accounting, required_cost_axes=list(AXES),
        objective_directions=OBJECTIVES, objective_scope='measured terminal RGB Q and literal native outcomes; modelQ diagnostic only',
        scope='six local baselines x two existing calibration roots; historical descriptive Pareto, not prospective equal budget',
        equal_total_compute=False, global_equal_budget_winner=False, independent_confirmation=False,
        author_reported_values_used=False, P1_admission=False, P2_admission=False,
        limitations=['No cross-unit total, cohort allocation, netting or refund.',
            'Incomplete costs mean unknown, NOT nondominated/frontier membership.',
            'Different action/proposal support and replay contracts retained; no global equal-budget winner.',
            'Task native_root_sha256 may be null; physical root-state SHA and latent context SHA are explicitly derived saved identities, not invented upstream hashes.',
            'No new model/physics/NPY recomputation; trusts pinned previously validated saved numerical metrics.'])


class SavedReader:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.refs = {}

    def raw(self, ref):
        p = Path(ref['path'])
        hash_value(ref['sha256'])
        require(p.is_absolute() and p == p.resolve() and p.is_relative_to(self.root) and p.is_file() and
                not p.is_symlink() and p.stat().st_size <= CAP, 'bounded nonsymlink saved input within results')
        data = p.read_bytes()
        require(len(data) <= CAP and sha(data) == ref['sha256'], 'saved input SHA mismatch')
        require(str(p) not in self.refs or self.refs[str(p)] == ref['sha256'], 'conflicting input SHA')
        self.refs[str(p)] = ref['sha256']
        return data

    def load(self, ref):
        raw = self.raw(ref)
        if ref['path'].endswith('.gz'):
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as f:
                raw = f.read(CAP + 1)
            require(len(raw) <= CAP, 'decompressed metadata cap')
        result = json.loads(raw, object_pairs_hook=budget.unique_pairs,
                            parse_constant=lambda _: require(False, 'nonfinite JSON'))
        encode(result)
        return result

    def recheck(self):
        for path, h in list(self.refs.items()): self.raw(dict(path=path, sha256=h))


def load_bundle(matrix_path, comparison_path):
    reader = SavedReader(ROOT / 'results')
    matrix = reader.load(dict(path=str(Path(matrix_path).absolute()), sha256=MATRIX_SHA))
    comparison = reader.load(dict(path=str(Path(comparison_path).absolute()), sha256=COMPARISON_SHA))
    require(matrix['input'] == dict(path=str(Path(comparison_path).absolute()), sha256=COMPARISON_SHA), 'exact comparison reference')
    tasks = []
    for h in budget.TASK_SHAS:
        paths = [p for p, v in comparison['bindings_sha256'].items() if v == h]
        require(len(paths) == 1, 'unique task export binding')
        tasks.append(reader.load(dict(path=paths[0], sha256=h)))
    consumes = {r['source_id']: reader.load(r['provenance']['consume']) for r in matrix['model_invocations']}
    physical = {r['id']: reader.load(r['provenance']['report']) for r in matrix['physical_cohorts']}
    rgb = {r['id']: reader.load(r['provenance']['report']) for r in matrix['RGB_cohorts']}
    return reader, (matrix, comparison, tasks, consumes, physical, rgb)


def markdown(report):
    def show(c):
        return 'NA' if c['status'] != 'actual' else str(c['value'])
    lines = ['# Saved 六 baseline 逐根资源 / 实测结果', '',
        '仅两个已知 calibration roots；不同单位不相加，不称全局等预算胜出。NA 不是零。', '',
        '| root | arm | predict completed | model Q（诊断） | real Q terminal10 | native success | native distance |',
        '|---|---|---:|---:|---:|---|---:|']
    for r in report['rows']:
        c = r['costs']['predict']; n = c['value']['completed'] if c['status'] == 'actual' else 'NA'
        lines.append('| ' + ' | '.join([str(r['binding']['source_slot']), r['binding']['arm'], str(n)] +
            [show(r['outcomes'][k]) for k in ('model_Q', 'realQ_terminal10', 'native_success', 'native_distance')]) + ' |')
    counts = {s: sum(p['status'] == s for p in report['pairwise']) for s in
              ('unknown', 'a_dominates_b', 'b_dominates_a', 'tie', 'nondominated_pair')}
    lines += ['', 'Pareto 完整成本轴门禁结果：`' + json.dumps(counts, sort_keys=True) + '`。', '',
        'unknown 不是 nondominated；不通过缺失值删轴来宣布 frontier。', '',
        f"共享账本 {len(report['ledger']['entries'])} 条唯一条目；逐根/standalone 为引用视图，不重复收费或退款。", '',
        '完整 SHA join、资源轴、缺口、原生指标和 overlap 关系见 pareto.json。', '']
    lines += ['- ' + s for s in report['limitations']]
    return ('\n'.join(lines) + '\n').encode()


def run(matrix_path, comparison_path, output):
    target = Path(output).absolute()
    require(target.parent == ROOT / 'results' and target.name.startswith(STEM + '_') and
            not target.exists() and not target.is_symlink(), 'new exclusive stem output under results only')
    reader, bundle = load_bundle(matrix_path, comparison_path)
    report = assemble(*bundle)
    report['inputs_sha256'] = deepcopy(reader.refs)
    report['implementation_sha256'] = {str(p): sha(p.read_bytes()) for p in (Path(__file__), Path(budget.__file__))}
    files = {'pareto.json': encode(report), 'pareto.md': markdown(report)}
    size = sum(map(len, files.values()))
    require(size <= OUTPUT_CAP, '2MiB output cap')
    require(shutil.disk_usage(target.parent).free >= FLOOR + size, '200MiB reserve')
    reader.recheck()
    for p, h in report['implementation_sha256'].items(): require(sha(Path(p).read_bytes()) == h, 'source changed')
    target.mkdir()
    for name, data in files.items():
        require(shutil.disk_usage(target).free >= FLOOR + len(data), 'write reserve; preserve partial output')
        with (target / name).open('xb') as f:
            require(f.write(data) == len(data), 'short write; preserve partial output')
            f.flush(); os.fsync(f.fileno())
    reader.recheck()
    fd = os.open(target, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return dict(status=report['status'], rows=len(report['rows']), pairs=len(report['pairwise']),
        unknown_pairs=sum(p['status'] == 'unknown' for p in report['pairwise']),
        files={str(target / n): sha(data) for n, data in files.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix', type=Path, default=ROOT / 'results/p1_moving_baseline_budget_matrix_20261027_v1_main01/budget_matrix.json')
    p.add_argument('--comparison', type=Path, default=ROOT / 'results/p1_moving_baseline_comparison_20261027_v3_main01/comparison.json')
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    print(encode(run(a.matrix, a.comparison, a.output)).decode(), end='')


if __name__ == '__main__': main()
