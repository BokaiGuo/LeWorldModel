"""Resource-only projection of pinned moving comparison v3; standard library only.

No outcomes, ownconsumer calls, neural/physics execution, or compute-equivalence
certification. Different module/API/directional counters are NEVER added.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil

NAME = 'p1_moving_baseline_budget_matrix_20261027_v1'
INPUT_SHA = '96cf27b143bb1a839edf7bc1e4de02f4a97e433c6475c4c731c78af76300d32e'
INPUT_SCHEMA = 'p1_moving_baseline_comparison_20261027_v3.comparison'
ARMS = ('moving_gradient_open_loop', 'CEM', 'kernel_faithful_mppi_nominal',
        'fixed_H2_kinodynamic_beam16x4', 'GRASP_H3_bounded', 'joint_H2_spectral_packet_mapped')
TASK_SHAS = ('a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b',
             '101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74')
CAP, INPUT_CAP, FLOOR = 256 * 1024, 1024 * 1024, 200 * 1024**2
COUNTERS = ('predict', 'forward_samples', 'action_encoder', 'pred_projector',
            'backward', 'grad_API', 'jacobian', 'jacobian_API', 'svd', 'bounded_solve')
SCALARS = ('AD_forward_in_predict', 'mathematical_reverse_directions')
CONTRACT_KEYS = ('history', 'history_source', 'horizon', 'horizon_packets', 'packet_steps',
    'horizon_semantics', 'raw_commands_shape', 'logits_shape', 'context_tokens_per_native_call',
    'packet_logit_cap', 'packet_cap', 'joint_cap', 'joint_radii', 'bounds', 'action_support',
    'bank', 'bank_size', 'beam_width', 'bank_mapping', 'raw', 'mapping', 'parameter_authority',
    'initialization', 'method', 'CEM', 'MPPI', 'selection', 'CEM_selection', 'MPPI_selection',
    'replay', 'calls', 'attempts', 'per_root_budget', 'total_budget', 'CPU_work_per_root',
    'consume_CPU_per_root', 'verification_CPU', 'linear_ridge', 'numerical', 'new_encoder',
    'new_projector', 'new_images', 'new_physics', 'AD_API_per_root', 'backward_accounting',
    'reverse_accounting', 'predict_accounting')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + '\n').encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def unknown(reason, status='unmeasured'):
    return dict(status=status, value=None, reason=reason)


def actual(value, basis):
    return dict(status='actual', value=deepcopy(value), basis=basis)


def count(value):
    require(isinstance(value, dict) and set(value) == {'attempted', 'completed'}, 'counter phases')
    require(all(type(v) is int and v >= 0 for v in value.values()), 'integer nonnegative counters')
    require(value['completed'] <= value['attempted'], 'completed exceeds attempted')
    return value


def cell(value, counter=False):
    require(isinstance(value, dict) and value.get('status') in ('actual', 'missing', 'partial', 'unmeasured')
            and 'value' in value, 'typed evidence cell')
    if value['status'] != 'actual':
        require(value['value'] is None and bool(value.get('reason')), 'unknown must be null with reason')
    elif counter:
        count(value['value'])
    else:
        require(type(value['value']) in (int, float) and math.isfinite(value['value'])
                and value['value'] >= 0, 'finite nonnegative scalar; bool forbidden')
    return deepcopy(value)


def total(cells, counter=False):
    checked = [cell(c, counter) for c in cells]
    if any(c['value'] is None for c in checked):
        return unknown('at least one fixed-root receipt unavailable; no partial sum presented as total')
    if counter:
        return actual({p: sum(c['value'][p] for c in checked) for p in ('attempted', 'completed')},
                      'sum SAME operation across both fixed roots, not across modules/APIs')
    return actual(sum(c['value'] for c in checked), 'sum SAME scalar unit across both fixed roots')


def resource_rows(row):
    enclosing = row['resources']
    require(isinstance(enclosing, dict), 'resource envelope')
    if enclosing.get('status') != 'actual':
        cell(enclosing)
        return {k: unknown('model resource envelope unavailable', enclosing['status'])
                for k in COUNTERS + SCALARS + ('predict_API', 'scalar_output_VJP_directions', 'scalar_loss_reverse_sweeps')}
    value = enclosing['value']
    require(isinstance(value, dict), 'actual resource map')
    result = {k: cell(value.get(k, unknown('not separately metered')), k in COUNTERS)
              for k in COUNTERS + SCALARS}
    for name, rawmap, key in [('predict_API', 'raw_predict_API_resources', 'predict'),
                              ('scalar_output_VJP_directions', 'raw_resources', 'vjp_directions')]:
        raw = value.get(rawmap, {}).get(key)
        result[name] = unknown('no separate receipt; absence is not zero') if raw is None else actual(count(raw), rawmap + '/' + key)
    result['scalar_loss_reverse_sweeps'] = (deepcopy(result['mathematical_reverse_directions'])
        if row['arm'] in (ARMS[0], ARMS[4]) else unknown('not a separately reported scalar-loss reverse unit'))
    return result


def project_cohorts(report, key):
    result, covered = [], set()
    for run in report[key]:
        arms = run['arms']
        require(arms and len(set(arms)) == len(arms) and set(arms) <= set(ARMS), 'cohort arms')
        require(not covered.intersection(arms), 'duplicate cohort would double count')
        covered.update(arms)
        resources = deepcopy(run['counts'] if key == 'physical_runs' else run['resources'])
        for v in resources.values():
            count(v)
        result.append(dict(id=run.get('source_id', digest(run['provenance'])), arms=deepcopy(arms),
            provenance=deepcopy(run['provenance']), counts=resources,
            latency_seconds=cell(run['latency_seconds']), scope='whole_cohort_once_not_per_arm_or_root'))
    return result


def assemble(report, reference):
    """Pure projection; RAM fixtures may call this without file IO or SHA pretense."""
    require(report['schema'] == INPUT_SCHEMA and report['fixed_baselines'] == list(ARMS)
            and report['cohort'] == list(ARMS), 'fixed six-method v3 cohort only')
    require(all(report[k] is False for k in ('equal_total_compute', 'equal_search_domain',
            'official_reproduction', 'independent_confirmation', 'P1_admission', 'P2_admission')), 'claim boundaries')
    rows = report['rows']
    require([(x['arm'], x['source_slot']) for x in rows] == [(a, s) for a in ARMS for s in (0, 1)]
            and all(type(x['source_slot']) is int for x in rows), 'all six methods x two roots in fixed order')
    runs = {r['source_id']: r for r in report['runs']}
    require(len(runs) == len(report['runs']), 'unique model invocations')
    methods = []
    for arm in ARMS:
        selected = [r for r in rows if r['arm'] == arm]
        require(all(r['task_sha256'] == TASK_SHAS[r['source_slot']] for r in selected), 'original moving tasks only')
        source_id = selected[0]['source_id']
        require(all(r['source_id'] == source_id for r in selected) and source_id in runs, 'same model source across roots')
        run = runs[source_id]
        require(arm in run['arms'], 'model run arm join')
        require(selected[0]['contract'] == selected[1]['contract'] == run['contract'], 'same sealed method contract')
        roots = [dict(source_slot=r['source_slot'], task_sha256=r['task_sha256'],
                      counters=resource_rows(r)) for r in selected]
        totals = {k: total([r['counters'][k] for r in roots], k not in SCALARS + ('scalar_loss_reverse_sweeps',))
                  for k in roots[0]['counters']}
        c = selected[0]['contract']
        fields = {k: deepcopy(c[k]) for k in CONTRACT_KEYS if k in c}
        # Dimensions are explicit contract evidence, not a claim of shared support.
        h = c.get('horizon', c.get('horizon_packets'))
        if h is None and c.get('logits_shape') == [2, 10] and c.get('raw_commands_shape') == [10, 2]:
            h = 2
        horizon = actual(h, 'sealed horizon/horizon_packets or explicit [2,10] logits + [10,2] raw commands') if h is not None else unknown('no explicit future-packet evidence')
        methods.append(dict(algorithm=arm, implementation='local_not_official_reproduction',
            source_id=source_id, root_receipts=roots, two_root_totals=totals,
            future_packets=horizon, contract_fields=fields,
            contract_pointer=f'/runs/{list(runs).index(source_id)}/contract',
            model_invocation_latency=cell(run['latency_seconds']), latency_scope='whole_model_invocation_not_per_method',
            physics_cohort_ids=[], RGB_cohort_ids=[]))
    require(set(runs) == {m['source_id'] for m in methods}, 'no extra model invocation')
    physical, rgb = project_cohorts(report, 'physical_runs'), project_cohorts(report, 'RGB_runs')
    for m in methods:
        for label, cohorts in [('physics', physical), ('RGB', rgb)]:
            m[label + '_cohort_ids'] = [r['id'] for r in cohorts if m['algorithm'] in r['arms']]
        # A single-arm cohort is exact; never allocate a multi-arm cohort.
        own = [r for r in physical if r['arms'] == [m['algorithm']]]
        m['per_method_physics'] = (actual(own[0]['counts'], 'entire single-arm cohort; same receipt, not extra cost')
            if len(own) == 1 else unknown('missing or multi-arm cohort; no proportional allocation'))
    inherited = deepcopy(report['inherited_encoding_shared_once'])
    require(inherited['status'] == 'actual' and set(inherited['value']) == {'encoder', 'projector'}, 'shared encoding receipt')
    for value in inherited['value'].values():
        count(value)
        require(value == dict(attempted=8, completed=8), 'original eight-image shared encoding once')
    for m in methods:
        m['standalone_inherited_encoding'] = deepcopy(inherited)
    pairs = []
    for a, b in itertools.combinations(methods, 2):
        av, bv = (m['two_root_totals']['predict']['value'] for m in (a, b))
        same = None if av is None or bv is None else av == bv
        pairs.append(dict(methods=[a['algorithm'], b['algorithm']], same_predict_counter=same,
            equal_total_compute_established=False, equal_search_domain_established=False,
            remaining_contracts=['predeclared matched total-compute accounting (AD, CPU, batching and latency)',
                'matched action/proposal support and horizon/replay/feedback schedule',
                'full pipeline encoding sharing versus standalone cost convention']))
    return dict(schema=NAME + '.matrix', status='saved_resource_projection_not_admitted',
        input=deepcopy(reference), input_manifest=deepcopy(report['manifest']), methods=methods,
        model_invocations=[dict(source_id=r['source_id'], arms=deepcopy(r['arms']),
            provenance=deepcopy(r['provenance']), latency_seconds=cell(r['latency_seconds'])) for r in report['runs']],
        physical_cohorts=physical, RGB_cohorts=rgb, inherited_encoding_shared_once=inherited,
        pairwise_budget_contracts=pairs, equal_total_compute=False, P1_admission=False,
        warnings=['AD forwards are a subset of predict, not extra forwards.',
            'predict/action_encoder/pred_projector are overlapping modules, never summed as forwards.',
            'backward/grad_API/J/VJP describe overlapping derivative work; scalar loss sweeps != output VJP directions.',
            'kino full16+4x16 enumeration is not adaptive-budget advantage; discrete repeated packets differ from per-time logits.',
            'GRASP H3 means observed history, not three future packets; nominal MPPI is local nonofficial.',
            'SRA joint bank then packet projection changes support; not exact doubly constrained ridge.',
            'Cohort latency and physics are not divided among arms; standalone inherited8 is not summed six times.',
            'Fixed two roots are calibration only; no outcome or scientific winner is emitted.'])


def markdown(matrix):
    def shown(c):
        v = c['value']
        return 'NA' if v is None else (str(v['completed']) + '/' + str(v['attempted']) if isinstance(v, dict) else str(v))
    lines = ['# 固定两根六方法 saved 资源矩阵', '', '计数为两根合计：完成/尝试；NA不是0。不等总算力，不输出科学胜负。', '',
        '| 方法 | predict | AE | pred-projector | ADforward子集 | scalar backward | grad API | J | output VJP方向 | SVD | future packets |',
        '|---|---|---|---|---|---|---|---|---|---|---|']
    for m in matrix['methods']:
        t = m['two_root_totals']
        lines.append('| ' + ' | '.join([m['algorithm']] + [shown(t[k]) for k in ('predict', 'action_encoder',
            'pred_projector', 'AD_forward_in_predict', 'backward', 'grad_API', 'jacobian', 'scalar_output_VJP_directions', 'svd')]
            + [shown(m['future_packets'])]) + ' |')
    lines += ['', '## 物理 / RGB / 历史编码分账', '', '逐方法只关联cohort ID，不将整次物理计数复制成该方法成本。', '']
    for kind, cohorts in [('physical', matrix['physical_cohorts']), ('RGB', matrix['RGB_cohorts'])]:
        for r in cohorts:
            lines.append(f"- {kind} `{r['id']}`；{', '.join(r['arms'])}；整次计数 `{json.dumps(r['counts'], sort_keys=True)}`。")
    lines += ['', '历史encoder8/projector8共享只计一次；standalone视图每方法列继承8/8，不相加。', '',
        '## 支持域及equal-budget剩余合同', '',
        '全部15方法对均未建立等总算力/同搜索域。预测计数是否相等只是一项字面比较，不足以证明公平预算。',
        'JSON保留每方法原始时域/动作映射/bank/replay合同字段及原comparison指针；缺字段不推断为零或相同。', '']
    lines += ['- ' + x for x in matrix['warnings']]
    return ('\n'.join(lines) + '\n').encode()


def unique_pairs(pairs):
    d = {}
    for k, v in pairs:
        require(k not in d, 'duplicate JSON key')
        d[k] = v
    return d


def read_pinned(path, expected):
    require(expected == INPUT_SHA, 'only approved comparison SHA accepted')
    p = Path(path).absolute()
    require(p.stat().st_size <= INPUT_CAP, 'input cap')
    data = p.read_bytes()
    require(len(data) <= INPUT_CAP and hashlib.sha256(data).hexdigest() == expected, 'comparison external SHA mismatch')
    r = json.loads(data, object_pairs_hook=unique_pairs,
                   parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    return r, dict(path=str(p), sha256=expected)


def run(path, expected, output):
    report, reference = read_pinned(path, expected)
    matrix = assemble(report, reference)
    files = {'budget_matrix.json': canonical(matrix), 'budget_matrix.md': markdown(matrix)}
    size = sum(map(len, files.values()))
    require(size <= CAP, 'output cap 256KiB')
    target = Path(output).absolute()
    require(not target.exists() and target.parent.is_dir(), 'NEW output directory under existing parent')
    require(shutil.disk_usage(target.parent).free >= FLOOR + size, '200MiB reserve')
    # Recheck the only input before publishing any output; no transitive artifact IO.
    read_pinned(path, expected)
    target.mkdir()
    for name, data in files.items():
        require(shutil.disk_usage(target).free >= FLOOR + len(data), 'per-write reserve; preserve partial output')
        with (target / name).open('xb') as f:
            require(f.write(data) == len(data), 'short write; preserve partial output')
            f.flush()
            os.fsync(f.fileno())
    fd = os.open(target, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return dict(status='budget_matrix_written_not_admitted', bytes=size,
        files={str(target / k): hashlib.sha256(v).hexdigest() for k, v in files.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison', required=True)
    p.add_argument('--comparison-sha', default=INPUT_SHA)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    print(canonical(run(a.comparison, a.comparison_sha, a.output)).decode(), end='')


if __name__ == '__main__':
    main()
