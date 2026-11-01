"""Current finite P5 dependency/wiring manifest, saved data only, stdout only.

0 = valid blocked inventory, 2 = invalid evidence, 3 = --require-admission blocked.
No execution authority or general-purpose scientific admission consumer. To add
new outcomes/protocols, review a NEW version; affirmative flags cannot unlock it.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import stat

from . import p5_native_stage_admission_20261027_v1 as native
from . import p2_baseline_evidence_inventory_20261027_v1 as baseline

NAME = 'p5_closure_readiness_20261027_v1'
ROOT = Path(__file__).resolve().parents[3]
CAP = 16 * 1024 * 1024
TOTAL_CAP = 64 * 1024 * 1024
RESERVE = 200 * 1024 * 1024
P1 = 'le-wm-main/results/p3_goal_patch_consistent_compiler_20261027_v1_cuda_main_v1/report.json'
P1_SHA = '64fd87f48f73674d98b8eb2ddce7415507bf1e6d457df8e78ad91da1c15c4fe1'
ROOT_DOC = 'docs/p1_conditional_root_preconditions_20261027_v1.md'
ROOT_DOC_SHA = '60238c499e093efd6d90c543481955405b834ecfa2543a7bdc1e6203db2fd1ac'
STRICT = 'le-wm-main/results/p5_jepa_strict_metadata_20261027_v3/pusht_native_strict_cuda_main_v1.json'
EQUIV = 'le-wm-main/results/p5_jepa_official_equivalence_20261027_v6/pusht_official_vs_hf_cuda_main_v1.json'
PINNED = {
    P1: P1_SHA, ROOT_DOC: ROOT_DOC_SHA,
    STRICT: 'c1498ecc867523b7b10cf9028543151e402ce593b09ac8a85bca58b55cb450e0',
    EQUIV: '78d5e5d8584e6dc78efe4e7b70c2a7d64e1247d84494d929afc07f61d8343b41',
}
# These documents may receive new appendices from main. Bind their current bytes
# and exact authority rows, rather than mistaking historical summaries for runs.
AUTHORITIES = {
    'plan': ('docs/SPECTRAL_REACHABILITY_ATLAS_RESEARCH_PLAN_20261023.md', '| P5 | 其他冻结模型和任务类别 |'),
    'closure': ('docs/D2P_RESEARCH_CLOSURE_AUDIT_20261025.md', '| P5：跨模型/跨任务 |'),
    'matrix': ('docs/D2P_RESEARCH_REQUIREMENT_EVIDENCE_MATRIX_20261026.md', '| P5 项目跨模型/任务复制 |'),
    'dependency_chain': ('docs/D2P_CVPR_CRITICAL_PATH_20261026.md', '依赖链：输入/度量合同与任务前提'),
}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def finite_tree(value):
    if isinstance(value, float):
        require(math.isfinite(value), 'nonfinite number')
    elif isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)


def decode(raw):
    def unique(pairs):
        out = {}
        for k, v in pairs:
            require(k not in out, 'duplicate JSON key')
            out[k] = v
        return out
    value = json.loads(raw, object_pairs_hook=unique)
    finite_tree(value)
    return value


class Reader:
    def __init__(self, root):
        self.root = Path(root).absolute()
        self.bindings = {}
        self.bytes = 0

    def raw(self, relative, expected=None):
        rel = Path(relative)
        require(not rel.is_absolute() and '..' not in rel.parts, 'nonlocal path')
        p = self.root / rel
        require(not any(q.is_symlink() for q in (p, *p.parents)), 'symlink input')
        require(stat.S_ISREG(p.stat().st_mode) and p.stat().st_size <= CAP, 'not bounded regular file')
        with p.open('rb') as stream:
            raw = stream.read(CAP + 1)
        self.bytes += len(raw)
        require(len(raw) <= CAP and self.bytes <= TOTAL_CAP, 'read budget')
        sha = digest(raw)
        require(expected is None or expected == sha, 'SHA mismatch: ' + relative)
        require(relative not in self.bindings or self.bindings[relative]['sha256'] == sha, 'input drift')
        self.bindings[relative] = dict(sha256=sha, bytes=len(raw))
        return raw

    def read(self, relative, expected=None):
        return decode(self.raw(relative, expected))

    def stable(self):
        for rel, rec in list(self.bindings.items()):
            self.raw(rel, rec['sha256'])


def task_slots():
    """Requirements transcribed from authority tables; no fitted numeric gates."""
    rows = [
        ('core_freeze', [], 'matrix', ['core_algorithm_source_sha256', 'component_scope_P1_P2_P3_P4', 'protocol_sha256'],
         'Bind admitted core algorithm, not just this diagnostic compiler.'),
        ('conditional_roots', [], 'matrix', ['finite_pool_and_order', 'source_episode_exclusions', 'calibration_outcome_split',
          'same_goal_action_domain', 'direct_gradient_plateau_trajectory', 'legal_two_stage_reach_witness', 'predeclared_threshold_authority'],
         'Opened four calibration roots lack direct plateau and two-stage task-reach certification; no beta tuning or automatic pool expansion.'),
        ('local_strong_controls', [], 'closure', ['methods_and_faithfulness', 'checkpoint_root_goal_bindings', 'feedback_and_action_support',
          'predict_encoder_AD_Jacobian_SVD_search_walltime_caps', 'paired_outcome_and_statistics'],
         'Existing calibration runs and paper cells do not provide full matched independent controls.'),
        ('formal_P1', ['conditional_roots', 'local_strong_controls'], 'matrix', ['calibration_freeze_receipt', 'independent_outcome_source_ids',
          'same_root_non_greedy_capability_task_benefit', 'original_admission_statistics'],
         'Same-root mechanism conjunction and independent physical benefit are not established by saved C1/FD.'),
        ('used_components', ['core_freeze', 'formal_P1'], 'dependency_chain', ['P2_composition_increment', 'P3_geometry_quality_cost',
          'P4_real_RGB_feedback_benefit', 'E1_E2_E3_required_scope'],
         'Freeze and admit each component actually replicated; full closure still needs P2/P3/P4 and E1-E3, not borrowed SRA gates.'),
        ('architecture_contract', [], 'plan', ['independent_dynamics_architecture', 'source_revision_and_local_edits', 'checkpoint_provenance_license',
          'RGB_proprio_privilege_disclosure', 'normalization_action_history', 'pretraining_extra_learning_cost'],
         'JEPA native frozen state identities exist; strict load/equivalence do not authenticate official weights or close interface provenance.'),
        ('native_P0_task', ['architecture_contract', 'conditional_roots'], 'matrix', ['model_specific_CUDA_FD_replay',
          'action_scale_and_order', 'task_metric_alignment', 'task_reach_prerequisites'],
         'C1/JVP/FD completed on one opened root; model-specific task admission remains unbound. Zero suffix contact is diagnostic, not a universal P5 gate.'),
        ('P5_independent_replication', ['formal_P1', 'used_components', 'native_P0_task', 'local_strong_controls'], 'matrix',
         ['preregistered_architecture_and_tasks', 'same_frozen_core', 'independent_root_panel', 'mechanism_and_real_task_gain',
          'paired_success_cost_time_budget', 'all_failures_retained', 'training_free_disclosure'],
         'Independent frozen architecture replication, not a single checkpoint multi-seed diagnostic.'),
        ('same_model_multitask', ['formal_P1', 'used_components'], 'closure', ['one_checkpoint_per_model',
          'goal_horizon_action_restriction_matrix', 'independent_roots_each_cell', 'real_success_cost_time_resource'],
         'E3 separate from independent architecture P5; neither substitutes for the other.'),
    ]
    return [dict(id=k, depends_on=deps, authority=src, required_manifest_fields=fields,
                 bound_completion_artifact=None, status='unbound_scientific_requirement', reason=reason)
            for k, deps, src, fields, reason in rows]


def graph(rows):
    ids = [r['id'] for r in rows]
    require(len(set(ids)) == len(ids), 'duplicate dependency slot')
    pending = {r['id']: set(r['depends_on']) for r in rows}
    require(all(d in pending for deps in pending.values() for d in deps), 'unknown dependency')
    order = []
    while pending:
        ready = sorted(k for k, deps in pending.items() if deps <= set(order))
        require(bool(ready), 'cyclic dependency')
        order.extend(ready)
        for k in ready:
            del pending[k]
    return order


def evidence_summary(n, inv, p1, strict, equiv):
    require(n['all_checks'] is True and n['check_count'] == 45 and
            len(n['checks']) == 45 and all(v is True for v in n['checks'].values()), 'native arithmetic incomplete')
    require(n['P5_admission'] is False and n['C2_established'] is False, 'native admission contamination')
    require(inv['scientific_admission'] is False and not inv['missing_assets'], 'baseline inventory invalid')
    require(p1['status'] == 'cuda_completed_h2_prerequisite' and p1['model_unchanged'] is True,
            'P1 diagnostic not completed')
    require(all(p1[k] is False for k in ('P1_admission', 'P3_admission', 'P4_admission')), 'P1 flags changed')
    require(p1['contract']['new_independent_roots'] == 0, 'different P1 cohort contract')
    pairs = {(r['root_sha256'], r['arm']) for r in p1['rows']}
    roots = sorted({r['root_sha256'] for r in p1['rows']})
    require(len(roots) == 4 and len(p1['rows']) == len(pairs) == 8 and
            pairs == {(r, a) for r in roots for a in ('old_Q', 'goal_patch_Q')}, 'P1 complete four-root two-arm census')
    require(all(r['status'] == 'completed' and len(r['compilation']['candidates']) == 10 for r in p1['rows']), '80 candidate census')
    methods = inv['methods']
    require(len(methods) == 10 and len({m['method'] for m in methods}) == 10, 'baseline method census')
    for m in methods:
        require(m['total_compute_matched'] is False and m['direct_paper_comparison'] is False and
                m['paper_protocol_reproduced'] is False, 'reported/local comparison contamination')
        if m['method'] in baseline.METHODS:
            require(len(m['local_evidence']) == 4 and
                    {r['root_sha256'] for r in m['local_evidence']} == set(roots), 'baseline/P1 root wiring')
    require(strict['status'] == 'native_dynamics_strict_load_only' and strict['P5_admission'] is False,
            'strict-load evidence scope changed')
    require(equiv['status'] == 'paired_equivalence_failed' and equiv['P5_admission'] is False,
            'equivalence failure cannot be promoted')
    return dict(native=dict(arithmetic_checks=45, input_files=n['input_files_checked'],
                            root_namespace='native_public_state_identity', root=n['root_identity_sha256'],
                            model_states=n['model_state_identities'], stages=n['stage_gates'],
                            observations=n['observations']),
                P1=dict(root_namespace='LeWM_calibration_RGB_identity', roots=roots, candidates=80,
                        protocol_id=p1['protocol_id'], independent_roots=0,
                        plateau='unverified', two_stage_task_reach='unverified',
                        mechanism_audit_recomputed=False, source=ROOT_DOC),
                baseline_checkpoint_bindings_recorded_not_rehashed=inv['checkpoint_bindings_recorded_not_rehashed'],
                baselines=[dict(method=m['method'], local_records=len(m['local_evidence']),
                                reported_cells=m['paper_reported_cells'], scope=m['local_roots_scope'],
                                budget_matched=False, gaps=m['missing_or_unmatched']) for m in methods],
                architecture=dict(strict_load=strict['status'], encoder_equivalence=equiv['status'],
                                  official_weight_authentication='unknown', independent_replication='unbound'))


def produce(root=ROOT):
    reader = Reader(root)
    # Bind actual validator implementations as well as inputs, without executing
    # any source located by manifests. Only the two explicit stdlib modules run.
    for filename in (NAME, native.NAME, baseline.NAME, 'd2p_reported_baselines'):
        reader.raw('le-wm-main/experiments/spectral_reachability/' + filename + '.py')
    authorities = {}
    for key, (path, marker) in AUTHORITIES.items():
        lines = reader.raw(path).decode().splitlines()
        hits = [(i + 1, line) for i, line in enumerate(lines) if line.startswith(marker)]
        require(len(hits) == 1, 'ambiguous/missing stage authority: ' + key)
        authorities[key] = dict(path=path, line=hits[0][0], excerpt=hits[0][1])
    reader.raw(ROOT_DOC, ROOT_DOC_SHA)
    p1 = reader.read(P1, P1_SHA)
    strict, equiv = reader.read(STRICT, PINNED[STRICT]), reader.read(EQUIV, PINNED[EQUIV])
    n = native.audit(Path(root) / 'le-wm-main')
    require(n.get('all_checks') is True, 'native saved-data recheck: ' + str(n.get('error', n.get('failed_checks'))))
    for rec in n['snapshot_files'].values():
        reader.read('le-wm-main/' + rec['path'], rec['sha256'])
    g = reader.read('le-wm-main/' + native.SPECS['geometry'][0], native.SPECS['geometry'][1])
    for path, sha in g['input_bindings'].items():
        rel = Path(path).relative_to(Path(root).absolute()).as_posix()
        reader.read(rel, sha)
    inv = baseline.build_inventory(Path(root) / 'le-wm-main')
    for path, rec in inv['inputs'].items():
        # Reparse all baseline JSON with strict duplicate/nonfinite handling.
        rel = 'le-wm-main/' + path
        if path.endswith('.json'):
            reader.read(rel, rec['sha256'])
        else:
            reader.raw(rel, rec['sha256'])
    summary = evidence_summary(n, inv, p1, strict, equiv)
    ready = reader.read('le-wm-main/' + native.SPECS['ready'][0], native.SPECS['ready'][1])
    summary['native']['opened_execution_contract'] = {k: ready['contract'][k] for k in
        ('seed', 'context_frames', 'decision_frame', 'terminal_frame', 'input_layout',
         'native_action_support', 'support_scope', 'evaluation')}
    rows = task_slots()
    # Every open obligation has concrete available support, not an imaginary
    # completed result. These paths resolve through the file-hash ledger below.
    refs = {
        'core_freeze': [P1],
        'conditional_roots': [ROOT_DOC, P1],
        'local_strong_controls': ['le-wm-main/' + baseline.COMPARISON, 'le-wm-main/' + baseline.GRASP,
                                  'le-wm-main/' + baseline.EXPORT + 'report.json'],
        'formal_P1': [ROOT_DOC, P1, 'le-wm-main/' + native.SPECS['outcome'][0]],
        'used_components': [P1, AUTHORITIES['matrix'][0]],
        'architecture_contract': [STRICT, EQUIV, 'le-wm-main/' + native.SPECS['compiler'][0]],
        'native_P0_task': ['le-wm-main/' + native.SPECS[k][0] for k in ('compiler', 'fd', 'fd_audit', 'geometry')],
        'P5_independent_replication': ['le-wm-main/' + native.SPECS['outcome'][0]],
        'same_model_multitask': [AUTHORITIES['closure'][0], AUTHORITIES['matrix'][0]],
    }
    for row in rows:
        row['available_support_not_completion'] = refs[row['id']]
        require(all(p in reader.bindings for p in refs[row['id']]), 'unbound support path')
    order = graph(rows)
    reader.stable()
    return dict(schema=NAME, status='saved_evidence_valid_P5_blocked',
                validity_scope='finite explicit snapshot; unbound != proven nonexistent anywhere',
                scientific_admission=False, execution_authorized=False, theory_failure=False,
                new_neural_calls=0, new_physical_steps=0, training=False,
                authorities=authorities, evidence=summary, bindings=reader.bindings,
                bytes_read=reader.bytes,
                verification_limits=['No full upstream source/weights/NPY/journal authentication',
                    'Native 101-file JSON geometry closure rerun; baseline existing inventory arithmetic rerun',
                    'P1 candidate census only; do not duplicate mechanism audit or infer theoretical failure',
                    'No same-root pairing between JEPA native root and LeWM calibration roots',
                    'Paper reported cells remain context only, never local paired controls'],
                execution_manifest=dict(kind='dependency_evidence_manifest_NOT_GPU_job',
                    execution_authorized=False, launch_command=None, topological_order=order,
                    requirements=rows,
                    offline_frontier=['Source/license/local-edit and model-interface provenance contract',
                        'Declare finite independent root protocol; freeze plateau/reach predicates without outcome tuning',
                        'Seal faithful baseline feedback/action/resource contracts and statistics',
                        'Declare component scope and separate same-model goal/horizon/action task matrix'],
                    main_only_after_admission=['New calibration/outcome CUDA and physical execution',
                        'Independent architecture mechanism AND task benefit replication'],
                    unknown_values_policy='No numeric thresholds invented; bind existing preregistered authority or stop'))


def verify_manifest(saved, current):
    canonical = lambda obj: json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False)
    require(canonical(saved) == canonical(current), 'manifest stale, incomplete, or modified; regenerate/review NEW snapshot')
    require(saved['scientific_admission'] is False and saved['execution_authorized'] is False,
            'offline manifest is not authority')
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-manifest', help='project-relative existing same-stem JSON; read-only')
    parser.add_argument('--require-admission', action='store_true')
    args = parser.parse_args(argv)
    try:
        require(shutil.disk_usage(ROOT).free >= RESERVE, 'disk reserve below 200 MiB; do not delete or run')
        report = produce()
        if args.verify_manifest:
            require(Path(args.verify_manifest).name == NAME + '.json', 'wrong manifest stem')
            verify_manifest(Reader(ROOT).read(args.verify_manifest), report)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
        return 3 if args.require_admission else 0
    except (ValueError, OSError, KeyError, TypeError, IndexError, RecursionError, OverflowError) as exc:
        print(json.dumps(dict(schema=NAME, status='invalid_evidence', scientific_admission=False,
                              execution_authorized=False, error=str(exc)), ensure_ascii=False))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
