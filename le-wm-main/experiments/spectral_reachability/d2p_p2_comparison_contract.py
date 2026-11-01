"""Read-only P2 paired-data validator; neither neural replay nor stage admission.

The independently retained manifest digest is mandatory. Hash consistency does
not establish prospective freezing, CUDA execution, P1, or physical efficacy.
This module imports no neural runtime and never falls back to CPU inference.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

from experiments.spectral_reachability import d2p_composition_witness_v2 as witness


BUDGET_KEYS = {'predict_calls', 'predict_samples', 'candidate_evaluations',
               'planning_predict_calls', 'planning_predict_samples',
               'replay_predict_calls', 'replay_predict_samples',
               'encoder_calls', 'backward_calls', 'optimizer_steps', 'flops',
               'attempted_expansions',
               'physical_env_steps', 'wall_time_limit_s'}
BASELINES = {'flat_search', 'no_composition_sra'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _identity(value):
    _require(isinstance(value, str) and bool(value.strip()), 'nonempty identity required')
    return value


def _equal(left, right):
    # Exact portable JSON, including numeric representation; no tolerances.
    return witness._canonical(left) == witness._canonical(right)


def _budget(value):
    _require(isinstance(value, dict) and set(value) == BUDGET_KEYS,
             'complete explicit budget required')
    for key, number in value.items():
        if key == 'wall_time_limit_s':
            _require(type(number) in (int, float) and math.isfinite(number) and number > 0,
                     'positive finite wall time limit required')
        else:
            _require(type(number) is int and number >= 0, 'nonnegative integer budget required')
    _require(value['candidate_evaluations'] > 0, 'nonempty candidate budget required')
    for unit in ('calls', 'samples'):
        _require(value['predict_'+unit] == value['planning_predict_'+unit] + value['replay_predict_'+unit],
                 'total predict accounting must include planning and full replay')


def _non_admission(value):
    for key in ('P2_admission', 'physical_efficacy_verified'):
        if key in value:
            _require(value[key] is False, 'input cannot claim stage admission or physical efficacy')


def _check_candidate(candidate, case, slot, arm, manifest):
    _require(candidate['candidate_id'] == slot['candidate_ids'][arm],
             'candidate identity differs from frozen slot')
    receipt, edges = candidate['receipt'], candidate['edges']
    _require(_equal(receipt['root'], case['root']), 'root/goal/full-context mismatch')
    _require(_equal(receipt['provenance'], manifest['provenance']), 'provenance mismatch')
    _require(_equal(receipt['action_domain'], manifest['action_domain']), 'action domain mismatch')
    _require(len(receipt['actions']) == slot['primitive_steps'], 'primitive total length mismatch')
    _require(isinstance(edges, list) and bool(edges), 'explicit action witness edges required')
    if arm == 'composition':
        _require(len(edges) >= 2, 'composition requires multiple primitive edges')
    else:
        _require(len(edges) == 1, 'no-composition baseline requires a single flat action witness')
    for edge in edges:
        # v2 edges inherit goal binding via the receipt root. Optional explicit
        # goal fields must agree too; they cannot override that binding.
        if 'goal_rgb_sha256' in edge:
            _require(edge['goal_rgb_sha256'] == case['root']['goal_rgb_sha256'],
                     'edge goal mismatch')
    outputs = iter(edge['output_context'] for edge in edges)

    def declared_output_only(context, actions):
        # This is NOT model replay. compose validates action/domain/context
        # chains and rebuilds hashes; the recorded outputs remain unverified.
        return deepcopy(next(outputs))

    rebuilt = witness.compose(root=receipt['root'], provenance=receipt['provenance'],
                              action_domain=receipt['action_domain'], edges=edges,
                              replay=declared_output_only, max_edges=receipt['max_edges'],
                              max_actions=receipt['max_actions'])
    _require(_equal(rebuilt, receipt), 'exact action witness/receipt consistency failed')
    return receipt


def validate_comparison(data, *, expected_manifest_sha256, p1_evidence=None):
    """Validate all declared cases, slots, arms and optional physical records.

    Raises ValueError on malformed or unmatched data. A valid return is only
    structural validation. P1 evidence is bound as an opaque external report,
    never trusted merely because it says passed. An independent stage auditor
    must interpret it; this validator always keeps P2 admission false.
    """
    try:
        return _validate(deepcopy(data), expected_manifest_sha256, deepcopy(p1_evidence))
    except (KeyError, TypeError, IndexError, StopIteration, OverflowError, RecursionError) as exc:
        raise ValueError('malformed P2 comparison contract') from exc


def _validate(data, expected_manifest_sha256, p1_evidence):
    witness._sha(expected_manifest_sha256)
    witness._canonical(data)
    _non_admission(data)
    manifest = data['manifest']
    _require(witness.digest(manifest) == expected_manifest_sha256, 'external manifest seal mismatch')
    _require(manifest['schema'] == 'd2p_p2_manifest.v1', 'unsupported manifest schema')
    _non_admission(manifest)
    _require(manifest['selection'] == dict(rule='pre_outcome_frozen', outcome_accessed=False),
             'pre-outcome frozen selection required; no outcome filtering')
    # Equality alone would admit 0 as False.
    _require(manifest['selection']['outcome_accessed'] is False, 'outcome access must be false')
    arms = manifest['arms']
    _require(isinstance(arms, list) and all(isinstance(a, str) for a in arms), 'arm list required')
    _require(len(set(arms)) == len(arms) and 'composition' in arms
             and bool(set(arms) & BASELINES) and set(arms) <= BASELINES | {'composition'},
             'composition and at least one explicit no-composition baseline required')
    _budget(manifest['budget'])
    cases = manifest['cases']
    _require(isinstance(cases, list) and bool(cases), 'nonempty frozen cohort required')
    expected, bound_roots = {}, set()
    for case in cases:
        case_id = _identity(case['case_id'])
        root = case['root']
        _identity(root['root_id'])
        for key in ('root_rgb_sha256', 'goal_rgb_sha256'):
            witness._sha(root[key])
        witness._context(root['context'])
        binding = (root['root_id'], root['root_rgb_sha256'], root['goal_rgb_sha256'])
        _require(binding not in bound_roots, 'duplicate root/goal case; no pseudoreplication')
        bound_roots.add(binding)
        slots = case['slots']
        _require(isinstance(slots, list) and bool(slots), 'nonempty frozen candidate slots required')
        slots_by_id, candidate_ids = {}, {arm: set() for arm in arms}
        for slot in slots:
            slot_id = _identity(slot['slot_id'])
            _require(slot_id not in slots_by_id, 'duplicate frozen slot')
            _require(type(slot['primitive_steps']) is int and slot['primitive_steps'] > 0,
                     'positive integer primitive steps required')
            _require(set(slot['candidate_ids']) == set(arms), 'candidate slots must cover all arms')
            for arm in arms:
                identity = _identity(slot['candidate_ids'][arm])
                _require(identity not in candidate_ids[arm], 'duplicate frozen candidate identity')
                candidate_ids[arm].add(identity)
            slots_by_id[slot_id] = slot
        _require(len(slots) == manifest['budget']['candidate_evaluations'],
                 'candidate budget must include every evaluated candidate, not just winners')
        for arm in arms:
            _require((case_id, arm) not in expected, 'duplicate frozen case identity')
            expected[case_id, arm] = (case, slots_by_id)
    rows = data['rows']
    _require(isinstance(rows, list), 'comparison rows required')
    seen, receipts = set(), {}
    for row in rows:
        _non_admission(row)
        key = (_identity(row['case_id']), _identity(row['arm']))
        _require(key in expected and key not in seen, 'unpaired, extra or duplicate arm/root row')
        seen.add(key)
        case, slots = expected[key]
        _budget(row['budget'])
        _require(_equal(row['budget'], manifest['budget']), 'same complete budget required in every arm')
        elapsed = row['actual_wall_time_s']
        _require(type(elapsed) in (int, float) and math.isfinite(elapsed)
                 and 0 <= elapsed <= manifest['budget']['wall_time_limit_s'],
                 'finite measured wall time within shared limit required')
        candidates = row['candidates']
        _require(isinstance(candidates, list), 'candidate list required')
        seen_slots = set()
        for candidate in candidates:
            _non_admission(candidate)
            slot_id = _identity(candidate['slot_id'])
            _require(slot_id in slots and slot_id not in seen_slots, 'extra/duplicate candidate slot')
            seen_slots.add(slot_id)
            receipts[key + (slot_id,)] = _check_candidate(candidate, case, slots[slot_id], key[1], manifest)
        _require(seen_slots == set(slots), 'missing paired candidate; no winner-only reporting')
    _require(seen == set(expected), 'missing paired arm/root row')

    physical = data.get('physical_observations')
    physical_pairs = 0
    if physical is not None:
        _require(isinstance(physical, list), 'physical observation list required')
        observed = set()
        for record in physical:
            _non_admission(record)
            key = (_identity(record['case_id']), _identity(record['arm']), _identity(record['slot_id']))
            _require(key in receipts and key not in observed, 'unpaired/duplicate physical observation')
            observed.add(key)
            receipt = receipts[key]
            _require(_equal(record['root'], receipt['root']), 'physical root/goal mismatch')
            _require(record['receipt_sha256'] == receipt['receipt_sha256'], 'physical witness mismatch')
            _require(_equal(record['executed_actions'], receipt['actions']), 'physical exact actions mismatch')
            witness._sha(record['execution_artifact_sha256'])
            _require(record['metric_sha256'] == manifest['provenance']['metric_sha256'], 'physical metric mismatch')
            objective = record['objective']
            _require(type(objective) in (int, float) and math.isfinite(objective), 'finite physical objective required')
            _require(type(record['success']) is bool, 'explicit physical success boolean required')
        _require(observed == set(receipts), 'physical outcomes must cover every paired candidate and arm')
        physical_pairs = len(receipts) // len(arms)

    for key, (_, slots) in expected.items():
        steps = sum(slot['primitive_steps'] for slot in slots.values()) if physical is not None else 0
        _require(manifest['budget']['physical_env_steps'] == steps,
                 'physical step budget must match paired executions, not model replay')

    if p1_evidence is not None:
        _require(isinstance(p1_evidence, dict) and bool(p1_evidence), 'nonempty external P1 report required')
        witness._canonical(p1_evidence)
    return dict(schema='d2p_p2_comparison_validation.v1', data_contract_valid=True,
                status='blocked_p1' if p1_evidence is None else 'p1_requires_independent_audit',
                manifest_sha256=expected_manifest_sha256, arms=arms,
                paired_candidates=len(receipts)//len(arms), physical_observation_pairs=physical_pairs,
                p1_evidence_sha256=None if p1_evidence is None else witness.digest(p1_evidence),
                p1_empirically_verified=False, P2_admission=False,
                model_replay_independently_verified=False, physical_efficacy_verified=False,
                evidence_scope='declared action/context/budget consistency only; hashes are not empirical proof')


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'duplicate JSON object key')
        result[key] = value
    return result


def _read(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_strict_object)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--p1-evidence', type=Path)
    args = parser.parse_args(argv)
    try:
        result = validate_comparison(_read(args.input), expected_manifest_sha256=args.manifest_sha256,
                                     p1_evidence=_read(args.p1_evidence) if args.p1_evidence else None)
    except (ValueError, OSError) as exc:
        print(json.dumps(dict(data_contract_valid=False, status='invalid_contract',
                              error=str(exc), P2_admission=False, physical_efficacy_verified=False)))
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    # Deliberately no exit-0 stage pass, including for self-declared P1 reports.
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
