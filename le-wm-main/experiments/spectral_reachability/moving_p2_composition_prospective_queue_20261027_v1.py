"""P2 prospective contract builder/validator, not a model or physics runner.

All evidence is externally supplied; SHA integrity is not scientific truth.
No execution or scientific admission is granted, even on a valid contract.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

STEM = 'moving_p2_composition_prospective_queue_20261027_v1'
AXES = ('predict_calls', 'predict_samples', 'action_encoder_calls', 'encoder_calls',
        'jacobian_calls', 'jvp_directions', 'vjp_directions', 'backward_calls',
        'svd_calls', 'solve_calls', 'physics_steps', 'host_wall_ns', 'cuda_ns', 'flops')
PHASES = ('prefix_search', 'suffix_search', 'derivatives', 'replay', 'execution')
ARMS = ('composition', 'no_composition_sra')
ASSETS = ('model', 'encoder', 'normalizer', 'q', 'source_rule', 'task_scope',
          'action_domain', 'goal_rule', 'success_rule', 'controller', 'observation_rule')
PROHIBITIONS = ('training', 'adapters', 'q_fitting', 'model_updates', 'cpu_neural',
                'oracle_selection', 'outcome_filtering', 'root_replacement')
P1_GATES = ('independent_calibration', 'independent_outcome', 'non_greedy',
            'real_capability_change', 'real_task_benefit', 'integrity')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def keys(obj, names):
    require(type(obj) is dict and set(obj) == set(names), 'exact fields required: ' + ','.join(names))


def label(value):
    require(type(value) is str and bool(value.strip()), 'nonempty string required')


def digest(value):
    require(type(value) is str and len(value) == 64 and
            all(c in '0123456789abcdef' for c in value), 'SHA256 required')


def integer(value, positive=False):
    require(type(value) is int and value >= (1 if positive else 0), 'integer count/time required')


def encode(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def parse(data):
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, 'duplicate JSON key: ' + k)
            result[k] = v
        return result

    def constant(value):
        raise ValueError('nonfinite JSON constant: ' + value)

    result = json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
    encode(result)  # Also reject overflowed exponents (e.g. 1e999).
    return result


def ref_bytes(ref, load):
    keys(ref, ('file', 'sha256'))
    label(ref['file']); digest(ref['sha256'])
    path = Path(ref['file'])
    require(not path.is_absolute() and '..' not in path.parts, 'relative contained evidence path required')
    data = load(ref['file'])
    require(type(data) is bytes and sha(data) == ref['sha256'], 'evidence byte SHA mismatch')
    return data


def vector(value):
    keys(value, AXES)
    for v in value.values():
        if v is not None:
            integer(v)
    return value


def add_vectors(values):
    # Unknown is never coerced to zero, even when other phases are available.
    return {k: None if any(v[k] is None for v in values) else sum(v[k] for v in values)
            for k in AXES}


def build_draft(slot_ids):
    """Build an intentionally blocked skeleton, never a pass certificate."""
    require(type(slot_ids) is list and bool(slot_ids), 'nonempty fixed slots required')
    for slot in slot_ids:
        label(slot)
    require(len(set(slot_ids)) == len(slot_ids), 'duplicate slots')
    unknown = {k: None for k in AXES}
    return dict(schema=STEM + '.manifest', sealed_at_ns=None, denominator=len(slot_ids),
                contract=dict(assets={k: None for k in ASSETS}, horizon_steps=None,
                              p1_required_metrics=None, preregistration=None,
                              prohibited={k: True for k in PROHIBITIONS},
                              execution_backend='HOST_CUDA_AFTER_MAIN_REVIEW'),
                p1=None, freshroot=None,
                slots=[dict(slot_id=s, identity=None, shared_cost_upper=deepcopy(unknown),
                            matched_total_cap=deepcopy(unknown),
                            arms={a: dict(shared_sha256=None, composition_enabled=a == 'composition',
                                          plan_rule=None,
                                          cost_upper={p: deepcopy(unknown) for p in PHASES})
                                  for a in ARMS}) for s in slot_ids])


def identity(row):
    keys(row, ('root_sha256', 'goal_sha256', 'history_sha256', 'blind_sha256', 'task_id'))
    for k, v in row.items():
        label(v) if k == 'task_id' else digest(v)


def shared_sha(contract, row):
    return sha(encode(dict(contract=contract, identity=row)))


def spec_sha(manifest):
    """Seal all prospective choices, excluding later external attestations."""
    return sha(encode({k: manifest[k] for k in
                       ('schema', 'sealed_at_ns', 'denominator', 'contract', 'slots')}))


def validate_contract(contract, load):
    keys(contract, ('assets', 'horizon_steps', 'p1_required_metrics', 'preregistration',
                    'prohibited', 'execution_backend'))
    keys(contract['assets'], ASSETS)
    for ref in contract['assets'].values():
        ref_bytes(ref, load)
    integer(contract['horizon_steps'], positive=True)
    metrics = contract['p1_required_metrics']
    require(type(metrics) is list and bool(metrics), 'MAIN preregistered P1 metrics required')
    for metric in metrics:
        label(metric)
    require(len(set(metrics)) == len(metrics), 'duplicate P1 metric')
    prereg = parse(ref_bytes(contract['preregistration'], load))
    keys(prereg, ('schema', 'p1_required_metrics', 'assets_sha256', 'horizon_steps'))
    require(prereg['schema'] == STEM + '.preregistration', 'preregistration schema')
    require(prereg['p1_required_metrics'] == metrics and
            prereg['assets_sha256'] == sha(encode(contract['assets'])) and
            type(prereg['horizon_steps']) is int and
            prereg['horizon_steps'] == contract['horizon_steps'], 'preregistration binding mismatch')
    keys(contract['prohibited'], PROHIBITIONS)
    require(all(v is True for v in contract['prohibited'].values()), 'training/selection prohibition missing')
    require(contract['execution_backend'] == 'HOST_CUDA_AFTER_MAIN_REVIEW', 'HOST CUDA/MAIN policy required')


def attestation(ref, kind, manifest, trusted, load):
    raw = ref_bytes(ref, load)
    # Trust is external to the manifest and must bind this exact contract.
    require(trusted is not None, 'external MAIN authority missing')
    keys(trusted, ('schema', 'reviewer', 'contract_sha256', 'spec_sha256', 'p1_sha256', 'freshroot_sha256'))
    require(trusted['schema'] == STEM + '.main_authority' and trusted['reviewer'] == 'MAIN',
            'MAIN authority schema/reviewer')
    for k in ('contract_sha256', 'spec_sha256', 'p1_sha256', 'freshroot_sha256'):
        digest(trusted[k])
    require(trusted['contract_sha256'] == sha(encode(manifest['contract'])), 'MAIN contract binding mismatch')
    require(trusted['spec_sha256'] == spec_sha(manifest), 'MAIN prospective spec binding mismatch')
    require(trusted[kind + '_sha256'] == sha(raw), 'untrusted ' + kind + ' attestation')
    return parse(raw)


def validate_p1(m, trusted, load):
    p = attestation(m['p1'], 'p1', m, trusted, load)
    keys(p, ('schema', 'contract_sha256', 'certified_at_ns', 'gates', 'roots',
             'ci', 'preregistration_sha256', 'evidence'))
    require(p['schema'] == STEM + '.p1', 'P1 schema')
    require(p['contract_sha256'] == sha(encode(m['contract'])), 'P1 contract mismatch')
    integer(p['certified_at_ns'], positive=True)
    require(p['certified_at_ns'] <= m['sealed_at_ns'], 'P1 must predate P2 seal')
    keys(p['gates'], P1_GATES)
    require(all(v is True for v in p['gates'].values()), 'P1 independent real mechanism incomplete')
    require(type(p['roots']) is list and bool(p['roots']), 'P1 root census missing')
    for root in p['roots']:
        digest(root)
    require(len(set(p['roots'])) == len(p['roots']), 'duplicate P1 roots')
    require(p['preregistration_sha256'] == m['contract']['preregistration']['sha256'], 'P1 preregistration mismatch')
    keys(p['ci'], m['contract']['p1_required_metrics'])
    for bounds in p['ci'].values():
        require(type(bounds) is list and len(bounds) == 2 and
                all(type(x) in (int, float) and math.isfinite(x) for x in bounds), 'finite CI bounds required')
        require(bounds[0] <= bounds[1] < 0, 'every preregistered P1 CI must be strictly negative')
    require(type(p['evidence']) is list and bool(p['evidence']), 'P1 evidence missing')
    for ref in p['evidence']:
        require(bool(ref_bytes(ref, load)), 'empty P1 evidence')
    return p


def validate_fresh(m, trusted, load):
    f = attestation(m['freshroot'], 'freshroot', m, trusted, load)
    keys(f, ('schema', 'issued_at_ns', 'contract_sha256', 'identities', 'excluded_roots',
             'coverage_complete', 'independent', 'not_used_for_design',
             'no_outcome_selection', 'exclusion_manifest'))
    require(f['schema'] == STEM + '.freshroot', 'freshroot schema')
    integer(f['issued_at_ns'], positive=True)
    require(f['issued_at_ns'] >= m['sealed_at_ns'], 'freshroot authority predates seal')
    require(f['contract_sha256'] == sha(encode(m['contract'])), 'freshroot contract mismatch')
    require(all(f[k] is True for k in ('coverage_complete', 'independent', 'not_used_for_design',
                                     'no_outcome_selection')), 'freshroot coverage/independence unknown')
    require(type(f['identities']) is list and type(f['excluded_roots']) is list and
            bool(f['excluded_roots']), 'complete identities/exclusions required')
    for ident in f['identities']:
        identity(ident)
    for root in f['excluded_roots']:
        digest(root)
    require(len(set(f['excluded_roots'])) == len(f['excluded_roots']), 'duplicate exclusions')
    exclusion = parse(ref_bytes(f['exclusion_manifest'], load))
    # This is a MAIN-approved metadata projection, NOT the existing history resolver schema.
    keys(exclusion, ('schema', 'excluded_roots', 'unresolved_sources', 'evidence'))
    require(exclusion['schema'] == STEM + '.exclusion_projection' and
            exclusion['excluded_roots'] == f['excluded_roots'] and
            exclusion['unresolved_sources'] == [], 'incomplete exclusion projection')
    require(type(exclusion['evidence']) is list and bool(exclusion['evidence']), 'exclusion provenance missing')
    for ref in exclusion['evidence']:
        require(bool(ref_bytes(ref, load)), 'empty exclusion provenance')
    return f


def validate_slot(row, contract, load):
    keys(row, ('slot_id', 'identity', 'shared_cost_upper', 'matched_total_cap', 'arms'))
    label(row['slot_id']); identity(row['identity'])
    shared = vector(row['shared_cost_upper']); cap = vector(row['matched_total_cap'])
    require(all(v is not None for v in cap.values()), 'matched total caps incomplete')
    keys(row['arms'], ARMS)
    totals = {}
    for name, arm in row['arms'].items():
        keys(arm, ('shared_sha256', 'composition_enabled', 'plan_rule', 'cost_upper'))
        require(arm['shared_sha256'] == shared_sha(contract, row['identity']), 'arm shared contract mismatch')
        require(arm['composition_enabled'] is (name == 'composition'), 'composition ablation mismatch')
        rule = parse(ref_bytes(arm['plan_rule'], load))
        keys(rule, ('schema', 'arm', 'program_count', 'action_witness_required',
                    'intermediate_states_from_actions_only', 'search_rule'))
        require(rule['schema'] == STEM + '.plan_rule' and rule['arm'] == name, 'plan rule binding')
        integer(rule['program_count'], positive=True)
        require(rule['program_count'] >= 2 if name == 'composition' else rule['program_count'] == 1,
                'multi-program vs no-composition count')
        require(rule['action_witness_required'] is True and
                rule['intermediate_states_from_actions_only'] is True, 'generating action witnesses required')
        require(bool(ref_bytes(rule['search_rule'], load)), 'empty search rule')
        keys(arm['cost_upper'], PHASES)
        totals[name] = add_vectors([shared] + [vector(arm['cost_upper'][p]) for p in PHASES])
        require(all(v is not None for v in totals[name].values()), 'full prefix/suffix/derivative cost missing')
        require(all(totals[name][k] <= cap[k] for k in AXES), 'full cost exceeds shared cap')
    return totals


def build_queue(manifest, load, trusted=None):
    """Retain every fixed slot; return diagnostics, never authorize dispatch.

    `trusted` must come from MAIN outside the candidate manifest. Verifying its
    SHA does not independently establish the truth of P1 or historical freshness.
    """
    m = deepcopy(manifest)
    issues, rows = [], []
    result = dict(schema=STEM + '.queue', manifest_sha256=None,
                  status='blocked', blockers=issues, rows=rows,
                  execution_authorized=False, P2_admission=False,
                  scientific_evidence_generated=False, actual_experiment_cost=None,
                  external_attestations_not_independently_proven=True)
    try:
        result['manifest_sha256'] = sha(encode(m))
        keys(m, ('schema', 'sealed_at_ns', 'denominator', 'contract', 'p1', 'freshroot', 'slots'))
        require(m['schema'] == STEM + '.manifest', 'manifest schema')
        integer(m['denominator'], positive=True)
        require(type(m['slots']) is list and len(m['slots']) == m['denominator'], 'fixed complete denominator required')
    except (ValueError, TypeError, OverflowError) as e:
        issues.append('manifest: ' + str(e))
        return result
    checks = [('contract', lambda: validate_contract(m['contract'], load)),
              ('seal', lambda: integer(m['sealed_at_ns'], positive=True)),
              ('p1', lambda: validate_p1(m, trusted, load)),
              ('freshroot', lambda: validate_fresh(m, trusted, load))]
    evidence = {}
    for name, check in checks:
        try:
            evidence[name] = check()
        except (ValueError, TypeError, KeyError, OSError, OverflowError) as e:
            issues.append(name + ': ' + str(e))
    seen = {k: set() for k in ('slot_id', 'root_sha256', 'history_sha256', 'blind_sha256', 'task_id')}
    for i, row in enumerate(m['slots']):
        out = dict(slot_index=i, slot_id=row.get('slot_id') if type(row) is dict else None,
                   status='blocked', errors=[], planned_total_upper=None)
        rows.append(out)
        try:
            totals = validate_slot(row, m['contract'], load)
            for k in seen:
                value = row[k] if k == 'slot_id' else row['identity'][k]
                require(value not in seen[k], 'duplicate slot identity: ' + k)
                seen[k].add(value)
            if 'p1' in evidence:
                require(row['identity']['root_sha256'] not in evidence['p1']['roots'], 'P1 root reused')
            if 'freshroot' in evidence:
                require(row['identity'] == evidence['freshroot']['identities'][i], 'freshroot identity/order mismatch')
                require(row['identity']['root_sha256'] not in evidence['freshroot']['excluded_roots'], 'historical root reused')
            out['planned_total_upper'] = totals
            out['status'] = 'contract_valid_pending_MAIN'
        except (ValueError, TypeError, KeyError, OSError, IndexError, OverflowError) as e:
            out['errors'].append(str(e))
    if 'freshroot' in evidence and len(evidence['freshroot']['identities']) != len(rows):
        issues.append('freshroot: complete fixed census mismatch')
    if issues or any(r['errors'] for r in rows):
        for row in rows:
            row['status'] = 'blocked'
    else:
        result['status'] = 'contract_valid_pending_MAIN'
    # No filtering, ranking, replacement, scientific success or dispatch list.
    return result


def file_loader(root):
    root = Path(root).resolve()

    def load(name):
        path = (root / name).resolve()
        require(path.is_relative_to(root), 'evidence symlink escapes input root')
        return path.read_bytes()
    return load


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    draft = sub.add_parser('draft', help='print a blocked skeleton to stdout')
    draft.add_argument('--slots', nargs='+', required=True)
    validate = sub.add_parser('validate', help='read metadata only; output JSON to stdout')
    validate.add_argument('--manifest', type=Path, required=True)
    validate.add_argument('--input-root', type=Path, required=True)
    validate.add_argument('--main-authority', type=Path)
    validate.add_argument('--main-authority-sha256')
    args = parser.parse_args(argv)
    if args.command == 'draft':
        print(encode(build_draft(args.slots)).decode())
        return 0  # Skeleton creation only, not validation or admission.
    try:
        trusted = None
        require((args.main_authority is None) == (args.main_authority_sha256 is None), 'MAIN path and external SHA required together')
        if args.main_authority is not None:
            raw = args.main_authority.read_bytes()
            digest(args.main_authority_sha256)
            require(sha(raw) == args.main_authority_sha256, 'MAIN external SHA mismatch')
            trusted = parse(raw)
        report = build_queue(parse(args.manifest.read_bytes()), file_loader(args.input_root), trusted)
    except (ValueError, TypeError, OSError) as e:
        print(encode(dict(schema=STEM + '.input_error', status='blocked', error=str(e),
                          execution_authorized=False, P2_admission=False)).decode())
        return 2
    print(encode(report).decode())
    return 0 if report['status'] == 'contract_valid_pending_MAIN' else 2


if __name__ == '__main__':
    raise SystemExit(main())
