"""Saved historical quarantine metadata, never a freshness certificate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

STEM = 'moving_historical_reset_quarantine_20261027_v1'
ROOT = Path(__file__).resolve().parents[2]
REASON = 'historical_reset_attempt_bound_not_attested'
AUTHORITY = 'results/root_exclusion_authority_resolution_20261027_v1_MAIN01.json'
AUTHORITY_SHA = '032ea722f53da7c93db90377d775b75f7ff7a40771a43fdfa38dfd7761e27d00'
DISPOSITION = 'innerdocs/moving_historical_reset_bound_disposition_20261027_v1.md'
DISPOSITION_SHA = '462508d664b2064544e2c246a885f1175026215de4bce5f77e4a0e3be256a5d3'
CAP = 2 * 1024**2
FLOOR = 200 * 1024**2
# ID, directory, original report hash, seed, requested roots, schema, reservation.
SOURCES = (
 ('A','pusht_d2p_p0_packet_contract_v2_20261026_cuda','2f29a195f291d040a666cc3db6673f9be12c315c96162b126ddaef9bcdb5e135',202753000,4,'pusht_d2p_p0_preflight.v2',202753100),
 ('B','pusht_d2p_p0_preflight_20261024_cuda','b1b0fbc29ceb050939ae1403c7644d0ca704b133e47a2411c079dbd626ee0eaf',202610241,4,'pusht_d2p_p0_preflight.v1',202610281),
 ('C','pusht_d2p_p0_preflight_packet_layout_repair_20261024_cuda','c499ececab67135a2f4d4d698359e6f62b557be66f9132480ef95b14a0dce080',202710241,4,'pusht_d2p_p0_preflight.v1',202710281),
 ('D','pusht_d2p_p1_outcome_admission_20261024_cuda','a9266fb3cd3e68ee197ec187e8b7916cf3ca8c11966778df3d4c92083b7d6b54',202710281,24,'pusht_d2p_p1_reconfiguration_preflight.v3',None),
 ('E','pusht_d2p_p1_packet_contract_v4_calibration_20261026_cuda','a0c9c53bd0161946ec4466878de49795abdf6db102dbb97b174035ed4028a006',202753010,4,'pusht_d2p_p1_reconfiguration_preflight.v4',None),
 ('F','pusht_d2p_p1_reconfiguration_replay_bound_20261024_cuda','4911ae7e945abfb042b332f71d7cda0163e14c9727ee8131ff7881bc67257a14',202710251,4,'pusht_d2p_p1_reconfiguration_preflight.v2',None),
 ('G','pusht_d2p_p1_rollout_alignment_repair_20261024_cuda','1c5fdd414a5104b6487c2c6e5cb38d05943f902a9d7a695b891ae73202ef7abb',202710251,4,'pusht_d2p_p1_reconfiguration_preflight.v3',None),
 ('H','pusht_d2p_p1_root_anchored_v5_calibration_20261026_cuda','99cbb6b6dc4653eb629427353f6f6d05022c219333f3f7362cf4ca43a3076bc2',202753010,4,'pusht_d2p_p1_reconfiguration_preflight.v5',None),
)
COMMIT = 'd26e4bff84f31138adf5d1bda809bccafd9be10a'
FAMILY = (
 ('experiments/audit_pusht_reachability_horizon.py','73871055f10b09aea15f03acf87b04e79e30d546668d44bb8160c7be3c439a67','collect_nonterminal_boundary_contact_roots'),
 ('experiments/run_pusht_mode_aware_shared_candidate.py','1b027be83024ea1d4b868c86f2c1340032596520256e5eb5d7a9be869ad203e6','make_boundary_contact_roots'),
 ('experiments/run_pusht_signed_torque_audit.py','f35fe78b70237fcb7f10cabc950b30c163318c5eb5121cbaff8a5081e5ad3493','make_terminal_roots'),
 ('experiments/run_causal_visual_pusht_world_model.py','261f609e91da9110f8ba5254033529527314d6bcf38e058c3a237c920d3c4190','terminal_pose_cases'),
)
HISTORICAL = (
 ('git:675cbb2d5d743ab5da8abd53b73d153c17bbdcff:le-wm-main/experiments/spectral_reachability/preflight_pusht_d2p_p0.py','6170c69352462b8755b729b0e8dd32cdb873364944a60be9756f6d8935ad1962'),
 (f'git:{COMMIT}:le-wm-main/experiments/spectral_reachability/preflight_pusht_d2p_p0.py','567bcbe1b4edc9241806f1b60882de5d971cab694620153d992199d984b5f3ac'),
 (f'git:{COMMIT}:le-wm-main/experiments/spectral_reachability/preflight_pusht_d2p_p1_reconfiguration.py','24171e996addc6910d30f800f8ad7c03817f9296b95ae4c675cf9dd3168006f3'),
 ('results/pusht_d2p_p1_packet_contract_v4_calibration_20261026_cuda/postrun_implementation/preflight_pusht_d2p_p1_reconfiguration.py','80b9674063360a2f51d2e146c830c29a5399bb53b9086b177f3b3f2dc3b0d5e4'),
 ('results/pusht_d2p_p1_packet_contract_v4_calibration_20261026_cuda/postrun_implementation/preflight_pusht_d2p_p0.py','5215f0b49b772d8def14997996774bfd49db614b52b9f615c164271591f7b83b'),
 ('results/pusht_d2p_p1_root_anchored_v3_physical_calibration_20261026_cuda/prerun_seal.json','4e024c0ded185e0bb58c6ceaeb33a0ab1d784165a962995c28f6216d439ef81e'),
 ('results/pusht_d2p_sealed_program_execution4_v1_20261026_cuda/prerun_source_seal/manifest.json','0f0f5b8b8a525990dcffd2393ada1a49270dd50a3d4865800343cfd9b8a39aef'),
 ('results/pusht_d2p_p1_outcome_physical_20261024_cuda/report.json','f67d22b56e8f8bc3f41f4890ccea40ddf0c24f8735f1ea7713dd6d6d5badfe5e'),
)
VERSION_DISPOSITION = {
 'A':'packet_P0_report_and_later_source_only_no_original_run_binding',
 'B':'old_P0_git_blob_available_v1_schema_not_unique_no_run_binding',
 'C':'layout_repair_git_blob_candidate_upstream_report_edges_not_source_binding',
 'D':'P1_v3_git_blob_matches_schema_later_physical_pins_report_not_source',
 'E':'postrun_implementation_not_prerun_and_missing_root_factory_closure',
 'F':'exact_P1_v2_producer_not_recovered_v3_cannot_substitute',
 'G':'P1_v3_git_blob_matches_schema_no_run_to_blob_binding',
 'H':'later_physical_prerun_pins_upstream_report_not_upstream_execution_source',
}
ASSUMPTIONS = [
 'historical_run_used_this_code_family_not_proven',
 'collector_oversample_factor_4', 'boundary_max_initial_angle_error_infinite',
 'terminal_pose_branch_not_radial_contact', 'support_gap_max_0',
 'no_unrecorded_alternative_producer_restart_or_environment_reset',
]
UNKNOWN = [
 'original_run_to_producer_blob_binding', 'original_run_full_root_factory_and_env_closure',
 'all_reset_attempts_including_unselected_failed_and_prior_processes',
 'complete_historical_scope_authority',
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def parse(data):
    def unique(items):
        out = {}
        for k, v in items:
            require(k not in out, 'duplicate JSON key')
            out[k] = v
        return out
    return json.loads(data, object_pairs_hook=unique,
                      parse_constant=lambda _: require(False, 'nonfinite JSON'))


def exact(obj, names):
    require(isinstance(obj, dict) and set(obj) == set(names.split()), 'strict keys: '+names)


def local_read(name):
    if name.startswith('git:'):
        allowed = {p for p, _ in HISTORICAL} | {f'git:{COMMIT}:le-wm-main/{p}' for p, _, _ in FAMILY}
        require(name in allowed, 'unapproved git object')
        _, commit, path = name.split(':', 2)
        return subprocess.run(['git', 'show', commit+':'+path], cwd=ROOT,
                              check=True, capture_output=True, timeout=10).stdout
    p = Path(name)
    require('..' not in p.parts, 'path traversal forbidden')
    p = p if p.is_absolute() else ROOT/p
    require(p.resolve().is_relative_to(ROOT.resolve()), 'outside project input')
    require(p.stat().st_size <= 8*1024**2, 'input exceeds 8MiB')
    return p.read_bytes()


def pinned(read, name, expected):
    data = read(name)
    require(sha(data) == expected, 'SHA mismatch: '+name)
    return data


def evidence(read):
    rows = []
    for name, expected in [(DISPOSITION, DISPOSITION_SHA)] + [
            (f'git:{COMMIT}:le-wm-main/{p}', h) for p, h, _ in FAMILY] + list(HISTORICAL):
        row = dict(path=name, sha256=expected, status='not_verified',
                   proves_original_eight_run_source_binding=False,
                   scope=('historical_git_blob_only' if name.startswith('git:') else
                          'postrun_snapshot_only' if '/postrun_implementation/' in name else
                          'written_disposition' if name == DISPOSITION else 'later_run_evidence_only'))
        try:
            pinned(read, name, expected)
            row['status'] = 'saved_bytes_verified'
            if name.endswith('prerun_seal.json'):
                seal = parse(read(name))
                for ref in seal['inputs']:
                    pinned(read, ref['snapshot'], ref['sha256'])
                row['verified_snapshot_count'] = len(seal['inputs'])
            if name.endswith('/manifest.json'):
                manifest = parse(read(name))
                for ref in manifest['files']:
                    pinned(read, str(Path(name).parent/ref['archive']), ref['sha256'])
                row['verified_archive_count'] = len(manifest['files'])
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
            row.update(status='evidence_unavailable', error=str(exc))
        rows.append(row)
    return rows


def span(base, count, kind, pointers):
    return dict(start=base, end=base+8*count-1, closed=True, kind=kind,
                formula='base + [0,8*root_count-1]', root_count=count,
                oversample_factor=4, terminal_attempt_factor=2,
                classification='conditional_code_family_envelope', certified=False,
                protocol_pointers=pointers, assumptions=list(ASSUMPTIONS), unknown=list(UNKNOWN),
                code_evidence=[dict(path=f'git:{COMMIT}:le-wm-main/{p}', sha256=h, function=fn)
                               for p, h, fn in FAMILY])


def build(read=local_read):
    authority = parse(pinned(read, AUTHORITY, AUTHORITY_SHA))
    selected = [x for x in authority['history_file_requests'] if any(
        r['original_reason'] == REASON for r in x['inherited_refs'])]
    require(len(selected) == 8, 'fixed eight historical sources required')
    by_path = {x['path']: x for x in selected}
    require(len(by_path) == 8, 'duplicate historical source')
    aux = evidence(read)
    rows = []
    for ident, directory, expected, base, n, schema, reserved in SOURCES:
        path = f'results/{directory}/report.json'
        item = by_path.get(str(ROOT/path))
        require(item is not None and item['sha256'] == expected, 'fixed roster/authority mismatch')
        refs = item['inherited_refs']
        require(len(refs) == 2 and {r['parent'] for r in refs} == {'reachable_prepare','moving_prepare'}
                and all(r['original_reason'] == REASON and r['historical_sha256'] == expected for r in refs),
                'two original references must remain unchanged')
        row = dict(source_id=ident, path=path, sha256=expected, inherited_refs=refs,
                   bound_facts=[], conditional_envelopes=[], actual_attempt_bound_certified=False,
                   historical_execution_source_bound=False, unknown=list(UNKNOWN),
                   historical_version_disposition=VERSION_DISPOSITION[ident],
                   historical_evidence_index=[e['path'] for e in aux])
        try:
            report = parse(pinned(read, path, expected))
            proto = report['protocol']; p0 = ident in ('A','B','C')
            seedkey, countkey = ('calibration_root_seed','calibration_roots') if p0 else ('root_seed','root_count')
            require(report['schema'] == schema and type(proto[seedkey]) is int and proto[seedkey] == base
                    and type(proto[countkey]) is int and proto[countkey] == n, 'historical protocol values changed')
            if not p0:
                require(type(proto['root_oversample_factor']) is int and proto['root_oversample_factor'] == 4,
                        'oversample protocol mismatch')
            pointer = ['/protocol/'+seedkey, '/protocol/'+countkey]
            row['bound_facts'] = [dict(classification='report_bytes_bound_fact_not_execution_source',
                                      pointer=p, value=v, evidence_sha256=expected)
                                  for p, v in zip(pointer, (base,n))]
            root_rows = report.get('calibration_roots' if p0 else 'rows', [])
            row['declared_selected_root_sha256'] = [r['root_sha256'] for r in root_rows]
            row['conditional_envelopes'].append(span(base,n,'candidate_random_reset_seed_envelope',pointer))
            if reserved is not None:
                outcome = proto['outcome']
                require(outcome['root_seed'] == reserved and outcome['root_count'] == 24
                        and outcome['oversample_factor'] == 4 and outcome['status'] == 'reserved_not_opened',
                        'reserved cohort protocol mismatch')
                row['bound_facts'].append(dict(classification='report_bytes_bound_fact_not_execution_source',
                    pointer='/protocol/outcome', value={k:outcome[k] for k in
                    ('root_seed','root_count','oversample_factor','status')}, evidence_sha256=expected))
                row['conditional_envelopes'].append(span(reserved,24,'reserved_not_observed_reset_envelope',
                    ['/protocol/outcome/root_seed','/protocol/outcome/root_count']))
            row['status'] = 'conditional_quarantine_metadata_assembled'
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row.update(status='blocked_report_evidence', error=str(exc))
        row['explicit_state_reset'] = dict(seed=0, classification='conditional_explicit_state_reconstruction',
            not_a_random_root_identity=True, certified=False,
            evidence=DISPOSITION, evidence_sha256=DISPOSITION_SHA,
            assumptions=list(ASSUMPTIONS), unknown=['original_requested_state_goal_history_attempt_ledger'])
        rows.append(row)
    require(sum(len(r['inherited_refs']) for r in rows) == 16, 'fixed 16-reference denominator')
    complete = all(x['status'] == 'saved_bytes_verified' for x in aux) and all(
        r['status'] == 'conditional_quarantine_metadata_assembled' for r in rows)
    return dict(schema=STEM+'.quarantine', status='freshness_not_certified',
                metadata_status='assembled' if complete else 'blocked_evidence',
                authority=dict(path=AUTHORITY,sha256=AUTHORITY_SHA),
                disposition=dict(path=DISPOSITION,sha256=DISPOSITION_SHA), evidence=aux, rows=rows,
                source_count=8, inherited_reference_count=16, certified_source_count=0,
                source_sha256=sha(Path(__file__).read_bytes()), historical_coverage_complete=False,
                freshness_certified=False, old_freshness_blockers_cleared=False,
                P1_admission=False, actual_neural_calls=0, actual_physics_calls=0,
                unknown=list(UNKNOWN))


def integer(value):
    require(type(value) is int and 0 <= value < 2**63, 'nonnegative int64 seed required')
    return value


def intersection(a, b):
    lo, hi = max(a['start'], b['start']), min(a['end'], b['end'])
    return dict(start=lo,end=hi,closed=True) if lo <= hi else None


def check_new(quarantine, request):
    require(quarantine.get('schema') == STEM+'.quarantine' and len(quarantine['rows']) == 8
            and quarantine['source_count'] == 8 and quarantine['inherited_reference_count'] == 16
            and quarantine['freshness_certified'] is False, 'fixed un-certified quarantine input required')
    exact(request, 'schema roots')
    require(request['schema'] == STEM+'.new_seed_spans' and isinstance(request['roots'],list)
            and 0 < len(request['roots']) <= 32, '1..32 explicit new root requests')
    checks, seen = [], set()
    for root in request['roots']:
        exact(root, 'new_root_id random_reset_spans explicit_state_reset_seeds reset_coverage_declared_complete')
        ident = root['new_root_id']
        require(isinstance(ident,str) and ident.strip() and ident not in seen, 'unique new_root_id')
        seen.add(ident)
        require(type(root['reset_coverage_declared_complete']) is bool, 'coverage declaration must be boolean')
        ranges = root['random_reset_spans']; seeds = root['explicit_state_reset_seeds']
        require(isinstance(ranges,list) and 0 < len(ranges) <= 16 and isinstance(seeds,list)
                and len(seeds) <= 16, 'bounded complete reset range list required')
        for s in ranges:
            exact(s,'start end'); integer(s['start']); integer(s['end'])
            require(s['start'] <= s['end'], 'reversed span')
        for s in seeds:
            integer(s)
        overlaps, state_labels = [], []
        for history in quarantine['rows']:
            for envelope in history['conditional_envelopes']:
                for i, s in enumerate(ranges):
                    hit = intersection(envelope, s)
                    if hit:
                        overlaps.append(dict(source_id=history['source_id'], source_sha256=history['sha256'],
                            kind=envelope['kind'], new_span_index=i, intersection=hit,
                            classification='conditional_seed_span_overlap_not_physical_identity_proof'))
            if 0 in seeds or any(s['start'] <= 0 <= s['end'] for s in ranges):
                state_labels.append(dict(source_id=history['source_id'], seed=0,
                    classification='explicit_state_seed_label_overlap_not_physical_identity_proof'))
        checks.append(dict(new_root_id=ident, request=root, status='freshness_not_certified',
            conditional_span_overlap=bool(overlaps), overlaps=overlaps, explicit_state_seed_labels=state_labels,
            no_known_span_intersection=not overlaps,
            fresh_coverage_unknown=True, request_coverage_only_self_declared=True,
            comparison_complete_for_listed_envelopes=quarantine['metadata_status']=='assembled',
            missing=['historical_full_coverage_and_execution_binding', 'physical_state_goal_history_dedup'] +
                    ([] if root['reset_coverage_declared_complete'] else ['new_reset_coverage_not_declared_complete'])))
    return dict(schema=STEM+'.seed_span_check', status='freshness_not_certified',
                request_canonical_sha256=sha(encode(request)), checks=checks,
                freshness_certified=False, old_freshness_blockers_cleared=False)


def write_new(output, value):
    data = encode(value)
    require(len(data) <= CAP, '2MiB output cap')
    path = Path(output)
    require(shutil.disk_usage(path.parent).free >= FLOOR+CAP, '200MiB floor plus output reserve')
    with path.open('xb') as handle:
        handle.write(data); handle.flush(); os.fsync(handle.fileno())
    return sha(data)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--new-seeds'); p.add_argument('--new-seeds-sha')
    args = p.parse_args(argv)
    require(bool(args.new_seeds) == bool(args.new_seeds_sha), 'new request requires external SHA')
    result = build()
    if args.new_seeds:
        data = local_read(args.new_seeds)
        require(sha(data) == args.new_seeds_sha, 'new seed request SHA mismatch')
        result['new_seed_check'] = check_new(result,parse(data))
        result['new_seed_request_file_sha256'] = args.new_seeds_sha
    else:
        result['new_seed_check'] = dict(status='not_requested',freshness_certified=False)
    h = write_new(args.output,result)
    print(json.dumps(dict(path=args.output,sha256=h,status=result['status'],metadata_status=result['metadata_status'])))
    # Metadata success is not a successful freshness check; machine-consumable exit 2.
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
