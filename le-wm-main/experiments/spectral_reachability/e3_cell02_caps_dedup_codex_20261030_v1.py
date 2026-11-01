"""Cell02 source-bound budget and bounded name-only duplicate review."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from experiments.spectral_reachability import e3_cell02_exclusive_candidate_codex_20261030_v1 as candidate
from experiments.spectral_reachability import e3_cell02_final_review_codex_20261030_v1 as final_review
from experiments.spectral_reachability import agent_e3_native_meter_20261029_v1 as physical
from experiments.spectral_reachability import agent_e3_gd_host_cuda_20261029_v1 as numeric_host
from experiments.spectral_reachability import agent_e3_feedback_node_20261029_v1 as feedback
from experiments.spectral_reachability import agent_e3_model_source_guard_20261029_v1 as guard

NAME = 'e3_cell02_caps_dedup_codex_20261030_v1'
PREPARED = candidate.RESULTS/'agent_e3_gd_host_cuda_20261029_v1_prepare_MAIN01.json'
PREPARED_SHA = 'ea8e4feae44e31876852c960db03ce4613184c55fc0745ef8f890f70485f8425'
ROOT_SHA = 'a3bd09470de75652506544b8bea0ff3c5fbae0c0d3b28775da47ee3e1adbb6f7'
MAX_JSON = 16*1024**2
CELL02 = re.compile(r'cell[_-]?0*2(?![0-9])', re.IGNORECASE)
CELL03 = re.compile(r'cell[_-]?0*3(?![0-9])', re.IGNORECASE)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def pinned_json(path, sha):
    p = Path(path)
    need(p.is_absolute() and p == p.resolve() and p.is_file() and not p.is_symlink(), 'canonical pinned JSON')
    with p.open('rb') as stream:
        data = stream.read(MAX_JSON+1)
    need(len(data) <= MAX_JSON and hashlib.sha256(data).hexdigest() == sha, 'pinned JSON bytes')
    return json.loads(data)


def manifest_binding(manifest):
    fields = {'schema', 'cell', 'root', 'claim', 'output', 'authority', 'authority_sha256',
              'python_invocation', 'python_executable', 'python_sha256', 'environment_sha256',
              'parent_sources', 'precheck_sources', 'precheck_code_sha256', 'child_code_sha256',
              'audit_code_sha256', 'production_authorization', 'scientific_admission'}
    need(type(manifest) is dict and set(manifest) == fields and manifest.get('schema') == candidate.SCHEMA
         and manifest.get('cell') == 2 and manifest.get('root') == str(candidate.ROOT)
         and manifest.get('claim') == str(candidate.CLAIM) and manifest.get('output') == str(candidate.OUTPUT)
         and manifest.get('authority') == str(candidate.AUTHORITY)
         and manifest.get('authority_sha256') == candidate.AUTHORITY_SHA
         and manifest.get('precheck_code_sha256') == candidate.sha(candidate.PRECHECK.encode())
         and manifest.get('child_code_sha256') == candidate.sha(candidate.CHILD.encode())
         and manifest.get('audit_code_sha256') == candidate.sha(candidate.AUDIT.encode())
         and manifest.get('parent_sources', {}).get(str(Path(candidate.__file__).resolve()))
             == candidate.file_sha(Path(candidate.__file__).resolve())
         and manifest.get('production_authorization') == 'external_decision_required'
         and manifest.get('scientific_admission') is False, 'fixed candidate manifest binding')


def contract(authority, prepared):
    need(type(authority) is dict and authority.get('cell') == 2
         and authority.get('schema') == 'agent_e3_continuous_host_launcher_20261029_v1.authority'
         and authority['control5']['prepare'] == {'path': str(PREPARED), 'sha256': PREPARED_SHA}
         and authority['resolver'] == prepared['actual_resolver'], 'prepared authority binding')
    need(len(prepared['bodies']) == prepared['fixed_cells'] == 16, 'fixed 16 denominator')
    body = prepared['bodies'][2]
    need(body['root_sha256'] == ROOT_SHA and body['horizon_packets'] == 3
         and body['domain']['name'] == 'Dxy' and body['feedback_boundary'] == 5
         and body['task_deadline_control'] == 20, 'cell02 prepared task')
    bindings = prepared['bindings_sha256']
    extra = authority['additional_sources']
    modules = (physical, numeric_host, numeric_host.numerical, feedback, guard)
    pins = {}
    for module in modules:
        path = Path(module.__file__)
        need(path.is_absolute() and path == path.resolve(), 'canonical cap source path')
        name = str(path)
        a, b = bindings.get(name), extra.get(name)
        need(a is not None or b is not None, 'cap implementation not presealed: '+name)
        need(a is None or b is None or a == b, 'conflicting source bindings: '+name)
        digest = a if a is not None else b
        need(path.is_file() and not path.is_symlink() and candidate.file_sha(path) == digest,
             'cap source drift: '+name)
        pins[name] = digest
    k = 3
    host_budget = numeric_host.budget(k)
    need(prepared['budgets'][str(k)] == host_budget, 'prepared numerical envelope mismatch')
    packet = numeric_host.numerical.packet_budget(k)
    need(packet == {'candidate_forwards': 33, 'replays': 3, 'AD_forwards': 32,
                    'predict': 204, 'reverse': 32}, 'numerical node plan mismatch')
    need(physical.CAPS == {'constructor': 1, 'close': 1, 'reset': 2, 'set_state': 4,
                           'goal_setter': 3, 'goal_render': 2, 'goal_bind': 1,
                           'public_step': 30, 'physics_step': 304, 'explicit_render': 32},
         'physical cap mismatch')
    return {'cell': 2, 'horizon_packets': k, 'fixed_denominator': 16,
            'physical_predispatch_caps': dict(physical.CAPS),
            'numerical_per_node_success_expectation': packet,
            'online_two_node_success_expectation': {'predict': 2*packet['predict'],
                                                    'reverse': 2*packet['reverse']},
            'prepared_three_node_transport_envelope': host_budget,
            'numerical_axes_per_successful_node': numeric_host.numerical.axes27(k),
            'guard_gpu_peak_caps_bytes': {'allocated': 3*1024**3, 'reserved': 4*1024**3},
            'operation_caps_present_directly_in_launcher_manifest': False,
            'binding': 'launcher manifest authority SHA -> prepared SHA and pinned implementation bytes',
            'numerical_counts_are_success_expectations_not_independently_proved_predispatch_caps': True,
            'source_pins': pins, 'gpu_capacity_proven': False}


def candidate_names(results):
    """Only filenames explicitly naming cell02; never open or stat other entries."""
    known = {
        candidate.CLAIM.name: 'candidate_claim',
        candidate.OUTPUT.name: 'candidate_output',
        candidate.AUTHORITY.name: 'prepared_authority',
        'agent_e3_gd_host_cuda_20261029_v1_cell02_MAIN01': 'upstream_control5',
        'agent_e3_gd_host_cuda_20261029_v1_consume_cell02_MAIN01.json': 'upstream_control5_receipt',
    }
    found = []
    with os.scandir(results) as directory:
        for entry in directory:
            name = entry.name
            if CELL03.search(name) or not CELL02.search(name):
                continue
            found.append({'name': name, 'classification': known.get(name, 'unclassified_possible_attempt')})
    found.sort(key=lambda x: x['name'])
    conflicts = [x for x in found if x['classification'] in
                 ('candidate_claim', 'candidate_output', 'unclassified_possible_attempt')]
    return {'scope': 'one results directory, names explicitly spelling cell02 only; no entry opened or statted',
            'entries': found, 'possible_conflicts': conflicts,
            'enumerated_names_clear': not conflicts, 'cross_entrypoint_unique_proven': False,
            'uninspected': ['arbitrary output names and paths', 'other hosts and processes',
                            'uncooperative entrypoints', 'future concurrent launches'],
            'cell03_entries_opened_or_statted': False}


def review(manifest_path, manifest_sha):
    review_sha = candidate.file_sha(Path(__file__).resolve())
    manifest_bytes, manifest = final_review.read_manifest(manifest_path, manifest_sha)
    manifest_binding(manifest)
    authority = pinned_json(candidate.AUTHORITY, candidate.AUTHORITY_SHA)
    prepared = pinned_json(PREPARED, PREPARED_SHA)
    caps = contract(authority, prepared)
    names = candidate_names(candidate.RESULTS)
    need(final_review.read_manifest(manifest_path, manifest_sha)[0] == manifest_bytes, 'manifest changed')
    pinned_json(candidate.AUTHORITY, candidate.AUTHORITY_SHA)
    pinned_json(PREPARED, PREPARED_SHA)
    need(contract(authority, prepared) == caps, 'source or contract drift')
    need(candidate_names(candidate.RESULTS) == names, 'bounded duplicate observations changed')
    need(candidate.file_sha(Path(__file__).resolve()) == review_sha, 'review source drift')
    return {'schema': NAME+'.review', 'manifest_path': str(manifest_path),
            'manifest_sha256': manifest_sha, 'authority_sha256': candidate.AUTHORITY_SHA,
            'prepared_sha256': PREPARED_SHA, 'caps': caps, 'bounded_duplicate_review': names,
            'review_source_sha256': review_sha,
            'final_environment_revalidated': False, 'production_ready': False,
            'production_authorized': False, 'scientific_admission': False,
            'full_saved_strict': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    try:
        result = review(args.manifest, args.manifest_sha256)
        print(candidate.canonical(result).decode())
        if result['bounded_duplicate_review']['possible_conflicts']:
            return 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({'error': str(exc), 'production_ready': False,
                          'production_authorized': False, 'scientific_admission': False}), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
