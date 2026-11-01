"""Offline localization of score/prediction/task ordering; no selector changes."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability.analyze_pusht_full_packet_20261027 import summarize, require
from experiments.spectral_reachability.p3_matched_physical_20261027_v1 import digest, write


def pair_counts(rows, left, right):
    counts = dict(agree=0, reverse=0, tied=0, unavailable=0)
    for a, b in itertools.combinations(rows, 2):
        vals = [a.get(left), b.get(left), a.get(right), b.get(right)]
        if any(v is None for v in vals):
            counts['unavailable'] += 1
            continue
        require(np.isfinite(vals).all(), 'nonfinite score')
        x, y = np.sign(vals[0]-vals[1]), np.sign(vals[2]-vals[3])
        counts['tied' if x == 0 or y == 0 else 'agree' if x == y else 'reverse'] += 1
    return counts


def audit(planning, physical):
    census = summarize(planning, physical)
    branches = {(b['root_sha256'], b['method']): b for b in physical['rows']}
    enc = {(e['root_sha256'], e['method'], e['kind']): e for e in physical['encodings']}
    groups = []
    for p in planning['plans']:
        for arm in p['arms']:
            rows = []
            for c in arm['candidates']:
                key = (p['root_sha256'], arm['arm']+'_'+c['candidate_id'])
                b, e = branches[key], enc[*key, 'terminal']
                rows.append(dict(candidate_id=c['candidate_id'],
                    objective=c['objective'], stage=c['stage_cost'],
                    linear_terminal=c['suffix_solve']['linear_terminal_cost'],
                    nonlinear_terminal=c['nonlinear_terminal_cost'],
                    nonlinear_objective=c['stage_cost']+c['nonlinear_terminal_cost']+
                        c['suffix_solve']['ridge_penalty'],
                    observed_terminal=e.get('observed_Q_goal_cost'),
                    pose=b['pose']['pose_cost'] if b['full_horizon_completed'] else None,
                    official_distance=b['official']['official_state_distance'] if b['full_horizon_completed'] else None,
                    completed_commands=b['completed_commands']))
            selected = min(rows, key=lambda r: r['objective'])
            require(selected['candidate_id'] == arm['selected_candidate_id'], 'selector differs')
            alternative = min(rows, key=lambda r: r['nonlinear_objective'])
            groups.append(dict(root_sha256=p['root_sha256'], root_index=p['root_index'], arm=arm['arm'],
                selected=selected, nonlinear_diagnostic_winner=alternative,
                selection_changes=selected['candidate_id'] != alternative['candidate_id'],
                comparisons={a+'__'+b: pair_counts(rows, a, b) for a, b in (
                    ('linear_terminal', 'nonlinear_terminal'),
                    ('nonlinear_terminal', 'observed_terminal'),
                    ('observed_terminal', 'pose'), ('observed_terminal', 'official_distance'))},
                candidates=rows))
    return dict(status='offline_ordering_diagnostic', original_physical_status=census['original_physical_status'],
        original_physical_gates=census['original_gates'], excluded_fixed_terminal_slots=census['missing_endpoints_explained_by_guard'],
        groups=groups, P1_admission=False, selector_changed=False, new_neural_calls=0,
        scope='four known roots, pairs dependent; alternate score is posthoc diagnosis, not validated method or efficacy')


def run(planning_path, physical_path, output):
    from experiments.spectral_reachability.pusht_full_packet_physical_20261027 import prepare
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal
    planning_path, physical_path, output = map(lambda p: Path(p).resolve(), (planning_path, physical_path, output))
    require(not output.exists(), 'exclusive output required')
    prepare(planning_path)
    inputs = {str(p): digest(p) for p in (planning_path, physical_path)}
    planning, physical = [json.loads(p.read_text()) for p in (planning_path, physical_path)]
    for s in physical['source_seals']:
        verify_seal(output_dir=s['output_dir'], expected_manifest_sha256=s['manifest_sha256'])
    result = audit(planning, physical)
    output.mkdir(exist_ok=False)
    seal = create_seal(files=[Path(__file__), Path(__file__).resolve().parents[2]/'tests/test_pusht_full_packet_objective_audit_20261027.py'],
        args_json=json.dumps(inputs, sort_keys=True), output_dir=output/'source_seal')
    result.update(inputs=inputs, source_seal=seal)
    require(inputs == {str(p): digest(p) for p in (planning_path, physical_path)}, 'input mutated')
    write(output/'report.json', result)
    print(json.dumps([dict(root=g['root_index'], arm=g['arm'], selection_changes=g['selection_changes'],
        comparisons=g['comparisons']) for g in result['groups']]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('planning-path', 'physical-path', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(**vars(parser.parse_args()))
