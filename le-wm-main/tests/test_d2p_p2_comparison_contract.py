"""Synthetic data-contract tests, NOT CUDA or physical efficacy experiments."""
from copy import deepcopy
import json
import subprocess
import sys

import pytest

from experiments.spectral_reachability import d2p_composition_witness_v2 as w
from experiments.spectral_reachability import d2p_p2_comparison_contract as p2


def fixture():
    sha = 'a' * 64
    context = dict(layout='states_with_between_state_actions_v1', context_length=2,
                   latent_dim=1, action_dim=1, latent_history=[[0.], [0.]],
                   action_history=[[0.]])
    root = dict(root_id='root-1', root_rgb_sha256=sha, goal_rgb_sha256='b'*64,
                context=context)
    provenance = {k: sha for k in w.PROVENANCE_KEYS}
    domain = dict(name='raw', units='command', low=[-1.], high=[1.],
                  action_block=1, normalization_sha256=sha)
    budget = dict(predict_calls=4, predict_samples=4, candidate_evaluations=1,
                  planning_predict_calls=2, planning_predict_samples=2,
                  replay_predict_calls=2, replay_predict_samples=2,
                  encoder_calls=1, backward_calls=0, optimizer_steps=0,
                  flops=100, attempted_expansions=2,
                  physical_env_steps=0, wall_time_limit_s=30)
    arms = ['composition', 'flat_search', 'no_composition_sra']
    manifest = dict(schema='d2p_p2_manifest.v1', arms=arms,
                    selection=dict(rule='pre_outcome_frozen', outcome_accessed=False),
                    provenance=provenance, action_domain=domain, budget=budget,
                    cases=[dict(case_id='case-1', root=root, slots=[dict(
                        slot_id='slot-1', primitive_steps=2,
                        candidate_ids={arm: arm+'-candidate' for arm in arms})])])

    def replay(ctx, actions):
        value = ctx['latent_history'][-1][0] + sum(row[0] for row in actions)
        return dict(ctx, latent_history=[ctx['latent_history'][-1], [value]],
                    action_history=[actions[-1]])

    rows = []
    for arm in arms:
        chunks = [[[.25]], [[.5]]] if arm == 'composition' else [[[.25], [.5]]]
        edges, ctx = [], deepcopy(context)
        for i, chunk in enumerate(chunks):
            out = replay(ctx, chunk)
            edges.append(dict(program_id=str(i), kind='progress', root_id=root['root_id'],
                              provenance=provenance, action_domain=domain,
                              input_context=ctx, output_context=out, actions=chunk))
            ctx = out
        receipt = w.compose(root=root, provenance=provenance, action_domain=domain,
                            edges=edges, replay=replay, max_edges=2, max_actions=2)
        rows.append(dict(case_id='case-1', arm=arm, budget=deepcopy(budget), actual_wall_time_s=1.,
                         candidates=[dict(slot_id='slot-1', candidate_id=arm+'-candidate',
                                          receipt=receipt, edges=edges)]))
    return deepcopy(dict(manifest=manifest, rows=rows))


def check(data, **kwargs):
    return p2.validate_comparison(data, expected_manifest_sha256=w.digest(data['manifest']),
                                  **kwargs)


def test_valid_contract_is_not_replay_or_p1_or_physical_evidence():
    data = fixture()
    before = deepcopy(data)
    result = check(data)
    assert result['data_contract_valid'] is True
    assert result['status'] == 'blocked_p1'
    assert result['paired_candidates'] == 1
    assert result['P2_admission'] is False
    assert result['physical_efficacy_verified'] is False
    assert result['model_replay_independently_verified'] is False
    assert data == before


@pytest.mark.parametrize('mutate', [
    lambda d: d['rows'].pop(),
    lambda d: d['rows'].append(deepcopy(d['rows'][0])),
    lambda d: d['rows'][1].update(case_id='other'),
    lambda d: d['rows'][1]['candidates'].clear(),
    lambda d: d['rows'][1]['candidates'].append(deepcopy(d['rows'][1]['candidates'][0])),
    lambda d: d['rows'][1]['candidates'][0].update(candidate_id='outcome-winner'),
    lambda d: d['rows'][1]['budget'].update(predict_samples=3),
    lambda d: d['rows'][1]['budget'].update(predict_calls=True),
    lambda d: d['rows'][1]['budget'].pop('physical_env_steps'),
    lambda d: d['rows'][1]['budget'].pop('flops'),
    lambda d: d['rows'][1]['budget'].update(encoder_calls=2),
    lambda d: d['rows'][1]['budget'].update(replay_predict_samples=0),
    lambda d: d['rows'][1].update(actual_wall_time_s=31),
    lambda d: d['rows'][1].update(actual_wall_time_s=True),
    lambda d: d['rows'][0]['candidates'][0]['receipt']['actions'][0].__setitem__(0, .3),
    lambda d: d['rows'][0]['candidates'][0]['edges'][0]['actions'][0].__setitem__(0, .3),
    lambda d: d['rows'][1]['candidates'][0]['receipt']['root'].update(root_id='other'),
    lambda d: d['rows'][1]['candidates'][0]['receipt']['root'].update(goal_rgb_sha256='c'*64),
    lambda d: d['rows'][1]['candidates'][0]['receipt'].update(P2_admission=True),
    lambda d: d['rows'][0]['candidates'][0]['edges'][1].update(goal_rgb_sha256='c'*64),
    lambda d: d['rows'][0]['candidates'][0]['edges'][1].update(root_id='other'),
    lambda d: d['manifest']['selection'].update(outcome_accessed=True),
    lambda d: d['manifest']['selection'].update(rule='best_outcome'),
    lambda d: d['manifest']['cases'][0]['slots'][0].update(primitive_steps=3),
    lambda d: d['manifest']['budget'].update(candidate_evaluations=2),
    lambda d: d['manifest'].update(arms=['composition']),
    lambda d: d.update(P2_admission=True),
    lambda d: d.update(physical_efficacy_verified=True),
])
def test_fail_closed_on_unpaired_mismatched_or_tampered_data(mutate):
    data = fixture()
    mutate(data)
    with pytest.raises(ValueError):
        check(data)


def test_external_manifest_digest_prevents_refreezing_selected_roots():
    data = fixture()
    seal = w.digest(data['manifest'])
    data['manifest']['cases'][0]['root']['root_id'] = 'outcome-selected-root'
    with pytest.raises(ValueError, match='manifest'):
        p2.validate_comparison(data, expected_manifest_sha256=seal)


def test_pair_by_identity_not_array_order():
    data = fixture()
    other = deepcopy(data['manifest']['cases'][0])
    other['case_id'] = 'case-2'
    other['root']['root_id'] = 'root-2'
    data['manifest']['cases'].append(other)
    for row in list(data['rows']):
        row = deepcopy(row)
        row['case_id'] = 'case-2'
        candidate = row['candidates'][0]
        candidate['receipt']['root']['root_id'] = 'root-2'
        for edge in candidate['edges']:
            edge['root_id'] = 'root-2'
        reseal(candidate)
        data['rows'].append(row)
    data['rows'].reverse()
    assert check(data)['paired_candidates'] == 2


def reseal(candidate):
    receipt = candidate['receipt']
    for transition, edge in zip(receipt['transitions'], candidate['edges']):
        transition['edge_sha256'] = w.digest(edge)
    receipt.pop('receipt_sha256', None)
    receipt['receipt_sha256'] = w.digest(receipt)


def test_rehashed_goal_mismatch_still_rejected():
    data = fixture()
    candidate = data['rows'][1]['candidates'][0]
    candidate['receipt']['root']['goal_rgb_sha256'] = 'c'*64
    reseal(candidate)
    with pytest.raises(ValueError, match='root'):
        check(data)


def test_baseline_cannot_relabel_composed_witness():
    data = fixture()
    data['rows'][1]['candidates'][0].update(deepcopy(data['rows'][0]['candidates'][0]))
    data['rows'][1]['candidates'][0]['candidate_id'] = 'flat_search-candidate'
    with pytest.raises(ValueError, match='single'):
        check(data)


def test_declared_p1_pass_cannot_self_authorize_stage():
    result = check(fixture(), p1_evidence={'status': 'passed', 'empirical': True})
    assert result['status'] == 'p1_requires_independent_audit'
    assert result['P2_admission'] is False
    assert result['p1_empirically_verified'] is False


def observations(data):
    data['manifest']['budget']['physical_env_steps'] = 2
    for row in data['rows']:
        row['budget']['physical_env_steps'] = 2
    result = []
    for row in data['rows']:
        candidate = row['candidates'][0]
        receipt = candidate['receipt']
        result.append(dict(case_id=row['case_id'], arm=row['arm'], slot_id='slot-1',
                           root=deepcopy(receipt['root']),
                           receipt_sha256=receipt['receipt_sha256'],
                           executed_actions=deepcopy(receipt['actions']),
                           execution_artifact_sha256='d'*64,
                           metric_sha256=data['manifest']['provenance']['metric_sha256'],
                           objective=1., success=False))
    return result


def test_physical_observations_are_paired_and_not_benefit_proof():
    data = fixture()
    data['physical_observations'] = observations(data)
    result = check(data)
    assert result['physical_observation_pairs'] == 1
    assert result['physical_efficacy_verified'] is False


@pytest.mark.parametrize('mutate', [
    lambda obs: obs.pop(),
    lambda obs: obs.append(deepcopy(obs[0])),
    lambda obs: obs[1]['executed_actions'].reverse(),
    lambda obs: obs[1]['root'].update(goal_rgb_sha256='e'*64),
    lambda obs: obs[1].update(receipt_sha256='e'*64),
    lambda obs: obs[1].update(objective=float('nan')),
    lambda obs: obs[1].update(objective=True),
    lambda obs: obs[1].update(success=1),
    lambda obs: obs[1].update(execution_artifact_sha256=''),
])
def test_invalid_physical_observations_rejected(mutate):
    data = fixture()
    data['physical_observations'] = observations(data)
    mutate(data['physical_observations'])
    with pytest.raises(ValueError):
        check(data)


def test_cli_missing_p1_nonzero_even_for_valid_data(tmp_path):
    data = fixture()
    path = tmp_path / 'input.json'
    path.write_text(json.dumps(data))
    result = subprocess.run([sys.executable, '-B', '-m', p2.__name__, str(path),
                             '--manifest-sha256', w.digest(data['manifest'])],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert json.loads(result.stdout)['status'] == 'blocked_p1'


def test_cli_malformed_data_nonzero(tmp_path):
    path = tmp_path / 'input.json'
    path.write_text('{bad json')
    result = subprocess.run([sys.executable, '-B', '-m', p2.__name__, str(path),
                             '--manifest-sha256', 'a'*64], capture_output=True, text=True)
    assert result.returncode == 1
    assert json.loads(result.stdout)['data_contract_valid'] is False


def test_variable_depth_best_cannot_enter_fixed_h2_slot_even_when_resealed():
    data = fixture()
    candidate = data['rows'][1]['candidates'][0]
    edge = candidate['edges'][0]
    edge['actions'] = edge['actions'][:1]
    receipt = candidate['receipt']
    candidate['receipt'] = w.compose(root=receipt['root'], provenance=receipt['provenance'],
        action_domain=receipt['action_domain'], edges=candidate['edges'],
        replay=lambda context, actions: edge['output_context'], max_edges=2, max_actions=2)
    with pytest.raises(ValueError, match='primitive total length'):
        check(data)


def test_physical_steps_cannot_be_hidden_as_model_only_budget():
    data = fixture()
    data['physical_observations'] = observations(data)
    data['manifest']['budget']['physical_env_steps'] = 0
    for row in data['rows']:
        row['budget']['physical_env_steps'] = 0
    with pytest.raises(ValueError, match='physical step'):
        check(data)


def test_disconnected_context_rejected_even_with_rehashed_receipt():
    data = fixture()
    candidate = data['rows'][0]['candidates'][0]
    candidate['edges'][1]['input_context']['latent_history'][-1][0] = .9
    reseal(candidate)
    with pytest.raises(ValueError):
        check(data)


def test_valid_receipt_for_another_context_cannot_replace_frozen_root():
    data = fixture()
    candidate = data['rows'][1]['candidates'][0]
    candidate['receipt']['root']['context']['latent_history'][0][0] = .1
    reseal(candidate)
    with pytest.raises(ValueError, match='root'):
        check(data)


def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path / 'duplicate.json'
    path.write_text('{"rows": [], "rows": []}')
    with pytest.raises(ValueError, match='duplicate JSON'):
        p2._read(path)
