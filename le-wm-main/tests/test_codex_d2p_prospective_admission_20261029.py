import copy

import pytest

from experiments.spectral_reachability import codex_d2p_prospective_admission_20261029 as gate


@pytest.fixture(scope='module')
def inputs():
    return {key: gate.read(gate.PROJECT / path, sha) for key, (path, sha) in gate.INPUTS.items()}


def candidate(inputs, slot=0):
    item = inputs['prereg']['candidate_order'][slot]
    return {'slot': slot, 'root_seed': item['root_seed'], 'reset_seed': item['reset_seed'],
            'identity': {'state7': [1., 2., 3., 4., 0., 0., 0.], 'snapshot_sha256': '1' * 64,
                         'rgb3_sha256': ['2' * 64, '3' * 64, '4' * 64],
                         'raw_history_sha256': '5' * 64, 'root_goal_sha256': None}}


def run(inputs, candidates):
    return gate.screen(inputs['prereg'], inputs['index'], inputs['manifest'], candidates)


def test_real_preflight_fails_closed(inputs):
    result = run(inputs, [None] * 4)
    assert all(row['screening_status'] == 'blocked_no_actual_identity' for row in result['rows'])
    assert not result['freshness_certified'] and not result['source_run_authorized']


def test_identity_without_collision_is_not_admitted(inputs):
    result = run(inputs, [candidate(inputs), None, None, None])
    assert result['rows'][0]['screening_status'] == 'no_known_collision_unverified'
    assert not result['rows'][0]['admitted'] and not result['witness_dispatch_authorized']


def test_known_physical_alias_including_angle_wrap(inputs):
    item = candidate(inputs)
    state = copy.deepcopy(inputs['index']['observed_prefix_records'][0]['state7'])
    state[4] += 2 * 3.141592653589793
    state[5] += 10
    item['identity']['state7'] = state
    assert 'known_historical_pose_collision' in run(inputs, [item, None, None, None])['rows'][0]['collision_reasons']


def test_all_batch_collisions_are_retained(inputs):
    items = [candidate(inputs, 0), candidate(inputs, 1), None, None]
    rows = run(inputs, items)['rows']
    assert all('batch_pose_collision' in row['collision_reasons'] for row in rows[:2])
    assert len(rows) == 4


def test_seed_envelope_collision_does_not_depend_on_candidate_identity(inputs):
    index = copy.deepcopy(inputs['index'])
    seed = inputs['prereg']['candidate_order'][0]['root_seed']
    index['conditional_seed_envelopes'].append({'start_inclusive': seed, 'end_inclusive': seed})
    result = gate.screen(inputs['prereg'], index, inputs['manifest'], [None] * 4)
    assert 'known_or_conditional_seed_conflict' in result['rows'][0]['collision_reasons']


@pytest.mark.parametrize('mutation', ['outcome', 'wrong_seed', 'nan'])
def test_reject_outcome_seed_and_invalid_state(inputs, mutation):
    item = candidate(inputs)
    if mutation == 'outcome':
        item['success'] = True
    elif mutation == 'wrong_seed':
        item['root_seed'] += 1
    else:
        item['identity']['state7'][0] = float('nan')
    with pytest.raises(ValueError):
        run(inputs, [item, None, None, None])
