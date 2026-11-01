"""Array-only contrast tests, no neural/physics or filesystem writes."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability import p1_constructive_feedback_contrast_20261027_v1 as m


def candidates(values):
    return [dict(index=i, kind=k, terminal_cost=v) for i, (k, v) in enumerate(zip(
        ('zero_nominal', 'original_imagined_suffix', 'fresh_observed_ridge'), values, strict=True))]


def test_actual_same_Q_inversion_is_not_theory_verdict():
    result = m.contrast(candidates([.9, .6, .5]), 2, .8, .85)
    assert result['strict_rank_reversal'] and result['changed_suffix']
    assert result['predicted_Q_delta'] < 0 < result['actual_Q_delta']


def test_same_suffix_control_is_not_inversion():
    result = m.contrast(candidates([.9, .5, .6]), 1, .8, .8)
    assert not result['strict_rank_reversal'] and result['unchanged_choice_control']


def test_stable_tie_chooses_zero():
    result = m.contrast(candidates([.5, .5, .6]), 0, .8, .9)
    assert result['changed_suffix'] and not result['strict_rank_reversal']


@pytest.mark.parametrize('bad', ['missing', 'nan', 'order', 'winner'])
def test_invalid_model_choice_rejected(bad):
    values = candidates([.9, .6, .5]); selected = 2
    if bad == 'missing': values.pop()
    if bad == 'nan': values[0]['terminal_cost'] = float('nan')
    if bad == 'order': values.reverse()
    if bad == 'winner': selected = 1
    with pytest.raises(ValueError):
        m.contrast(values, selected, .8, .9)


@pytest.mark.parametrize('bad', [np.zeros(191), np.full(192, np.nan)])
def test_native_cost_shapes_and_finite(bad):
    with pytest.raises(ValueError):
        m.cost(bad, np.zeros(192))


def test_cost_uses_float64_original_identity_Q():
    x = np.arange(192, dtype=np.float32)/192
    assert m.cost(x, np.zeros(192)) == float(np.mean(x.astype(float)**2))


def test_changed_suffix_actual_improvement_is_not_reversal():
    r = m.contrast(candidates([.8, .6, .4]), 2, .8, .3)
    assert not r['strict_rank_reversal'] and r['actual_Q_delta'] < 0


@pytest.mark.parametrize('damage', ['none', 'winner', 'tail', 'prefix'])
def test_winner_command_bound_to_executed_full_plan(damage):
    suffix = np.zeros((5, 2)).tolist()
    winner = dict(commands=deepcopy(suffix))
    result = dict(selected_index=0, winner=winner, candidates=[winner])
    plan = dict(selected_index=0, commands=suffix+suffix,
                suffix_commands=deepcopy(suffix), original_prefix_commands=deepcopy(suffix))
    commands = deepcopy(plan['commands'])
    if damage == 'winner': winner['commands'][0][0] = .125
    if damage == 'tail': plan['suffix_commands'][0][0] = .125
    if damage == 'prefix': plan['original_prefix_commands'][0][0] = .125
    if damage == 'none':
        m.check_suffix(result, plan, commands)
    else:
        with pytest.raises(ValueError, match='winner commands'):
            m.check_suffix(result, plan, commands)


@pytest.mark.parametrize('damage', [False, True])
def test_actual_full_analyze_and_resealed_winner_counterexample(monkeypatch, damage):
    root = Path(__file__).resolve().parents[1]/'results'
    paths = [root/'p1_constructive_selected_rgb_cuda_20261027_v1_cuda_main01/report.json',
             root/'p1_constructive_observed_suffix_rgb_20261027_v2_cuda_main01/report.json',
             root/'p1_constructive_observed_suffix_cuda_20261027_v1_cuda_main01/report.json',
             root/'p1_constructive_feedback_prefix_replay_audit_20261027_v1_main01.json']
    if not all(p.is_file() for p in paths):
        pytest.skip('explicit actual saved chain unavailable')
    original = m.Reader.read
    def read(reader, path, sha):
        value = original(reader, path, sha)
        if damage and Path(path).name == 'branch_1.result.json.gz':
            # Simulate semantic tampering after the outer hash layer accepted
            # a resealed result. Winner/candidate agree; dispatched plan does not.
            value['winner']['commands'][0][0] += .125
            value['candidates'][value['selected_index']]['commands'] = deepcopy(value['winner']['commands'])
        return value
    monkeypatch.setattr(m.Reader, 'read', read)
    args = [paths[0], 'c77a5d208acb813ad968afe24ef0c547f08a83ff9968a984b29502baf0dda770',
            paths[1], '2cc16ea94dc76640a09ddb6d3d3cf8c70f171d9423b3896702715d48132306b1',
            paths[2], 'ac8a65900b94639f94b3797e90f80e73fa0500e13a055629750dc4dc418aaf79',
            paths[3], '69adf417bb8551926b5023485f95e1302692c80677f0d195188598ce97e1a318']
    if damage:
        with pytest.raises(ValueError, match='winner commands bind'):
            m.analyze(*args)
    else:
        result = m.analyze(*args)
        assert result['changed_branches'] == result['strict_rank_reversals'] == 2
