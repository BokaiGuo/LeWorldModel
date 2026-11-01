"""Synthetic arithmetic only, never a production receipt or a neural experiment."""
from copy import deepcopy
import pytest
from experiments.spectral_reachability import main_observed_suffix_effect_20261028_v1 as m


def fixture():
    def context(x): return dict(latent_history=[[x]*192]*3)
    values = [dict(iteration=i, Q=m.qvalue([0]*192, context(1/(i+1))),
                   context=context(1/(i+1)), commands=[[i/10, 0]]*5) for i in range(9)]
    prefix = [[0., 0.]]*5
    plan = dict(commands=prefix+values[8]['commands'], winner_iteration=8,
                new_suffix_Q=values[8]['Q'], new_terminal_context=values[8]['context'])
    spec = dict(index=0, root_slot=0, condition='off', observed_incumbent_Q=1.,
                request=dict(task=dict(goal=[0]*192), incumbent=dict(prefix=prefix, commands=values[0]['commands'])))
    numeric = dict(iterates=values, winner_iteration=8, observed_incumbent_Q=1., new_suffix_Q=values[8]['Q'])
    row = dict(index=0, status='sealed', result=dict(numeric=numeric, plan=plan))
    return spec, row


def test_independent_cost_and_prefix():
    result = m.analyze(*fixture())
    assert result['model_Q_delta'] < 0 and result['prefix_exact'] and result['suffix_changed']
    assert result['physical_benefit'] is None and len(result['all_nine_Q']) == 9


@pytest.mark.parametrize('bad', ['prefix', 'incumbent', 'new_context', 'score', 'drop', 'winner', 'raw', 'slot'])
def test_saved_numerical_mutations_rejected(bad):
    spec, row = deepcopy(fixture()); r = row['result']
    if bad == 'prefix': r['plan']['commands'][0] = [.1, 0]
    elif bad == 'incumbent': spec['request']['incumbent']['commands'] = [[.1, 0]]*5
    elif bad == 'new_context': r['plan']['new_terminal_context'] = dict(latent_history=[[.9]*192]*3)
    elif bad == 'score': r['numeric']['iterates'][2]['Q'] += .1
    elif bad == 'drop': r['numeric']['iterates'].pop()
    elif bad == 'winner': r['numeric']['winner_iteration'] = True
    elif bad == 'raw': r['plan']['commands'][5] = [.9, 0]
    else: row['index'] = 1
    with pytest.raises(ValueError): m.analyze(spec, row)


def test_goal_dimension_and_nonfinite_rejected():
    with pytest.raises(ValueError): m.qvalue([0]*191, dict(latent_history=[[0]*192]*3))
    with pytest.raises(ValueError): m.qvalue([0]*192, dict(latent_history=[[float('nan')]*192]*3))
