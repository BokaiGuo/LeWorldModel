"""RAM semantic regression only; no torch/CUDA/environment execution."""
from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p1_api_endpoint_semantics_20261028_v1 as m


def case(missing=False,delta=0.):
    source = np.repeat(np.asarray([1.,2.,3.],np.float32)[:,None],192,axis=1)
    old_native = np.repeat(np.asarray([2.25,3.5,4.],np.float32)[:,None],192,axis=1)
    old_context = np.concatenate((source[1:],old_native[-1:]))
    native = old_native.copy(); native[-1] += delta
    goal = np.full(192,.5,np.float32)
    actual = None if missing else np.full(192,3.25,np.float32)
    terms = m.api.history.rank.scalar_terms(native[-1],actual,goal)
    legacy = dict(**terms,vs_original_prediction=m.api.stageb.discrepancy(native,old_context))
    return source,native,old_context,old_native,goal,actual,legacy


def test_same_endpoint_does_not_inherit_false_full_context_drift():
    args = case(); out = m.resolve(*args)
    assert out['deprecated_v1_field']['original_value']['exact_equal'] is False
    assert out['deprecated_v1_field']['causal_drift_evidence'] is False
    assert out['endpoint_vs_original_endpoint']['exact_equal'] is True
    assert out['native_full3_vs_original_native_full3']['exact_equal'] is True
    assert out['reconstructed_shifted_context_vs_original_shifted_context']['exact_equal'] is True
    assert out['endpoint_Q']['predicted_Q_delta'] == 0.
    assert out['shifted_context_semantics'][0] == dict(row=0,kind='preserved_observation',absolute_step=10)
    assert out['native_output_semantics']['rows'][0]['kind'] == 'model_prediction_not_preserved_observation'


def test_genuine_endpoint_difference_preserved_no_threshold_filter():
    out = m.resolve(*case(delta=.125))
    assert out['endpoint_vs_original_endpoint']['exact_equal'] is False
    assert out['endpoint_vs_original_endpoint']['max_abs_difference'] == .125
    assert out['native_full3_vs_original_native_full3']['exact_equal'] is False
    assert out['reconstructed_shifted_context_vs_original_shifted_context']['exact_equal'] is False
    assert out['endpoint_Q']['predicted_Q_delta'] > 0


def test_first_two_native_differences_not_endpoint_difference():
    args = list(case()); args[1][0,0] += .125
    args[-1]['vs_original_prediction'] = m.api.stageb.discrepancy(args[1],args[2])
    out = m.resolve(*args)
    assert out['native_full3_vs_original_native_full3']['exact_equal'] is False
    assert out['endpoint_vs_original_endpoint']['exact_equal'] is True
    assert out['reconstructed_shifted_context_vs_original_shifted_context']['exact_equal'] is True


def test_missing_actual_remains_null_but_all_predictions_compared():
    out = m.resolve(*case(missing=True))
    for key in ('current','original'):
        terms = out['endpoint_Q'][key]
        assert terms['actual_Q'] is terms['Q_error'] is terms['cross_term'] is terms['quadratic_term'] is None
        assert terms['predicted_Q'] is not None
    assert len(out['native_rows_vs_original_native_rows']) == 3
    assert out['endpoint_vs_original_endpoint']['exact_equal'] is True


@pytest.mark.parametrize('pos',range(4))
@pytest.mark.parametrize('mutation',['float64','flatten','drop_row'])
def test_strict_semantic_shapes_not_relaxed(pos,mutation):
    args = list(case())
    args[pos] = args[pos].astype(np.float64) if mutation == 'float64' else args[pos].reshape(-1) if mutation == 'flatten' else args[pos][:-1]
    with pytest.raises(ValueError): m.resolve(*args)


@pytest.mark.parametrize('mutation',['old_history','old_endpoint','goal','legacy_value','legacy_Q','legacy_actual','native_nonfinite'])
def test_broken_old_closure_or_numeric_receipt_rejected(mutation):
    args = list(case())
    if mutation == 'old_history': args[2][0,0] += 1
    elif mutation == 'old_endpoint': args[3][-1,0] += 1
    elif mutation == 'goal': args[4][0] += 1
    elif mutation == 'legacy_value': args[6]['vs_original_prediction']['max_abs_difference'] = 0.
    elif mutation == 'legacy_Q': args[6]['predicted_Q'] = 0.
    elif mutation == 'legacy_actual': args[6]['actual_Q'] = None
    else: args[1][0,0] = np.nan
    with pytest.raises(ValueError): m.resolve(*args)


def rows():
    members = []
    for i in range(40):
        task,rem = divmod(i,10); role,word = divmod(rem,5)
        semantic = m.resolve(*case(missing=i == 12))
        members.append(dict(index=i,root_slot=task//2,condition=('off','on')[task%2],role=('Z','S')[role],
            library_index=word,library_label=m.teacher.LIBRARY[word],routes={r:deepcopy(semantic) for r in m.api.ROUTES}))
    return members


def test_full40_two_routes_missing12_and_invalid_legacy_counts():
    out = m.assemble(rows())
    assert out['fixed_slots'] == 40 and out['route_slots'] == 80 and out['known_roots'] == 2
    for r in m.api.ROUTES:
        assert out['per_route'][r]['endpoint_exact_slots'] == 40
        assert out['per_route'][r]['deprecated_cross_semantics_unequal_slots'] == 40
        assert out['per_route'][r]['max_abs_predicted_Q_delta'] == 0.


@pytest.mark.parametrize('mutation',['drop','duplicate','condition','role','word','missing_route','fill12','drop_actual'])
def test_full_denominator_mutations_fail_closed(mutation):
    a = rows()
    if mutation == 'drop': a.pop()
    elif mutation == 'duplicate': a[39] = deepcopy(a[38])
    elif mutation == 'condition': a[0]['condition'] = 'on'
    elif mutation == 'role': a[0]['role'] = 'S'
    elif mutation == 'word': a[0]['library_label'] = '-uZ'
    elif mutation == 'missing_route': del a[0]['routes']['cached_suffix']
    elif mutation == 'fill12': a[12]['routes']['cached_suffix']['endpoint_Q']['current']['actual_Q'] = 0.
    else: a[0]['routes']['cached_suffix']['endpoint_Q']['current']['actual_Q'] = None
    with pytest.raises(ValueError): m.assemble(a)


def test_signed_zero_is_separate_from_numeric_exact():
    out = m.discrepancy(np.asarray([0.],np.float32),np.asarray([-0.],np.float32))
    assert out['exact_equal'] is True and out['bitwise_equal'] is False
