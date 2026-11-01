"""Arithmetic and saved-asset inventory; no learned-model execution."""
import builtins

from experiments.spectral_reachability import p2_composition_prerequisites_20261027_v1 as m


def test_bank_equal_predictions_do_not_equal_terminal_coverage():
    row = m.arithmetic()['bank_per_root']
    assert row['planning_predict']==row['flat_planning_predict']==80
    assert row['terminal_candidates']==64 and row['flat_terminal_candidates']==40
    assert not row['terminal_candidate_counts_equal']


def test_native_includes_shared_fd_and_all_candidate_replays():
    row = m.arithmetic()['native_four_roots']
    assert row['predict']==572
    assert row['jacobian_api_calls']==88
    assert row['scalar_output_vjp_directions']==16896
    assert row['explicit_svd_calls_from_source']==96
    assert row['matrix_rank_diagnostic_svd_calls_from_source']==8
    assert row['bounded_ridge_calls_from_source']==88


def test_sra_overhead_and_all_slot_physical_arithmetic():
    row = m.arithmetic()
    assert row['sra_four_roots']==dict(predict=352,jacobian_api_calls=4,scalar_output_vjp_directions=768,svd=8)
    assert row['all_slot_example_only']['physical_steps']==3200


def test_real_allowlisted_inventory_imports_no_neural_runtime(monkeypatch):
    original = builtins.__import__
    def guard(name,*args,**kwargs):
        assert not name.startswith(('torch','stable_worldmodel','stable_pretraining')), name
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guard)
    result = m.inventory()
    assert result['status']=='inventory_only_not_admission'
    assert result['new_neural_calls']==result['new_physical_steps']==0
    assert not result['P2_admission'] and not result['fair_total_budget_comparison']
    assert result['details']['native']['predict_from_phase_records']==572
    assert result['details']['native']['candidates']==80
    assert result['details']['root_metric_identity_count']==4
    assert result['details']['provenance']['same_root_set']
    assert not result['details']['provenance']['equal_six_field_provenance_each_root']
    for root in result['details']['bank']:
        arms = {a['arm']:a for a in root['arms']}
        assert arms['composition']['H2_candidates']==64
        assert arms['flat_search']['H2_candidates']==40
        assert all(a['attempt_fields']==['completed','cost','path'] and a['saved_witnesses']==1 for a in arms.values())
