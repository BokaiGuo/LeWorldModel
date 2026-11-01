"""Saved-input verification only; no model, GPU, or physical execution."""
from experiments.spectral_reachability import p3_matched_order_feedback_source_preflight_codex_20261029 as preflight


def test_actual_source_reconstruction_has_fixed_two_roots_four_slots():
    p = preflight.build()
    assert [r['slot'] for r in p['roster']] == [0,1,2,3]
    assert [r['root'] for r in p['roster']] == [0,0,1,1]
    assert len(p['roots']) == len(p['histories']) == 2
    assert all(h['saved_history_binding_verified'] and not h['live_history_verified'] for h in p['histories'])
    assert not p['physical_authorized'] and not p['gpu_coordinated'] and not p['scientific_pass']
