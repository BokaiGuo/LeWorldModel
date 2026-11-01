"""Synthetic packet contracts only, not GPU or physical evidence."""
import copy
import numpy as np
import pytest

from experiments.spectral_reachability.audit_reacher_d2p_masked_packet_fidelity import family_indices,validate_packets


def packets():
    vectors = {k:np.zeros(192).tolist() for k in ("free","actual_history","observed","goal")}
    return [{"packet":i+1,"vectors":copy.deepcopy(vectors),"costs":{k:0. for k in ("free","actual_history","observed")}} for i in range(3)]


def test_all_complete_packet_coordinates_and_same_first_forecast():
    validate_packets(packets())


@pytest.mark.parametrize("mutation",["partial","order","shape","nan","cost","first"])
def test_partial_changed_or_nonfinite_packets_rejected(mutation):
    p = packets()
    if mutation=="partial": p.pop()
    elif mutation=="order": p[1]["packet"]=1
    elif mutation=="shape": p[0]["vectors"]["free"].pop()
    elif mutation=="nan": p[0]["vectors"]["free"][0]=float("nan")
    elif mutation=="cost": p[1]["costs"]["observed"]=1.
    else:
        p[0]["vectors"]["actual_history"][0]=1.
        p[0]["costs"]["actual_history"]=1/192
    with pytest.raises(ValueError): validate_packets(p)


def test_all_fixed_family_decisions_are_model_only_and_zero_is_literal_index0():
    p = {"bank":[{"index":i} for i in range(17)],"decisions":{"cem_then_gn4":{"native_index":11,"greedy_index":10,"physical_labels_used":False}}}
    assert family_indices(p)=={"common_native":11,"same_bank_greedy":10,"zero_prefix":0}
    p["decisions"]["cem_then_gn4"]["physical_labels_used"]=True
    with pytest.raises(ValueError): family_indices(p)
