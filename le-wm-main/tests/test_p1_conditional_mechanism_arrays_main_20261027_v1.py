"""Algebra tests only; no model or physical capability evidence."""
from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import p1_conditional_mechanism_arrays_main_20261027_v1 as m
from test_p3_goal_patch_consistent_compiler_20261027_v1 import packet, LinearBackend


def test_complete_arrays_and_indices():
    p=packet();q=np.eye(192)
    c=m.compiler.compile_arm(p,q,LinearBackend(),lambda e:None)
    r=m.describe_arm(p,q,c)
    assert len(r['candidates'])==10
    assert r['selected_minus_greedy_stage']>=0
    assert r['selected_minus_greedy_objective']<=1e-12
    assert not r['P1_admission'] and not r['actual_task_gain_measured']


def test_stage_tie_is_not_non_greedy_even_if_selected_index_differs(monkeypatch):
    p=packet();q=np.eye(192)
    c=m.compiler.compile_arm(p,q,LinearBackend(),lambda e:None)
    c=deepcopy(c)
    # Isolate reporting definition with synthetic costs, not a certified compiler.
    for row in c['candidates']:
        row.update(stage_cost=1.,own_reach_value=2.,objective=3.,nonlinear_terminal_cost=1.)
    c['candidates'][1].update(own_reach_value=1.,objective=2.)
    c['winner']=deepcopy(c['candidates'][1])
    monkeypatch.setattr(m.compiler,'validate_compilation',lambda *args:c['winner'])
    r=m.describe_arm(p,q,c)
    assert r['selected_index']==1 and r['greedy_index']==0
    assert not r['strict_model_non_greedy_prefix']


def test_saved_cost_tampering_rejected():
    p=packet();q=np.eye(192)
    c=m.compiler.compile_arm(p,q,LinearBackend(),lambda e:None)
    c['candidates'][0]['nonlinear_terminal_cost']=-10
    with pytest.raises(ValueError):m.describe_arm(p,q,c)
