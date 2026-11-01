"""Pure protocol fixtures, not CPU neural/physical experiments."""
import copy

import numpy as np
import pytest

from experiments.spectral_reachability.audit_reacher_d2p_prefix_capability import roles, probe_commands, expand_roles
from experiments.spectral_reachability import reacher_d2p_masked_action_chart as chart


def plan():
    return {"bank": [{"index":i,"prefix_cost":float(abs(i-4))} for i in range(17)],
        "decisions":{"cem_then_gn4":{"native_index":9}}}


def test_four_roles_preserve_identity_and_never_filter_by_outcome():
    assert roles(plan(),2)=={"common_candidate":9,"feasible_selector":2,"greedy":4,"zero":0}
    p=plan()
    for b in p["bank"]: b["prefix_cost"]=0.
    assert roles(p,0)["greedy"]==0


@pytest.mark.parametrize("bad", [-1,17,True,1.5])
def test_rejects_changed_role_index(bad):
    with pytest.raises(ValueError): roles(plan(),bad)


def test_all_seventeen_prefixes_required():
    p=copy.deepcopy(plan())
    p["bank"].pop()
    with pytest.raises(ValueError): roles(p,2)


@pytest.mark.parametrize("task",list(chart.TASKS))
@pytest.mark.parametrize("scale",[.1,.25])
def test_probe_set_is_complete_reduced_raw_basis(task,scale):
    probes=probe_commands(task,scale)
    n=len(chart.active_columns(task,2))
    assert [i for i,_ in probes]==list(range(n))
    values=np.stack([v for _,v in probes])
    np.testing.assert_array_equal(values,np.eye(n,dtype=np.float32)*np.float32(scale))


def test_probe_contract_cannot_be_outcome_tuned():
    with pytest.raises(ValueError): probe_commands("full_drive",.2)
    with pytest.raises(ValueError): probe_commands("partial_shoulder",.1)


def test_all_programs_expansion_preserves_roles_and_retains_every_index():
    mapping=roles(plan(),2)
    assert expand_roles(mapping,False)==mapping
    expanded=expand_roles(mapping,True)
    assert set(expanded.values())==set(range(17))
    assert all(expanded[k]==v for k,v in mapping.items())
    with pytest.raises(ValueError): expand_roles(mapping,1)
